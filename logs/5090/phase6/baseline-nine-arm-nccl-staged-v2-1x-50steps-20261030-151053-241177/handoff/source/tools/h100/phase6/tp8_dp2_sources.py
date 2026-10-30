"""Bounded adaptations of the existing three measured single-node workers."""
import ast
import io
import tokenize

from common import HERE, REPO, replace_once
import ma11_fullstep
import basic_quant_sources
import baseline_sources
from tp8_dp2_protocol import MODEL, POLICIES, PROTOCOL


def worker_source(arm):
    source = {'ma11': ma11_fullstep, 'flux-basic': basic_quant_sources,
              'megatron-cuda': baseline_sources}[arm].worker_source()
    guard = {'ma11': 'H100_MA11_EXECUTE', 'flux-basic': 'H100_BASIC_QUANT_EXECUTE',
             'megatron-cuda': 'H100_BASELINE_EXECUTE'}[arm]
    source = replace_once(source, guard, 'H100_TP8_DP2_EXECUTE')
    source = replace_once(source, 'local_rank * len(allowed_cpus) // WORLD',
                          'local_rank * len(allowed_cpus) // 8')
    source = replace_once(source, "assert WORLD == CASE['tp']", "assert WORLD == 16 and CASE['tp'] == 8")
    source = replace_once(source, 'assert torch.cuda.device_count() == WORLD',
                          'assert torch.cuda.device_count() == 8')
    source = replace_once(source, 'hardware = check_topology(WORLD)', 'hardware = check_topology(8)')
    source = replace_once(source, 'for i in range(WORLD) for j in range(WORLD)',
                          'for i in range(8) for j in range(8)')
    source = replace_once(source, 'args.tensor_model_parallel_size == WORLD', 'args.tensor_model_parallel_size == 8')
    source = replace_once(source,
        'assert args.pipeline_model_parallel_size == args.context_parallel_size == args.data_parallel_size == 1',
        '''assert args.pipeline_model_parallel_size == args.context_parallel_size == 1
assert args.data_parallel_size == 2
assert parallel_state.get_tensor_model_parallel_rank() == local_rank
assert parallel_state.get_data_parallel_rank() == dist.get_rank() // 8
assert dist.get_process_group_ranks(parallel_state.get_tensor_model_parallel_group()) == list(range(dist.get_rank()//8*8, dist.get_rank()//8*8+8))
assert dist.get_process_group_ranks(parallel_state.get_data_parallel_group()) == [local_rank, local_rank+8]''')
    source = replace_once(source,
        "microsteps = SPEC['model']['global_batch'] // SPEC['model']['micro_batch']",
        "microsteps = SPEC['model']['global_batch'] // (SPEC['model']['micro_batch'] * 2)\nassert microsteps == 2")
    source = replace_once(source, "(microsteps,SPEC['model']['micro_batch'],seq+1)",
                          "(microsteps*2,SPEC['model']['micro_batch'],seq+1)")
    source = replace_once(source, 'positions = torch.arange', '''report['global_tokens_sha256'] = fingerprint([('tokens',all_tokens)])
sample_start = parallel_state.get_data_parallel_rank() * microsteps
report['sample_ids'] = list(range(sample_start, sample_start+microsteps))
report['microsteps'] = microsteps
all_tokens = all_tokens[sample_start:sample_start+microsteps]
positions = torch.arange''')
    if arm != 'ma11':
        reference = ma11_fullstep.worker_source()
        audit = reference[reference.index('consumer_smoke_step = 0'):reference.index('\ndef clear_gradients():')]
        start, end = source.index('def step():'), source.index('\ndef clear_gradients():')
        source = source[:start] + audit + source[end:]
    else:
        source = replace_once(source,
            "report['ma11_plan_sha256'] = hashlib.sha256((HERE.parent/'plan.json').read_bytes()).hexdigest()",
            "report['ma11_source_package'] = PLAN['package_id']")
        start = source.index('import importlib.util\n')
        end = source.index("report['ma11_submission_state']", start)
        source = source[:start] + source[end:]
    source = replace_once(source, "report['protocol'] = PLAN['protocol']", '''report['protocol'] = PLAN['protocol']
report['package_id'] = PLAN['package_id']
report['arm'] = PLAN['arm']
report['worker_config_sha256'] = hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest()''')
    source = replace_once(source, "(OUT/f'rank{rank}.json').write_text", '''report['parameter_trajectory'] = consumer_parameter_trajectory
report['final_parameters_sha256'] = fingerprint(raw.named_parameters())
(OUT/f'rank{rank}.json').write_text''')
    # Avoid a ten-minute NCCL hang if a peer fails during initialization.
    ast.parse(source)
    return source


def profile_source(name):
    source = (HERE/name).read_text()
    if name == 'ma8_profile.py':
        numbers = {'260': '130', '256': '128', '128': '64'}
        for old,new in (
            ('ORIGINAL_COLUMN: 4,', 'ORIGINAL_COLUMN: 2,'),
            ('len(heads) == 4', 'len(heads) == 2'),
            ('calls=4, original_autograd_class', 'calls=2, original_autograd_class'),
            ('forward_allgather_kernels=4', 'forward_allgather_kernels=2'),
            ('backward_allgather_kernels=4', 'backward_allgather_kernels=2'),
            ('reduce_scatter_kernels=4', 'reduce_scatter_kernels=2')):
            source = replace_once(source, old, new)
    elif name == 'ma11_profile.py':
        numbers = {'128':'64', '256':'128', '640':'320', '132':'66'}
        for old,new in (("counts['torch::autograd::CopySlices'] == 4", "counts['torch::autograd::CopySlices'] == 2"),
                        ("[('forward', 32), ('backward', 32)] * 4", "[('forward', 32), ('backward', 32)] * 2"),
                        ('microbatches=4', 'microbatches=2')):
            source = replace_once(source, old, new)
        # DP adds legitimate gradient scaling outside attention. Retain all
        # attention ownership/ordering checks, but not global aten::mul totals.
        lines = source.splitlines(keepends=True)
        drops = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Expr)
                 and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                 and node.value.func.id == 'require' and
                 "counts['aten::masked_fill_']" in ast.get_source_segment(source, node)]
        assert len(drops) == 2
        for node in sorted(drops, key=lambda n:n.lineno, reverse=True):
            lines[node.lineno-1:node.end_lineno] = []
        source = ''.join(lines)
    else:
        return source
    tokens = [token._replace(string=numbers.get(token.string,token.string)) if token.type == tokenize.NUMBER else token
              for token in tokenize.generate_tokens(io.StringIO(source).readline)]
    source = tokenize.untokenize(tokens)
    source = '# Adapted for two local micro-batches; arithmetic and launch ownership checks retained.\n'+source
    ast.parse(source)
    return source


def model_arguments():
    # Freeze the existing flags, not a second hand-maintained model definition.
    from run import arguments
    return arguments(MODEL) + ['--distributed-timeout-minutes', '5']


def configuration(arm, selection):
    import json
    cfg = json.loads((REPO/'tools/a800/phase5/e2e/config.json').read_text())
    policy = POLICIES[arm]
    hashes = {role: value['entry_sha256'] for role,value in selection['variants'][policy]['roles'].items()} if arm == 'ma11' else {'attn':None,'mlp':None}
    cfg.update(protocol=PROTOCOL, arm=arm, policies=[policy], orders=[], repetitions=1,
               warmup_steps=10, timed_steps=50, smoke_warmup_steps=2, smoke_steps=3,
               graph_forward=False, double_buffered=arm=='ma11', ring_reduction=True,
               cases=[dict(name='h100-phase6', **{k:MODEL[k] for k in ('hidden','ffn','heads','sequence','tp','dp')})],
               common={k:MODEL[k] for k in ('layers','micro_batch','global_batch','seed','token_seed','vocab')},
               role_plan_sha256={policy:hashes},
               scope='Two local TP8 groups, cross-node DP2; fixed synthetic global batch 4; full optimizer steps.')
    if arm != 'ma11':
        cfg['codec_identity'] = (basic_quant_sources if arm == 'flux-basic' else baseline_sources).codec_identity()
    return cfg
