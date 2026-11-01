"""Four workers with unchanged model/backward/optimizer/timing and explicit quantization routes."""
import ast
import hashlib
from pathlib import Path
import shutil
from rtx5090_loader import HERE, H100, REPO, load
import flux_baseline_protocol as p
import flux_baseline_routing as routing
from flux_baseline_native import replace

base = load('sources')
read, write, sha, require = p.read, p.write, p.sha, p.require
BASIC_ARMS = p.BASIC_ARMS


def base_arm(arm):
    require(arm in p.ARMS, 'Unknown Flux baseline arm')
    return p.NATIVE.get(arm, 'flux-basic')


def implementation(arm):
    original = base_arm(arm)
    return dict(protocol=p.PROTOCOL, original_arm=original,
        quantization='none' if arm == 'flux-bf16' else 'all_remote',
        interleaved=False, arrival=False, selective=False, calibration_used=False,
        cta_order='original-Flux-rank-offset' if arm in p.NATIVE else 'original-Flux-GemmOnly',
        decoder={'c': 'original-C-warp', 'a': 'original-A-scalar'}.get(original,
                    'none' if arm == 'flux-bf16' else 'basic-receiver-v2'),
        collective='NCCL-BF16-ReduceScatter' if arm == 'flux-bf16' else 'NCCL-byte-AllToAll',
        workspaces=2 if original == 'c' else 1,
        owns_configuration=original != 'a', owns_output=original != 'a',
        packet_layout='none' if arm == 'flux-bf16' else 'staged-tile' if arm in p.NATIVE else 'peer-soa',
        op1_direct_wire=False, gemm_submission_cache=False, original_peer_overlap_preserved=False)


def definition(arm):
    return dict(routing.definition(arm), experiment=implementation(arm))


def selection_metadata():
    return dict(protocol=p.PROTOCOL, calibration_used=False,
                roles={role: dict(role=role, shape=shape, K_local=shape[2]//8)
                       for role, shape in routing.SHAPES.items()})


def taco_source(arm):
    source = base.taco_source(base_arm(arm))
    if arm in p.NATIVE:
        tree = ast.parse(source)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TacoForward')
        init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
        constructor = 'GemmRSTacoDoubleBuffered' if arm == 'flux-improved' else 'GemmRSTaco'
        init.body = ast.parse(f'''
from flux.gemm_rs_taco import {constructor}
if policy != {arm!r}:
    raise ValueError('Wrong isolated native policy')
self.role, self.plan_sha256 = role, None
self.codec = {constructor}(group, m, n, k_local, placement='fused', quantization_mode='all_remote')
self.owns_output = self.codec.owns_output
self.owns_configuration = self.codec.owns_configuration
assert self.owns_output is {arm == 'flux-improved'} and self.owns_configuration is {arm == 'flux-improved'}
self.workspace_info = {"self.codec.local_workspace_info()" if arm == 'flux-improved' else "[self.codec.local_workspace_info()]"}
''').body
        source = ast.unparse(ast.fix_missing_locations(tree))+'\n'
    elif arm == 'flux-bf16':
        # Compare against an independent unquantized GEMM + NCCL reference.
        # The baseline's 1% protocol budget is retained, with no codec reference.
        start = source.index('@torch.no_grad()\ndef check_taco_output(')
        source = source[:start]+'''@torch.no_grad()
def check_taco_output(partial, actual, group, mask=None):
    if mask is not None:
        raise ValueError('Unquantized Flux must not receive a mask')
    reference = torch.empty_like(actual)
    dist.reduce_scatter_tensor(reference, partial.contiguous(), group=group)
    denominator = max(reference.float().norm().item(), 1e-20)
    delta = actual.float() - reference.float()
    relative = delta.norm().item()/denominator
    error = dict(max_abs=delta.abs().max().item(), relative_l2=relative,
                 protocol_relative_l2=relative, budget_vs_bf16=.05, budget_vs_codec=.01,
                 reference='unquantized_torch_matmul_nccl_reduce_scatter', quantization='none')
    error['passed'] = bool(torch.isfinite(actual).all()) and relative <= .01
    assert error['passed'], error
    return error
'''
    return source


def adapter_source(arm):
    source = base.adapter_source(base_arm(arm))
    if arm not in p.NATIVE:
        return source
    reference = base.adapter_source('flux-basic')
    begin, end = reference.index('    def record_smoke_output('), reference.index('    def stop_audit(')
    source = replace(source, 'import types\n', 'import types\nimport hashlib\n')
    source = replace(source, '        self.scope_ops = {}', '''        self.smoke_outputs = {}
        self.retained_outputs = {}
        self.retention_targets = {}
        self.retention_checks = {}
        self.scope_ops = {}''')
    source = replace(source, '            if module.skip_bias_add:', '''            if os.environ.get('E2E_MODE') == 'smoke':
                adapter.record_smoke_output(name, output)
            if module.skip_bias_add:''')
    return replace(source, '    def stop_audit(self):\n', reference[begin:end]+'    def stop_audit(self):\n')


def codec_source(arm):
    if arm == 'flux-bf16':
        return (HERE/'flux_baseline_bf16.py').read_text()
    source = base.basic_quant_sources.codec_source()
    return replace(source, "self.policy = os.environ['E2E_POLICY']", "self.policy = 'basic'  # Fixed receiver-v2 algorithm.")


def codec_identity(arm):
    source = codec_source(arm)
    value = (base.basic_quant_sources.codec_identity() if arm == 'flux-basic' else
             dict(implementation='unquantized-nccl-bf16-reduce-scatter-v1',
                  reference_sha256=sha(REPO/'tools/a800/phase4/reference.py')))
    value['source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
    return value


def worker_source(arm):
    source = base.worker_source(base_arm(arm))
    source = replace(source, "if PLAN['arm'] != 'flux-basic' else {}",
                     "if PLAN['arm'] in ('flux-improved', 'flux-fused') else {}")
    anchor = "report['arm'] = PLAN['arm']"
    source = replace(source, anchor, anchor+"\nreport['flux_implementation'] = PLAN['flux_implementation']")
    if arm in p.NATIVE:
        source = replace(source, "report['native_workspace_bytes'] = sum(slot['native_workspace_bytes']", """report['smoke_output_sha256'] = adapter.smoke_outputs
report['retained_output_checks'] = adapter.retention_checks
if MODE == 'smoke':
    assert set(adapter.smoke_outputs) == set(adapter.scope_ops)
    assert set(adapter.retention_checks) == {'attn', 'mlp'}
else:
    assert not adapter.smoke_outputs and not adapter.retention_checks
report['native_workspace_bytes'] = sum(slot['native_workspace_bytes']""")
    if arm == 'flux-bf16':
        source = replace(source, "if is_tensor(POLICY) and MODE == 'smoke':\n"
            '    from basic_quant_validation import validate as validate_codec\n'
            '    import reference as codec_reference\n'
            '    codec_validation = validate_codec(standalone_codec, codec_reference, local_rank)',
            '# No codec exists in the BF16 control; distributed and projection references validate it.')
        source = replace(source, "report['codec_validation'] = codec_validation",
            "report['codec_validation'] = codec_validation\n"
            "report['bf16_collectives'] = {name: op.codec.tensor_codec.state() for name, op in adapter.scope_ops.items()}")
    base.validate_worker(source)
    return source


def configuration(arm, selection, communication, codec=None):
    cfg = base.configuration('a' if arm in p.NATIVE else 'flux-basic', {}, communication, codec)
    cfg.update(protocol=p.PROTOCOL, arm=arm, policies=[arm], scope_definition=definition(arm),
               flux_implementation=implementation(arm), double_buffered=arm == 'flux-improved',
               role_plan_sha256={arm: {'attn': None, 'mlp': None}},
               scope='RTX5090 original-order Flux quantization comparison; no calibration or selective masks')
    if arm in BASIC_ARMS:
        cfg.update(codec_identity=codec_identity(arm), codec_library=codec if arm == 'flux-basic' else None)
    return cfg


def prepare_workers(build, out, plan):
    import flux_baseline_build as builder
    build, out = Path(build), Path(out)
    original = builder.base_root(build)
    selection, workers = selection_metadata(), {}
    for arm in p.ARMS:
        scripts = out/'workers'/arm/'scripts'
        scripts.mkdir(parents=True)
        runtime = f'TRANSPORT = {p.TRANSPORT!r}\nTRANSPORT_ENV = {p.TRANSPORT_ENV!r}\n'+base.runtime_source()
        generated = {'worker.py': worker_source(arm), 'adapter.py': adapter_source(arm),
            'taco_support.py': taco_source(arm), 'runtime.py': runtime,
            'routing.py': (HERE/'flux_baseline_routing.py').read_text(),
            'megatron_taco.py': codec_source(arm) if arm in BASIC_ARMS else
                (REPO/'tools/a800/phase5/e2e/megatron_taco.py').read_text().replace('self.world==4', 'self.world in (4,8)')}
        for name, source in generated.items():
            (scripts/name).write_text(source)
        paths = [(H100/'ablation4_protocol.py', 'ablation4_protocol.py'),
                 (H100/'ablation4_runtime.py', 'communication.py'),
                 (REPO/'tools/a800/phase4/reference.py', 'reference.py')]
        if arm in BASIC_ARMS:
            paths.append((H100/'ablation6_flux_tensor.py', 'flux_tensor_taco.py'))
        if arm == 'flux-basic':
            paths += [(H100/name, name) for name in ('baseline_codec_validation.py', 'basic_quant_validation.py')]
        for source, name in paths:
            shutil.copy2(source, scripts/name)
        (scripts/'common.py').write_text('def require(ok, message):\n    if not ok: raise ValueError(message)\n')
        write(scripts/'selection-plan.json', selection)
        codec = None
        if arm == 'flux-basic':
            library = original/'codec-basic/libbasic_quant.so'
            codec = dict(path=str(library), sha256=sha(library))
        cfg = configuration(arm, selection, plan['communication'], codec)
        write(scripts/'config.json', cfg)
        write(scripts.parent/'plan.json', plan)
        (scripts/'build').mkdir()
        (scripts/'build'/arm).symlink_to(builder.native_root(build, arm), target_is_directory=True)
        workers[arm] = dict(root=str(scripts), config=cfg, definition=definition(arm),
            libraries=builder.libraries(build, arm), selection=selection,
            files={path.name: sha(path) for path in scripts.iterdir() if path.is_file()},
            plan_sha256=sha(scripts.parent/'plan.json'))
    return workers
