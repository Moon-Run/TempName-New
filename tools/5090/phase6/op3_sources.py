"""Combined OP3 worker on the A/MA89/MA11 foundation and frozen training loop."""
import ast
import copy
import hashlib
from pathlib import Path
import shutil

from rtx5090_loader import HERE, H100, REPO, load
import op1_abc_sources as projection
import op2_abc_sources as previous
import op2_abc_build as optimizers
import op3_build as builder
import op3_native as native
import op3_protocol as p
import ma4_f_adapter as consumer

base = load('sources')
require, write, sha = p.require, p.write, p.sha
STACK = previous.STACK
BASE_ARM = 'flux-basic-interleaved'


def implementation(arm):
    require(arm in p.ALL_ARMS, 'Unknown OP3 arm')
    value = dict(projection.implementation(BASE_ARM))
    value.update(protocol=p.PROTOCOL, foundation='OP2-A-MA89-MA11', optimizations=STACK,
        repeated_ma89_installation=False, packet_layout='H128-AoS-136',
        encoder='upstream-cta128' if arm == 'op3-r' else 'warp-owned-four-groups',
        decoder='r-packed-source-prefetch' if p.fused(arm) else 'basic-ring-bf16-h128-direct-v1',
        r_decoder=p.fused(arm), cr3_consumer_fusion=p.fused(arm),
        consumer_rounding='BF16-projection-plus-bias-then-residual',
        original_peer_overlap_preserved=False)
    return value


def definition(arm):
    require(arm in p.ALL_ARMS, 'Unknown OP3 arm')
    value = dict(projection.routing.definition(BASE_ARM))
    value.update(policy=arm, implementation='rtx5090-'+arm+'-v1', experiment=implementation(arm))
    return value


def selection_metadata():
    return dict(projection.selection_metadata(), protocol=p.PROTOCOL)


def routing_source():
    source = (HERE/'op1_abc_routing.py').read_text()
    tree = ast.parse(source)
    replacements = ast.parse(f'''PROTOCOL = {p.PROTOCOL!r}
POLICIES = {p.ARMS!r}
DEFINITIONS = { {arm: definition(arm) for arm in p.ARMS}!r}
def definition(policy):
    require(policy in POLICIES, 'Unknown OP3 policy')
    return DEFINITIONS[policy]
def is_flux_tensor(policy):
    definition(policy)
    return True
def selection_entry(*args):
    raise ValueError('OP3 always quantizes all remote groups; no selection table')
''').body
    functions = {node.name: node for node in replacements if isinstance(node, ast.FunctionDef)}
    assignments = {node.targets[0].id: node for node in replacements if isinstance(node, ast.Assign)}
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            node = functions[node.name]
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            node = assignments.get(node.targets[0].id, node)
        body.append(node)
    body.insert(3, assignments['DEFINITIONS'])
    tree.body = body
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'


def adapter_source(arm):
    require(arm in p.ALL_ARMS, 'Unknown OP3 arm')
    source = previous.adapter_source(BASE_ARM)
    if p.fused(arm):
        source = consumer.adapter_source(source)
        source = projection.native.replace(source, "os.environ.get('H100_MA4_F_ARM') == 'F4'",
                                             "os.environ.get('E2E_POLICY') in ('op3-r', 'op3-combined')")
    return source


def taco_source(arm):
    require(arm in p.ALL_ARMS, 'Unknown OP3 arm')
    tree = ast.parse(base.taco_source('flux-basic'))
    if p.fused(arm):
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TacoForward')
        cls.body.extend(ast.parse('''def forward_consumer(self, input_, weight, bias, residual, reduce_scatter_option=None):
    if not self.owns_output or not self.codec.owns_configuration:
        raise RuntimeError('OP3 requires owned output and configuration')
    return self.codec.forward_consumer(input_, weight, bias, residual)
''').body)
        check = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                                   and n.name == 'check_taco_output'))
        check.name = 'check_consumer_output'
        check.args.args += [ast.arg(arg='bias'), ast.arg(arg='residual')]
        check.args.defaults = []
        index = next(i for i, n in enumerate(check.body) if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'denominator')
        check.body[index:index] = ast.parse('bf16 = residual + (bf16 + bias)\ncoded = residual + (coded + bias)').body
        tree.body.append(check)
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'


def flux_source():
    source = (H100/'ablation6_flux_tensor.py').read_text()
    return source+'''
    def forward_consumer(self, input_, weight, bias, residual):
        assert input_.shape == (self.m, self.k_local) and weight.shape == (self.n, self.k_local)
        assert input_.dtype == weight.dtype == torch.bfloat16
        with torch.profiler.record_function('ablation6.flux_gemm_only') if self.profile else nullcontext():
            self.gemm.forward(input_, weight, output_buf=self.partial, fast_accum=False)
        return self.tensor_codec.reduce(self.partial, bias=bias, residual=residual)

    def ma4_consumer_state(self):
        return self.tensor_codec.state()
'''


def worker_source(arm):
    require(arm in p.ALL_ARMS, 'Unknown OP3 arm')
    source = previous.worker_source(BASE_ARM)
    source = source.replace('op1_abc_implementation', 'op3_implementation')
    source = source.replace('op1_abc_transport', 'op3_transport')
    source = source.replace('op2_implementation', 'op3_optimization_stack')
    source = source.replace('op2_optimization_libraries', 'op3_optimization_libraries')
    source = source.replace('op2_libraries', 'op3_libraries').replace('op2_native', 'op3_native')
    source = source.replace('final ABC', 'final OP3').replace('OP2 optimization', 'OP3 optimization')
    final = "report['op3_consumer_state'] = "+('adapter.consumer_state()' if p.fused(arm) else '{}')+'\n'
    source = projection.native.replace(source, "report['rtx5090_transport'] = ", final+"report['rtx5090_transport'] = ")
    base.validate_worker(source)
    previous.validate_installation(source)
    return source


def codec_identity(arm):
    require(arm in p.ALL_ARMS, 'Unknown OP3 arm')
    value = base.basic_quant_sources.codec_identity()
    value.update(implementation='rtx5090-op3-direct-h128-v1',
        source_sha256=sha(HERE/'op3_codec.py'), reference_codec_sha256=sha(HERE/'op1_abc_codec.py'),
        cuda_source_identity=native.identity(), receiver_implementation=implementation(arm)['decoder'])
    return value


def configuration(arm, selection, communication, codec, plan):
    cfg = projection.configuration(BASE_ARM, selection, communication, codec)
    cfg.pop('op1_abc_implementation')
    cfg.update(protocol=p.PROTOCOL, arm=arm, policies=[arm], scope_definition=definition(arm),
        role_plan_sha256={arm: {'attn': None, 'mlp': None}}, codec_identity=codec_identity(arm),
        op3_implementation=implementation(arm), op3_optimization_stack=STACK,
        op3_optimization_libraries=optimizers.libraries(plan),
        scope='OP3 A plus MA89/MA11: warp-owned encoder with R consumer fusion')
    return cfg


def prepare_workers(build, out, plan):
    build, out = Path(build), Path(out)
    selection, workers = selection_metadata(), {}
    for arm in p.ARMS:
        scripts = out/'workers'/arm/'scripts'
        scripts.mkdir(parents=True)
        generated = {'worker.py': worker_source(arm), 'adapter.py': adapter_source(arm),
            'taco_support.py': taco_source(arm), 'routing.py': routing_source(),
            'runtime.py': f'TRANSPORT = {p.TRANSPORT!r}\nTRANSPORT_ENV = {p.TRANSPORT_ENV!r}\n'+base.runtime_source(),
            'megatron_taco.py': (HERE/'op3_codec.py').read_text(),
            'op3_reference_codec.py': (HERE/'op1_abc_codec.py').read_text(),
            'flux_tensor_taco.py': flux_source(),
            'ma11_adapter.py': base.softmax_adapter_source(plan['environment'])}
        for name, content in generated.items():
            compile(content, name, 'exec')
            (scripts/name).write_text(content)
        for name in ('ablation4_protocol.py', 'ablation4_runtime.py', 'baseline_codec_validation.py',
                     'basic_quant_validation.py', 'ma8_adapter.py', 'ma9_adapter.py'):
            shutil.copy2(H100/name, scripts/('communication.py' if name == 'ablation4_runtime.py' else name))
        shutil.copy2(REPO/'tools/a800/phase4/reference.py', scripts/'reference.py')
        (scripts/'common.py').write_text('def require(ok, message):\n    if not ok: raise ValueError(message)\n')
        write(scripts/'selection-plan.json', selection)
        cfg = configuration(arm, selection, plan['communication'], builder.codec_library(build), plan)
        write(scripts/'config.json', cfg)
        write(scripts.parent/'plan.json', plan)
        (scripts/'build').mkdir()
        (scripts/'build'/arm).symlink_to(builder.native_root(build, p.build_arm(arm)), target_is_directory=True)
        workers[arm] = dict(root=str(scripts), config=cfg, definition=definition(arm),
            libraries=builder.libraries(build, p.build_arm(arm)), selection=selection,
            files={path.name: sha(path) for path in scripts.iterdir() if path.is_file()},
            plan_sha256=sha(scripts.parent/'plan.json'))
    return workers
