"""Install the MA89 improvements plus MA11 softmax once on each ABC path."""
import ast
from pathlib import Path
import shutil

from rtx5090_loader import HERE, H100, load
import op1_abc_sources as original
import op1_abc_build as builder
import op2_abc_protocol as p
import op2_abc_build as optimizers

base = load('sources')
require, read, write, sha = p.require, p.read, p.write, p.sha
STACK = dict(ma8='retain-owned-forward-column-input-v2',
             ma9='ordered-fp32-gradient-zero-count', ma11='pytorch28-exact-causal-softmax')


def implementation(arm):
    require(arm in p.ARMS, 'Unknown OP2 arm')
    return dict(protocol=p.PROTOCOL, projection=original.implementation(arm),
                optimizations=STACK, repeated_ma89_installation=False, cr3_consumer_fusion=False)


def definition(arm):
    return dict(original.definition(arm), op2=implementation(arm))


def selection_metadata():
    return dict(original.selection_metadata(), protocol=p.PROTOCOL)


def adapter_source(arm):
    require(arm in p.ARMS, 'Unknown OP2 arm')
    return original.adapter_source(arm)


def taco_source(arm):
    require(arm in p.ARMS, 'Unknown OP2 arm')
    return original.taco_source(arm)


def worker_source(arm):
    require(arm in p.ARMS, 'Unknown OP2 arm')
    source = original.worker_source(arm)
    anchor = "adapter = Adapter(raw, POLICY, SPEC['shapes'], SPEC['ring_reduction'])"
    setup = '''
import ctypes
import ma8_adapter
import ma9_adapter
import ma11_adapter
op2_libraries = {}
op2_native = {}
for _mechanism, _identity in PLAN['op2_optimization_libraries'].items():
    _library_path = Path(_identity['path']).resolve()
    _actual_identity = dict(path=str(_library_path), sha256=hashlib.sha256(_library_path.read_bytes()).hexdigest())
    assert _actual_identity == _identity, 'OP2 optimization library changed: ' + _mechanism
    op2_libraries[_mechanism] = _actual_identity
    op2_native[_mechanism] = ctypes.CDLL(str(_library_path))
assert set(op2_native) == {'ma9', 'ma11'}
ma11_activation_setup = ma8_adapter.install_activation_cache(raw, 'F4-cache')
ma11_zero_tracker = ma9_adapter.install_zero_count(optimizer, op2_native['ma9'], MODE)
ma11_softmax_tracker = ma11_adapter.install_softmax(raw, op2_native['ma11'], MODE)
'''
    source = original.native.replace(source, anchor, anchor+setup)
    anchor = "report['rtx5090_transport'] = "
    final = '''report['op2_implementation'] = PLAN['op2_implementation']
report['op2_optimization_libraries'] = op2_libraries
report['ma11_activation_setup'] = ma11_activation_setup
report['ma11_zero_count_state'] = ma11_zero_tracker.state()
report['ma11_softmax_state'] = ma11_softmax_tracker.state()
'''
    source = original.native.replace(source, anchor, final+anchor)
    base.validate_worker(source)
    validate_installation(source)
    return source


def validate_installation(source):
    tree = ast.parse(source)
    expected = ('ma8_adapter.install_activation_cache', 'ma9_adapter.install_zero_count', 'ma11_adapter.install_softmax')
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and ast.unparse(node.func) in expected]
    require(len(calls) == 3 and {ast.unparse(node.func) for node in calls} == set(expected),
            'OP2 must install each optimization exactly once')
    statements = [node for node in tree.body if isinstance(node, ast.Assign)
                  and node.value in calls]
    require(len(statements) == 3 and [ast.unparse(node.value.func) for node in statements] == list(expected),
            'OP2 installation must be unconditional, ordered, and outside training loops')
    first_step = next(node.lineno for node in tree.body if isinstance(node, ast.For)
                      and ast.unparse(node.iter) == 'range(warmup)')
    require(max(node.lineno for node in statements) < first_step, 'OP2 setup happened after first training step')


def prepare_workers(build, out, plan):
    build, out = Path(build), Path(out)
    selection, workers = selection_metadata(), {}
    routing = (HERE/'op1_abc_routing.py').read_text()
    routing = original.native.replace(routing, "PROTOCOL = 'rtx5090-phase6-op1-abc-staged-v1'", f'PROTOCOL = {p.PROTOCOL!r}')
    routing = original.native.replace(routing,
        "POLICIES = ('flux-basic-interleaved', 'flux-basic-arrival', 'flux-basic-selective', 'flux-improved')",
        f'POLICIES = {p.ARMS!r}')
    for arm in p.ARMS:
        scripts = out/'workers'/arm/'scripts'
        scripts.mkdir(parents=True)
        generated = dict(**{'worker.py': worker_source(arm), 'adapter.py': adapter_source(arm),
            'taco_support.py': taco_source(arm), 'routing.py': routing,
            'runtime.py': f'TRANSPORT = {p.TRANSPORT!r}\nTRANSPORT_ENV = {p.TRANSPORT_ENV!r}\n'+base.runtime_source(),
            'megatron_taco.py': original.codec_source(arm),
            'ma11_adapter.py': base.softmax_adapter_source(plan['environment'])})
        for name, content in generated.items():
            (scripts/name).write_text(content)
        for name in ('ablation4_protocol.py', 'ablation4_runtime.py', 'baseline_codec_validation.py',
                     'basic_quant_validation.py', 'ablation6_flux_tensor.py', 'ma8_adapter.py', 'ma9_adapter.py'):
            target = {'ablation4_runtime.py': 'communication.py', 'ablation6_flux_tensor.py': 'flux_tensor_taco.py'}.get(name, name)
            shutil.copy2(H100/name, scripts/target)
        shutil.copy2(original.REPO/'tools/a800/phase4/reference.py', scripts/'reference.py')
        (scripts/'common.py').write_text('def require(ok, message):\n    if not ok: raise ValueError(message)\n')
        write(scripts/'selection-plan.json', selection)
        cfg = original.configuration(arm, selection, plan['communication'], builder.codec_library(build))
        cfg.update(protocol=p.PROTOCOL, scope_definition=definition(arm), op2_implementation=implementation(arm),
                   op2_optimization_libraries=optimizers.libraries(plan), scope='OP2 ABC plus MA8, MA9 and MA11 once')
        write(scripts/'config.json', cfg)
        write(scripts.parent/'plan.json', plan)
        (scripts/'build').mkdir()
        (scripts/'build'/arm).symlink_to(builder.native_root(build, arm), target_is_directory=True)
        workers[arm] = dict(root=str(scripts), config=cfg, definition=definition(arm),
            libraries=builder.libraries(build, arm), selection=selection,
            files={path.name: sha(path) for path in scripts.iterdir() if path.is_file()},
            plan_sha256=sha(scripts.parent/'plan.json'))
    return workers
