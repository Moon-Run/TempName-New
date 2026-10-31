"""Freeze the nine numerical workers with the RTX 5090 NCCL staging transport.

The numerical functions and full-step timing loop retain their original AST
fingerprints. Workspace plumbing, the hardware gate and execution evidence use
the new local producer / NCCL exchange / local decoder protocol.
"""
import ast
import hashlib
from pathlib import Path
import shutil

from rtx5090_loader import load

protocol = load('protocol')
import ablation4_protocol
import basic_quant_prepare
import basic_quant_protocol
import basic_quant_sources
import ma2_ablation_fullstep
import ma4_f_fullstep
import ma8_fullstep
import ma9_fullstep
import ma11_fullstep
from common import HERE, REPO, read, replace_once, require, sha, write

ARMS = protocol.ARMS
CR3_FAMILY = protocol.CR3_FAMILY
QUANT_SELECTIVE = protocol.QUANT_SELECTIVE
PREFIX = {'ma11': 'ma11', 'ma89': 'ma11', 'ma8': 'ma8', 'ma9': 'ma9', 'cr3': 'ma4_f'}


def parent(arm):
    require(arm in ARMS, 'Unknown RTX 5090 baseline arm')
    return {'ma11': ma11_fullstep, 'ma89': ma11_fullstep, 'ma8': ma8_fullstep,
            'ma9': ma9_fullstep, 'cr3': ma4_f_fullstep, 'r': ma2_ablation_fullstep,
            'c': ma2_ablation_fullstep, 'a': ma2_ablation_fullstep,
            'flux-basic': basic_quant_sources}[arm]


def routing(arm):
    return basic_quant_protocol if arm == 'flux-basic' else ablation4_protocol


def definition(arm):
    value = dict(routing(arm).definition(protocol.POLICY[arm]))
    value['implementation'] = ('rtx5090-nccl-staged-v2-basic-receiver' if arm == 'flux-basic'
                               else 'rtx5090-nccl-staged-v2-taco-local-producer')
    value['transport'] = protocol.TRANSPORT
    return value


def taco_source(arm):
    """Keep arithmetic references; obtain only explicitly local workspace pointers."""
    source = parent(arm).taco_source()
    if arm == 'flux-basic':
        tree = ast.parse(source)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TacoForward')
        init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
        branch = next(index for index, node in enumerate(init.body) if isinstance(node, ast.If)
                      and isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Name)
                      and node.test.func.id == 'is_flux_tensor')
        init.body = init.body[:branch+1] + ast.parse("raise RuntimeError('Basic worker only supports the local GemmOnly codec')").body
        return ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    tree = ast.parse(source)
    replacements = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not any(
                isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                and target.value.id == 'self' and target.attr == 'workspace_info' for target in node.targets):
            continue
        call = ast.parse('self.codec.local_workspace_info()', mode='eval').body
        if isinstance(node.value, ast.ListComp):
            node.value = call
        else:
            require(isinstance(node.value, ast.List) and len(node.value.elts) == 1,
                    'Unexpected original single workspace declaration')
            node.value = ast.List(elts=[call], ctx=ast.Load())
        replacements += 1
    require(replacements == 2, 'Workspace source contract changed')
    result = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    require('.peers' not in result, 'Worker still reads legacy peer pointers')
    return result


def adapter_source(arm):
    source = parent(arm).adapter_source()
    # The historical adapter initialized IPC before constructing any projection.
    # Local staging and GemmOnly need no Flux shared-memory process group.
    call = '            flux.init_flux_shm(parallel_state.get_tensor_model_parallel_group())'
    if call in source:
        source = replace_once(source, call, '            # Local NCCL staging: no Flux IPC initialization.')
    require('init_flux_shm(' not in source and 'create_tensor_list(' not in source,
            'Legacy IPC allocation survived in a generated adapter')
    return source


def softmax_adapter_source(environment):
    """Bind dispatch to this campaign, retaining all exact native/GPU audits.

    The original MA11 kernel was derived from PyTorch 2.6. Compatibility with
    the selected newer runtime is established by the existing smoke self-test
    and every full-model forward/backward audit, never by the version string.
    """
    version = environment['torch']
    require(isinstance(version, str) and version, 'Frozen PyTorch identity is required')
    source = (HERE/'ma11_adapter.py').read_text()
    source = replace_once(source, "    if not torch.__version__.split('+')[0] == '2.6.0':",
                          f'    if str(torch.__version__) != {version!r}:')
    source = replace_once(source,
        "raise RuntimeError('MA11 requires the pinned PyTorch 2.6.0 softmax dispatch')",
        "raise RuntimeError('MA11 requires the frozen RTX5090 PyTorch environment and exact smoke validation')")
    source = replace_once(source, '        derivative = native.backward(grad, out, mask)',
                          '        derivative = native.backward(grad, ref.detach(), mask)')
    return replace_once(source,
        "        if not exact_equal(torch, ref, out) or not exact_equal(torch, ref_grad, derivative):\n"
        "            raise RuntimeError('MA11 native arithmetic differs from installed ATen, case ' + str(case))",
        "        forward_equal = exact_equal(torch, ref, out)\n"
        "        backward_equal = exact_equal(torch, ref_grad, derivative)\n"
        "        if not forward_equal or not backward_equal:\n"
        "            fwd_mismatches = (torch.count_nonzero(ref.contiguous().view(torch.int16) !=\n"
        "                out.contiguous().view(torch.int16)).item() if not forward_equal else 0)\n"
        "            bwd_mismatches = (torch.count_nonzero(ref_grad.contiguous().view(torch.int16) !=\n"
        "                derivative.contiguous().view(torch.int16)).item() if not backward_equal else 0)\n"
        "            raise RuntimeError('MA11 native arithmetic differs from installed ATen, case ' + str(case)\n"
        "                + ', forward_bitwise=' + str(forward_equal) + ', backward_bitwise=' + str(backward_equal)\n"
        "                + ', forward_mismatches=' + str(fwd_mismatches)\n"
        "                + ', backward_mismatches=' + str(bwd_mismatches))")


def validate_worker(source):
    tree = ast.parse(source)
    require(not any(isinstance(node, ast.Attribute) and node.attr == 'can_device_access_peer'
                    for node in ast.walk(tree)),
            'Worker peer access must remain diagnostic through runtime.check_topology')
    require(sum(isinstance(node, ast.Assign) and ast.unparse(node) == 'hardware = check_topology(WORLD)'
                for node in tree.body) == 1, 'Missing unique RTX5090 topology gate')
    # The frozen H100 fingerprints use Python 3.10's AST schema. Python 3.12
    # adds type_params=[] even to ordinary functions/classes. Normalize only
    # that empty field on this private tree, retaining nonempty type parameters
    # and every executable node in the original hash contract.
    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and getattr(node, 'type_params', None) == []):
            node._fields = tuple(field for field in node._fields if field != 'type_params')
    functions = {node.name: hashlib.sha256(ast.dump(node).encode()).hexdigest()
                 for node in tree.body if isinstance(node, ast.FunctionDef)}
    changed = sorted(name for name in functions.keys() | ma11_fullstep.BASELINE_FUNCTIONS.keys()
                     if functions.get(name) != ma11_fullstep.BASELINE_FUNCTIONS.get(name))
    require(not changed, 'Original training functions changed: ' + ', '.join(changed))
    timing = [node for node in tree.body if isinstance(node, ast.If)
              and ast.unparse(node.test) == "MODE == 'profile'"]
    require(len(timing) == 1 and hashlib.sha256(ast.dump(timing[0]).encode()).hexdigest()
            == ma11_fullstep.BASELINE_TIMING_SHA256, 'Original full-step timing loop changed')


def worker_source(arm):
    source = parent(arm).worker_source()
    guard = {'ma11': 'H100_MA11_EXECUTE', 'ma89': 'H100_MA11_EXECUTE',
             'ma8': 'H100_MA8_EXECUTE', 'ma9': 'H100_MA9_EXECUTE',
             'cr3': 'H100_MA4_F_EXECUTE', 'r': 'H100_MA2C_EXECUTE',
             'c': 'H100_MA2C_EXECUTE', 'a': 'H100_MA2C_EXECUTE',
             'flux-basic': 'H100_BASIC_QUANT_EXECUTE'}[arm]
    source = replace_once(source, guard, 'RTX5090_BASELINE_EXECUTE')
    if arm == 'ma11':
        source = replace_once(source,
            '    ma11_softmax_tracker = ma11_adapter.install_softmax(raw, ma11_library, MODE)',
            "    ma11_softmax_library = ctypes.CDLL(os.environ['RTX5090_MA11_SOFTMAX_LIBRARY'])\n"
            '    ma11_softmax_tracker = ma11_adapter.install_softmax(raw, ma11_softmax_library, MODE)')
    # The inherited model worker has a second P2P gate after check_topology.
    # Only the RTX5090 overlay removes it: our runtime records peer access and
    # enforces the device/protocol requirements for NCCL staging instead.
    source = replace_once(source,
        'assert all(torch.cuda.can_device_access_peer(i,j) for i in range(WORLD) for j in range(WORLD) if i != j)',
        '# Peer access is diagnostic in hardware; NCCL staging requires no remote GPU pointers.')
    source = replace_once(source, "report['scope_definition'] = definition(POLICY)",
                          "report['scope_definition'] = PLAN['scope_definition']")
    if arm in CR3_FAMILY:
        # Only remove the obsolete post-timing lookup of the H100 operator
        # campaign. The new build manifest validates the native implementation.
        begin = source.index('import importlib.util\n')
        end = source.index("report['" + PREFIX[arm] + "_submission_state']", begin)
        source = source[:begin] + source[end:]
    else:
        original = ma4_f_fullstep.worker_source()
        audit = original[original.index('consumer_smoke_step = 0'):original.index('\ndef clear_gradients():')]
        begin, end = source.index('def step():'), source.index('\ndef clear_gradients():')
        source = source[:begin] + audit + source[end:]
    source = replace_once(source, "report['protocol'] = PLAN['protocol']", """report['protocol'] = PLAN['protocol']
report['arm'] = PLAN['arm']
report['transport'] = PLAN['transport']
report['trial'] = int(os.environ['E2E_TRIAL'])
report['worker_config_sha256'] = hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest()""")
    if arm == 'ma11':
        source = replace_once(source, "report['arm'] = PLAN['arm']", """report['arm'] = PLAN['arm']
_compat_library = Path(os.environ['RTX5090_MA11_SOFTMAX_LIBRARY']).resolve()
report['rtx5090_ma11_softmax'] = dict(path=str(_compat_library),
    sha256=hashlib.sha256(_compat_library.read_bytes()).hexdigest())""")
    source = replace_once(source, "(OUT/f'rank{rank}.json').write_text", """report['parameter_trajectory'] = consumer_parameter_trajectory
report['trainable_parameters'] = consumer_parameter_universe
report['final_parameters_sha256'] = fingerprint(raw.named_parameters())
report['rtx5090_transport'] = ({name: op.codec.transport_state() for name, op in adapter.scope_ops.items()}
                             if PLAN['arm'] != 'flux-basic' else {})
report['native_workspace_bytes'] = sum(slot['native_workspace_bytes']
    for transport in report['rtx5090_transport'].values()
    for slot in transport.get('slots', [transport]))
(OUT/f'rank{rank}.json').write_text""")
    validate_worker(source)
    return source


def runtime_source():
    return '''"""RTX 5090 TP8 NCCL staging worker gate; peer access is diagnostic."""
import os
import socket

def check_topology(world):
    import torch
    if world != 8 or torch.cuda.device_count() != world:
        raise RuntimeError('Expose exactly eight RTX 5090 devices for TP8 DP1')
    devices = []
    for index in range(world):
        prop = torch.cuda.get_device_properties(index)
        if ('5090' not in prop.name or (prop.major, prop.minor) != (12, 0)
                or prop.multi_processor_count != 170):
            raise RuntimeError('Expected full RTX 5090 SM120 with 170 SMs: ' + prop.name)
        devices.append(dict(name=prop.name, uuid=str(prop.uuid), sm_count=prop.multi_processor_count,
                            capability=[prop.major, prop.minor]))
    if len({device['uuid'] for device in devices}) != world:
        raise RuntimeError('Eight distinct GPUs are required')
    peers = [[i == j or torch.cuda.can_device_access_peer(i, j) for j in range(world)]
             for i in range(world)]
    if any(os.environ.get(key) != value for key, value in TRANSPORT_ENV.items()):
        raise RuntimeError('NCCL staging transport settings changed')
    if os.environ.get('RTX5090_TRANSPORT') != TRANSPORT:
        raise RuntimeError('Wrong RTX5090 transport protocol')
    return dict(host=socket.gethostname(), devices=devices, local_rank=int(os.environ['LOCAL_RANK']),
                peer_access=peers, transport=TRANSPORT)
'''


def configuration(arm, selection, communication, codec=None):
    policy = protocol.POLICY[arm]
    cfg = read(REPO/'tools/a800/phase5/e2e/config.json')
    hashes = ({role: value['entry_sha256'] for role, value in selection['variants'][policy]['roles'].items()}
              if arm in QUANT_SELECTIVE else {'attn': None, 'mlp': None})
    cfg.update(protocol=protocol.PROTOCOL, transport=protocol.TRANSPORT, scope_definition=definition(arm),
               arm=arm, policies=[policy], orders=[], repetitions=1,
               **protocol.TIMING, graph_forward=False, double_buffered=arm != 'flux-basic',
               ring_reduction=True,
               cases=[dict(name='rtx5090-phase6', **{k: protocol.MODEL[k]
                            for k in ('hidden', 'ffn', 'heads', 'sequence', 'tp', 'dp')})],
               common={k: protocol.MODEL[k] for k in
                       ('layers', 'micro_batch', 'global_batch', 'seed', 'token_seed', 'vocab')},
               role_plan_sha256={policy: hashes}, communication=communication,
               scope='RTX 5090 TP8 DP1 NCCL-staged v2; retained arithmetic with new transport/synchronization.')
    if arm == 'flux-basic':
        cfg.update(codec_identity=basic_quant_sources.codec_identity(), codec_library=codec)
    return cfg


def prepare_workers(build, out, plan):
    builder = load('build')
    build, out = Path(build), Path(out)
    selection = read(build/'selective/selection-plan.json')
    workers = {}
    for arm in ARMS:
        scripts = out/'workers'/arm/'scripts'
        scripts.mkdir(parents=True)
        module = parent(arm)
        runtime = (f'TRANSPORT = {protocol.TRANSPORT!r}\nTRANSPORT_ENV = {protocol.TRANSPORT_ENV!r}\n'
                   + runtime_source())
        generated = {'worker.py': worker_source(arm), 'adapter.py': adapter_source(arm),
                     'taco_support.py': taco_source(arm), 'runtime.py': runtime}
        generated['megatron_taco.py'] = (basic_quant_sources.codec_source() if arm == 'flux-basic' else
            (REPO/'tools/a800/phase5/e2e/megatron_taco.py').read_text().replace('self.world==4', 'self.world in (4,8)'))
        # Original routing predicates use their implementation names to choose
        # A/R versus selective constructors; campaign identity is recorded in
        # config/worker evidence rather than altering those predicates.
        generated['routing.py'] = Path(routing(arm).__file__).read_text()
        for name, source in generated.items():
            (scripts/name).write_text(source)
        paths = [(HERE/'ablation4_protocol.py', 'ablation4_protocol.py'),
                 (HERE/'ablation4_runtime.py', 'communication.py'),
                 (REPO/'tools/a800/phase4/reference.py', 'reference.py')]
        helpers = {'ma11': ('ma8_adapter.py', 'ma9_adapter.py', 'ma11_adapter.py'),
                   'ma89': ('ma8_adapter.py', 'ma9_adapter.py', 'ma11_adapter.py'),
                   'ma8': ('ma8_adapter.py',), 'ma9': ('ma9_adapter.py',)}.get(arm, ())
        paths += [(HERE/name, name) for name in helpers]
        if arm == 'flux-basic':
            paths += [(HERE/name, name) for name in ('baseline_codec_validation.py', 'basic_quant_validation.py')]
            paths.append((HERE/'ablation6_flux_tensor.py', 'flux_tensor_taco.py'))
        for source, name in paths:
            if name == 'ma11_adapter.py':
                (scripts/name).write_text(softmax_adapter_source(plan['environment']))
            else:
                shutil.copy2(source, scripts/name)
        (scripts/'common.py').write_text('def require(ok, message):\n    if not ok: raise ValueError(message)\n')
        write(scripts/'selection-plan.json', basic_quant_prepare.shape_metadata() if arm == 'flux-basic' else selection)
        codec = None
        if arm == 'flux-basic':
            library = build/'codec-basic/libbasic_quant.so'
            codec = dict(path=str(library), sha256=sha(library))
        cfg = configuration(arm, selection, plan['communication'], codec)
        write(scripts/'config.json', cfg)
        write(scripts.parent/'plan.json', plan)
        (scripts/'build').mkdir()
        (scripts/'build'/protocol.POLICY[arm]).symlink_to(builder.native_root(build, arm), target_is_directory=True)
        workers[arm] = dict(root=str(scripts), config=cfg, definition=definition(arm),
                            libraries=builder.libraries(build, arm), selection=selection,
                            files={p.name: sha(p) for p in scripts.iterdir() if p.is_file()},
                            plan_sha256=sha(scripts.parent/'plan.json'))
    return workers
