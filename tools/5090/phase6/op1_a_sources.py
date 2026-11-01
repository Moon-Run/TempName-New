"""Three isolated OP1 workers, retaining the frozen training and timing AST."""
from pathlib import Path
import shutil

from rtx5090_loader import HERE, H100, REPO, load
import op1_a_protocol as p

base = load('sources')
require, read, write, sha = p.require, p.read, p.write, p.sha
BASIC_ARMS = ('flux-basic', 'flux-basic-interleaved')


def replace(source, old, new):
    require(source.count(old) == 1, 'OP1 source anchor changed: ' + old)
    return source.replace(old, new, 1)


def base_arm(arm):
    require(arm in p.ARMS, 'Unknown OP1 arm')
    return 'a' if arm == 'a' else 'flux-basic'


def implementation(arm):
    base_arm(arm)
    return dict(protocol=p.PROTOCOL,
                producer='fused-taco-sm80-v2' if arm == 'a' else 'flux-gemmonly-sm89-v2',
                gemm_schedule='frozen-A' if arm == 'a' else 'StreamkSK',
                may_change_streamk_reduction_assignment=arm == 'flux-basic-interleaved',
                cta_order='original-flux' if arm == 'flux-basic' else 'rank-rotated-destination-interleaved',
                packet_layout='compact-peer-soa-h128-v1',
                pack_unpack_kernels=0,
                owns_output=True, owns_configuration=True,
                receiver='original-a-scalar' if arm == 'a' else 'basic-decode-reduce-v2',
                gemm_submission_cache=False,
                peer_overlap_preserved=False)


def definition(arm):
    value = base.definition(base_arm(arm))
    value['implementation'] = 'rtx5090-op1-' + arm
    value['op1'] = implementation(arm)
    return value


def taco_source(arm):
    source = base.taco_source(base_arm(arm))
    if arm == 'a':
        source = replace(source,
            "(os.environ['H100_MA2C_ARM'] == 'R',) * 2", '(True, True)')
        source = source.replace("Wrong A/R ownership", "OP1 A requires instance configuration and independent output")
    return source


def adapter_source(arm):
    source = base.adapter_source(base_arm(arm))
    if arm != 'a':
        return source
    source = replace(source,
        "assert self.scope_ops[name].owns_output == (os.environ['H100_MA2C_ARM'] != 'A')",
        'assert self.scope_ops[name].owns_output is True')
    source = replace(source,
        "assert self.scope_ops[name].owns_configuration == (os.environ['H100_MA2C_ARM'] != 'A')",
        'assert self.scope_ops[name].owns_configuration is True')
    # Use the same smoke-only ownership audit already used by both basic arms.
    reference = base.adapter_source('flux-basic')
    begin = reference.index('    def record_smoke_output(')
    end = reference.index('    def stop_audit(', begin)
    source = replace(source, 'import types\n', 'import types\nimport hashlib\n')
    source = replace(source, '        self.scope_ops = {}', '''        self.smoke_outputs = {}
        self.retained_outputs = {}
        self.retention_targets = {}
        self.retention_checks = {}
        self.scope_ops = {}''')
    source = replace(source, '            if module.skip_bias_add:', '''            if os.environ.get('E2E_MODE') == 'smoke':
                adapter.record_smoke_output(name, output)
            if module.skip_bias_add:''')
    return replace(source, '    def stop_audit(self):\n', reference[begin:end] + '    def stop_audit(self):\n')


def worker_source(arm):
    source = base.worker_source(base_arm(arm))
    source = replace(source, "if PLAN['arm'] != 'flux-basic' else {}", "if PLAN['arm'] == 'a' else {}")
    anchor = "report['arm'] = PLAN['arm']"
    source = replace(source, anchor, anchor + "\nreport['op1_implementation'] = PLAN['op1_implementation']")
    if arm == 'a':
        source = replace(source, "report['native_workspace_bytes'] = sum(slot['native_workspace_bytes']", """report['smoke_output_sha256'] = adapter.smoke_outputs
report['retained_output_checks'] = adapter.retention_checks
if MODE == 'smoke':
    assert set(adapter.smoke_outputs) == set(adapter.scope_ops)
    assert set(adapter.retention_checks) == {'attn', 'mlp'}
else:
    assert not adapter.smoke_outputs and not adapter.retention_checks
report['native_workspace_bytes'] = sum(slot['native_workspace_bytes']""")
    base.validate_worker(source)
    return source


def configuration(arm, selection, communication, codec=None):
    cfg = base.configuration(base_arm(arm), selection, communication, codec)
    cfg.update(protocol=p.PROTOCOL, arm=arm, scope_definition=definition(arm),
               op1_implementation=implementation(arm), double_buffered=False,
               scope='RTX5090 OP1: direct peer packets / Flux StreamkSK basic / Flux StreamkSK basic with interleaved tiles and permitted reduction reassignment')
    return cfg


def prepare_workers(build, out, plan):
    import op1_a_build as builder
    build, out = Path(build), Path(out)
    original = builder.base_root(build)
    selection = read(original/'selective/selection-plan.json')
    workers = {}
    for arm in p.ARMS:
        scripts = out/'workers'/arm/'scripts'
        scripts.mkdir(parents=True)
        runtime = (f'TRANSPORT = {p.TRANSPORT!r}\nTRANSPORT_ENV = {p.TRANSPORT_ENV!r}\n'
                   + base.runtime_source())
        generated = {'worker.py': worker_source(arm), 'adapter.py': adapter_source(arm),
                     'taco_support.py': taco_source(arm), 'runtime.py': runtime,
                     'routing.py': Path(base.routing(base_arm(arm)).__file__).read_text()}
        generated['megatron_taco.py'] = (base.basic_quant_sources.codec_source() if arm in BASIC_ARMS else
            (REPO/'tools/a800/phase5/e2e/megatron_taco.py').read_text().replace('self.world==4', 'self.world in (4,8)'))
        for name, source in generated.items():
            (scripts/name).write_text(source)
        paths = [(H100/'ablation4_protocol.py', 'ablation4_protocol.py'),
                 (H100/'ablation4_runtime.py', 'communication.py'),
                 (REPO/'tools/a800/phase4/reference.py', 'reference.py')]
        if arm in BASIC_ARMS:
            paths += [(H100/name, name) for name in ('baseline_codec_validation.py', 'basic_quant_validation.py')]
            paths.append((H100/'ablation6_flux_tensor.py', 'flux_tensor_taco.py'))
        for source, name in paths:
            shutil.copy2(source, scripts/name)
        (scripts/'common.py').write_text('def require(ok, message):\n    if not ok: raise ValueError(message)\n')
        write(scripts/'selection-plan.json', base.basic_quant_prepare.shape_metadata() if arm in BASIC_ARMS else selection)
        codec = None
        if arm in BASIC_ARMS:
            library = original/'codec-basic/libbasic_quant.so'
            codec = dict(path=str(library), sha256=sha(library))
        cfg = configuration(arm, selection, plan['communication'], codec)
        write(scripts/'config.json', cfg)
        write(scripts.parent/'plan.json', plan)
        (scripts/'build').mkdir()
        (scripts/'build'/p.POLICY[arm]).symlink_to(builder.native_root(build, arm), target_is_directory=True)
        workers[arm] = dict(root=str(scripts), config=cfg, definition=definition(arm),
                            libraries=builder.libraries(build, arm), selection=selection,
                            files={path.name: sha(path) for path in scripts.iterdir() if path.is_file()},
                            plan_sha256=sha(scripts.parent/'plan.json'))
    return workers
