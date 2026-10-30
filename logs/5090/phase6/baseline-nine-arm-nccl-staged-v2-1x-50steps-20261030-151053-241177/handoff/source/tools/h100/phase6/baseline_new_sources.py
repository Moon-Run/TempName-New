"""Keep existing numerical workers; change only campaign identity/evidence."""
import ast
import hashlib
from pathlib import Path
import shutil

import basic_quant_prepare
import basic_quant_sources
import baseline_sources
import baseline_protocol
import basic_quant_protocol
import ablation4_protocol
import ma4_f_fullstep
import ma11_fullstep
from common import HERE, REPO, replace_once, require, read, write, sha
from baseline_new_protocol import ARMS, SELECTIVE, POLICY, MODEL, PROTOCOL, TIMING
import baseline_new_build as builder


def parent(arm):
    require(arm in ARMS,'Unknown baseline arm')
    return (ma11_fullstep if arm in ('ma11','ma89') else ma4_f_fullstep if arm=='cr3'
            else basic_quant_sources if arm=='flux-basic' else baseline_sources)


def routing(arm):
    return ablation4_protocol if arm in SELECTIVE else basic_quant_protocol if arm=='flux-basic' else baseline_protocol


def validate_worker(text):
    tree=ast.parse(text)
    functions={n.name:hashlib.sha256(ast.dump(n).encode()).hexdigest()
               for n in tree.body if isinstance(n,ast.FunctionDef)}
    require(functions==ma11_fullstep.BASELINE_FUNCTIONS,'Original training functions changed')
    timing=[n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=="MODE == 'profile'"]
    require(len(timing)==1 and hashlib.sha256(ast.dump(timing[0]).encode()).hexdigest()==
            ma11_fullstep.BASELINE_TIMING_SHA256,'Original full-step timing loop changed')


def worker_source(arm):
    text=parent(arm).worker_source()
    guards={'ma11':'H100_MA11_EXECUTE','ma89':'H100_MA11_EXECUTE','cr3':'H100_MA4_F_EXECUTE',
            'flux-basic':'H100_BASIC_QUANT_EXECUTE','megatron-cuda':'H100_BASELINE_EXECUTE'}
    text=replace_once(text,guards[arm],'H100_BASELINE_NEW_EXECUTE')
    if arm in SELECTIVE:
        prefix='ma4_f' if arm=='cr3' else 'ma11'
        # The native build is verified by the new campaign. Remove only the
        # old, post-timing operator-controller lookup, not any training code.
        begin=text.index('import importlib.util\n')
        end=text.index("report['"+prefix+"_submission_state']",begin)
        text=text[:begin]+text[end:]
    else:
        original=ma4_f_fullstep.worker_source()
        audit=original[original.index('consumer_smoke_step = 0'):original.index('\ndef clear_gradients():')]
        begin,end=text.index('def step():'),text.index('\ndef clear_gradients():')
        text=text[:begin]+audit+text[end:]
    text=replace_once(text,"report['protocol'] = PLAN['protocol']", """report['protocol'] = PLAN['protocol']
report['arm'] = PLAN['arm']
report['trial'] = int(os.environ['E2E_TRIAL'])
report['worker_config_sha256'] = hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest()""")
    text=replace_once(text,"(OUT/f'rank{rank}.json').write_text", """report['parameter_trajectory'] = consumer_parameter_trajectory
report['trainable_parameters'] = consumer_parameter_universe
report['final_parameters_sha256'] = fingerprint(raw.named_parameters())
(OUT/f'rank{rank}.json').write_text""")
    validate_worker(text)
    return text


def configuration(arm, selection, communication, codec=None):
    policy=POLICY[arm]
    cfg=read(REPO/'tools/a800/phase5/e2e/config.json')
    hashes=({r:x['entry_sha256'] for r,x in selection['variants'][policy]['roles'].items()}
            if arm in SELECTIVE else {'attn':None,'mlp':None})
    cfg.update(protocol=PROTOCOL,arm=arm,policies=[policy],orders=[],repetitions=1,**TIMING,
               graph_forward=False,double_buffered=arm in SELECTIVE,ring_reduction=True,
               cases=[dict(name='h100-phase6',**{k:MODEL[k] for k in ('hidden','ffn','heads','sequence','tp','dp')})],
               common={k:MODEL[k] for k in ('layers','micro_batch','global_batch','seed','token_seed','vocab')},
               role_plan_sha256={policy:hashes},communication=communication,
               scope='Single-node TP8 DP1 five-arm full optimizer-step baseline; frozen original arithmetic.')
    if arm not in SELECTIVE:
        cfg.update(codec_identity=parent(arm).codec_identity(),codec_library=codec)
    return cfg


def prepare_workers(build, out, plan):
    build,out=Path(build),Path(out)
    selection=read(build/'selective/selection-plan.json')
    workers={}
    for arm in ARMS:
        scripts=out/'workers'/arm/'scripts';scripts.mkdir(parents=True)
        module=parent(arm)
        for name,source in [('worker.py',worker_source(arm)),('adapter.py',module.adapter_source()),
                            ('taco_support.py',module.taco_source())]:
            (scripts/name).write_text(source)
        codec=((REPO/'tools/a800/phase5/e2e/megatron_taco.py').read_text().replace('self.world==4','self.world in (4,8)')
               if arm in SELECTIVE else module.codec_source())
        (scripts/'megatron_taco.py').write_text(codec)
        paths=[(Path(routing(arm).__file__),'routing.py'),(HERE/'ablation4_protocol.py','ablation4_protocol.py'),
               (HERE/'runtime.py','runtime.py'),(HERE/'ablation4_runtime.py','communication.py'),
               (REPO/'tools/a800/phase4/reference.py','reference.py')]
        if arm in ('ma11','ma89'):
            paths += [(HERE/name,name) for name in ('ma8_adapter.py','ma9_adapter.py','ma11_adapter.py')]
        if arm not in SELECTIVE:
            paths += [(HERE/name,name) for name in ('baseline_codec_validation.py','basic_quant_validation.py')]
            paths.append((HERE/'ablation6_flux_tensor.py','flux_tensor_taco.py'))
        for source,name in paths:shutil.copy2(source,scripts/name)
        (scripts/'common.py').write_text('def require(ok, message):\n    if not ok: raise ValueError(message)\n')
        write(scripts/'selection-plan.json',basic_quant_prepare.shape_metadata() if arm=='flux-basic' else selection)
        codec=None
        if arm not in SELECTIVE:
            lib=build/('codec-basic/libbasic_quant.so' if arm=='flux-basic' else 'codec-cuda/libbaseline_taco.so')
            codec=dict(path=str(lib),sha256=sha(lib))
        cfg=configuration(arm,selection,plan['communication'],codec)
        write(scripts/'config.json',cfg); write(scripts.parent/'plan.json',plan)
        (scripts/'build').mkdir()
        if arm!='megatron-cuda':(scripts/'build'/POLICY[arm]).symlink_to(builder.native_root(build,arm),target_is_directory=True)
        workers[arm]=dict(root=str(scripts),config=cfg,definition=routing(arm).definition(POLICY[arm]),
                         libraries=builder.libraries(build,arm),
                         files={p.name:sha(p) for p in scripts.iterdir() if p.is_file()},
                         plan_sha256=sha(scripts.parent/'plan.json'))
    return workers
