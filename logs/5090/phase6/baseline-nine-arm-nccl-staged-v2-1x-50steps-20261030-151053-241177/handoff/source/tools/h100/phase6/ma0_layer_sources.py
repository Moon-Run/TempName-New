"""Freeze the existing single-node workers, changing only measurement control."""
import ast
from pathlib import Path
import shutil

from common import HERE, replace_once
import basic_quant_sources
import ma11_fullstep
import tp8_dp2_sources
from ma0_layer_protocol import ARMS, PROTOCOL, read, write, sha, require
from tp8_dp2_protocol import POLICIES


def worker_source(arm):
    parent = ma11_fullstep if arm == 'ma11' else basic_quant_sources
    text = parent.worker_source()
    guard = 'H100_MA11_EXECUTE' if arm == 'ma11' else 'H100_BASIC_QUANT_EXECUTE'
    text = replace_once(text,guard,'H100_MA0_LAYER_EXECUTE')
    text = replace_once(text,"report['protocol'] = PLAN['protocol']",
                        "report['protocol'] = PLAN['protocol']\nreport['arm'] = PLAN['arm']")
    if arm == 'ma11':
        # Native provenance is checked by the campaign/build manifests. The
        # original standalone MA11 operator controller is not part of this run.
        start = text.index('import importlib.util\n')
        end = text.index("report['ma11_submission_state']",start)
        text = text[:start]+text[end:]
    anchor = "clear_gradients()\nreport['timing_protocol']"
    text = replace_once(text,anchor,"""if MODE == 'timing':
    from ma0_layer_worker import run as measure_layers
    result = measure_layers(globals())
    (OUT/f'rank{rank}.json').write_text(json.dumps(result,indent=2)+'\\n')
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0)

clear_gradients()
report['timing_protocol']""")
    ast.parse(text)
    return text


def prepare_workers(package, build, out, plan, communication):
    workers = {}
    selection = read(package/'plans/selection-plan.json')
    for arm in ARMS:
        scripts = out/'workers'/arm/'scripts'
        parent = ma11_fullstep if arm == 'ma11' else basic_quant_sources
        for name,expected in [('adapter.py',parent.adapter_source()),('taco_support.py',parent.taco_source())]:
            require((package/'workers'/arm/name).read_text() == expected,
                    'Source package adapter differs from this checkout; export a fresh package: '+arm+'/'+name)
        if arm == 'ma11':
            for name in ('ma8_adapter.py','ma9_adapter.py','ma11_adapter.py'):
                require(sha(package/'workers'/arm/name) == sha(HERE/name),
                        'Source package mechanism differs from this checkout: '+name)
        else:
            require((package/'workers'/arm/'megatron_taco.py').read_text() == parent.codec_source(),
                    'Source package basic codec differs from this checkout')
        # Reuse the exact native adapters/dependencies, but not DP2 workers.
        shutil.copytree(package/'workers'/arm,scripts)
        (scripts/'worker.py').write_text(worker_source(arm))
        (scripts/'adapter.py').write_text(parent.adapter_source())
        (scripts/'taco_support.py').write_text(parent.taco_source())
        for name in ('ma0_layer_worker.py',): shutil.copy2(HERE/name,scripts/name)
        if arm == 'ma11':
            for name in ('ma8_adapter.py','ma9_adapter.py','ma11_adapter.py'):
                shutil.copy2(HERE/name,scripts/name)
        else:
            (scripts/'megatron_taco.py').write_text(parent.codec_source())
        cfg = tp8_dp2_sources.configuration(arm,selection)
        cfg['cases'][0]['dp'] = 1
        cfg.update(protocol=PROTOCOL,arm=arm,communication=communication,
                   scope='Single-node TP8 complete sublayers; MA11 versus basic receiver v2.',
                   layer_measurement={k:plan[k] for k in ('warmup','iterations')})
        if arm == 'flux-basic':
            lib = build/'codec-basic/libbasic_quant.so'
            cfg['codec_library'] = dict(path=str(lib),sha256=sha(lib))
        write(scripts/'config.json',cfg)
        write(scripts.parent/'plan.json',plan)
        (scripts/'build').mkdir()
        (scripts/'build'/POLICIES[arm]).symlink_to(build/arm,target_is_directory=True)
        workers[arm] = dict(root=str(scripts),files={p.name:sha(p) for p in scripts.iterdir() if p.is_file()})
    return workers
