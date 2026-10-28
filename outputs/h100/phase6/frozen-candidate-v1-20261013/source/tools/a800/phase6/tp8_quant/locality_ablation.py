"""Paired full-step locality/SM controls, with identical physical masks."""
import argparse
import ast
import json
from pathlib import Path
import types

HERE = Path(__file__).resolve().parent


def prepare(args):
    source = HERE.parent/'decoder_ablation.py'
    text = source.read_text()
    variants = [64,16] if args.build80 is None else [64,80,16]
    text = text.replace('{0,3,4,5,6,10,11,12,13}', '{16,64,80}')
    text = text.replace('--nproc_per_node=4', '--nproc_per_node=8').replace('range(4)', 'range(8)')
    text = text.replace('E2E_DECODE_VARIANT','E2E_LOCALITY_CONDITION').replace('decoder_variant','configuration_id')
    text = text.replace('configure_configuration_id', 'configure_decoder_variant')
    text = text.replace('trajectories.setdefault((policy, rank), trajectory)',
                        'trajectories.setdefault((policy, variant, rank), trajectory)')
    text = text.replace("            window = len(windows)", """            build = Path(info['builds'][str(variant)])
            manifest = json.loads((build/'manifest.json').read_text())
            assert sha(build/'manifest.json') == info['build_sha256'][str(variant)]
            window = len(windows)""")
    anchor = "    config = json.loads((root/'scripts/config.json').read_text())"
    assert text.count(anchor) == 1
    text = text.replace(anchor, """    for width, path in info['builds'].items():
        folder = Path(path)
        assert sha(folder/'manifest.json') == info['build_sha256'][width]
        candidate = json.loads((folder/'manifest.json').read_text())
        for name, digest in candidate['files'].items():
            assert sha(folder/name) == digest, name
""" + anchor)
    module = types.ModuleType('tp8_locality_ablation')
    module.__file__ = str(source)
    exec(compile(text,str(source),'exec'),module.__dict__)
    module.prepare(args.source,args.root,args.blocks,variants,args.build64,args.job_id)
    root = args.root.resolve()
    plans = [json.loads((folder/'scripts/selection-plan.json').read_text())
             for folder in (args.source,args.window16_source)]
    assert [p['window'] for p in plans] == [64,16]
    for policy in ('remote_first','interleaved'):
        assert plans[0]['policies'][policy][0]['mask'] == plans[1]['policies'][policy][0]['mask']
    (root/'run.py').write_text(text)
    control = (root/'scripts/decoder_control.py').read_text()
    control = control.replace("variant = int(os.environ['E2E_DECODE_VARIANT'])", 'variant = 0')
    (root/'scripts/decoder_control.py').write_text(control)
    p = root/'scripts/decoder_entry.py'
    s = p.read_text().replace('E2E_DECODE_VARIANT','E2E_LOCALITY_CONDITION').replace('decoder_variant','configuration_id')
    s = s.replace('(0, 3, 4, 5, 6, 10, 11, 12, 13)', '(16,64,80)')
    s = s.replace("report['configuration_id'] = variant", "report['configuration_id'] = variant\nreport['arrival_window'] = 64 if variant == 80 else variant\nreport['gemm_sms'] = 80 if variant == 80 else 0\nreport['decoder_variant'] = 0")
    p.write_text(s)
    assert 'from decoder_control import configure_decoder_variant' in (root/'scripts/taco_support.py').read_text()
    assert 'def configure_decoder_variant()' in (root/'scripts/decoder_control.py').read_text()
    for script in root.rglob('*.py'):
        ast.parse(script.read_text(), filename=str(script))
    p = root/'manifest.json'
    info = json.loads(p.read_text())
    info.update(world=8, builds={'64':str(args.build64.resolve()),'16':str(args.build16.resolve())},
        scope='TP8 full optimizer-step controls: window64/default SMS, window16/default SMS, optional window64/80 SMS; identical mask, BF16 GEMM K32/stage3, original decoder, synchronization and inputs. Frozen BF16 priority, no new calibration. Numerical trajectory repeatability checked within each condition.')
    if args.build80 is not None:
        info['builds']['80'] = str(args.build80.resolve())
    info['conditions'] = {str(v):dict(arrival_window=64 if v == 80 else v,
                                    gemm_sms=80 if v == 80 else 0, decoder_variant=0) for v in variants}
    info['build_sha256'] = {w:module.sha(Path(folder)/'manifest.json') for w,folder in info['builds'].items()}
    info['files'] = {str(f.relative_to(root)):module.sha(f) for f in root.rglob('*') if f.is_file() and f != p}
    info['sources'] = {str(f):module.sha(f) for f in (source,Path(__file__).resolve())}
    p.write_text(json.dumps(info,indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--window16-source',type=Path,required=True)
    p.add_argument('--build64',type=Path,required=True)
    p.add_argument('--build16',type=Path,required=True)
    p.add_argument('--build80',type=Path)
    p.add_argument('--job-id',type=int,required=True)
    p.add_argument('--blocks',type=int,default=4)
    prepare(p.parse_args())
