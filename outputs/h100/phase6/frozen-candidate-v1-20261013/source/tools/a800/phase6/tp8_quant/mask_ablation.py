"""Freeze a two-mask full-step experiment with unchanged GPU permutations."""
import argparse
import ast
import json
from pathlib import Path
import types

HERE = Path(__file__).resolve().parent


def prepare(args):
    source = HERE.parent/'decoder_ablation.py'
    text = source.read_text().replace('{0,3,4,5,6,10,11,12,13}', '{0,1}')
    text = text.replace('--nproc_per_node=4','--nproc_per_node=8').replace('range(4)','range(8)')
    text = text.replace('E2E_DECODE_VARIANT','E2E_MASK_VARIANT').replace("row['decoder_variant']","row['mask_variant']")
    text = text.replace('trajectories.setdefault((policy, rank), trajectory)',
                        'trajectories.setdefault((policy, variant, rank), trajectory)')
    text = text.replace('            window = len(windows)', '''            build = Path(info['builds'][str(variant)])
            manifest = json.loads((build/'manifest.json').read_text())
            assert sha(build/'manifest.json') == info['build_sha256'][str(variant)]
            selection = json.loads(Path(info['plans'][str(variant)]).read_text())
            mask = selection['policies']['remote_first' if policy.startswith('remote_') else 'interleaved'][0]['mask']
            mask_digest = hashlib.sha256(bytes(v for row in mask for v in row)).hexdigest()
            window = len(windows)''')
    text = text.replace('            directory.mkdir()', '''            directory.mkdir()
            shutil.copytree(root/'scripts',directory/'scripts',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            shutil.copy2(info['plans'][str(variant)],directory/'scripts/selection-plan.json')''')
    text = text.replace("str(root/'scripts/decoder_entry.py')", "str(directory/'scripts/decoder_entry.py')")
    text = text.replace("                signature = tuple(row[k]", """                assert len(row['selection_checks']) == 12
                assert all(c['mask_sha256'] == mask_digest and
                           c['wire_bytes'] == 7340032-15360*sum(mask[rank])
                           for c in row['selection_checks'].values())
                signature = tuple(row[k]""")
    anchor = "    config = json.loads((root/'scripts/config.json').read_text())"
    text = text.replace(anchor, '''    for variant,path in info['builds'].items():
        folder = Path(path)
        assert sha(folder/'manifest.json') == info['build_sha256'][variant]
        assert sha(Path(info['plans'][variant])) == info['plan_sha256'][variant]
        candidate = json.loads((folder/'manifest.json').read_text())
        for name,digest in candidate['files'].items(): assert sha(folder/name) == digest,name
''' + anchor)
    module = types.ModuleType('tp8_mask_ablation');module.__file__ = str(source)
    exec(compile(text,str(source),'exec'),module.__dict__)
    module.prepare(args.source,args.root,args.blocks,[0,1],args.build3,args.job_id)
    root = args.root.resolve()
    old_plan = args.source/'scripts/selection-plan.json'
    plans = [json.loads(p.read_text()) for p in (old_plan,args.plan1)]
    for policy in ('remote_first','interleaved'):
        a,b = [p['policies'][policy][0] for p in plans]
        assert a['maps'] == b['maps'] and a['coords'] == b['coords']
        assert all(sum(row) == 1 for row in b['mask'])
        assert all(not b['mask'][r][t] or a['mask'][r][t] for r in range(8) for t in range(256))
    (root/'run.py').write_text(text)
    p = root/'scripts/decoder_control.py'
    p.write_text(p.read_text().replace("variant = int(os.environ['E2E_DECODE_VARIANT'])", 'variant = 0'))
    p = root/'scripts/decoder_entry.py'
    s = p.read_text().replace('E2E_DECODE_VARIANT','E2E_MASK_VARIANT').replace("report['decoder_variant'] = variant","report['mask_variant'] = variant\nreport['decoder_variant'] = 0")
    p.write_text(s.replace('(0, 3, 4, 5, 6, 10, 11, 12, 13)', '(0,1)'))
    for script in root.rglob('*.py'):ast.parse(script.read_text(),filename=str(script))
    p = root/'manifest.json';info = json.loads(p.read_text())
    info.update(world=8,builds={'0':str(args.build3.resolve()),'1':str(args.build1.resolve())},
                plans={'0':str(old_plan.resolve()),'1':str(args.plan1.resolve())},
                scope='Paired TP8 full optimizer steps: 3 versus 1 selected remote tiles per source; unchanged base/arrival mappings, original decoder, BF16 GEMM and synchronization. Mask selected from training passes only; no speed acceptance claim from this ablation.')
    info['build_sha256'] = {v:module.sha(Path(f)/'manifest.json') for v,f in info['builds'].items()}
    info['plan_sha256'] = {v:module.sha(Path(f)) for v,f in info['plans'].items()}
    info['files'] = {str(f.relative_to(root)):module.sha(f) for f in root.rglob('*') if f.is_file() and f != p}
    info['sources'] = {str(f):module.sha(f) for f in (source,Path(__file__).resolve())}
    p.write_text(json.dumps(info,indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--build3',type=Path,required=True)
    p.add_argument('--build1',type=Path,required=True)
    p.add_argument('--plan1',type=Path,required=True)
    p.add_argument('--job-id',type=int,required=True)
    p.add_argument('--blocks',type=int,default=4)
    prepare(p.parse_args())
