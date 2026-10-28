"""Adapt the frozen paired decoder experiment to eight ranks, without editing TP4."""
import argparse
import json
from pathlib import Path
import types

HERE = Path(__file__).resolve().parent


def prepare(args):
    source = HERE.parent/'decoder_ablation.py'
    text = source.read_text()
    text = text.replace('{0,3,4,5,6,10,11,12,13}', '{0,14,15,16,17,18,19}')
    text = text.replace('--nproc_per_node=4', '--nproc_per_node=8').replace('range(4)', 'range(8)')
    module = types.ModuleType('tp8_decoder_ablation')
    module.__file__ = str(source)
    exec(compile(text, str(source), 'exec'), module.__dict__)
    module.prepare(args.source, args.root, args.blocks, args.variants, args.build, args.job_id)
    root = args.root.resolve()
    (root/'run.py').write_text(text)
    for name in ('decoder_entry.py', 'decoder_control.py'):
        p = root/'scripts'/name
        p.write_text(p.read_text().replace('(0, 3, 4, 5, 6, 10, 11, 12, 13)', '(0, 14, 15, 16, 17, 18, 19)'))
    p = root/'manifest.json'
    info = json.loads(p.read_text())
    info['world'] = 8
    info['files'] = {str(f.relative_to(root)):module.sha(f) for f in root.rglob('*')
                     if f.is_file() and f != p}
    info['sources'] = {str(f):module.sha(f) for f in (source, Path(__file__).resolve())}
    p.write_text(json.dumps(info, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--job-id', type=int, required=True)
    p.add_argument('--blocks', type=int, default=4)
    p.add_argument('--variants', type=int, nargs='+', default=[0,18])
    prepare(p.parse_args())
