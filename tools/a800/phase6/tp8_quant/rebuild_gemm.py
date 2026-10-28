"""Clone a TP8 candidate and tune only its selected MLP launch geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main(args):
    base, out = args.base.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for policy in ('remote_arrival', 'interleaved_arrival'):
        old, new = base/policy, out/policy
        m = json.loads((old/'manifest.json').read_text())
        assert m['world'] == 8
        for name, digest in m['files'].items():
            assert sha(old/name) == digest, name
        shutil.copytree(old, new, symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        cc = new/'taco/overlay/gemm_rs/ths_op/gemm_reduce_scatter.cc'
        text = cc.read_text()
        old_gate = '''    const bool tuned_taco_shape = taco.enabled && taco.selected && taco.world == 4 &&
        rt_conf.m() == 8192 && rt_conf.n() == 2048 && rt_conf.k() == 2048;
    if (tuned_taco_shape && taco_gemm_stages())
      hparams_.mainloop_stage() = taco_gemm_stages();'''
        assert text.count(old_gate) == 1
        text = text.replace(old_gate, f'''    const bool tp8_tuned_shape = taco.enabled && taco.selected && taco.world == 8 &&
        rt_conf.m() == 2048 && rt_conf.n() == 2048 && rt_conf.k() == 1024;
    const bool tuned_taco_shape = tp8_tuned_shape || (taco.enabled && taco.selected && taco.world == 4 &&
        rt_conf.m() == 8192 && rt_conf.n() == 2048 && rt_conf.k() == 2048);
    if (tuned_taco_shape && taco_gemm_stages())
      hparams_.mainloop_stage() = taco_gemm_stages();
    else if (tp8_tuned_shape && {args.stages})
      hparams_.mainloop_stage() = {args.stages};
    if (tp8_tuned_shape) cute::get<2>(hparams_.tile_shape()) = {args.tile_k};''')
        old_sms = '''    if (tuned_taco_shape && taco_gemm_avail_sms())
      args.avail_sms = taco_gemm_avail_sms();'''
        assert text.count(old_sms) == 1
        text = text.replace(old_sms, old_sms + f'''
    else if (tp8_tuned_shape && {args.sms}) args.avail_sms = {args.sms};''')
        cc.write_text(text)
        for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so'):
            link = new/'taco/python/flux/lib'/name
            link.unlink()
            link.symlink_to(new/'taco'/name)
        commands = [dict(args=[v.replace(str(old),str(new)) for v in c['args']],
                         cwd=c['cwd'].replace(str(old),str(new))) for c in m['commands']]
        for command in commands:
            argv = command['args']
            if '-o' in argv and Path(argv[argv.index('-o')+1]).name in (
                    'gemm_reduce_scatter.cc.o', 'libflux_cuda_ths_op.so'):
                print('BUILD', policy, Path(argv[argv.index('-o')+1]).name, flush=True)
                subprocess.run(argv, cwd=command['cwd'], check=True)
        m.update(commands=commands, tp8_gemm_tuning=dict(tile_k=args.tile_k, stages=args.stages, sms=args.sms),
                 gemm_parent_manifest=dict(path=str(old/'manifest.json'),sha256=sha(old/'manifest.json')))
        m['inputs'][str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
        m['files'] = {str(p.relative_to(new)):sha(p) for p in new.rglob('*')
                     if p.is_file() and not p.is_symlink() and p.name not in ('manifest.json','build-progress.json')}
        (new/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
        print('TP8 GEMM BUILD COMPLETE',policy,flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('base', type=Path)
    p.add_argument('out', type=Path)
    # The existing TACO epilogue explicitly supports K32 only. Keep its guard.
    p.add_argument('--tile-k', type=int, choices=[32], default=32)
    p.add_argument('--stages', type=int, choices=[0,3,4], default=0)
    p.add_argument('--sms', type=int, choices=[0,64,72,80,84,88,92,96,100,104,108], default=0)
    main(p.parse_args())
