"""Independent ablation builds derived from the a–h measured candidate sources.

prepare is CPU only; compile hides all GPUs and never loads a built extension.
No old build, plan, baseline, or shared A800 implementation is edited.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import build
from ablation_plan import BASES, FREEZE, FREEZE_ID, PROTOCOL, definitions, derive, parent_plan, experiment
from common import HERE, REPO, policy, read, replace_once, require, sha, verify_files, write


def codec_source(text):
    """Explicit wire modes on the SAME frozen instance-owned GemmV2 path.

    All-zero/all-remote physical masks drive existing taco_selected branches.
    In BF16 mode config stays enabled: disabling it would route MLP to V3.
    Workspace lifetime, kernel implementation and publication barriers stay fixed.
    """
    text = replace_once(text, '_defer_reuse_barrier=False):',
                        '_defer_reuse_barrier=False, quantization_mode=None):')
    text = replace_once(text, '        self.group = group', '''        if quantization_mode not in ('bf16', 'targeted', 'random_matched', 'all_remote'):
            raise ValueError('Ablation requires an explicit quantization_mode')
        if selected is None:
            raise ValueError('Ablation requires an explicit physical mask, including BF16/G6')
        self.quantization_mode = quantization_mode
        self.group = group''')
    text = replace_once(text, '            mask = mask.to(torch.uint8)', '''            if quantization_mode == 'bf16' and bool(mask.any()):
                raise ValueError('BF16 requires an all-zero physical mask')
            if quantization_mode == 'all_remote':
                per = tiles // self.world
                if any(int(mask[src].sum()) != tiles-per for src in range(self.world)):
                    raise ValueError('all_remote requires EVERY remote physical tile')
            mask = mask.to(torch.uint8)''')
    text = replace_once(text, "        if graph and not hasattr(self.lib, 'taco_allow_capture'):",
                        "        if not self.owns_configuration:\n"
                        "            raise RuntimeError('Ablation requires the frozen instance-owned V2 implementation')\n"
                        "        if graph and not hasattr(self.lib, 'taco_allow_capture'):")
    return text


def sources():
    freeze = read(FREEZE/'manifest.json')
    old = REPO/freeze['candidate']['build']
    require(sha(old/'manifest.json') == freeze['candidate']['manifest_sha256'], 'Measured candidate manifest changed')
    data = read(old/'manifest.json')
    for base in BASES:
        verify_files(old/policy(base), data['variants'][policy(base)]['source_sha256'])
    a, b = (data['variants'][policy(base)]['source_sha256'] for base in BASES)
    different = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    require(different == {'src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp', 'phase6_host.hpp'},
            'Measured bases unexpectedly differ in their compute implementation')
    require(data['source_commit'] == freeze['candidate']['source_commit'], 'Wrong measured source identity')
    for name, commit in freeze['submodules'].items():
        require(subprocess.check_output(['git', '-C', str(REPO/'3rdparty'/name), 'rev-parse', 'HEAD'],
                                        text=True).strip() == commit, f'Wrong {name} revision')
    return freeze, old, data


def prepare(out, plan_path=None, seeds=(17, 29, 43), bases=BASES, protocol=PROTOCOL):
    definitions = experiment(protocol).definitions
    require(bases and len(set(bases)) == len(bases) and set(bases) <= set(BASES), 'Invalid bases')
    for base in bases:
        definitions(base, seeds)
    parent, plan_path, evidence = parent_plan(plan_path)
    freeze, old, measured = sources()
    c = parent['config']
    upstream = 'src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp'
    verify_files(FREEZE/'source', {upstream: freeze['source_files'][upstream]['sha256']})
    upstream_text = (FREEZE/'source'/upstream).read_text()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'plans').mkdir()
    (out/'builds').mkdir()
    shutil.copy2(plan_path, out/'plans/parent.json')
    write(out/'config.json', c)
    record = dict(protocol=protocol, parent_freeze=FREEZE_ID, config=c, bases=list(bases), seeds=list(seeds),
                  parent_plan=dict(path=str(plan_path), sha256=sha(plan_path), evidence=evidence),
                  candidate_source=dict(root=str(old), source_commit=measured['source_commit'],
                                        manifest_sha256=sha(old/'manifest.json')),
                  maintenance_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                  calibration_hardware=parent['hardware'], variants={}, files={}, gpu_validated=False)
    # Freeze the actual source/compile recipe. Model/report controllers are
    # independently frozen by ablation_prepare, not dependencies of nvcc.
    inputs = {str(HERE/name): sha(HERE/name) for name in
              ('candidate.py', 'ablation_plan.py', 'build.py', 'common.py')}
    if protocol != PROTOCOL:
        inputs[str(HERE/'ablation2_plan.py')] = sha(HERE/'ablation2_plan.py')
    for base in bases:
        for key, definition in definitions(base, seeds).items():
            identity = base+'-'+key
            record['variants'][identity] = dict(definition, policy=key)
            if key in ('native', 'native_taco'):
                continue
            entry = derive(parent, definition)
            derived = dict(protocol=protocol, parent_freeze=FREEZE_ID, parent_plan_sha256=sha(plan_path),
                           definition=definition, entry=entry,
                           rule='physical mask independent of arrival; G5 matches source/destination actual counts')
            plan_file = out/'plans'/f'{identity}.json'
            write(plan_file, derived)
            root = out/'builds'/identity
            dest = root/identity
            root.mkdir()
            source = old/policy(base)
            hashes = measured['variants'][policy(base)]['source_sha256']
            # Copy only registered sources; never bring old libraries or caches.
            for name in hashes:
                target = dest/name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source/name, target)
            (dest/'3rdparty').mkdir(exist_ok=True)
            (dest/'3rdparty/cutlass').symlink_to(REPO/'3rdparty/cutlass', target_is_directory=True)
            (dest/upstream).write_text(build.swizzle(upstream_text, c, definition['order'], entry))
            (dest/'phase6_host.hpp').write_text('#pragma once\n'+build.table(entry, c['tp'], False))
            codec = dest/'python/flux/gemm_rs_taco.py'
            codec.write_text(codec_source(codec.read_text()))
            result_hashes = {name: sha(dest/name) for name in hashes}
            changes = {name: dict(before=hashes[name], after=result_hashes[name])
                       for name in hashes if hashes[name] != result_hashes[name]}
            require(set(changes) <= {upstream, 'phase6_host.hpp', 'python/flux/gemm_rs_taco.py'},
                    'Unexpected compute implementation change')
            shutil.copy2(plan_file, root/'derived-plan.json')
            write(root/'config.json', c)
            variant = dict(definition, role='candidate', sampling=False, python=f'{identity}/python',
                           source_sha256=result_hashes, source_changes=changes,
                           parent_source=str(source), parent_source_sha256=hashes,
                           derived_plan_sha256=sha(plan_file))
            info = dict(protocol=protocol, arch='sm90', kind='ablation', status='prepared',
                        config=c, gpu_validated=False, variants={identity: variant}, inputs=inputs,
                        parent_freeze=FREEZE_ID, source_commit=record['maintenance_commit'],
                        parent_source_commit=measured['source_commit'], submodules=freeze['submodules'],
                        parent_plan_sha256=sha(plan_path),
                        files={name: sha(root/name) for name in ('config.json', 'derived-plan.json')})
            write(root/'manifest.json', info)
            record['variants'][identity].update(build=str(root.relative_to(out)),
                                               plan=str(plan_file.relative_to(out)),
                                               derived_plan_sha256=sha(plan_file),
                                               mask_sha256=entry['mask_sha256'])
    record['files'] = {str(p.relative_to(out)): sha(p) for p in (out/'plans').iterdir()}
    record['files']['config.json'] = sha(out/'config.json')
    write(out/'ablation-manifest.json', record)
    return out


def verify(out, built=False):
    out = Path(out).resolve()
    info = read(out/'ablation-manifest.json')
    spec = experiment(info['protocol'])
    require(info['parent_freeze'] == FREEZE_ID, 'Wrong ablation session')
    if info['protocol'] != PROTOCOL:
        require(info['bases'] == ['interleaved'] and info['seeds'] == [], 'Wrong five-group scope')
        expected = {f'interleaved-{k}': d for k, d in spec.definitions().items()}
        require(set(info['variants']) == set(expected), 'Missing/extra five-group variant')
        parent = read(out/'plans/parent.json')
    verify_files(out, info['files'])
    common_compute = None
    for key, v in info['variants'].items():
        if info['protocol'] != PROTOCOL:
            require(all(v.get(k) == value for k, value in expected[key].items()), 'Variant definition changed')
        if v['group'] in ('native', 'native_taco'):
            continue
        root = out/v['build']
        m = read(root/'manifest.json')
        require(m['protocol'] == v.get('build_protocol', info['protocol']) and m['config'] == info['config'] and
                m['parent_plan_sha256'] == info['parent_plan']['sha256'], 'Build/plan mismatch')
        if built:
            require(m['status'] == 'built' and not m.get('diagnostic_only'), 'Unbuilt/diagnostic build')
        require(m['variants'][key]['derived_plan_sha256'] == v['derived_plan_sha256'] ==
                sha(out/v['plan']) == sha(root/'derived-plan.json'), 'Derived plan changed')
        verify_files(root, m['files'])
        verify_files(root/key, m['variants'][key]['source_sha256'])
        if info['protocol'] != PROTOCOL:
            mv = m['variants'][key]
            compute = {name: digest for name, digest in mv['source_sha256'].items()
                       if name not in ('phase6_host.hpp', 'src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp')}
            if common_compute is None:
                common_compute = compute
            require(compute == common_compute, 'All four candidates must share the same compute implementation')
            require(mv['role'] == 'candidate' and mv['sampling'] is False and
                    all(mv.get(k) == value for k, value in expected[key].items()), 'Wrong candidate route')
            require(read(out/v['plan'])['entry'] == derive(parent, expected[key]), 'Physical plan changed')
            require(m['parent_source_commit'] == info['candidate_source']['source_commit'], 'Compute source changed')
            if 'reuse_manifest_sha256' in v:
                require(sha(root/'manifest.json') == v['reuse_manifest_sha256'], 'Reused build manifest changed')
    return info


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('prepare'); q.add_argument('out', type=Path)
    q.add_argument('--plan', type=Path)
    q.add_argument('--seeds', nargs='+', type=int, default=[17, 29, 43])
    q.add_argument('--bases', nargs='+', choices=BASES, default=list(BASES))
    q = sub.add_parser('compile'); q.add_argument('out', type=Path)
    q.add_argument('--cuda-home', type=Path, required=True); q.add_argument('--jobs', type=int, default=2)
    q.add_argument('--nccl-root', type=Path); q.add_argument('--variants', nargs='+')
    a = p.parse_args()
    if a.action == 'prepare':
        print(prepare(a.out, a.plan, a.seeds, a.bases))
    else:
        require(a.jobs > 0, 'jobs must be positive')
        info = verify(a.out)
        keys = a.variants or [k for k, v in info['variants'].items() if v['group'] != 'native']
        require(keys and all(k in info['variants'] and info['variants'][k]['group'] != 'native' for k in keys), 'Unknown build')
        for key in keys:
            root = a.out/info['variants'][key]['build']
            if read(root/'manifest.json')['status'] == 'built':
                continue
            build.compile_build(root, a.cuda_home, a.jobs, a.nccl_root)
