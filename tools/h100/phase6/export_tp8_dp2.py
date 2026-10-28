#!/usr/bin/env python3
"""Export only the frozen sources needed for the two-node three-arm benchmark."""
import argparse
import ast
import io
import inspect
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

from common import HERE, REPO, require as common_require
import ablation6_build
import basic_quant_build
import basic_quant_prepare
import baseline_codec_build
import ma11_build
import tp8_dp2_sources as sources
from tp8_dp2_protocol import ARMS, MODEL, PROTOCOL, MEGATRON_COMMIT, digest, read, require, sha, verify_package, write


def archive_snapshot(repo, target, revision='HEAD', paths=()):
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse',revision],text=True).strip()
    data=subprocess.check_output(['git','-C',str(repo),'archive','--format=tar',commit,*paths])
    target.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:') as handle:
        members=handle.getmembers()
        require(all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and
                    (m.isdir() or m.isfile() or m.issym()) for m in members),'Unexpected dependency archive member')
        for member in members:
            if member.issym():
                require(not Path(member.linkname).is_absolute() and
                        (target/Path(member.name).parent/member.linkname).resolve().is_relative_to(target.resolve()),
                        'Dependency symlink escapes the exported snapshot')
        handle.extractall(target,members=members)
    return dict(commit=commit,archive_sha256=__import__('hashlib').sha256(data).hexdigest(),paths=list(paths))


def copy_native(source,target):
    shutil.copytree(source,target,symlinks=True)
    link=target/'3rdparty/cutlass'
    require(link.is_symlink(),'Expected frozen CUTLASS dependency link')
    link.unlink()
    link.symlink_to('../../../deps/cutlass',target_is_directory=True)


def export(root,megatron):
    root=Path(root).resolve()
    require(not root.exists(),'Use a fresh export directory')
    root.mkdir(parents=True)
    provenance={}
    with tempfile.TemporaryDirectory(prefix='tp8-dp2-export-') as temp:
        temp=Path(temp)
        print('Prepare exact MA11 native sources',flush=True)
        ma=ma11_build.prepare(temp/'ma11')
        selection=read(ma/'selection-plan.json')
        copy_native(ma/'F4-cache-zcount-csoftmax',root/'native/ma11')
        write(root/'plans/selection-plan.json',selection)
        provenance['ma11']=read(ma/ma11_build.MANIFEST)
        shutil.copy2(ma/'TACO_LICENSE.txt',root/'native/ma11/TACO_LICENSE.txt')
        print('Prepare Flux GemmOnly and standalone codec sources',flush=True)
        flux=ablation6_build.prepare(temp/'flux')
        copy_native(flux/'flux-gemm-only',root/'native/flux-basic')
        provenance['flux-basic']=read(flux/'manifest.json')
        for name,builder in [('codec-basic',basic_quant_build),('codec-cuda',baseline_codec_build)]:
            origin=builder.prepare(temp/name)
            shutil.copytree(origin,root/'native'/name)
            provenance[name]=read(origin/'manifest.json')
    print('Archive pinned dependency sources (no Git history)',flush=True)
    deps={}
    deps['Megatron-LM']=archive_snapshot(megatron,root/'deps/Megatron-LM',MEGATRON_COMMIT)
    deps['cutlass']=archive_snapshot(REPO/'3rdparty/cutlass',root/'deps/cutlass',paths=(
        'include','tools/util/include','tools/library/include','tools/profiler/include','LICENSE.txt'))
    deps['nccl']=archive_snapshot(REPO/'3rdparty/nccl',root/'deps/nccl')
    require(all(provenance['flux-basic']['submodules'][name]==deps[name]['commit'] for name in ('cutlass','nccl')),
            'Native snapshots and dependency revisions differ')
    require(all(read_value==deps[name]['commit'] for name,read_value in
                provenance['ma11'].get('submodules',provenance['flux-basic']['submodules']).items()),
            'MA11 dependency revisions differ')
    app=root/'app';app.mkdir()
    for name in ('tp8_dp2_protocol.py','tp8_dp2_train.py','tp8_dp2_build.py','tp8_dp2_profile.py',
                 'test_tp8_dp2_validation.py','test_tp8_dp2_profile.py','test_tp8_dp2_campaign.py'):
        shutil.copy2(HERE/name,app/name)
    shutil.copy2(HERE/'ablation4_runtime.py',app/'communication.py')
    shutil.copy2(HERE/'check_tp8_dp2_link.py',app/'check_tp8_dp2_link.py')
    (app/'common.py').write_text(inspect.getsource(common_require))
    for name in ('basic_quant_validation.py','baseline_codec_validation.py'):
        shutil.copy2(HERE/name,app/name)
    for name in ('ma8_profile.py','ma9_profile.py','ma11_profile.py'):
        (app/name).write_text(sources.profile_source(name))
    write(app/'model-arguments.json',sources.model_arguments())
    for arm in ARMS:
        directory=root/'workers'/arm;directory.mkdir(parents=True)
        (directory/'common.py').write_text(inspect.getsource(common_require))
        parent={'ma11':sources.ma11_fullstep,'flux-basic':sources.basic_quant_sources,
                'megatron-cuda':sources.baseline_sources}[arm]
        (directory/'worker.py').write_text(sources.worker_source(arm))
        (directory/'adapter.py').write_text(parent.adapter_source())
        (directory/'taco_support.py').write_text(parent.taco_source())
        if arm=='ma11':
            for name in ('ma8_adapter.py','ma9_adapter.py','ma11_adapter.py'):
                shutil.copy2(HERE/name,directory/name)
            codec=(REPO/'tools/a800/phase5/e2e/megatron_taco.py').read_text().replace('self.world==4','self.world in (4,8)')
            routing='ablation4_protocol.py'
        else:
            codec=parent.codec_source()
            routing='basic_quant_protocol.py' if arm=='flux-basic' else 'baseline_protocol.py'
            for name in ('baseline_codec_validation.py','basic_quant_validation.py'):
                shutil.copy2(HERE/name,directory/name)
            shutil.copy2(HERE/'ablation6_flux_tensor.py',directory/'flux_tensor_taco.py')
        (directory/'megatron_taco.py').write_text(codec)
        for source,name in ((HERE/routing,'routing.py'),(HERE/'ablation4_protocol.py','ablation4_protocol.py'),
                            (HERE/'runtime.py','runtime.py'),(HERE/'ablation4_runtime.py','communication.py'),
                            (REPO/'tools/a800/phase4/reference.py','reference.py')):
            shutil.copy2(source,directory/name)
        write(directory/'selection-plan.json',basic_quant_prepare.shape_metadata() if arm=='flux-basic' else selection)
        write(directory/'config.json',sources.configuration(arm,selection))
    linkroot=root/'tools/h100/phase6';linkroot.mkdir(parents=True)
    for name in ('check_tp8_dp2_link.py','check_tp8_dp2_link.sh','probe_tp8_dp2.py','probe_tp8_dp2.sh'):
        shutil.copy2(HERE/name,linkroot/name)
    for name,target in [('run_training.sh','app/tp8_dp2_train.py'),('run_link.sh','tools/h100/phase6/check_tp8_dp2_link.py')]:
        (root/name).write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
            'bench_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)\n'
            f'exec python3 "$bench_root/{target}" "$@"\n')
        (root/name).chmod(0o755)
    shutil.copy2(REPO/'docs/design/design-h100/New-Mechanism/tp8-dp2-package.md',root/'README.md')
    write(root/'provenance.json',provenance)
    return seal(root,deps)


def seal(root,deps):
    root=Path(root)
    require(not (root/'build').exists() and not (root/'runs').exists(),'Cannot export a used runtime directory')
    # Only the newly created export is traversed. Nothing walks the repository's
    # logs, outputs, Git objects, environments, or other working directories.
    files={};links={}
    for path in sorted(root.rglob('*')):
        name=str(path.relative_to(root))
        if name in ('package.json','SHA256SUMS'):
            continue
        if path.is_symlink(): links[name]=os.readlink(path)
        elif path.is_file():
            require(not any(part in ('.git','__pycache__') for part in path.parts),'Unexpected export file')
            # Upstream snapshots also contain unused example scripts (including
            # examples for other Python versions). Parse our executable closure;
            # preserve the dependency commit byte-for-byte instead of editing it.
            if path.suffix=='.py' and path.relative_to(root).parts[0] in ('app','workers'):
                ast.parse(path.read_bytes(),filename=name)
            files[name]=sha(path)
    info=dict(protocol=PROTOCOL,model=MODEL,arms=list(ARMS),dependencies=deps,files=files,symlinks=links,
        source_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
        generator_sha256={name:sha(HERE/name) for name in ('export_tp8_dp2.py','tp8_dp2_sources.py')},
        adaptation='TP8 per node, DP2, global batch 4 split into two local microbatches; same native arithmetic. '
                   'Profile call counts halved; global mask/mul totals not gated because DP adds gradient scaling.',
        gpu_validated=False)
    info['package_id']=digest(info)
    write(root/'package.json',info)
    verify_package(root)
    (root/'SHA256SUMS').write_text(''.join(f'{value}  {name}\n' for name,value in files.items())+
                                 f'{sha(root/"package.json")}  package.json\n')
    archive=root.with_suffix('.tar.gz')
    require(not archive.exists(),'Export archive already exists')
    # Numeric ownership avoids looking up thousands of cluster NIS accounts;
    # it also makes the archive independent of the exporting user's IDs.
    subprocess.run(['tar','--numeric-owner','--owner=0','--group=0','--sort=name',
                    '-czf',str(archive),'-C',str(root.parent),root.name],check=True)
    (archive.with_name(archive.name+'.sha256')).write_text(f'{sha(archive)}  {archive.name}\n')
    print(dict(folder=str(root),archive=str(archive),bytes=archive.stat().st_size,
               files=len(files),package_id=info['package_id'],archive_sha256=sha(archive)),flush=True)
    return root


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--megatron',type=Path,default=REPO.parent/'Megatron-LM')
    a=p.parse_args();export(a.out,a.megatron)
