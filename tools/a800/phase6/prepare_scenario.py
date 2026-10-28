"""Freeze one standard-4H model scenario and all calibrated base-order candidates."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'phase5/e2e'))
from routing import mapping_policy
from model_config import DEFAULT_MODEL, read_model


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(args):
    out,build_root,plan_path=args.out.resolve(),args.build.resolve(),args.plan.resolve()
    assert REPO/'logs/a800' in out.parents
    plan=json.loads(plan_path.read_text());assert plan['schema']==2 and plan['world']==4
    for p,h in plan['inputs'].items():assert sha(Path(p))==h,p
    profile = getattr(args, 'model_config', None)
    if profile is not None or args.hidden is None:
        profile = Path(profile or DEFAULT_MODEL).resolve()
        model = read_model(profile, 4)
    else:
        # Explicit --hidden retains the historical scenario-suite defaults.
        model = dict(layers=12, hidden=args.hidden, ffn=4*args.hidden,
                     heads=args.hidden//64, sequence=2048, micro_batch=4,
                     global_batch=4, vocab=8192, seed=1234, token_seed=4321)
    hidden = args.hidden or model['hidden']
    sequence = args.sequence or model['sequence']
    m = args.tokens_per_microbatch or (model['sequence']*model['micro_batch'] if profile else 8192)
    layers = getattr(args, 'layers', None) or model['layers']
    heads = getattr(args, 'heads', None) or model['heads']
    vocab = getattr(args, 'vocab', None) or model['vocab']
    assert m%sequence==0 and sequence%4==0
    micro=m//sequence;ffn=hidden*4
    global_batch = getattr(args, 'global_batch', None) or (model['global_batch'] if profile else micro)
    assert layers>0 and hidden%heads==0 and heads%4==0 and global_batch%micro==0
    assert m%512==0 and (m//128)*(hidden//128)<=2048
    assert args.decoder!='compact-v1' or (m,hidden)==(8192,2048), 'compact-v1 is a historical exact-shape specialization'
    bases=args.bases or list(plan['policies'])
    assert len(set(bases))==len(bases) and set(bases)<=set(plan['policies'])
    if not args.pilot:
        assert {'remote_first','interleaved'}<=set(bases), 'Formal acceptance requires both candidate routes'
    assert 0 < args.target_percent < 100
    candidates=['remote_arrival_selective' if p=='remote_first' else p+'_arrival_selective' for p in bases]
    policies=['native','original','native_taco','taco_fused']+candidates
    # The underlying compiler writes its first manifest before this builder
    # finishes the GPU mapping checker and arrival-plan metadata.
    for policy in candidates:
        manifest_path=build_root/mapping_policy(policy)/'manifest.json'
        deadline=time.monotonic()+180
        while True:
            try:
                ready=json.loads(manifest_path.read_text())
                if 'arrival_plan_sha256' in ready:
                    assert ready['arrival_plan_sha256']==sha(plan_path)
                    break
            except (FileNotFoundError,json.JSONDecodeError):pass
            if time.monotonic()>deadline:raise RuntimeError(f'Incomplete build: {manifest_path}')
            time.sleep(.2)
    for entries in plan['policies'].values():assert any(s['shape']==[m,hidden,ffn] for s in entries)
    out.mkdir(parents=True,exist_ok=False);(out/'results').mkdir();(out/'scripts').mkdir()
    artifact=REPO/'outputs/a800/phase6'/out.name;build=artifact/'build';build.mkdir(parents=True,exist_ok=False)
    (build/'mapping').mkdir()
    stock=REPO/'outputs/a800/phase3/three-way-build'
    fused=REPO/'outputs/a800/phase4/taco-fused-warp-20261003'
    manifest=dict(files={},policies={},source_manifests={},scope='MLP-only scenario extension, original-order frozen fused-v2 required baseline')
    for policy in policies:
        if policy in ('native','native_taco'):continue
        source=stock/'original' if policy=='original' else fused/'taco' if policy=='taco_fused' else build_root/mapping_policy(policy)/'taco'
        parent=source.parent;src=json.loads((parent/'manifest.json').read_text())
        for name,h in src['files'].items():assert sha(parent/name)==h,(policy,name)
        if policy=='taco_fused':
            assert src['placement']=='fused' and not src['reorder']
            assert src['original_swizzle_sha256']==sha(stock/'original/overlay/gemm_rs/tile_scheduler/threadblock_swizzle.hpp')
        elif policy!='original':assert src['arrival_plan_sha256']==sha(plan_path) and src['mapping']==mapping_policy(policy)
        (build/policy).symlink_to(source,target_is_directory=True)
        prefix=source.name+'/'
        manifest['files'].update({policy+'/'+name[len(prefix):]:h for name,h in src['files'].items() if name.startswith(prefix)})
        manifest['policies'][policy]=dict(library_sha256=sha(source/'libflux_cuda.so'),mapping=mapping_policy(policy))
        manifest['source_manifests'][policy]=dict(path=str(parent/'manifest.json'),sha256=sha(parent/'manifest.json'))
        if policy in candidates:
            name=mapping_policy(policy);checker=parent/'check_mapping'
            (build/'mapping'/name).symlink_to(checker);manifest['files']['mapping/'+name]=sha(checker)
    (build/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    common=HERE.parent/'phase5/e2e'
    for name in ('adapter.py','worker.py','launch.py','summarize.py','routing.py','taco_support.py','megatron_taco.py'):
        shutil.copy2(common/name,out/'scripts'/name)
    cfg=json.loads((common/'config.json').read_text())
    cfg['policies']=policies;cfg['orders']=[policies[i:]+policies[:i] for i in range(len(policies))]
    if args.pilot:cfg['orders']=cfg['orders'][:2]
    cfg['common'].update(layers=layers, micro_batch=micro, global_batch=global_batch,
                         vocab=vocab, seed=model['seed'], token_seed=model['token_seed'])
    cfg['cases']=[dict(name=f'l{layers}-h{hidden}-s{sequence}-mb{micro}-gb{global_batch}-tp4',hidden=hidden,ffn=ffn,heads=heads,sequence=sequence,tp=4)]
    if profile:
        shutil.copy2(profile, out/'scripts/model-profile.json')
    cfg.update(scenario_extension=True,graph_forward=False,double_buffered=True,base_orders=bases,
               selective_decoder=args.decoder,
               acceptance=dict(version='paired-full-step-v2-20261004',target_percent=args.target_percent),
               scope=f'Standard FFN=4H compact/batched GPT; all policies share model, TP, batch, tokens and precision. Full optimizer steps; MLP-only arrival/selective TACO with double-buffered workspaces. {len(bases)} independently calibrated base orders. '+
                     ('Exploratory pilot.' if args.pilot else 'Balanced paired confirmation; no convergence claim.'))
    (out/'scripts/config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    shutil.copy2(plan_path,out/'scripts/selection-plan.json')
    shutil.copy2(plan_path,out/'scripts/arrival-plan.json')
    for source,name in ((HERE.parent/'phase5/validate.py','validate.py'),(HERE.parent/'phase5/validate_megatron.py','validate_megatron.py'),
                        (HERE.parent/'phase4/validate.py','validate_separate.py'),(HERE.parent/'phase4/reference.py','reference.py')):
        shutil.copy2(source,out/'scripts'/name)
    # All baselines must check the actual new model, not only the old H2048 case.
    for name in ('validate_megatron.py','validate_separate.py'):
        path=out/'scripts'/name
        text=path.read_text()
        old='(2048,2048,512),(2048,2048,2048)'
        assert text.count(old)==1
        path.write_text(text.replace(old,f'({m},{hidden},{hidden//4}),({m},{hidden},{ffn//4})'))
    for name in ('campaign.py','verify.py','profile_components.py'):shutil.copy2(common/name,out/name)
    shutil.copy2(HERE/'scenario_mapping.py',out/'check_mappings.py')
    shutil.copy2(HERE/'scenario_report.py',out/'report.py')
    shutil.copy2(HERE/'assess.py',out/'assess.py')
    for p in out.rglob('*.py'):ast.parse(p.read_text())
    mappings=[mapping_policy(p) for p in candidates]
    info=dict(job_id=args.job_id,repo=str(REPO),artifact_root=str(artifact),
              files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},
              build_manifest_sha256=sha(build/'manifest.json'),policies=policies,mappings=mappings,
              codec_checks=mappings+['fused','native_taco'],repetitions=2,blocks=len(cfg['orders']),
              scope=cfg['scope'],original_campaign=str(REPO/'outputs/a800/arrival-v2-tp4-20261003'))
    (out/'submission.json').write_text(json.dumps(info,indent=2)+'\n');print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('out',type=Path)
    p.add_argument('--build',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--model-config',type=Path,help='Default is GPT 6.7B when --hidden is omitted')
    p.add_argument('--hidden',type=int,choices=[512,1024,2048,4096]);p.add_argument('--pilot',action='store_true')
    p.add_argument('--layers',type=int);p.add_argument('--heads',type=int);p.add_argument('--vocab',type=int)
    p.add_argument('--global-batch',type=int)
    p.add_argument('--sequence',type=int)
    p.add_argument('--tokens-per-microbatch',type=int)
    p.add_argument('--bases',nargs='+',choices=['remote_first','interleaved','interleaved_remote','interleaved_remote_group'])
    p.add_argument('--target-percent',type=float,default=4.)
    p.add_argument('--decoder',choices=['legacy','compact-v1'],default='legacy')
    p.add_argument('--job-id',type=int,required=True)
    main(p.parse_args())
