#!/usr/bin/env python3
"""Paired full MA11 versus frozen MA20 order/mask, using the original train loop."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import uuid

import ablation4_runtime as communication
import ma11_fullstep as full
import ma20_build as builder
import ma20_plan as planner
import ma20_worker
from common import HERE, REPO, read, require, sha, write
from run import arguments as model_arguments

PROTOCOL = 'h100-ma20-codec-first-fullstep-v1'
NATIVE_ARM = 'F4-cache-zcount-csoftmax'
POLICY = full.C_POLICY
BLOCKS, STEPS = 8, 50


def _profile_reader():
    """Add path measurements to the unchanged launch/stream ownership checks."""
    source = full._trace_source
    source = full.replace_once(source,
        ".append(dict(launch=launch,cpu_us=call['dur'],decoder_us=decoder['dur'],",
        ".append(dict(launch=launch,cpu_us=call['dur'],decoder_us=decoder['dur'],\n"
        "            producer_us=gemm['dur'],barrier_us=barrier['dur'],\n"
        "            producer_start_to_layernorm_end_us=end-gemm['ts'],")
    source = full.replace_once(source,
        "('cpu_us','decoder_us','decoder_start_to_layernorm_end_us','gemm_end_to_layernorm_end_us')",
        "('cpu_us','decoder_us','decoder_start_to_layernorm_end_us','gemm_end_to_layernorm_end_us',\n"
        "                          'producer_us','barrier_us','producer_start_to_layernorm_end_us')")
    source = full.replace_once(source,
        "layernorm_successors=128,fused_consumer=True,launch=values[0]['launch'],",
        "layernorm_successors=128,fused_consumer=True,launch=values[0]['launch'],\n"
        "                      path_distribution_us={key:dict(median=statistics.median(row[key] for row in values),\n"
        "                          p95=sorted(row[key] for row in values)[math.ceil(.95*len(values))-1],\n"
        "                          maximum=max(row[key] for row in values),sum=sum(row[key] for row in values))\n"
        "                          for key in ('producer_us','barrier_us','producer_start_to_layernorm_end_us')},")
    namespace = dict(full.__dict__)
    exec(compile(source, str(HERE/'ma20_experiment.py') + ':profile_reader', 'exec'), namespace)
    return namespace['trace_projection_records']


trace_forward_records = _profile_reader()


def trace_projection_records(events, tuning):
    forward = trace_forward_records(events, NATIVE_ARM, tuning)
    evidence = full.ma11_profile.trace_composed_records(events, NATIVE_ARM)
    result = {role: dict(value, backward=evidence['row_backward'][role])
              for role,value in forward.items()}
    result.update(activation_cache=evidence['activation_cache'],zero_count=evidence['zero_count'],
                  attention_softmax=evidence['attention_softmax'])
    return result


def schedule(candidate, trials=1, candidate_only=False):
    require(candidate in builder.VARIANTS and candidate != 'MA11', 'Unknown MA20 candidate')
    require(type(trials) is int and trials in (1, 2), 'Use one or two independent trials')
    arms = ('MA11', candidate)
    first = [list(arms[b % 2:] + arms[:b % 2]) for b in range(BLOCKS // 2)]
    orders = first + [row[::-1] for row in first[::-1]]
    return [dict(trial=t, block=b, round=1 if b < 4 else 2, index=2*b+i,
                 arm=arm, mode='timing', directory=f'compare{t}/block{b:02d}-{arm}')
            for t in range(1, trials+1) for b, order in enumerate(orders)
            for i, arm in enumerate(order) if not candidate_only or arm==candidate]


def libraries(bank, arm):
    root = builder.runtime_root(bank, arm)
    paths = [root / n for n in ('libflux_cuda.so', 'libflux_cuda_ths_op.so', 'check_mapping')]
    paths += sorted((root/'python').glob('flux_ths_pybind*.so'))
    require(len(paths) >= 4 and all(p.is_file() for p in paths), 'Incomplete bank runtime: ' + arm)
    return dict(root=str(root), files={str(p.resolve()): sha(p) for p in paths},
                mapping=dict(path=str((root/'check_mapping').resolve()), sha256=sha(root/'check_mapping')))


def controller_inputs():
    paths = {Path(module.__file__).resolve() for module in tuple(sys.modules.values())
             if getattr(module, '__file__', None) and str(module.__file__).endswith('.py')
             and Path(module.__file__).resolve().is_relative_to(HERE)}
    paths.update((full._BASE_PATH, full._F4_PATH, full.bench.TEMPLATE,
                  full.bench.DRIVER, full.bench.base_path))
    paths.update(HERE / name for name in builder.INPUTS)
    return {str(p): sha(p) for p in sorted(paths)}


def candidate_selection(decision,candidate,manual=False):
    require(decision.get('completed') is True,'Operator calibration/heldout did not complete')
    require(candidate in builder.VARIANTS and candidate!='MA11','Unknown MA20 candidate')
    confirmed=decision.get('candidate')==candidate
    require(manual or confirmed,'A completed independent operator selection must nominate this candidate')
    return dict(mode='manual' if manual else 'calibrated',
                operator_selected_candidate=decision.get('candidate'),
                operator_confirmation_passed=confirmed)


def reuse_ma11(source,bank,bundle,decision,trials):
    """Validate saved MA11 records without rerunning or rewriting the old batch.

    Historical controller hashes describe the old controller version. Validate
    its frozen worker, native library and raw rank evidence, not those hashes
    against newly installed launcher/report code.
    """
    source=Path(source).resolve();plan=read(source/'plan.json');result=read(source/'report.json')
    require(plan['protocol']==PROTOCOL and plan['arms']==['MA11',plan['candidate']] and
            plan['trials']==trials and plan['schedule']==schedule(plan['candidate'],trials) and
            (plan['blocks'],plan['timed_steps'],plan['warmup_steps'])==(8,50,10),
            'Reused MA11 requires a complete original paired batch with matching trials')
    require(result['protocol']==PROTOCOL and result['completed'] is True and
            result['run_id']==plan['run_id'] and result['formal_optimizer_steps']==trials*800 and
            result['timing_rank_records']==trials*128,'Reused fullstep report is incomplete')
    require(plan['bundle_sha256']==sha(bank/'ma20-plan.json')==sha(source/'ma20-plan.json') and
            plan['selection_sha256']==sha(source/'operator-selection.json') and
            read(source/'operator-selection.json')==decision,'Reused MA11 belongs to a different operator bank')
    subroot=Path(plan['arm_roots']['MA11']);item=read(subroot/'plan.json')
    require(sha(subroot/'plan.json')==plan['arm_plan_sha256']['MA11'] and
            item['selection']==planner.selection_for(bundle,'MA11') and
            item['libraries'][NATIVE_ARM]==libraries(bank,'MA11'),
            'Reused MA11 plan, mask or library differs')
    paths=[source/name for name in ('plan.json','report.json','ma20-plan.json','operator-selection.json')]
    paths.append(subroot/'plan.json')
    for name,digest in item['source_files'].items():
        path=subroot/name
        require(sha(path)==digest,'Reused MA11 worker changed: '+name)
        paths.append(path)
    ma20_worker.validate_sources((subroot/'scripts/worker.py').read_text())
    samples=[];reference=None
    for job in plan['schedule']:
        if job['arm']!='MA11':continue
        rows=window(source,plan,job)
        for row in rows:validate_operator_identity(row,decision)
        identity=full.identity(rows)
        reference=identity if reference is None else reference
        require(identity==reference,'Reused MA11 hardware/input identity changed')
        sample=dict(job,ms_per_step=max(r['elapsed_seconds'] for r in rows)*1000/STEPS)
        require(sum(saved==sample for saved in result['all_samples'])==1,
                'Reused MA11 timing differs from raw rank records')
        samples.append(sample)
        for rank in range(8):
            path=subroot/'results'/job['directory']/f'rank{rank}.json'
            require(result['files'].get(str(path))==sha(path),'Reused MA11 rank record changed')
            paths.append(path)
    require(len(samples)==trials*BLOCKS,'Incomplete reused MA11 windows')
    return dict(source=str(source),run_id=plan['run_id'],samples=samples,rank_identities=reference,
                formal_optimizer_steps=len(samples)*STEPS,timing_rank_records=len(samples)*8,
                files={str(path):sha(path) for path in paths})


def prepare(args):
    root, bank = args.root.resolve(), args.build.resolve()
    builder.verify(bank, built=True)
    bundle = read(bank/'ma20-plan.json')
    decision = read(args.selection)
    selection_metadata=candidate_selection(decision,args.candidate,manual=getattr(args,'manual_candidate',False))
    require(decision.get('bundle_sha256') == sha(bank/'ma20-plan.json') and
            decision.get('build_manifest_sha256') == sha(bank/builder.MANIFEST),
            'Operator selection belongs to a different plan or compiled bank')
    import ma20_probe
    require(ma20_probe.verify_selection(args.selection.parent) == decision,
            'Operator selection differs from the complete calibration/heldout evidence')
    baseline_source=getattr(args,'reuse_ma11_from',None)
    candidate_only=getattr(args,'candidate_only',False)
    require(not (candidate_only and baseline_source),'Candidate-only mode must not borrow MA11 timing')
    baseline=reuse_ma11(baseline_source,bank,bundle,decision,args.trials) if baseline_source else None
    work = schedule(args.candidate, args.trials,candidate_only=candidate_only or baseline is not None)
    root.mkdir(parents=True, exist_ok=False)
    (root/'arms').mkdir()
    write(root/'ma20-plan.json', bundle)
    shutil.copy2(args.selection, root/'operator-selection.json')
    if baseline is not None:write(root/'reused-ma11.json',baseline)
    megatron = full.snapshot_megatron(args.megatron.resolve(), root/'megatron')
    comm = communication.capture('0')
    arms = [args.candidate] if candidate_only or baseline is not None else ['MA11', args.candidate]
    arm_roots = {}
    for arm in arms:
        subroot = root/'arms'/arm
        scripts = subroot/'scripts'
        scripts.mkdir(parents=True)
        (subroot/'results').mkdir()
        selection = planner.selection_for(bundle, arm)
        library = libraries(bank, arm)
        config = read(REPO/'tools/a800/phase5/e2e/config.json')
        config.update(protocol=full.PROTOCOL, policies=[POLICY], orders=[], repetitions=args.trials,
                      warmup_steps=10, timed_steps=STEPS, smoke_warmup_steps=2, smoke_steps=3,
                      graph_forward=False, double_buffered=True, ring_reduction=True,
                      cases=[dict(name='h100-phase6', **{k: full.MODEL[k] for k in ('hidden','ffn','heads','sequence','tp')})],
                      common={k: full.MODEL[k] for k in ('layers','micro_batch','global_batch','seed','token_seed','vocab')},
                      role_plan_sha256=full.role_hashes(selection), communication=comm,
                      scope='MA20 frozen order/mask on complete MA11; full optimizer steps',
                      ma20_arm=arm, ma20_bundle_sha256=sha(root/'ma20-plan.json'))
        write(scripts/'config.json', config)
        write(scripts/'selection-plan.json', selection)
        (scripts/'adapter.py').write_text(full.adapter_source())
        (scripts/'taco_support.py').write_text(full.taco_source())
        worker = ma20_worker.worker_source()
        ma20_worker.validate_sources(worker)
        (scripts/'worker.py').write_text(worker)
        native = (REPO/'tools/a800/phase5/e2e/megatron_taco.py').read_text()
        (scripts/'megatron_taco.py').write_text(full.replace_once(native, 'self.world==4', 'self.world in (4,8)'))
        full.freeze_helpers(scripts)
        for source, name in ((HERE/'ablation4_protocol.py','routing.py'),
                             (HERE/'runtime.py','runtime.py'),
                             (HERE/'ablation4_runtime.py','communication.py'),
                             (REPO/'tools/a800/phase4/reference.py','reference.py'),
                             (Path(full.bench.__file__),'operator-controller.py'),
                             (Path(ma20_worker.__file__),'ma20_worker.py')):
            shutil.copy2(source, scripts/name)
        native_dir = subroot/'build'
        native_dir.mkdir()
        (native_dir/POLICY).symlink_to(library['root'], target_is_directory=True)
        arm_plan = dict(protocol=full.PROTOCOL, run_id=uuid.uuid4().hex,
                        trial_ids={str(i): uuid.uuid4().hex for i in range(1,args.trials+1)},
                        selection=selection, tuning=read(bank/'tuning.json'),
                        communication_record=comm, communication=comm['environment'],
                        shared_environment=full.SHARED_ENVIRONMENT,
                        libraries={NATIVE_ARM: library}, model=full.MODEL,
                        source_files={str(p.relative_to(subroot)):sha(p) for p in scripts.iterdir()},
                        ma20_arm=arm)
        write(subroot/'plan.json', arm_plan)
        arm_roots[arm] = str(subroot)
    # Pair run identities, seeds, inputs, and binaries; only order/mask differ.
    run_id = uuid.uuid4().hex
    trials = {str(i): uuid.uuid4().hex for i in range(1,args.trials+1)}
    for arm in arms:
        path = Path(arm_roots[arm])/'plan.json'
        item = read(path); item.update(run_id=run_id, trial_ids=trials); write(path,item)
    plan = dict(protocol=PROTOCOL, run_id=run_id, trial_ids=trials, candidate=args.candidate,
                candidate_selection=selection_metadata,operator_selection_source=str(args.selection.resolve()),
                arms=arms, arm_roots=arm_roots, trials=args.trials, blocks=BLOCKS,
                timed_steps=STEPS, warmup_steps=10, schedule=work, build=str(bank),
                bundle_sha256=sha(root/'ma20-plan.json'),
                selection_sha256=sha(root/'operator-selection.json'),
                megatron_snapshot=megatron, communication_record=comm,
                controller_inputs=controller_inputs(),
                arm_plan_sha256={a:sha(Path(p)/'plan.json') for a,p in arm_roots.items()},
                numerical_rule='Independent codec reference <=1%; BF16 <=5%; finite full-model trajectories; no bitwise cross-mask claim',
                timing_rule='All eight paired blocks retained; max eight-rank wall time / 50; paired log-ratio bootstrap')
    if baseline is not None:
        plan.update(comparison_mode='historical_ma11',reused_ma11_sha256=sha(root/'reused-ma11.json'),
                    timing_rule='Candidate only; eight windows per trial; descriptive mean-time ratio to saved MA11; no paired CI')
    elif candidate_only:
        plan.update(comparison_mode='candidate_only',
                    timing_rule='Eight candidate windows per trial; no MA11 run or historical baseline; no speedup or paired CI')
    write(root/'plan.json', plan)
    return root


def verify(root, offline=False):
    root = Path(root).resolve()
    plan = read(root/'plan.json')
    reused=plan.get('comparison_mode')=='historical_ma11'
    single=plan.get('comparison_mode')=='candidate_only'
    require(plan.get('comparison_mode','paired') in ('paired','historical_ma11','candidate_only'),'Unknown comparison mode')
    expected_arms=[plan['candidate']] if reused or single else ['MA11',plan['candidate']]
    require(plan['protocol'] == PROTOCOL and plan['arms'] == expected_arms and
            plan['schedule'] == schedule(plan['candidate'],plan['trials'],candidate_only=reused or single) and
            (plan['blocks'],plan['timed_steps'],plan['warmup_steps']) == (8,50,10), 'Changed MA20 protocol')
    if reused:
        require(sha(root/'reused-ma11.json')==plan['reused_ma11_sha256'],'Reused MA11 snapshot changed')
        for name,digest in read(root/'reused-ma11.json')['files'].items():
            require(sha(name)==digest,'Reused MA11 evidence changed: '+name)
    require(sha(root/'ma20-plan.json') == plan['bundle_sha256'] and
            sha(root/'operator-selection.json') == plan['selection_sha256'], 'Frozen selection changed')
    decision=read(root/'operator-selection.json')
    selection=plan.get('candidate_selection')
    if selection is None:
        candidate_selection(decision,plan['candidate'])
    else:
        require(selection.get('mode') in ('manual','calibrated'),'Unknown candidate selection mode')
        require(selection==candidate_selection(decision,plan['candidate'],selection['mode']=='manual'),
                'Candidate selection provenance changed')
    bundle = read(root/'ma20-plan.json')
    for arm in plan['arms']:
        subroot = Path(plan['arm_roots'][arm])
        require(sha(subroot/'plan.json') == plan['arm_plan_sha256'][arm], 'Arm plan changed')
        item = read(subroot/'plan.json')
        require(item['selection'] == planner.selection_for(bundle,arm), 'Wrong per-arm selection')
        for name,digest in item['source_files'].items():
            require(sha(subroot/name) == digest, 'Frozen worker source changed: '+name)
        require(read(subroot/'scripts/selection-plan.json') == item['selection'], 'Worker selected the wrong mask')
        ma20_worker.validate_sources((subroot/'scripts/worker.py').read_text())
        if not offline:
            require(item['libraries'][NATIVE_ARM] == libraries(Path(plan['build']),arm), 'Changed bank libraries')
    for name,digest in plan['controller_inputs'].items():
        require(sha(name) == digest, 'Controller dependency changed: '+name)
    if not offline:
        builder.verify(Path(plan['build']),built=True)
        full.verify_megatron_snapshot(plan['megatron_snapshot'])
    return plan


def jobs(plan,stage):
    if stage in ('all','preflight','profile'):
        mode='profile' if stage=='profile' else 'smoke'
        for arm in plan['arms']:
            yield dict(arm=arm,mode=mode,trial=0,block=-1,index=-1,directory=mode)
    if stage in ('all','timing'):
        yield from plan['schedule']


def validate_device_masks(row, records):
    actual=row['ma20_device_masks']
    require(set(actual)==set(row['target_modules']), 'Missing device mask coverage')
    for name,slots in actual.items():
        role=full.routing.module_role(name)
        record=records[role]
        counts=list(map(sum,record['entry']['mask']))
        expected=[dict(slot=i,mask_sha256=record['mask_sha256'],selected_per_source=counts,
                       wire_bytes_per_source=[14680064-15360*q for q in counts]) for i in range(2)]
        require(slots==expected, 'Actual device mask/count/bytes differ from frozen '+name)


def window(root,plan,job):
    subroot=Path(plan['arm_roots'][job['arm']])
    item=read(subroot/'plan.json')
    # This validates every original MA11 single-arm invariant, including the
    # independent codec reference. It does not call its bitwise paired report.
    rows=full.window(subroot,item,dict(job,arm=NATIVE_ARM))
    records=item['selection']['variants'][POLICY]['roles']
    for row in rows:
        require(row['ma20_arm']==job['arm'] and row['ma20_bundle_sha256']==plan['bundle_sha256'],
                'Worker used the wrong MA20 arm or bundle')
        validate_device_masks(row,records)
    return rows


def preflight(root,offline=False):
    root=Path(root);plan=verify(root,offline);reference=None;files={};numeric={}
    if plan.get('comparison_mode')=='historical_ma11':
        reference=read(root/'reused-ma11.json')['rank_identities']
    operator_selection=read(root/'operator-selection.json')
    for job in jobs(plan,'preflight'):
        rows=window(root,plan,job)
        for row in rows:validate_operator_identity(row,operator_selection)
        identity=full.identity(rows)
        reference=identity if reference is None else reference
        require(reference==identity,'Fullstep arms differ in hardware, software, initial parameters, RNG or tokens')
        numeric[job['arm']]=dict(losses=rows[0]['losses'],grad_norms=rows[0]['grad_norms'])
        directory=Path(plan['arm_roots'][job['arm']])/'results'/job['directory']
        files.update({str(p):sha(p) for p in directory.glob('rank*.json')})
    gate=dict(protocol=PROTOCOL,run_id=plan['run_id'],completed=True,rank_identities=reference,
              independent_codec_reference_validated=True,full_parameter_gradient_coverage_validated=True,
              device_masks_read_back=True,operator_hardware_software_matched=True,
              bitwise_equivalence_required=False,trajectories=numeric,files=files)
    path=root/'preflight.json'
    if path.exists(): require(read(path)==gate,'Saved preflight differs from evidence')
    else: write(path,gate)
    return gate


def validate_operator_identity(row,selection):
    import ma20_probe
    require(ma20_probe.hardware_identity(row['hardware'])==selection['hardware'],
            'Fullstep host or ordered GPU identity differs from operator calibration')
    actual=dict(torch=row['torch'],cuda=row['cuda'],nccl=list(row['nccl']),
                python=row['ma20_python_version'])
    require(actual==selection['software'],
            'Fullstep Torch/CUDA/NCCL/Python differs from operator calibration')


def profile_summary(root,offline=False):
    root=Path(root);plan=verify(root,offline);gate=preflight(root,offline);profiles={};files={}
    operator_selection=read(root/'operator-selection.json')
    for job in jobs(plan,'profile'):
        rows=window(root,plan,job)
        for row in rows:validate_operator_identity(row,operator_selection)
        require(full.identity(rows)==gate['rank_identities'],'Profile hardware/input identity changed')
        directory=Path(plan['arm_roots'][job['arm']])/'results'/job['directory']
        for rank in range(8):
            path=directory/f'rank{rank}-profile.json.gz'
            with gzip.open(path,'rt') as handle: events=json.load(handle)['traceEvents']
            profiles[f'{job["arm"]}/rank{rank}']=trace_projection_records(
                events,tuning=read(Path(plan['arm_roots'][job['arm']])/'plan.json')['tuning'])
            files[str(path)]=sha(path)
    return dict(protocol=PROTOCOL,run_id=plan['run_id'],completed=True,execution_validated=True,
                profiles=profiles,files=files,
                path_measurements='Per-role 128 launch-correlated producer, publication barrier, and producer-start to LayerNorm-end paths; median/p95/max/sum per rank.',
                limitation='Independent profiled GPU work and per-rank local-clock path spans are not unprofiled full-step saving; ranks are not timestamp-aligned.')


def trajectory_drift(a,b):
    require(len(a)==len(b) and bool(a),'Incomplete paired trajectory')
    return dict(max_abs=max(abs(x-y) for x,y in zip(a,b)),
                relative_l2=math.sqrt(sum((x-y)**2 for x,y in zip(a,b))) /
                            max(math.sqrt(sum(x*x for x in a)),1e-20))


def timing_comparison(values,candidate,historical=False,candidate_only=False):
    result=dict(mean_ms={a:statistics.mean(v) for a,v in values.items()},
                median_ms={a:statistics.median(v) for a,v in values.items()})
    if candidate_only:
        require(set(values)=={candidate} and not historical,'Candidate-only timing contains a baseline')
        result.update(positive_increment=False,comparison_mode='candidate_only',
                      limitation='No MA11 fullstep measured or borrowed; no speedup or paired CI.')
    elif historical:
        result.update(descriptive_percent=100*(1-result['mean_ms'][candidate]/result['mean_ms']['MA11']),
                      positive_increment=False,
                      limitation='Historical baseline; no concurrent pairing or performance significance claim.')
    else:
        paired=full.bench.paired(values[candidate],values['MA11'])
        result.update(paired=paired,positive_increment=paired['percent']>0 and paired['ci95'][0]>0)
    return result


def report(root,offline=False,save=True):
    root=Path(root);plan=verify(root,offline);gate=preflight(root,offline)
    evidence=profile_summary(root,offline)
    operator_selection=read(root/'operator-selection.json')
    baseline=read(root/'reused-ma11.json') if plan.get('comparison_mode')=='historical_ma11' else None
    single=plan.get('comparison_mode')=='candidate_only'
    rows_by_block={};samples=[];files=dict(gate['files']);numerics=defaultdict(list)
    for job in jobs(plan,'timing'):
        rows=window(root,plan,job)
        for row in rows:validate_operator_identity(row,operator_selection)
        require(full.identity(rows)==gate['rank_identities'],'Timing hardware/input identity changed')
        samples.append(dict(job,ms_per_step=max(r['elapsed_seconds'] for r in rows)*1000/STEPS))
        rows_by_block[job['trial'],job['block'],job['arm']]=rows[0]
        numerics[job['arm']].append(dict(mean_loss=statistics.mean(rows[0]['losses']),
            final_loss=rows[0]['losses'][-1],final_grad_norm=rows[0]['grad_norms'][-1]))
        directory=Path(plan['arm_roots'][job['arm']])/'results'/job['directory']
        files.update({str(p):sha(p) for p in directory.glob('rank*.json')})
    comparisons={};drift=[]
    for trial in range(1,plan['trials']+1):
        selected=[s for s in samples if s['trial']==trial]
        values={arm:[s['ms_per_step'] for s in sorted(selected,key=lambda s:s['block']) if s['arm']==arm]
                for arm in plan['arms']}
        if baseline is not None:
            values['MA11']=[s['ms_per_step'] for s in baseline['samples'] if s['trial']==trial]
        comparisons[f'compare{trial}']=timing_comparison(values,plan['candidate'],historical=baseline is not None,candidate_only=single)
        if baseline is not None or single:continue
        for block in range(BLOCKS):
            baseline_row=rows_by_block[trial,block,'MA11'];candidate=rows_by_block[trial,block,plan['candidate']]
            drift.append(dict(trial=trial,block=block,
                loss=trajectory_drift(baseline_row['losses'],candidate['losses']),
                grad_norm=trajectory_drift(baseline_row['grad_norms'],candidate['grad_norms']),
                final_parameters_identical=baseline_row['ma11_final_parameters_sha256']==candidate['ma11_final_parameters_sha256']))
    positive=all(c['positive_increment'] for c in comparisons.values())
    result=dict(protocol=PROTOCOL,run_id=plan['run_id'],completed=True,candidate=plan['candidate'],
                independent_codec_reference_validated=True,execution_validated=evidence['execution_validated'],
                increment_over_MA11_validated=positive,
                performance_increment_over_MA11_validated=positive,
                bitwise_equivalence_required=False,accuracy_equivalence_validated=False,
                numerical_limitation='Codec gates and finite short synthetic training do not establish accuracy equivalence or long-term convergence.',
                comparisons=comparisons,all_samples=samples,paired_trajectory_drift=drift,
                training_summary={a:{key:statistics.mean(row[key] for row in values) for key in values[0]}
                                  for a,values in numerics.items()},
                formal_optimizer_steps=len(samples)*STEPS,timing_rank_records=len(samples)*8,
                selected_configuration=read(root/'ma20-plan.json')['configurations'][plan['candidate']],
                exploratory=plan['trials']==1,files=files)
    if 'candidate_selection' in plan:
        result['candidate_selection']=plan['candidate_selection']
        result['operator_selection_source']=plan['operator_selection_source']
        result['exploratory']=result['exploratory'] or plan['candidate_selection']['mode']=='manual'
    if baseline is not None:
        result.update(comparison_mode='historical_ma11',executed_arms=plan['arms'],exploratory=True,
                      reused_ma11=dict(source=baseline['source'],run_id=baseline['run_id'],
                          formal_optimizer_steps=baseline['formal_optimizer_steps'],
                          timing_rank_records=baseline['timing_rank_records'],
                          evidence_sha256=plan['reused_ma11_sha256']),
                      comparison_limitation='Only the candidate was run in this batch. MA11 times are historical; mean-time differences are descriptive, with no paired CI or validated performance increment.')
    elif single:
        result.update(comparison_mode='candidate_only',executed_arms=plan['arms'],
                      comparison_limitation='Only the selected candidate was measured. No MA11 fullstep or historical baseline; no paired comparison or speedup claim.')
    if save:
        path=root/'report.json'
        if path.exists():require(read(path)==result,'Saved report differs from evidence')
        else:write(path,result)
        lines=['# MA20 candidate only' if single else '# MA20 vs MA11','',f"Candidate: `{plan['candidate']}`; complete optimizer steps: {len(samples)*STEPS}."]
        if 'candidate_selection' in plan:
            selection=plan['candidate_selection']
            lines.extend(['',f"Candidate selection mode: {selection['mode']}; original operator-selected candidate: "
                          f"`{selection['operator_selected_candidate']}`; operator confirmation for this candidate: "
                          f"{selection['operator_confirmation_passed']}."])
        metric='Descriptive time reduction %' if baseline is not None else 'Paired speedup %, 95% CI'
        if baseline is not None or single:lines.extend(['',result['comparison_limitation']])
        lines.extend(['','| Trial | MA20 mean ms | MA20 median ms |','| --- | ---: | ---: |'] if single else
                     ['',f'| Trial | MA11 ms | MA20 ms | {metric} |','| --- | ---: | ---: | --- |'])
        for label,c in comparisons.items():
            if single:
                lines.append(f"| {label} | {c['mean_ms'][plan['candidate']]:.3f} | {c['median_ms'][plan['candidate']]:.3f} |")
                continue
            if baseline is not None:estimate=f"{c['descriptive_percent']:.3f}"
            else:
                p=c['paired'];estimate=f"{p['percent']:.3f} [{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}]"
            lines.append(f"| {label} | {c['mean_ms']['MA11']:.3f} | {c['mean_ms'][plan['candidate']]:.3f} | {estimate} |")
        lines.extend(['',('Performance comparison: not evaluated (no MA11 fullstep control).' if single else
                         f"Performance increment validated: {result['performance_increment_over_MA11_validated']}."),'',
                      '| Arm | Mean loss | Final loss | Final gradient norm |',
                      '| --- | ---: | ---: | ---: |'])
        for arm,item in result['training_summary'].items():
            lines.append(f"| {arm} | {item['mean_loss']:.8g} | {item['final_loss']:.8g} | {item['final_grad_norm']:.8g} |")
        if drift:
            lines.extend(['','| Paired trajectory drift across all blocks | Maximum absolute difference | Maximum relative L2 |',
                          '| --- | ---: | ---: |'])
            for key in ('loss','grad_norm'):
                lines.append(f"| {key} | {max(d[key]['max_abs'] for d in drift):.8g} | {max(d[key]['relative_l2'] for d in drift):.8g} |")
        lines.extend(['','Accuracy equivalence validated: False. No short-run drift threshold is inferred.',
                      '',result['numerical_limitation']])
        (root/'report.md').write_text('\n'.join(lines)+'\n')
    return result


def run(args):
    root=args.root.resolve();plan=verify(root);work=list(jobs(plan,args.stage))
    if not args.execute:
        print(json.dumps(dict(plan_only=True,jobs=work),indent=2));return
    marker=root/f'{args.stage}-started.json'
    require(not marker.exists(),'Do not reuse an executed stage')
    write(marker,dict(run_id=plan['run_id'],jobs=work))
    gate=preflight(root) if args.stage in ('timing','profile') else None
    if args.stage=='timing':
        require(read(root/'profile-summary.json')==profile_summary(root),'Independent profile gate required before timing')
    for job in work:
        if job['mode']=='timing' and gate is None:
            gate=preflight(root)
        verify(root)
        subroot=Path(plan['arm_roots'][job['arm']]);item=read(subroot/'plan.json')
        env=full.bench.environment(subroot,item,NATIVE_ARM,args.python,args.cuda_home)
        env=communication.apply(env,plan['communication_record'])
        library=Path(item['libraries'][NATIVE_ARM]['root'])
        destination=subroot/'results'/job['directory'];destination.mkdir(parents=True,exist_ok=False)
        env.update(H100_MA11_EXECUTE='1',H100_MA11_ARM=NATIVE_ARM,H100_MA11_PROTOCOL=full.bench.SOURCE_PROTOCOL,
            H100_TP='8',H100_SM_COUNT='132',H100_RUN_ID=plan['trial_ids'].get(str(job['trial']),plan['run_id']),
            E2E_POLICY=POLICY,E2E_MODE=job['mode'],E2E_CASE='h100-phase6',E2E_BUILD=str(subroot/'build'),
            E2E_PROCESS_OUT=str(destination),E2E_BLOCK=str(job['block']),E2E_WINDOW=str(job['index']),
            PYTHONPATH=f"{subroot}/scripts:{root}/megatron:{library}/python")
        command=[str(args.python),'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                 str(subroot/'scripts/worker.py'),*model_arguments(full.MODEL)]
        write(destination/'command.json',dict(command=command,job=job,ma20_arm=job['arm']))
        print('START',job['arm'],job['directory'],flush=True)
        with (destination/'process.log').open('x') as log:
            full.bench.run_worker(command,root/'megatron',env,log,args.worker_timeout)
        window(root,plan,job)
    if args.stage=='preflight':preflight(root)
    elif args.stage=='profile':write(root/'profile-summary.json',profile_summary(root))
    elif args.stage=='timing':report(root)


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('root',type=Path);p.add_argument('--build',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--manual-candidate',action='store_true',
                   help='Explicit fullstep experiment; verify prior operator evidence without requiring its winner')
    p.add_argument('--reuse-ma11-from',type=Path,
                   help='Read MA11 from a completed fullstep directory and execute only --candidate')
    p.add_argument('--candidate-only',action='store_true',
                   help='Measure only the selected candidate without running or borrowing an MA11 fullstep baseline')
    p.add_argument('--megatron',type=Path,required=True);p.add_argument('--trials',type=int,choices=(1,2),default=1)
    p=sub.add_parser('run');p.add_argument('root',type=Path)
    p.add_argument('--stage',choices=('preflight','profile','timing'),required=True)
    p.add_argument('--python',type=Path,default=Path(sys.executable));p.add_argument('--cuda-home',type=Path,default=Path('/usr/local/cuda-12.4'))
    p.add_argument('--worker-timeout',type=int,default=1800);p.add_argument('--execute',action='store_true')
    p=sub.add_parser('report');p.add_argument('root',type=Path);p.add_argument('--offline',action='store_true')
    args=parser.parse_args(argv)
    if args.action=='prepare':print(prepare(args))
    elif args.action=='run':run(args)
    else:print(json.dumps(report(args.root,args.offline),indent=2))


if __name__=='__main__':main()
