"""Verify two complete TP8 BF16 rounds and report each paired comparison separately."""
import csv,hashlib,json
from pathlib import Path
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def report(root):
 read=lambda p:json.loads(p.read_text())
 info=read(root/'submission.json');cfg=read(root/'scripts/config.json');policies=cfg['policies'];rounds=[];signatures={};records=0
 protocols=[read(root/f'model-{i}/protocol.json') for i in (1,2)]
 assert protocols[1]['plan']['orders']==[list(reversed(o)) for o in reversed(protocols[0]['plan']['orders'])]
 for repetition,protocol in enumerate(protocols,1):
  directory=root/f'model-{repetition}';result=read(directory/'analysis.json');assert result['completed'] and result['windows']==36
  assert len(list(directory.glob('window-*')))==36
  w=0
  for block,order in enumerate(protocol['plan']['orders']):
   for policy in order:
    path=directory/f'window-{w:02d}-{policy}';assert (path/'exit-code.txt').read_text().strip()=='0'
    for rank in range(8):
     row=read(path/f'rank{rank}.json');assert row['completed'] and row['passed'] and row['job_id']==str(info['job_id'])
     assert row['host']==info['node'] and row['rank']==row['local_rank']==row['tp_rank']==rank and row['dp_rank']==0
     assert row['tp_group_ranks']==list(range(8)) and row['dp_group_ranks']==[rank]
     assert row['padded_vocab_size']==9216 and row['tokens_per_optimizer_step']==8192 and row['timed_steps']==20
     assert row['warmup_steps']==10 and row['skips']==row['fallback_calls']==0
     assert row['policy']==policy and row['window']==w and row['block']==block
     if policy!='native':
      for lib in ['libflux_cuda.so','libflux_cuda_ths_op.so']:
       expected=(Path(info['build'])/policy/lib).resolve();assert row[lib]['path']==str(expected) and row[lib]['sha256']==sha(expected)
     sig=tuple(row[k] for k in ['initial_parameters_sha256','initial_rng_sha256','tokens_sha256','gpu_uuid','host','timing_protocol'])
     assert signatures.setdefault(rank,sig)==sig
     records+=1
    w+=1
  rounds.append(dict(round=repetition,**result))
 def mapping(policy,k):
  with (root/'preflight'/f'mapping-{policy}-k{k}.csv').open() as f:
   return [tuple(int(row[x]) for x in ['rank','index','m','n','destination']) for row in csv.DictReader(f)]
 original=mapping('original',2048)
 for policy in policies[1:]:assert mapping(policy,2048)==original,policy
 assert records==576
 verification=dict(completed=True,formal_windows=72,rank_records=records,timed_optimizer_steps=1440,preflight_windows=12,reverse_order_checked=True,cross_round_initialization_checked=True,attention_original_mapping_checked=True,quantization_tested=False,performance_tuning=False,padded_vocab_size=9216,scope='Six BF16 policies; not six-policy quantization acceptance. Repetitions within one allocation.')
 (root/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
 result=dict(completed=True,rounds=rounds,verification=verification,submission=info)
 (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 names={'native':'原生Megatron','original':'原始Flux','remote_first':'MLP远端BF16','interleaved':'MLP交替BF16','remote_arrival':'MLP远端＋到达优先BF16','interleaved_arrival':'MLP交替＋到达优先BF16'}
 lines=['# 单节点TP8 BF16完整训练step测量','','179139 / '+info['node']+'，8×A800，TP8、DP1；12层/H2048/FFN8192，S2048/mb1/global4，4次微批累积、8192 token/step，实际词表9216。复用已有TP8到达表，仅将映射限定到MLP，attention为原始Flux；未调优、未量化。','', '| 策略 | 轮次 | ms/step | 对Megatron下降% [95% CI] | 对原始Flux下降% [95% CI] | 对对应基础排序下降% [95% CI] |','| --- | --- | --- | --- | --- | --- |']
 def ci(row,base):return f"{row[f'latency_reduction_vs_{base}_percent']:+.3f} [{row[f'ci95_low_vs_{base}']:+.3f},{row[f'ci95_high_vs_{base}']:+.3f}]"
 for policy in policies:
  for rnd in rounds:
   row=next(x for x in rnd['summary'] if x['policy']==policy);base={'remote_arrival':'remote_first','interleaved_arrival':'interleaved'}.get(policy)
   parts=[names[policy],str(rnd['round']),f"{row['median_ms_per_step']:.3f}",ci(row,'native'),ci(row,'original'),ci(row,base) if base else '—']
   lines.append('| '+' | '.join(parts)+' |')
 lines+=['','每轮6个平衡块，第二轮反序；窗口10步预热＋20步计时，取8rank最大耗时。百分比按配对比值的几何平均计算，10000次块bootstrap；正值表示更快，不能用中位数相除。固定合成token，不证明真实收敛。与此前mb4、词表8704的TP4/双节点TP8配置不同，不直接作强扩展加速比。', '', '72正式窗口/576份rank记录/1440计时step全部核验，另有6个smoke及6个profile窗口。旧算子速度准入没有用来筛除模型测量；数值、映射和kernel检查仍全部保留。']
 (root/'summary.md').write_text('\n'.join(lines)+'\n')
 print(root/'summary.md',flush=True)
if __name__=='__main__':
 import sys
 report(Path(sys.argv[1]).resolve())
