"""Generate an evidence-bounded acceptance report; do not turn missing counters into claims."""
import csv,json,statistics,sys
from pathlib import Path
out=Path(sys.argv[1])
def read(name):return list(csv.DictReader((out/name).open()))
def pct(n,d):return f'{100*n/d:.1f}%' if d else '未解析'
def median(rows,key):
 values=[float(r[key]) for r in rows if r[key]!=''];return statistics.median(values) if values else None
def fmt(x):return '无样本' if x is None else f'{x:+.2f}'
a=json.loads((out/'analysis.json').read_text());cfg=json.loads((out/'experiment.json').read_text())
over=read('sampling_overhead.csv');controls=read('build_controls.csv');joins=read('join_summary.csv');pairs=read('arrival_policy_pairs.csv');admission=read('measurement_admission.csv')
cap=json.loads((out/'counter-capability.json').read_text()) if (out/'counter-capability.json').exists() else {'available':False,'permission_denied':False}
lines=[f'阶段3机制诊断验收：作业{out.name}', '',
       f"采样forward正确性记录：{a['sample_forward_checks']}；首次扫描截断fragment：{a['first_scan_visible']}。",
       f"低扰动联合筛查窗口：{a['accepted_measurement_windows']}/{a['sparse_windows']}。该筛查同时考虑构建变化、采样开销、控制漂移与波动，不等于统计显著性。",
       '到达时刻仅在同一接收GPU时基下比较；覆盖采样子集，不能外推全体tile。',
       '文中的算子边界是GPU marker给出的诊断forward边界，不是模型端到端时间。', '',
       '一、扰动与构建控制',
       '| 策略 | M/N/K_global | 构建关闭采样开销范围% | 稀疏开启中位开销范围% | 联合筛查通过 |',
       '| --- | --- | --- | --- | --- |']
for shape in cfg['shapes']:
 m,n,k=shape
 for policy in ('original','remote_first','interleaved'):
  match=lambda r:(int(r['M']),int(r['N']),int(r['K_global']),r['policy'])==(m,n,k,policy)
  cs=[r for r in controls if match(r)];ps=[r for r in over if match(r) and r['mode']=='receiver_sparse'];ads=[r for r in admission if match(r)]
  c=[float(r['instrumented_off_overhead_percent']) for r in cs];p=[float(r['median_overhead_percent']) for r in ps]
  lines.append(f"| {policy} | {m}/{n}/{k} | {min(c):+.2f} 至 {max(c):+.2f} | {min(p):+.2f} 至 {max(p):+.2f} | {sum(r['accepted']=='True' for r in ads)}/{len(ads)} |")
lines+=['','二、四来源观察','| 策略 | M/N/K_global | 可解析最后来源 | 其中远端最后 | skew P50（各块中位数的中位数，μs） | 分区最后tile观察时刻（相对接收端起点，μs） |','| --- | --- | --- | --- | --- | --- |']
for shape in cfg['shapes']:
 m,n,k=shape
 for policy in ('original','remote_first','interleaved'):
  rs=[r for r in joins if (int(r['M']),int(r['N']),int(r['K_global']),r['policy'],r['mode'])==(m,n,k,policy,'receiver_sparse')]
  resolved=sum(int(r['resolved']) for r in rs);total=sum(int(r['observations']) for r in rs);remote=sum(int(r['remote_last']) for r in rs)
  lines.append(f"| {policy} | {m}/{n}/{k} | {resolved}/{total} | {remote}/{resolved} | {fmt(median(rs,'skew_p50_us'))} | {fmt(median(rs,'last_tile_arrival_p50_us'))} |")
lines+=['','三、相同输出tile的配对尾部变化','负的arrival_delta表示候选更早被观察到；表中逐块保留，不混为新加速比。','| 块 | 策略 | M/N/K_global | 尾tile样本对 | 观察时刻差μs | 诊断算子边界时长差μs | 轮询不确定性P95 μs | 两侧低扰动筛查 |','| --- | --- | --- | --- | --- | --- | --- | --- |']
for r in pairs:
 if r['last_tiles_only']!='True':continue
 ads=[x for x in admission if all(x[key]==r[key] for key in ('block','M','N','K_global')) and x['policy'] in ('original',r['policy'])]
 accepted=len(ads)==2 and all(x['accepted']=='True' for x in ads)
 lines.append(f"| {r['block']} | {r['policy']} | {r['M']}/{r['N']}/{r['K_global']} | {r['pairs']} | {fmt(float(r['arrival_delta_us'])) if r['arrival_delta_us'] else '无样本'} | {fmt(float(r['operator_span_delta_us'])) if r['operator_span_delta_us'] else '无样本'} | {r['poll_uncertainty_p95_us']} | {'通过' if accepted else '受限'} |")
lines+=['','四、验收边界',
 '记录完整、数值正确、hparams/grid保持一致，只说明测量原型可运行。',
 '采样关闭时构建偏差、采样开启开销、控制漂移或轮间噪声未达标的配置，时序结果仅作探索性证据。',
 '单独减少join skew不等于完整算子改善；须关注尾tile的最后贡献是否提前，以及与诊断算子边界是否同向。',
 '观察时刻包含system fence、标记传播和轮询；没有逐tile归约消费时间戳。rank间启动偏斜仍可能影响相对起点的时刻。',
 '此前无插桩三策略性能仍以178474/178475为正式依据，不用本报告单次采样forward替代。']
if not cap['available']:
 lines+=['硬件缓存/访存计数器未取得'+('（ERR_NVGPUCTRPERM）' if cap.get('permission_denied') else '')+'。不能宣布完整机制验收通过，也不能将收益唯一归因于缓存或传输。']
else:
 lines+=['权限探测可用，但简单probe不是实际GEMM-RS的访存证据；仍须检查实际算子的计数器报告。']
lines+=['原始证据：sampling_overhead.csv、build_controls.csv、measurement_admission.csv、receiver_tiles.csv、producer_tiles.csv、join_summary.csv、arrival_policy_pairs.csv、每rank JSON及scripts快照。']
(out/'report.txt').write_text('\n'.join(lines)+'\n')
print('Acceptance report saved:',out/'report.txt')
