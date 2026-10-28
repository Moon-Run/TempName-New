"""Aggregate verified campaigns and full-step ablations without pooling nodes."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root):
    root = root.resolve()
    spec = json.loads((root/'report-inputs.json').read_text())
    session = json.loads((root/'session.json').read_text())
    result = dict(updated_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                  session_status=session['status'],formal=[],ablations=[],pending=[],
                  formal_windows=0,formal_rank_records=0,formal_timed_optimizer_steps=0,
                  ablation_windows=0,ablation_rank_records=0,ablation_timed_optimizer_steps=0,
                  source_sha256={})
    result['not_run_due_to_shortened_timebox'] = spec.get('not_run_due_to_shortened_timebox',[])
    lines = ['# TP4 第二轮限时优化汇总','',
             f"窗口：{session['started_at']} 至 {session['deadline']}。状态：`{session['status']}`。",'',
             '完整 optimizer step，MLP-only。每组 baseline 与候选在同节点、同模型内配对；不跨节点或跨场景相除。', '',
             '| 实验 / 作业 | 轮次 | 远端组合 ms/step | 对 Megatron＋量化下降% [95% CI] | 对冻结融合v2下降% [95% CI] | 比原生Megatron耗时增加% | 比原生Flux耗时增加% |',
             '| --- | --- | ---: | --- | --- | ---: | ---: |']
    for entry in spec['formal']:
        run = Path(entry['root'])
        acceptance, verification = run/'acceptance.json', run/'verification.json'
        if not acceptance.exists() or not verification.exists():
            result['pending'].append(entry['label'])
            continue
        data, checked = json.loads(acceptance.read_text()), json.loads(verification.read_text())
        assert data['completed'] and checked['completed']
        assert checked['rank_records'] == 4*checked['windows']
        assert checked['global_timed_optimizer_steps'] == 20*checked['windows']
        result['formal_windows'] += checked['windows']
        result['formal_rank_records'] += checked['rank_records']
        result['formal_timed_optimizer_steps'] += checked['global_timed_optimizer_steps']
        result['source_sha256'].update({str(p):sha(p) for p in (acceptance,verification)})
        result['formal'].append(dict(entry,acceptance=data,verification=checked))
        for row in data['candidates']['remote_arrival_selective']['rounds']:
            def fmt(key):
                value = row['required'][key]
                return f"{value['reduction_percent']:.3f} [{value['ci95'][0]:.3f},{value['ci95'][1]:.3f}]"
            lines.append(f"| {entry['label']} / {entry['job_id']} | {row['round']} | {row['median_ms_per_step']:.3f} | {fmt('native_taco')} | {fmt('taco_fused')} | {row['reference_gap_percent']['native']:+.3f} | {row['reference_gap_percent']['original']:+.3f} |")
    lines += ['', '## 验收与完整性', '',
              '判定使用未四舍五入的配对几何平均耗时比。每轮点估计均须达到4%，且各95%区间下界大于0；置信区间按配对块 bootstrap 10000次。','',
              '| 实验 | 验收 | 窗口 / rank记录 / 计时step |','| --- | --- | --- |']
    for entry in result['formal']:
        run = Path(entry['root']);checked=entry['verification'];data=entry['acceptance']
        passed=[policy for policy,value in data['candidates'].items() if value['meets_target_both_rounds']]
        verdict=', '.join(passed)+' 通过' if passed else '未通过'
        lines.append(f"| [{entry['label']}](../{run.name}/acceptance.md) | {verdict} | {checked['windows']} / {checked['rank_records']} / {checked['global_timed_optimizer_steps']} |")
    lines += ['', '## 新实现的独立增量', '',
              '这些配对消融没有替代必须baseline验收。除明确标为整套构建比较者外，仅在同库中切换解码器，固定基础顺序、到达表和物理量化mask。','',
              '| 实验 | 对比 | 完整step下降% [95% CI] |','| --- | --- | --- |']
    for entry in spec['ablations']:
        run=Path(entry['root']);path=run/'results.json'
        if not path.exists():
            result['pending'].append(entry['label']);continue
        data=json.loads(path.read_text());assert data['completed']
        count=len(data['windows']);assert data['rank_records']==4*count
        result['ablation_windows']+=count;result['ablation_rank_records']+=4*count
        result['ablation_timed_optimizer_steps']+=20*count
        result['source_sha256'][str(path)]=sha(path)
        result['ablations'].append(dict(entry,results=data))
        for name,row in data['comparisons'].items():
            lines.append(f"| [{entry['label']}](../{run.name}/results.json) | {name} | {row['reduction_percent']:+.3f} [{row['ci95'][0]:+.3f},{row['ci95'][1]:+.3f}] |")
    for key in ('windows','rank_records','timed_optimizer_steps'):
        result['all_full_step_'+key]=result['formal_'+key]+result['ablation_'+key]
    lines += ['',f"正式验收共 {result['formal_windows']} 窗口 / {result['formal_rank_records']} 份rank记录 / {result['formal_timed_optimizer_steps']} 个计时step；另有 {result['ablation_windows']} 个完整step消融窗口。",'',
              '## 范围与限制','',
              '所有计时均为固定合成token，不含数据加载、checkpoint或评估；反向保持BF16 STE。有限loss和codec预算通过不等于真实数据收敛。', '',
              '到达表与mask沿用既有同形状BF16校准，未在本轮重排或量化后重新拟合。短列表只缓存每个接收rank的混合tile集合，不改变选中的来源、tile或通信字节。新增收益来自接收端实现，不能作为到达优先/选择规则独立收益的证据。', '',
              '默认新路径限TP4、M8192/N2048、每个接收rank最多32个混合tile；其余保持旧解码。S1024/mb8与S2048/mb4均为8192 token/step，但attention工作量不同，结果各自验收。']
    if result['pending']:
        lines += ['', '尚未汇入已完成验收：'+', '.join(result['pending'])+'。']
    if result['not_run_due_to_shortened_timebox']:
        lines += ['', '用户将总时限从约2.5小时缩短为约2小时；以下仅准备、未运行，也没有自动续跑队列：'+
                  '、'.join(e['label'] for e in result['not_run_due_to_shortened_timebox'])+'。']
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('formal','ablations','source_sha256')},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('root',type=Path)
    main(parser.parse_args().root)
