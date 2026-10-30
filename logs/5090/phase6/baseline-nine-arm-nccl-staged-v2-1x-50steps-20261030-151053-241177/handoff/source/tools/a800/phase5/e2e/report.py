"""Full-step phase5 comparisons; positive percentages mean lower latency."""
import json
from pathlib import Path

LABELS={'native':'原始Megatron','original':'原始Flux','taco_separate':'Flux＋独立量化',
        'taco_fused':'原始顺序Flux＋融合量化v2（必须baseline）',
        'remote_bf16':'仅MLP远端重排','interleaved_bf16':'仅MLP交替重排',
        'remote_all':'远端＋融合全量量化','interleaved_all':'交替＋融合全量量化',
        'remote_selective':'远端＋融合选择性量化','interleaved_selective':'交替＋融合选择性量化',
        'remote_arrival_selective':'远端＋到达优先＋选择性量化',
        'interleaved_arrival_selective':'交替＋到达优先＋选择性量化','native_taco':'Megatron＋张量TACO量化'}


def write_report(root):
    info=json.loads((root/'submission.json').read_text())
    plan=json.loads((root/'scripts/selection-plan.json').read_text())
    config=json.loads((root/'scripts/config.json').read_text())
    stage='Phase6' if 'taco_fused' in info['policies'] else 'Phase5'
    balance='位置平衡' if info['blocks']==len(info['policies']) else '探索性配对（未覆盖全部位置）'
    lines=[f'# {stage}：MLP重排＋全量/选择性TACO端到端','',
        '仅MLP linear_fc2前向远端贡献量化；Flux组attention保留BF16原始Flux映射，原始Megatron和Megatron量化组的attention保持Megatron原生路径。BF16 GEMM、E4M3/H128通信codec、BF16直通近似反向；固定合成token，不代表真实数据收敛。',
        f"12层GPT，H2048、FFN8192、S2048、TP4、global batch4，8192 token/step。每轮{info['blocks']}个{balance}块，每窗口10步预热＋20步完整optimizer step；第二轮反转块与组顺序。同一分配内两轮新进程，不是独立作业。",
        f"选择表在正式计时前冻结：两种基础各自的训练采样0/1要求来源均近临界，按接收端相对join时刻排序，每来源最多{100*plan['budget_fraction']:g}%远端tile；第2遍索引（第三次采样）只用于验证。未选中贡献保持BF16，按相同BF16 ring顺序累加。它是尾部候选启发式，尚无逐tile正净收益保证；不等同于完整自适应联合优化。",
        '远端基础沿用rank+1分区偏移，不承诺任意shape下严格全部远端在前；交替覆盖本地及三个远端分区。GEMM构建可能影响未量化attention资源；同顺序BF16控制与全量/选择性量化使用同一构建。','',
        '| 选择表 | 每来源选中tile数 | 远端贡献选中比例 | 第三遍近临界命中率 | 采样中位扰动 |',
        '| --- | --- | ---: | ---: | ---: |']
    for policy,r in plan['policies'].items():
        hit='N/A' if r['heldout_near_critical_fraction'] is None else f"{100*r['heldout_near_critical_fraction']:.2f}%"
        lines.append(f"| {policy} | {r['selected_per_source']} | {100*r['selected_remote_fraction']:.3f}% | {hit} | {r['sampled_median_overhead_percent']:.2f}% |")
    if 'remote_arrival_selective' in info['policies']:
        ordering=json.loads((root/'scripts/arrival-plan.json').read_text())
        lines += ['', f"到达优先采用v2 tail、每目标分区{ordering['window']}个合法槽位内的静态置换，仅用于MLP。量化mask以物理tile为索引，不按逻辑槽位移动；尚未在新顺序/量化后重新校准。构建资源和运行时查表成本计入实际耗时。"]
    if config.get('graph_forward'):
        lines += ['', '选择性量化的MLP前向使用CUDA Graph；输入复制、图重放、输出复制均计时，一次性建图在预热阶段。attention、反向和optimizer保持原路径。']
    if config.get('double_buffered'):
        lines += ['', '候选交替使用两套完整BF16/FP8通信工作区。每次GEMM发布barrier保留；第k+1次发布barrier确保所有rank已完成第k次解码，保护第k+2次复用，从而去除解码后单独barrier。GPU验证包含不同rank延迟下的12次连续调用和输出逐位一致性；profile核验每step的Flux barrier由144次降至96次。增加工作区内存，峰值内存另报。']
    if 'native_taco' in info['policies']:
        lines += ['', 'Megatron量化为朴素张量基线：保留原生RowParallelLinear GEMM、bias与线性层反向，仅在MLP替换RS；用PyTorch张量算子执行同一TACO codec，NCCL all-to-all交换远端压缩贡献，再BF16解码归约。本地贡献不量化，通信反向仍是原生AllGather的直通近似，不加载Flux库。张量临时分配、打包、Q/DQ和collective开销均计入step；不代表经过优化的Megatron量化实现。']
    for i in (1,2):
        path=root/f'results/model-{i}/analysis.json'
        if not path.exists():continue
        a=json.loads(path.read_text());rows={r['policy']:r for r in a['summary']}
        def pct(r,b):return f"{r[f'latency_reduction_vs_{b}_percent']:+.3f} [{r[f'ci95_low_vs_{b}']:+.3f}, {r[f'ci95_high_vs_{b}']:+.3f}]"
        lines += ['',f'## 第{i}轮','', '| 策略 | ms/step | tokens/s | 相对原始Flux下降% [95%区间] | 相对原始Megatron下降% [95%区间] |', '| --- | ---: | ---: | --- | --- |']
        for p in info['policies']:
            r=rows[p]
            lines.append(f"| {LABELS[p]} | {r['median_ms_per_step']:.3f} | {r['median_tokens_per_second']:,.0f} | {pct(r,'original') if 'original' in rows else 'N/A'} | {pct(r,'native') if 'native' in rows else 'N/A'} |")
        lines += ['','| 候选 / 对应基线 | 配对下降% [95%区间] |','| --- | --- |']
        for prefix in ('remote','interleaved'):
            for p,b in ((prefix+'_all',prefix+'_bf16'),(prefix+'_selective',prefix+'_bf16'),(prefix+'_selective',prefix+'_all')):
                if p in rows and b in rows:lines.append(f'| {LABELS[p]} / {LABELS[b]} | {pct(rows[p],b)} |')
            p=prefix+'_arrival_selective'
            if p in rows and prefix+'_selective' in rows:
                b=prefix+'_selective'
                lines.append(f'| {LABELS[p]} / {LABELS[b]} | {pct(rows[p],b)} |')
            if p in rows:
                for b in ('native_taco','taco_fused'):
                    if b in rows:lines.append(f'| {LABELS[p]} / {LABELS[b]} | {pct(rows[p],b)} |')
        if 'native_taco' in rows and 'native' in rows:
            lines.append(f"| Megatron＋量化 / 原始Megatron | {pct(rows['native_taco'],'native')} |")
        lines += ['',f"{a['windows']}个窗口通过；配对统计来自同一块耗时比的几何均值，不能以两个中位数相除重算。完整step包含运行时查表、配置、Q/DQ、同步、反向、optimizer及清梯度；离线拟合和一次性mask上传不在step中。"]
    lines += ['',f"运行于{info['job_id']}内部step，原驻留循环保留。mask和packet驻留内存开销保留，不宣称峰值显存下降或逻辑字节等于实测链路流量。"]
    (root/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    import sys
    write_report(Path(sys.argv[1]))
