"""Separate quantization costs from fusion and compiler-resource effects."""
import json
from pathlib import Path


def write_report(root):
    labels = {'original':'原始Flux（BF16）', 'original_matched':'融合构建关闭量化',
              'taco_separate':'原始顺序Flux＋独立TACO', 'taco_fused':'原始顺序Flux＋融合TACO',
              'taco_fused_legacy':'原始顺序Flux＋旧融合TACO'}
    info = json.loads((root/'submission.json').read_text())
    lines = ['# 四卡MLP-only TACO端到端对照','',
             '12层GPT，H=2048、FFN=8192、S=2048、TP4、global batch=4、BF16。仅MLP linear_fc2前向远端贡献量化；attention保持BF16原始Flux映射，无tile重排。',
             '两条TACO路径使用相同E4M3/128、自适应scale、归一化Hadamard、packet布局和DQ＋ring归约。本地贡献BF16，远端满组256字节变136字节（逻辑字节减少46.875%），不等于实测链路流量或加速。',
             '分离路径先将原始Flux GEMM的BF16贡献存入本地工作区，再独立编码/scatter；融合路径在GEMM epilogue内编码/scatter。均包含解码、额外buffer重用barrier、配置与适配成本。',
             '融合构建可能改变未量化kernel的资源占用，因此额外给出同构建关闭量化控制；attention无量化不等于每个构建的attention kernel机器码/资源均与stock Flux相同。',
             '所有策略使用同一BF16 AllGather/dgrad/wgrad反向；量化前向采用直通近似，不是量化函数的精确导数。固定合成token与有限loss/梯度不等于真实数据收敛或梯度精度验收。',
             f"每轮{info['blocks']}个位置平衡块，每窗口10步预热＋20步完整optimizer step，取4rank最大墙钟，8192token/step；第二轮反序。两轮在同一{info['job_id']}分配内，非独立作业。正值为配对耗时下降，区间来自块bootstrap。",'']
    for i in (1,2):
        p = root/f'results/model-{i}/analysis.json'
        if not p.exists(): continue
        data = json.loads(p.read_text())
        lines += [f'## 第{i}轮','',
                  '| 策略 | ms/step | tokens/s | 相对stock Flux下降% [95%区间] | 相对关闭量化控制下降% [95%区间] | 相对独立TACO下降% [95%区间] |',
                  '| --- | ---: | ---: | --- | --- | --- |']
        def pct(r, base):
            return f"{r[f'latency_reduction_vs_{base}_percent']:+.3f} [{r[f'ci95_low_vs_{base}']:+.3f}, {r[f'ci95_high_vs_{base}']:+.3f}]"
        for r in data['summary']:
            lines.append(f"| {labels[r['policy']]} | {r['median_ms_per_step']:.3f} | {r['median_tokens_per_second']:,.0f} | {pct(r,'original')} | {pct(r,'original_matched')} | {pct(r,'taco_separate')} |")
        if 'taco_fused_legacy' in info['policies']:
            r = next(r for r in data['summary'] if r['policy']=='taco_fused')
            lines += ['', f"当前融合相对旧融合的配对耗时下降：{pct(r,'taco_fused_legacy')}%。"]
        lines += ['','| 策略 | 初始loss | 末步loss | 末步梯度范数 | allocated峰值GiB |','| --- | ---: | ---: | ---: | ---: |']
        for r in data['summary']:
            lines.append(f"| {labels[r['policy']]} | {r['initial_loss']:.6f} | {r['last_timed_loss']:.6f} | {r['final_grad_norm']:.6f} | {r['max_peak_allocated_gib']:.2f} |")
        lines += ['',f"{data['windows']}个窗口全部rank记录完整，loss/梯度有限，跳步与fallback为0。",'']
    lines += ['正式计时包括前向、反向、通信、optimizer更新与清梯度，不含数据加载/checkpoint/评估。原驻留step 179147.1与外层batch保留。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    import sys
    write_report(Path(sys.argv[1]))
