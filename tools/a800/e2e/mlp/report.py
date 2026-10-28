"""Report MLP-only paired full-step results, retaining both rounds separately."""
import json
from pathlib import Path


def write_report(root):
    labels = {'native': '原始Megatron', 'original': '原始Flux',
              'mlp_remote': 'MLP远端优先＋attention原始Flux',
              'mlp_remote_arrival': 'MLP远端＋到达优先＋attention原始Flux'}
    lines = ['# 四卡端到端：仅MLP重排', '',
             '12层GPT，H=2048、FFN=8192、S=2048、TP4，micro batch=1、global batch=4，BF16。',
             '每计时窗口10步预热+20步完整optimizer step；每轮8个位置平衡块，第二轮反序，取4 rank最慢墙钟；8192 token/step。',
             '仅MLP linear_fc2使用rank+1或rank+1＋v2 tail，attention linear_proj保持原始Flux映射。复用预先冻结的MLP tail表，不根据本轮性能拟合。',
             '叠加策略仅MLP命中查表；attention沿原始映射执行，共享kernel的索引判断成本仍计入完整step。无压缩。',
             '模块选择在host swizzle构造时依据本模型两种已核验的不同shape完成；不是任意模型通用的模块路由。',
             '两轮是同一179147分配内的独立进程重复，不是两个独立Slurm作业。正值为配对耗时下降；区间来自每轮8个配对块。',
             '计时窗口与16槽位tile重排窗口不同：前者测完整训练，后者限制离线换位范围。点估计来自块内耗时比的几何均值；95%区间为10000次配对块bootstrap的2.5%/97.5%分位，上界不是最佳实测加速。', '']
    for i in (1, 2):
        p = root/f'results/model-{i}/analysis.json'
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        assert d['completed']
        lines += [f'## 第{i}轮', '',
                  '| 策略 | ms/step | tokens/s | 相对Megatron下降% [95%区间] | 相对原始Flux下降% [95%区间] | 相对MLP远端下降% [95%区间] |',
                  '| --- | ---: | ---: | --- | --- | --- |']
        def pct(r, baseline):
            return f"{r[f'latency_reduction_vs_{baseline}_percent']:+.3f} [{r[f'ci95_low_vs_{baseline}']:+.3f}, {r[f'ci95_high_vs_{baseline}']:+.3f}]"
        for r in d['summary']:
            lines.append(f"| {labels[r['policy']]} | {r['median_ms_per_step']:.3f} | {r['median_tokens_per_second']:,.0f} | {pct(r, 'native')} | {pct(r, 'original')} | {pct(r, 'mlp_remote')} |")
        lines += ['', '| 策略 | 初始loss | 末步loss | 末步梯度范数 | allocated峰值GiB |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for r in d['summary']:
            lines.append(f"| {labels[r['policy']]} | {r['initial_loss']:.6f} | {r['last_timed_loss']:.6f} | {r['final_grad_norm']:.6f} | {r['max_peak_allocated_gib']:.2f} |")
        lines += ['', f"本轮{d['windows']}个窗口全部4 rank记录完整，loss/梯度有限、跳步为0。", '']
    lines += ['固定GPU驻留合成token；完整step包含前向、反向、通信、optimizer更新与清梯度，不包含数据加载、checkpoint及评估。',
              '原始Megatron保持原生路径；三组Flux使用相同适配和反向，原始Flux基线在本轮重新计时。',
              '不以有限loss/梯度代替配对梯度精度预算或真实数据收敛验收。原驻留step 179147.1和外层batch保留。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    import sys
    write_report(Path(sys.argv[1]))
