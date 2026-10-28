"""Freeze TP8 optimization runs, including exact decoder launch expectations."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('tp8_prepare', HERE/'prepare.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def main(args):
    arrival_window = json.loads(args.plan.read_text())['window']
    decoder_label = '原通用解码（v0）' if args.variant == 0 else f'TP8专用解码v{args.variant}'
    tuning = None
    for policy in ('remote_arrival', 'interleaved_arrival'):
        m = json.loads((args.build/policy/'manifest.json').read_text())
        assert m.get('tp8_decoder_default', 0) == args.variant
        current = m.get('tp8_gemm_tuning', dict(tile_k=32, stages=0, sms=0))
        if tuning is None:
            tuning = current
        assert tuning == current
    base.main(args)
    root = args.out.resolve()
    cfg_path = root/'scripts/config.json'
    cfg = json.loads(cfg_path.read_text())
    cfg.update(selective_decoder=f'tp8-v{args.variant}',
        optimization=dict(decoder_variant=args.variant, gemm_tuning=tuning, arrival_window=arrival_window,
                          user_goal_percent=3., historical_target_percent=4.),
        scope=f'Single-node TP8 selective decoder v{args.variant}, MLP launch {tuning}, arrival window {arrival_window}; frozen mask budget1/64 and baseline libraries. Two reversed six-policy full-step rounds; no CUDA Graphs.')
    cfg_path.write_text(json.dumps(cfg, indent=2)+'\n')
    p = root/'scripts/summarize.py'
    s = p.read_text()
    s = base.replace(s, 'result = dict(legacy=0, bf16=0, compact=0)',
        'result = dict(legacy=0, bf16=0, compact=0, tp8=0, materialize=0)')
    anchor = "    result['legacy'] = invocations"
    s = base.replace(s, anchor, '''    if policy.endswith('_selective') and (m,n,world) == (2048,2048,8):
        variant = plan.get('optimization', {}).get('decoder_variant', 0)
        if 14 <= variant <= 19:
            result['tp8'] = invocations
            if variant == 18:
                mask = selection_entry(selection,policy,m,n,model['ffn'])['mask']
                count = sum(any(mask[src][tile] for src in range(world))
                            for tile in range(rank*32,(rank+1)*32))
                result['materialize'] = invocations if count else 0
            return result
''' + anchor)
    s = base.replace(s,
        "compact=sum('taco_decode_compact4_kernel' in e['name'] for e in all_kernels))",
        """compact=sum('taco_decode_compact4_kernel' in e['name'] for e in all_kernels),
                    tp8=sum(any(name in e['name'] for name in ('taco_decode_ring8_kernel', 'taco_decode_tile8_kernel', 'taco_decode_flat8_kernel', 'taco_reduce_materialized8_kernel', 'taco_decode_cooperative8_kernel')) for e in all_kernels),
                    materialize=sum('taco_materialize8_kernel' in e['name'] for e in all_kernels))""")
    anchor = "                    assert hparams=={'64x64x32_16x8x16_streamksk_nil_128x128x32_gemmstreamk_3_rasterheuristic'}"
    s = base.replace(s, anchor, '''                    original = '64x64x32_16x8x16_streamksk_nil_128x128x32_gemmstreamk_3_rasterheuristic'
                    expected = {original}
                    if policy.endswith('_selective'):
                        tuning = plan['optimization']['gemm_tuning']
                        tuned = f"64x64x32_16x8x16_streamksk_nil_128x128x{tuning['tile_k']}_gemmstreamk_{tuning['stages'] or 3}_rasterheuristic"
                        expected.add(tuned)
                        if tuned != original:
                            assert sum(tuned in e['name'] for e in kernels) == calls//2
                            assert sum(original in e['name'] for e in kernels) == calls//2
                    assert hparams == expected''')
    p.write_text(s)
    p = root/'assess.py'
    s = p.read_text()
    anchor = '    assess(p.parse_args().root.resolve())'
    s = base.replace(s, anchor, '''    root = p.parse_args().root.resolve()
    result = assess(root)
    user = dict(target_percent=3., historical_target_percent=result['target_percent'], candidates={})
    for name, candidate in result['candidates'].items():
        user['candidates'][name] = all(
            all(v['reduction_percent'] >= 3. and v['ci95'][0] > 0 for v in r['required'].values())
            for r in candidate['rounds'])
    user['target_achieved'] = any(user['candidates'].values())
    (root/'user-goal.json').write_text(json.dumps(user,indent=2)+'\\n')''')
    p.write_text(s)
    p = root/'report.py'
    s = p.read_text().replace('单节点 TP8 完整量化组合首测', '单节点 TP8 选择性解码优化')
    s = s.replace('仅适配并测量，未做参数扫描或性能优化。',
                  f'本轮执行成本/局部性优化，冻结{decoder_label}及window{arrival_window}；本次用户目标约3%，同时保留历史4%验收。')
    s = s.replace('候选沿用两套完整工作区和通用解码',
                  f'候选沿用两套完整工作区，使用{decoder_label}')
    s = s.replace('window64及1/64预算沿用既有规则，未调参。',
                  f'到达优先限制在同目标分区window{arrival_window}，1/64量化预算及物理mask保持冻结。')
    tuning_text = ', '.join(f'{key}={value}' for key,value in tuning.items())
    s = s.replace('每轮6个位置平衡块', f'MLP执行参数：{tuning_text}。每轮6个位置平衡块')
    p.write_text(s)
    for p in root.rglob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    info_path = root/'submission.json'
    info = json.loads(info_path.read_text())
    info['scope'] = cfg['scope']
    info['optimization'] = cfg['optimization']
    info['protected_sources'][str(Path(__file__).resolve())] = base.sha(Path(__file__).resolve())
    info['files'] = {str(p.relative_to(root)):base.sha(p) for p in root.rglob('*')
                     if p.is_file() and p != info_path}
    info_path.write_text(json.dumps(info, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--job-id', type=int, required=True)
    p.add_argument('--node', required=True)
    p.add_argument('--variant', type=int, choices=[0,14,15,16,17,18,19], required=True)
    main(p.parse_args())
