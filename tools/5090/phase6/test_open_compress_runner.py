"""CPU-only adversarial records; these are not hardware measurements."""
from collections import Counter
import copy
from contextlib import redirect_stdout
import io
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import open_compress as controller
import open_compress_protocol as p
import open_compress_sources as sources
import open_compress_validation as validation
from test_baseline_workers import fixture as original_fixture


def fixture(arm, mode='timing'):
    rows,job,plan,worker = original_fixture('flux-basic',mode)
    job['arm'] = arm
    codec = None if arm == p.ARMS[0] else dict(path='/CPU-ONLY/codec.so',sha256='a'*64)
    worker.update(selection=sources.selection_metadata(),definition=sources.definition(arm))
    worker['config'] = sources.configuration(arm,worker['selection'],plan['communication'],codec)
    plan['protocol'] = p.PROTOCOL
    for row in rows:
        calls = (row['warmup_steps']+len(row['losses']))*4
        row.update(protocol=p.PROTOCOL,arm=arm,policy=arm,scope_definition=worker['definition'],
            role_plan_sha256=worker['config']['role_plan_sha256'],open_compress_implementation=sources.implementation(arm),
            codec_identity=worker['config']['codec_identity'],codec_library=codec,codec_validation=None,
            smoke_output_sha256={},retained_output_checks={},compression_states={})
        names = row['target_modules']
        row['selection_checks'] = {name:dict(mask_sha256=None,plan_sha256=None,
            wire_bytes=7*256*4096*2 if arm == p.ARMS[0] else 7*p.PACKET_BYTES,
            bf16_wire_bytes=7*256*4096*2) for name in names}
        for name in names:
            state = dict(calls=calls,owns_output=True,encode_calls=0,decode_calls=0,all_to_all_calls=0,
                         collective='reduce_scatter_tensor',quantization='none')
            if arm != p.ARMS[0]:
                splits = [0 if r == row['rank'] else p.PACKET_BYTES for r in range(8)]
                state.update(encode_calls=calls,decode_calls=calls,all_to_all_calls=calls,wait_calls=calls,
                    logical_remote_bytes=7*p.PACKET_BYTES,send_split_bytes=splits,receive_split_bytes=list(splits),
                    protocol='open-block-int8-soa128-v1',arm=arm,source_order=list(range(8)),
                    packet_layout='peer-soa-q8-scale32',dtype='uint8',collective='all_to_all_single',
                    quantization='all-remote',caller_stream=100,current_stream=100,
                    finite_input_checks=calls if mode == 'smoke' else 0,arrival=False,selective=False,
                    calibration_used=False,peer_overlap_preserved=False,failed=False,pending_exchange=False)
            row['compression_states'][name] = state
        if mode == 'smoke':
            row['smoke_output_sha256'] = dict.fromkeys(names,'a'*64)
            for role,suffix in (('attn','.self_attention.linear_proj'),('mlp','.mlp.linear_fc2')):
                row['retained_output_checks'][role] = dict(module=next(n for n in names if n.endswith(suffix)),
                    outputs=6,passed=True,independent_storage=True,unchanged_after_subsequent_call=True)
    return rows,job,plan,worker


def samples(blocks):
    return [dict(job,ms_per_step=100+10*p.ARMS.index(job['arm']),peak_allocated_gib=17,
        peak_reserved_gib=18,native_workspace_gib=0,peak_allocated_plus_native_gib=17)
        for job in p.schedule(1,blocks)]


class Protocol(unittest.TestCase):
    def test_balanced_cycles_and_single_block(self):
        for blocks in (1,6,12,13):
            jobs = p.schedule(2,blocks)
            self.assertEqual(len(jobs),6*blocks)
            for trial in (1,2):
                for block in range(blocks):
                    self.assertEqual({j['arm'] for j in jobs if j['trial']==trial and j['block']==block},set(p.ARMS))
                if blocks % 6 == 0:
                    own = [j for j in jobs if j['trial']==trial]
                    for position in range(3):
                        self.assertEqual(Counter(j['arm'] for j in own if j['index']%3==position),
                                         Counter(dict.fromkeys(p.ARMS,blocks//3)))
        for trials,blocks in ((0,1),(1,0),(-1,1),(True,1),(1,1.5)):
            with self.assertRaises(ValueError): p.schedule(trials,blocks)

    def test_one_block_has_no_ci_and_retains_slowdowns(self):
        result = p.report(samples(1),1,1)
        self.assertTrue(result['exploratory'])
        self.assertEqual(result['formal_optimizer_steps'],150)
        for pair in result['comparisons']['compare1']['pairs'].values():
            self.assertIsNone(pair['ci95'])
            self.assertLess(pair['percent'],0)
        self.assertIn('不足六个',p.markdown(result))

    def test_six_blocks_ci_and_invalid_evidence_rejected(self):
        result = p.report(samples(6),1,6)
        self.assertTrue(result['balanced_order'])
        self.assertTrue(all(s['ci95'] is not None for s in result['comparisons']['compare1']['pairs'].values()))
        for bad in (samples(1)[:-1], samples(1)[::-1], [samples(1)[0]]*3):
            with self.assertRaises(ValueError): p.report(bad,1,1)
        bad = samples(1); bad[0]['ms_per_step'] = float('nan')
        with self.assertRaises(ValueError): p.report(bad,1,1)

    def test_preview_does_not_create_output_or_query_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'never-created'
            with patch.object(controller.base,'gpu_idle',side_effect=AssertionError('GPU queried')), \
                 redirect_stdout(io.StringIO()) as output:
                controller.main(['--trials','1','--blocks','1','--out',str(out)])
            self.assertFalse(out.exists())
            self.assertIn('PLAN ONLY',output.getvalue())


class Evidence(unittest.TestCase):
    def test_complete_modes_pass(self):
        for arm,mode in itertools.product(p.ARMS,('smoke','profile','timing')):
            with self.subTest(arm=arm,mode=mode):
                self.assertEqual(len(validation.validate_rows(*fixture(arm,mode))),8)

    def test_bad_codec_lifecycle_precision_or_selection_is_rejected(self):
        for changes in ({'wait_calls':0},{'source_order':list(reversed(range(8)))},
                        {'current_stream':99},{'pending_exchange':True},{'selective':True},
                        {'finite_input_checks':1},{'packet_layout':'other'},{'owns_output':False}):
            rows,job,plan,worker = fixture(p.ARMS[1])
            next(iter(rows[0]['compression_states'].values())).update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validation.validate_rows(rows,job,plan,worker)
        rows,job,plan,worker = fixture(p.ARMS[1])
        next(iter(rows[0]['selection_checks'].values()))['mask_sha256'] = 'a'*64
        with self.assertRaises(ValueError): validation.validate_rows(rows,job,plan,worker)

    def test_missing_rank_wrong_input_coverage_and_timing_rejected(self):
        rows,job,plan,worker = fixture(p.ARMS[0])
        with self.assertRaises(ValueError): validation.validate_rows(rows[:-1],job,plan,worker)
        for changes in ({'skips':1},{'fallback_calls':1},{'timed_steps':49},{'native_workspace_bytes':1}):
            bad = copy.deepcopy(rows); bad[0].update(changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError): validation.validate_rows(bad,job,plan,worker)

    def test_bf16_cannot_claim_compression(self):
        rows,job,plan,worker = fixture(p.ARMS[0])
        next(iter(rows[0]['compression_states'].values()))['encode_calls'] = 240
        with self.assertRaises(ValueError): validation.validate_rows(rows,job,plan,worker)


if __name__ == '__main__':
    unittest.main()
