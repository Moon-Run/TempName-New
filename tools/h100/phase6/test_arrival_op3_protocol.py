"""CPU safety model for op3; these tests are not GPU race or performance evidence.

The model deliberately permits every rank and fragment to progress out of order.
Only producer release/acquire, stream order, the local producer -> tail edge, and
the final all-rank reuse barrier constrain the schedules. Device execution,
CUDA's memory implementation, and the codec arithmetic require GPU validation.
"""
import random
import unittest


class ProtocolError(RuntimeError):
    pass


class ProtocolModel:
    """Executable ordering contract, independent of the generated CUDA source."""

    def __init__(self, world=3, fragments=4):
        self.world = world
        self.fragments = fragments
        self.groups = fragments * 16
        self.epoch = 0
        self.ready = {}
        self.visible_payload = {}
        self.done = {}
        self.reused = set(range(world))
        self.begin(1)

    def begin(self, epoch):
        if epoch != self.epoch + 1 or not 0 < epoch < 2**32:
            raise ProtocolError('epochs must increase without repetition or wrap')
        if len(self.reused) != self.world:
            raise ProtocolError('a receiver can still be reading the old workspace')
        self.epoch = epoch
        self.producer_pending = {
            src: {(dst, frag) for dst in range(self.world)
                  for frag in range(self.fragments)}
            for src in range(self.world)}
        self.try_pass = [0] * self.world
        self.try_pending = {dst: set(range(self.fragments)) for dst in range(self.world)}
        self.tail_entered = set()
        self.tail_pending = {dst: set(range(self.groups)) for dst in range(self.world)}
        self.reused = set()
        self.reads = {}
        self.early_groups = 0
        self.blocked_tail = 0
        # ready/done are intentionally not cleared between epochs.

    def publish(self, src, dst, fragment):
        if (dst, fragment) not in self.producer_pending[src]:
            raise ProtocolError('a source fragment must be published exactly once')
        key = dst, src, fragment
        # All payload writers fence before the CTA barrier and release store.
        self.visible_payload[key] = self.epoch
        self.ready[key] = self.epoch
        self.producer_pending[src].remove((dst, fragment))

    def can_read(self, dst, fragment):
        return all(self.ready.get((dst, src, fragment)) == self.epoch
                   for src in range(self.world))

    def _read_group(self, dst, group, phase):
        fragment = group // 16
        if not self.can_read(dst, fragment):
            raise ProtocolError('consumer observed incomplete or stale contributions')
        if any(self.visible_payload.get((dst, src, fragment)) != self.epoch
               for src in range(self.world)):
            raise ProtocolError('release was visible before all payload writers')
        if (dst, group) in self.reads:
            raise ProtocolError('try and tail decoded the same output group')
        self.reads[dst, group] = phase

    def try_fragment(self, dst, fragment):
        if self.try_pass[dst] >= 2 or fragment not in self.try_pending[dst]:
            raise ProtocolError('try CTAs must execute exactly once in each pass')
        decoded = False
        if self.done.get((dst, fragment)) != self.epoch and self.can_read(dst, fragment):
            for group in range(fragment * 16, (fragment + 1) * 16):
                self._read_group(dst, group, 'try')
            self.done[dst, fragment] = self.epoch
            self.early_groups += 16
            decoded = True
        # Missing readiness always returns: no dependency on any producer.
        self.try_pending[dst].remove(fragment)
        if not self.try_pending[dst]:
            self.try_pass[dst] += 1
            if self.try_pass[dst] < 2:
                self.try_pending[dst] = set(range(self.fragments))
        return decoded

    def enter_tail(self, dst):
        if self.producer_pending[dst]:
            raise ProtocolError('tail cannot precede its local GEMM completion')
        if self.try_pass[dst] != 2:
            raise ProtocolError('tail must join both finite auxiliary passes')
        if dst in self.tail_entered:
            raise ProtocolError('tail already entered')
        self.tail_entered.add(dst)

    def tail_group(self, dst, group):
        if dst not in self.tail_entered or group not in self.tail_pending[dst]:
            raise ProtocolError('tail CTA is not pending')
        fragment = group // 16
        if self.done.get((dst, fragment)) != self.epoch:
            if not self.can_read(dst, fragment):
                self.blocked_tail += 1
                return False
            self._read_group(dst, group, 'tail')
        elif (dst, group) not in self.reads:
            raise ProtocolError('done became visible before the full fragment output')
        self.tail_pending[dst].remove(group)
        return True

    def arrive_reuse(self, dst):
        if (dst not in self.tail_entered or self.tail_pending[dst] or
                self.try_pass[dst] != 2 or self.producer_pending[dst]):
            raise ProtocolError('reuse barrier cannot precede all local reads')
        self.reused.add(dst)

    def finish_tries(self, dst):
        while self.try_pass[dst] < 2:
            self.try_fragment(dst, min(self.try_pending[dst]))

    def publish_rank(self, src):
        for dst, fragment in sorted(self.producer_pending[src], reverse=True):
            self.publish(src, dst, fragment)

    def finish(self):
        for src in range(self.world):
            self.publish_rank(src)
        for dst in range(self.world):
            self.finish_tries(dst)
            if dst not in self.tail_entered:
                self.enter_tail(dst)
            for group in sorted(self.tail_pending[dst], reverse=True):
                if not self.tail_group(dst, group):
                    raise ProtocolError('all producers completed but tail cannot progress')
            self.arrive_reuse(dst)


class ArrivalOp3ProtocolTests(unittest.TestCase):
    def test_exact_fragment_rows_partition_all_8192_h128_groups(self):
        seen = {}
        for local_tile in range(64):
            tile_m, tile_n = divmod(local_tile, 32)
            for fragment in range(8):
                for compact_row in range(16):
                    row = tile_m * 128 + fragment * 8 + compact_row % 8 + (compact_row // 8) * 64
                    group = row * 32 + tile_n
                    self.assertNotIn(group, seen)
                    owner = (row // 128) * 32 + group % 32
                    step = (row % 64) // 8
                    self.assertEqual((owner, step), (local_tile, fragment))
                    seen[group] = local_tile * 8 + fragment
        self.assertEqual(set(seen), set(range(8192)))
        self.assertEqual(set(seen.values()), set(range(512)))

    def test_ready_and_done_fit_only_the_unused_local_packet_slot(self):
        source_bytes = 8192 * 136
        readiness_bytes = 8 * 512 * 4
        done_bytes = 512 * 4
        self.assertLessEqual(readiness_bytes + done_bytes, source_bytes)
        for dst in range(8):
            begin = dst * source_bytes
            end = begin + readiness_bytes + done_bytes
            self.assertEqual(begin % 4, 0)
            for src in range(8):
                if src != dst:
                    payload_begin = src * source_bytes
                    payload_end = payload_begin + source_bytes
                    self.assertTrue(end <= payload_begin or begin >= payload_end)

    def test_try_consumes_ready_fragment_before_any_rank_finishes_producing(self):
        model = ProtocolModel()
        for src in range(model.world):
            model.publish(src, 0, 2)
        self.assertTrue(all(model.producer_pending.values()))
        self.assertTrue(model.try_fragment(0, 2))
        self.assertEqual(len(model.reads), 16)
        self.assertNotIn(0, model.tail_entered)
        model.finish()
        self.assertEqual(len(model.reads), model.world * model.groups)

    def test_try_never_waits_and_all_serialized_misses_safely_fall_back(self):
        model = ProtocolModel()
        for dst in range(model.world):
            model.finish_tries(dst)
        self.assertFalse(model.reads)
        self.assertEqual(model.try_pass, [2] * model.world)
        model.finish()
        self.assertEqual(set(model.reads.values()), {'tail'})
        self.assertEqual(len(model.reads), model.world * model.groups)

    def test_second_try_and_tail_never_decode_an_early_fragment_twice(self):
        model = ProtocolModel()
        for src in range(model.world):
            model.publish(src, 1, 1)
        self.assertTrue(model.try_fragment(1, 1))
        model.finish_tries(1)
        self.assertEqual(sum(dst == 1 for dst, _ in model.reads), 16)
        model.finish()
        self.assertEqual(sum(dst == 1 and phase == 'try'
                             for (dst, _), phase in model.reads.items()), 16)
        self.assertEqual(len(model.reads), model.world * model.groups)

    def test_tail_requires_every_local_producer_fragment_and_aux_join(self):
        model = ProtocolModel()
        model.finish_tries(0)
        with self.assertRaisesRegex(ProtocolError, 'local GEMM'):
            model.enter_tail(0)
        model.publish_rank(1)
        with self.assertRaisesRegex(ProtocolError, 'auxiliary'):
            model.enter_tail(1)
        model.publish_rank(0)
        model.enter_tail(0)
        # Rank 2 is still producing: no global publication barrier was assumed.
        self.assertTrue(model.producer_pending[2])

    def test_fast_rank_tail_cannot_block_the_last_rank_producer(self):
        model = ProtocolModel(world=3, fragments=2)
        for dst in range(model.world):
            model.finish_tries(dst)
        for src in (0, 1):
            model.publish_rank(src)
            model.enter_tail(src)
            self.assertFalse(model.tail_group(src, 0))
        self.assertEqual(model.blocked_tail, 2)
        # The only missing rank has no tail running on its GPU. Its producer
        # remains enabled even if every other GPU is entirely occupied by tail.
        self.assertNotIn(2, model.tail_entered)
        model.publish_rank(2)
        for dst in (0, 1):
            self.assertTrue(model.tail_group(dst, 0))
        model.finish()

    def test_stale_epoch_flags_and_done_never_authorize_a_read(self):
        model = ProtocolModel()
        model.finish()
        self.assertTrue(model.done)
        old_ready, old_done = dict(model.ready), dict(model.done)
        model.begin(2)
        self.assertEqual(model.ready, old_ready)
        self.assertEqual(model.done, old_done)
        self.assertFalse(model.try_fragment(0, 0))
        model.publish(0, 0, 1)
        self.assertFalse(model.try_fragment(0, 1))
        self.assertFalse(model.reads)
        model.finish()
        self.assertEqual(len(model.reads), model.world * model.groups)

    def test_epoch_repetition_zero_and_wrap_are_rejected(self):
        model = ProtocolModel()
        model.finish()
        for bad_epoch in (0, 1, 3, 2**32):
            with self.subTest(epoch=bad_epoch), self.assertRaises(ProtocolError):
                model.begin(bad_epoch)
        model.epoch = 2**32 - 1
        with self.assertRaises(ProtocolError):
            model.begin(2**32)

    def test_reuse_waits_for_every_receiver_including_late_tail(self):
        model = ProtocolModel()
        for src in range(model.world):
            model.publish_rank(src)
        for dst in range(model.world):
            model.finish_tries(dst)
            model.enter_tail(dst)
        with self.assertRaises(ProtocolError):
            model.arrive_reuse(0)
        for group in list(model.tail_pending[0]):
            model.tail_group(0, group)
        model.arrive_reuse(0)
        with self.assertRaisesRegex(ProtocolError, 'old workspace'):
            model.begin(2)
        model.finish()
        model.begin(2)
        self.assertEqual(model.epoch, 2)

    def test_thread_zero_fence_alone_has_a_visible_counterexample(self):
        # Writer 0 publishes a flag while another payload writer's stores are
        # still buffered. An acquire flag read cannot invent the missing data.
        model = ProtocolModel()
        for src in range(model.world):
            model.publish(src, 0, 0)
        model.visible_payload[0, 1, 0] = 0
        with self.assertRaisesRegex(ProtocolError, 'payload writers'):
            model.try_fragment(0, 0)
        model.visible_payload[0, 1, 0] = model.epoch
        self.assertTrue(model.try_fragment(0, 0))

    def test_random_rank_fragment_and_stream_interleavings_preserve_coverage(self):
        for seed in range(32):
            rng = random.Random(seed)
            model = ProtocolModel(world=4, fragments=5)
            for epoch in range(1, 4):
                if epoch > 1:
                    model.begin(epoch)
                steps = 0
                while len(model.reused) < model.world:
                    enabled = []
                    for rank in range(model.world):
                        if model.producer_pending[rank]:
                            enabled.append(('produce', rank))
                        if model.try_pass[rank] < 2:
                            enabled.append(('try', rank))
                        elif not model.producer_pending[rank] and rank not in model.tail_entered:
                            enabled.append(('enter', rank))
                        if rank in model.tail_entered:
                            pending = model.tail_pending[rank]
                            ready = [g for g in pending if model.can_read(rank, g // 16)]
                            if ready:
                                enabled.append(('tail', rank))
                            elif pending:
                                # Every blocking dependency points at a source
                                # whose producer has not finished on its GPU.
                                for group in pending:
                                    frag = group // 16
                                    for src in range(model.world):
                                        if model.ready.get((rank, src, frag)) != model.epoch:
                                            self.assertTrue(model.producer_pending[src])
                                            self.assertNotIn(src, model.tail_entered)
                            elif rank not in model.reused:
                                enabled.append(('reuse', rank))
                    self.assertTrue(enabled, 'nonterminal state with no enabled progress')
                    action, rank = rng.choice(enabled)
                    if action == 'produce':
                        dst, frag = rng.choice(sorted(model.producer_pending[rank]))
                        model.publish(rank, dst, frag)
                    elif action == 'try':
                        model.try_fragment(rank, rng.choice(sorted(model.try_pending[rank])))
                    elif action == 'enter':
                        model.enter_tail(rank)
                    elif action == 'tail':
                        ready = [g for g in model.tail_pending[rank] if model.can_read(rank, g // 16)]
                        self.assertTrue(model.tail_group(rank, rng.choice(ready)))
                    else:
                        model.arrive_reuse(rank)
                    steps += 1
                    self.assertLess(steps, 1000)
                self.assertEqual(len(model.reads), model.world * model.groups)


if __name__ == '__main__':
    unittest.main()
