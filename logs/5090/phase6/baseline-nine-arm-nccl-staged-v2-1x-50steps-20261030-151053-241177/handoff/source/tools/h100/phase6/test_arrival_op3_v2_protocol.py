"""CPU protocol model for bounded op3 v2; not GPU race/performance evidence.

The model separates checking stop, completing a group, and joining the auxiliary
stream. Once a warp observes stop, only its already checked group can remain.
Stop loads default to the latest epoch; a test can supply an older observation
to model propagation delay. There is no wall-clock bound on that propagation.
Memory visibility below models the documented barrier/release/acquire contract;
it does not emulate the CUDA memory system or validate codec arithmetic.
"""
import random
import unittest


def group_of(fragment, compact):
    tile, step = divmod(fragment, 8)
    row = (tile // 32) * 128 + step * 8 + compact % 8 + (compact // 8) * 64
    return row * 32 + tile % 32


def fragment_of(group):
    row, column = divmod(group, 32)
    return ((row // 128) * 32 + column) * 8 + (row % 64) // 8


class ProtocolError(RuntimeError):
    pass


class TryCTA:
    def __init__(self, model, fragment, reader_barrier=True):
        self.model = model
        self.fragment = fragment
        self.reader_barrier = reader_barrier
        self.position = [0] * 4
        self.inflight = {}
        self.closed = False

    def claim(self, warp, stop_observation=None):
        if self.closed or warp in self.inflight:
            raise ProtocolError("warp cannot claim before its previous group completes")
        while self.position[warp] < 4:
            observed = self.model.stop if stop_observation is None else stop_observation
            if observed == self.model.epoch:
                self.position[warp] = 4
                return None
            compact = warp + 4 * self.position[warp]
            self.position[warp] += 1
            group = group_of(self.fragment, compact)
            if self.model.done.get(group) == self.model.epoch:
                continue
            self.inflight[warp] = group
            return group
        return None

    def complete(self, warp):
        group = self.inflight[warp]
        self.model.consume(group, "try", self.reader_barrier)
        self.model.done[group] = self.model.epoch
        del self.inflight[warp]

    def close(self):
        if self.closed or self.inflight or self.position != [4] * 4:
            raise ProtocolError("auxiliary CTA still owns unfinished work")
        self.closed = True
        self.model.active.remove(self)
        self.model.finish_cta(self.fragment)

    def run(self):
        for warp in range(4):
            while self.claim(warp) is not None:
                self.complete(warp)
        self.close()


class V2Model:
    """One receiver, eight independent producer ranks, two ordered try rounds."""

    def __init__(self, fragments=512, budget=64):
        self.fragments = fragments
        self.budget = min(budget, fragments)
        self.groups = {group_of(f, c) for f in range(fragments) for c in range(16)}
        self.ready = {}
        self.visible = {}
        self.done = {}
        self.stop = 0
        self.epoch = 0
        self.joined = True
        self.outputs = {g: "initial" for g in self.groups}
        self.begin(1)

    def begin(self, epoch):
        if epoch != self.epoch + 1 or not 0 < epoch <= 0x7FFFFFFF:
            raise ProtocolError("epoch must increase without repetition or wrap")
        if not self.joined or set(self.outputs) != self.groups:
            raise ProtocolError("workspace cannot be reused before all consumers finish")
        self.epoch = epoch
        self.outputs = {}
        self.pending = {(src, frag) for src in range(8)
                        for frag in range(self.fragments)}
        self.round = 0
        self.pending_try = set(range(self.budget))
        self.active = set()
        self.local_ended = False
        self.joined = False
        self.tail_entered = False
        # In particular, do not clear old stop, ready, or done epochs.

    def publish(self, src, fragment, writer_barrier=True):
        key = src, fragment
        if key not in self.pending:
            raise ProtocolError("producer publishes a fragment twice")
        self.pending.remove(key)
        # A collective barrier makes all writers precede leader release.sys.
        # Without it, an adversarial schedule may expose only leader's writes.
        self.visible[key] = (self.epoch, 4 if writer_barrier else 1)
        self.ready[key] = self.epoch

    def publish_rank(self, src):
        for rank, fragment in sorted(self.pending):
            if rank == src:
                self.publish(src, fragment)

    def is_ready(self, fragment):
        return all(self.ready.get((src, fragment)) == self.epoch for src in range(8))

    def consume(self, group, phase, reader_barrier=True):
        fragment = fragment_of(group)
        if not self.is_ready(fragment):
            raise ProtocolError("consumer read incomplete or stale producer data")
        if any(self.visible.get((src, fragment)) != (self.epoch, 4)
               for src in range(8)):
            raise ProtocolError("leader published before other payload writers")
        if not reader_barrier:
            raise ProtocolError("leader acquire was not broadcast to payload readers")
        if group in self.outputs:
            raise ProtocolError("an output group was decoded twice")
        self.outputs[group] = phase

    def finish_cta(self, fragment):
        self.pending_try.remove(fragment)
        if not self.pending_try:
            if self.active:
                raise ProtocolError("next round cannot overtake an active CTA")
            self.round += 1
            if self.round < 2:
                self.pending_try = set(range(self.budget))

    def open_cta(self, fragment, reader_barrier=True):
        if (self.round >= 2 or fragment not in self.pending_try or
                any(cta.fragment == fragment for cta in self.active)):
            raise ProtocolError("CTA is outside its bounded ordered round")
        if self.stop == self.epoch or not self.is_ready(fragment):
            self.finish_cta(fragment)  # Miss exits; it never polls readiness.
            return None
        cta = TryCTA(self, fragment, reader_barrier)
        self.active.add(cta)
        return cta

    def stop_after_local_gemm(self):
        if any(src == 0 for src, _ in self.pending):
            raise ProtocolError("stop cannot precede local GEMM completion")
        self.local_ended = True
        self.stop = self.epoch

    def join(self):
        if not self.local_ended or self.round != 2 or self.active:
            raise ProtocolError("tail requires local GEMM and auxiliary stream join")
        self.joined = True
        self.tail_entered = True

    def tail(self, group):
        if not self.tail_entered:
            raise ProtocolError("tail cannot run before stream join")
        if self.done.get(group) == self.epoch:
            if group not in self.outputs:
                raise ProtocolError("done became visible before output")
            return True
        if not self.is_ready(fragment_of(group)):
            return False
        self.consume(group, "tail")
        return True

    def drain_try(self):
        for cta in tuple(self.active):
            cta.run()
        while self.round < 2:
            cta = self.open_cta(min(self.pending_try))
            if cta:
                cta.run()

    def finish(self):
        for src in range(8):
            self.publish_rank(src)
        self.stop_after_local_gemm()
        self.drain_try()
        self.join()
        for group in sorted(self.groups):
            if group not in self.outputs or self.done.get(group) == self.epoch:
                if not self.tail(group):
                    raise ProtocolError("tail failed after every producer completed")


class ArrivalOp3V2ProtocolTests(unittest.TestCase):
    def test_four_warp_mapping_is_a_bijection_over_all_8192_groups(self):
        owner = {}
        for fragment in range(512):
            for warp in range(4):
                for iteration in range(4):
                    group = group_of(fragment, warp + 4 * iteration)
                    self.assertNotIn(group, owner)
                    owner[group] = fragment, warp, iteration
                    self.assertEqual(fragment_of(group), fragment)
        self.assertEqual(set(owner), set(range(8192)))
        self.assertEqual(len({g for g, (f, _, _) in owner.items() if f < 64}), 1024)

    def test_control_layout_alignment_and_unused_source_slot_capacity(self):
        ready_bytes = 8 * 512 * 4
        done_bytes = 8192 * 4
        stop_offset = ready_bytes + done_bytes
        stats_offset = stop_offset + 4 + 4
        control_bytes = stats_offset + 7 * 8
        self.assertEqual((stop_offset, stats_offset, control_bytes), (49152, 49160, 49216))
        self.assertEqual(stats_offset % 8, 0)
        source_bytes = 8192 * 136
        self.assertLess(control_bytes, source_bytes)
        for dst in range(8):
            control = range(dst * source_bytes, dst * source_bytes + control_bytes)
            for src in range(8):
                if src != dst:
                    self.assertTrue(control.stop <= src * source_bytes or
                                    control.start >= (src + 1) * source_bytes)

    def test_bounded_two_rounds_decode_at_most_1024_groups(self):
        model = V2Model()
        for src in range(8):
            model.publish_rank(src)
        model.drain_try()
        self.assertEqual(model.round, 2)
        self.assertEqual(len(model.outputs), 1024)
        self.assertEqual(set(model.outputs),
                         {group_of(f, c) for f in range(64) for c in range(16)})
        model.finish()
        self.assertEqual(len(model.outputs), 8192)

    def test_partial_done_must_not_skip_the_remaining_fragment_in_tail(self):
        model = V2Model(fragments=2)
        for src in range(8):
            model.publish(src, 0)
        cta = model.open_cta(0)
        early_group = cta.claim(2)
        cta.complete(2)
        model.publish_rank(0)
        model.stop_after_local_gemm()
        cta.run()
        model.drain_try()
        model.join()
        for compact in range(16):
            self.assertTrue(model.tail(group_of(0, compact)))
        self.assertEqual(sum(phase == "try" for phase in model.outputs.values()), 1)
        self.assertEqual(sum(phase == "tail" for phase in model.outputs.values()), 15)
        self.assertEqual(model.done[early_group], model.epoch)
        self.assertEqual(len(model.done), 1)

    def test_stop_race_allows_one_already_claimed_group_per_warp(self):
        model = V2Model(fragments=2)
        for src in range(8):
            model.publish(src, 0)
        cta = model.open_cta(0)
        claimed = {cta.claim(warp) for warp in range(4)}
        model.publish_rank(0)
        model.stop_after_local_gemm()
        for warp in range(4):
            cta.complete(warp)
            self.assertIsNone(cta.claim(warp))
        self.assertEqual(set(model.outputs), claimed)
        self.assertEqual(len(claimed), 4)
        cta.close()
        model.finish()

    def test_stop_publication_time_does_not_imply_immediate_warp_observation(self):
        model = V2Model(fragments=2)
        for src in range(8):
            model.publish(src, 0)
        cta = model.open_cta(0)
        cta.claim(0)
        model.publish_rank(0)
        model.stop_after_local_gemm()
        cta.complete(0)
        # The first post-store check can still precede observing the stop.
        self.assertIsNotNone(cta.claim(0, stop_observation=0))
        cta.complete(0)
        self.assertIsNone(cta.claim(0, stop_observation=model.epoch))
        # Once stopped, no later attempt can reopen this warp's iteration.
        self.assertIsNone(cta.claim(0, stop_observation=0))
        self.assertEqual(len(model.outputs), 2)
        cta.run()
        model.finish()

    def test_second_claim_before_completion_is_rejected(self):
        model = V2Model(fragments=1)
        for src in range(8):
            model.publish_rank(src)
        cta = model.open_cta(0)
        cta.claim(0)
        with self.assertRaisesRegex(ProtocolError, "previous group"):
            cta.claim(0)

    def test_second_round_reuses_group_done_without_duplicate_output(self):
        model = V2Model(fragments=2)
        for src in range(8):
            model.publish(src, 0)
        model.open_cta(0).run()
        self.assertIsNone(model.open_cta(1))
        self.assertEqual(model.round, 1)
        for src in range(8):
            model.publish(src, 1)
        model.drain_try()
        self.assertEqual(len(model.outputs), 32)
        self.assertEqual(set(model.outputs.values()), {"try"})
        model.finish()

    def test_stop_before_auxiliary_launch_leaves_all_work_for_tail(self):
        model = V2Model(fragments=3)
        for src in range(8):
            model.publish_rank(src)
        model.stop_after_local_gemm()
        model.drain_try()
        self.assertFalse(model.outputs)
        model.finish()
        self.assertEqual(set(model.outputs.values()), {"tail"})

    def test_auxiliary_before_any_producer_misses_twice_without_polling(self):
        model = V2Model(fragments=3)
        model.drain_try()
        self.assertEqual(model.round, 2)
        self.assertEqual(len(model.pending), 24)
        self.assertFalse(model.outputs)
        model.finish()
        self.assertEqual(set(model.outputs.values()), {"tail"})

    def test_tail_can_wait_for_remote_rank_only_after_local_gemm_and_join(self):
        model = V2Model(fragments=2)
        with self.assertRaisesRegex(ProtocolError, "local GEMM"):
            model.stop_after_local_gemm()
        model.publish_rank(0)
        model.stop_after_local_gemm()
        with self.assertRaisesRegex(ProtocolError, "stream join"):
            model.join()
        model.drain_try()
        model.join()
        self.assertFalse(model.tail(group_of(0, 0)))
        self.assertTrue(all(src != 0 for src, _ in model.pending))
        for src in range(1, 8):
            model.publish(src, 0)
        self.assertTrue(model.tail(group_of(0, 0)))

    def test_old_stop_ready_and_done_cannot_affect_a_new_epoch(self):
        model = V2Model(fragments=2)
        for src in range(8):
            model.publish_rank(src)
        model.drain_try()
        model.finish()
        previous_done = dict(model.done)
        model.begin(2)
        self.assertEqual(model.stop, 1)
        self.assertEqual(model.done, previous_done)
        self.assertIsNone(model.open_cta(1))  # Stale ready cannot authorize reads.
        for src in range(8):
            model.publish(src, 0)
        cta = model.open_cta(0)
        self.assertIsNotNone(cta)  # Stale stop must not suppress new ready work.
        cta.run()
        self.assertEqual(len(model.outputs), 16)  # Stale done must not skip it.
        model.finish()

    def test_epoch_reuse_repetition_zero_and_wrap_are_rejected(self):
        model = V2Model(fragments=1)
        with self.assertRaisesRegex(ProtocolError, "consumers"):
            model.begin(2)
        model.finish()
        for epoch in (0, 1, 3, 0x80000000):
            with self.subTest(epoch=epoch), self.assertRaises(ProtocolError):
                model.begin(epoch)

    def test_missing_producer_barrier_has_a_visibility_counterexample(self):
        model = V2Model(fragments=1)
        for src in range(8):
            model.publish(src, 0, writer_barrier=(src != 3))
        cta = model.open_cta(0)
        cta.claim(0)
        with self.assertRaisesRegex(ProtocolError, "payload writers"):
            cta.complete(0)

    def test_missing_consumer_barrier_has_a_visibility_counterexample(self):
        model = V2Model(fragments=1)
        for src in range(8):
            model.publish_rank(src)
        cta = model.open_cta(0, reader_barrier=False)
        cta.claim(1)
        with self.assertRaisesRegex(ProtocolError, "payload readers"):
            cta.complete(1)

    def test_random_publish_claim_complete_and_stop_interleavings(self):
        for seed in range(32):
            rng = random.Random(seed)
            model = V2Model(fragments=4, budget=3)
            for epoch in (1, 2, 3):
                if epoch > 1:
                    model.begin(epoch)
                while model.pending or model.round < 2 or not model.local_ended:
                    choices = []
                    if model.pending:
                        choices.append(("publish", None))
                    if not model.local_ended and not any(src == 0 for src, _ in model.pending):
                        choices.append(("stop", None))
                    unopened = model.pending_try - {c.fragment for c in model.active}
                    if model.round < 2 and unopened:
                        choices.append(("open", None))
                    for cta in model.active:
                        for warp in range(4):
                            if warp in cta.inflight:
                                choices.append(("complete", (cta, warp)))
                            elif cta.position[warp] < 4:
                                choices.append(("claim", (cta, warp)))
                        if cta.position == [4] * 4 and not cta.inflight:
                            choices.append(("close", cta))
                    self.assertTrue(choices, "finite try protocol cannot deadlock")
                    action, value = rng.choice(choices)
                    if action == "publish":
                        model.publish(*rng.choice(sorted(model.pending)))
                    elif action == "stop":
                        model.stop_after_local_gemm()
                    elif action == "open":
                        model.open_cta(rng.choice(sorted(unopened)))
                    elif action == "close":
                        value.close()
                    else:
                        cta, warp = value
                        getattr(cta, action)(warp)
                model.finish()
                self.assertEqual(set(model.outputs), model.groups)


if __name__ == "__main__":
    unittest.main()
