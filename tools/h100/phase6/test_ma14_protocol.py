"""Exhaustive small-model checks for MA14's two-slot epoch protocol.

This checks payload lifetime under arbitrary rank/lane interleavings, rather
than inspecting CUDA text. It does not model CUDA's memory system: production
code must additionally use system release/acquire and the fixed-stream contract.
"""
from collections import deque
from dataclasses import dataclass
import unittest


@dataclass(frozen=True)
class Result:
    states: int
    terminal_states: int
    overshoots: int
    violation: tuple | None


def explore(*, ranks=2, slots=2, calls=4, rendezvous=True,
            constant_epoch=False, reverse_second_rank=False):
    """Explore every enabled single-lane action, using independent IPC slots.

    Each rank executes producer -> publish -> acquire all peers -> decoder on
    one stream. Producer destinations, publication lanes, acquired sources and
    decoder source loads can independently interleave. Payload is indexed by
    (slot, receiver, source), just like the data protected by the ready words.
    ``calls`` counts global invocations, not the private epoch of either slot.

    Negative variants deliberately violate a protocol precondition. They are
    required to produce counterexamples, so a vacuous model cannot pass.
    """
    all_bits = (1 << ranks) - 1
    words = slots * ranks * ranks
    # rank cursor = (invocation, stage, completed lanes); stages are ordered.
    start = (tuple((0, 0, 0) for _ in range(ranks)),
             (-1,) * words, (0,) * words)
    queue = deque([start])
    parents = {start: None}
    terminal = overshoots = 0

    def witness(state, last):
        path = [last]
        while parents[state] is not None:
            state, action = parents[state]
            path.append(action)
        return tuple(reversed(path))

    while queue:
        state = queue.popleft()
        cursors, payload, ready = state
        if all(c == calls for c, _, _ in cursors):
            terminal += 1
            continue
        progressed = False
        for rank, (call, stage, done) in enumerate(cursors):
            if call == calls:
                continue
            slot = call % slots
            if reverse_second_rank and rank == 1:
                slot = slots - 1 - slot
            epoch = 1 if constant_epoch else call // slots + 1
            for lane in range(ranks):
                bit = 1 << lane
                if done & bit:
                    continue
                data, flags = list(payload), list(ready)
                # For stores lane names a receiver; for reads it names source.
                receiver, source = ((lane, rank) if stage < 2 else (rank, lane))
                index = (slot * ranks + receiver) * ranks + source
                action = (rank, call, slot, epoch,
                          ('write', 'publish', 'acquire', 'read')[stage], lane)
                if stage == 0:
                    data[index] = call
                elif stage == 1:
                    flags[index] = epoch
                elif stage == 2:
                    if ready[index] < epoch:
                        continue
                    if ready[index] > epoch:
                        overshoots += 1
                elif payload[index] != call:
                    return Result(len(parents), terminal, overshoots,
                                  witness(state, action + (payload[index],)))

                next_stage, next_done, next_call = stage, done | bit, call
                if next_done == all_bits:
                    next_stage, next_done = stage + 1, 0
                    if next_stage == 2 and not rendezvous:
                        next_stage = 3
                    if next_stage == 4:
                        next_call, next_stage = call + 1, 0
                next_cursors = list(cursors)
                next_cursors[rank] = (next_call, next_stage, next_done)
                successor = (tuple(next_cursors), tuple(data), tuple(flags))
                progressed = True
                if successor not in parents:
                    parents[successor] = state, action
                    queue.append(successor)
        if not progressed:
            return Result(len(parents), terminal, overshoots,
                          witness(state, ('deadlock',)))
    return Result(len(parents), terminal, overshoots, None)


class EpochProtocolTests(unittest.TestCase):
    def test_all_two_rank_interleavings_preserve_reused_payload(self):
        # Six calls visit each separate slot three times. A rank may enqueue
        # arbitrarily far ahead; stream order only enables its next action.
        result = explore(calls=6)
        self.assertIsNone(result.violation, result.violation)
        self.assertGreater(result.states, 1000)
        self.assertGreater(result.terminal_states, 0)
        # A rank cannot publish the next epoch of this slot before all peers
        # have completed this slot's prior decoder and entered the other slot.
        self.assertEqual(result.overshoots, 0)

    def test_three_rank_rendezvous(self):
        # A third rank exercises peer lanes beyond a two-rank handshake;
        # the third invocation also reuses the first slot.
        result = explore(ranks=3, calls=3)
        self.assertIsNone(result.violation, result.violation)
        self.assertGreater(result.terminal_states, 0)
        self.assertEqual(result.overshoots, 0)

    def test_single_slot_overwrites_a_slow_consumer(self):
        result = explore(slots=1, calls=2)
        self.assertIsNotNone(result.violation)
        self.assertEqual(result.violation[-1][4], 'read')
        self.assertGreater(result.overshoots, 0)

    def test_missing_rendezvous_can_read_unpublished_payload(self):
        result = explore(rendezvous=False, calls=2)
        self.assertIsNotNone(result.violation)
        self.assertEqual(result.violation[-1][4], 'read')

    def test_reusing_a_boolean_epoch_admits_stale_publications(self):
        result = explore(constant_epoch=True, calls=4)
        self.assertIsNotNone(result.violation)
        self.assertEqual(result.violation[-1][4], 'read')

    def test_inconsistent_slot_order_fails_collective_progress(self):
        result = explore(reverse_second_rank=True, calls=2)
        self.assertIsNotNone(result.violation)
        self.assertEqual(result.violation[-1], ('deadlock',))


if __name__ == '__main__':
    unittest.main()
