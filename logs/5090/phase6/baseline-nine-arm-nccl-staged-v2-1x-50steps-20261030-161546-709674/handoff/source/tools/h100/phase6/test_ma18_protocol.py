"""Exhaustive interleavings for two-slot IPC staging and pull lifetimes.

Models the existing publication barrier's semantic contract, not a new epoch
implementation. CUDA system visibility and real peer-copy ordering additionally
require GPU validation. Arbitrarily delayed ranks and partial peer pulls are
represented by all enabled action orders.
"""
from collections import deque
from dataclasses import dataclass
import unittest


@dataclass(frozen=True)
class Result:
    states: int
    terminal_states: int
    violation: tuple | None


def explore(*, ranks=2, slots=2, calls=6, publication=True,
            reverse_second_rank=False, pull_before_exit=False):
    """Every rank executes stage -> arrive -> exit barrier -> pull each peer.

    Payload is local to its source rank and slot. A pull transfers a source
    snapshot to a separately owned output. Logical arrival generations express
    the barrier contract only; the implementation retains its binary barrier.
    Completing barrier k+1 implies every peer completed its pulls at k, which
    protects slot k before a producer can stage k+2 into that slot.
    """
    all_bits = (1 << ranks) - 1
    start = (tuple((0, 0, 0) for _ in range(ranks)),
             (-1,) * (slots * ranks), (-1,) * ranks)
    queue = deque([start])
    parents = {start: None}
    terminals = 0

    def witness(state, last):
        path = [last]
        while parents[state] is not None:
            state, action = parents[state]
            path.append(action)
        return tuple(reversed(path))

    while queue:
        state = queue.popleft()
        cursors, payload, arrived = state
        if all(call == calls for call, _, _ in cursors):
            terminals += 1
            continue
        progress = False
        for rank, (call, phase, done) in enumerate(cursors):
            if call == calls:
                continue
            slot = call % slots
            if reverse_second_rank and rank == 1:
                slot = slots - 1 - slot
            lanes = range(ranks) if phase == 3 else (None,)
            for source in lanes:
                data, arrivals = list(payload), list(arrived)
                next_call, next_phase, next_done = call, phase + 1, 0
                action = (rank, call, slot, ('stage', 'arrive', 'barrier_exit', 'pull')[phase], source)
                if phase == 0:
                    data[slot * ranks + rank] = call
                elif phase == 1:
                    arrivals[rank] = call
                elif phase == 2:
                    if publication and not pull_before_exit and any(a < call for a in arrived):
                        continue
                else:
                    bit = 1 << source
                    if done & bit:
                        continue
                    observed = payload[slot * ranks + source]
                    if observed != call:
                        return Result(len(parents), terminals,
                                      witness(state, action + (observed,)))
                    next_done = done | bit
                    if next_done == all_bits:
                        next_call, next_phase, next_done = call + 1, 0, 0
                    else:
                        next_phase = phase
                next_cursors = list(cursors)
                next_cursors[rank] = (next_call, next_phase, next_done)
                successor = (tuple(next_cursors), tuple(data), tuple(arrivals))
                progress = True
                if successor not in parents:
                    parents[successor] = state, action
                    queue.append(successor)
        if not progress:
            return Result(len(parents), terminals, witness(state, ('deadlock',)))
    return Result(len(parents), terminals, None)


class PullProtocolTests(unittest.TestCase):
    def test_two_ranks_three_reuses_each_slot(self):
        result = explore(calls=6)
        self.assertIsNone(result.violation, result.violation)
        self.assertGreater(result.states, 100)
        self.assertGreater(result.terminal_states, 0)

    def test_three_ranks_delayed_partial_pulls_and_slot_reuse(self):
        result = explore(ranks=3, calls=4)
        self.assertIsNone(result.violation, result.violation)
        self.assertGreater(result.states, 1000)
        self.assertGreater(result.terminal_states, 0)

    def test_single_slot_can_overwrite_before_a_peer_pull(self):
        result = explore(slots=1, calls=3)
        self.assertIsNotNone(result.violation)
        self.assertEqual(result.violation[-1][3], 'pull')
        # The failing source value is a future invocation, not uninitialized.
        self.assertGreater(result.violation[-1][-1], result.violation[-1][1])

    def test_missing_publication_reads_unstaged_source(self):
        result = explore(publication=False, calls=3)
        self.assertIsNotNone(result.violation)
        self.assertEqual(result.violation[-1][3], 'pull')
        self.assertEqual(result.violation[-1][-1], -1)

    def test_arrival_does_not_authorize_pull_before_barrier_exit(self):
        result = explore(pull_before_exit=True, calls=3)
        self.assertIsNotNone(result.violation)
        self.assertEqual(result.violation[-1][3], 'pull')

    def test_inconsistent_module_slot_sequence_is_detected(self):
        result = explore(reverse_second_rank=True, calls=3)
        self.assertIsNotNone(result.violation)
        self.assertEqual(result.violation[-1][3], 'pull')


if __name__ == '__main__':
    unittest.main()
