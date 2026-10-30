"""Address coverage and all-interleavings checks for the MA12 epoch protocol.

The state model checks the protocol, not CUDA memory ordering. The host-native
test executes the actual header's producer/consumer address helpers; neither
test claims to execute device instructions or replace the eight-GPU smoke.
"""
from collections import deque
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


def explore(ranks, slots, rounds, guard_reuse=True):
    # A rank begins, publishes two fragments, consumes two fragments, and
    # publishes completion in stream order. Other ranks can pause at any stage.
    # This includes consuming a remote early fragment before its last fragment.
    width = ranks * slots
    start = ((0,) * ranks, (0,) * (width * 2), (0,) * width)
    pending, seen, terminal, overlapped = deque([start]), {start}, 0, False
    while pending:
        progress, payload, done = pending.popleft()
        if all(position == rounds * 6 for position in progress):
            terminal += 1
            continue
        successors = 0
        for rank, position in enumerate(progress):
            if position == rounds * 6:
                continue
            call, phase = divmod(position, 6)
            slot, epoch = call % slots, call // slots + 1
            values, completions = list(payload), list(done)
            if phase == 0:
                if guard_reuse and any(done[slot * ranks + peer] < epoch - 1
                                       for peer in range(ranks)):
                    continue
            elif phase in (1, 2):
                fragment = phase - 1
                values[(slot * ranks + rank) * 2 + fragment] = epoch
            elif phase in (3, 4):
                fragment = phase - 3
                sources = [payload[(slot * ranks + peer) * 2 + fragment]
                           for peer in range(ranks)]
                if min(sources) < epoch:
                    continue
                # An acquire may accept >=epoch only if a future writer cannot
                # overwrite a payload still needed by this reader.
                if any(value != epoch for value in sources):
                    return dict(corruption=True, states=len(seen), terminal=terminal)
                if fragment == 0 and any(
                        payload[(slot * ranks + peer) * 2 + 1] < epoch
                        for peer in range(ranks)):
                    overlapped = True
            else:
                completions[slot * ranks + rank] = epoch
            positions = list(progress)
            positions[rank] += 1
            state = tuple(positions), tuple(values), tuple(completions)
            successors += 1
            if state not in seen:
                seen.add(state)
                pending.append(state)
        if not successors:
            return dict(deadlock=True, states=len(seen), terminal=terminal)
    return dict(corruption=False, deadlock=False, states=len(seen),
                terminal=terminal, remote_fragment_overlap_observed=overlapped)


class EpochProtocol(unittest.TestCase):
    def test_double_slot_arbitrary_three_rank_skew_has_no_deadlock_or_overwrite(self):
        result = explore(ranks=3, slots=2, rounds=5)
        self.assertFalse(result['corruption'])
        self.assertFalse(result['deadlock'])
        self.assertGreater(result['states'], 1000)
        self.assertEqual(result['terminal'], 1)
        self.assertTrue(result['remote_fragment_overlap_observed'])

    def test_reuse_guard_also_protects_one_slot(self):
        result = explore(ranks=2, slots=1, rounds=4)
        self.assertFalse(result['corruption'])
        self.assertFalse(result['deadlock'])
        self.assertEqual(result['terminal'], 1)

    def test_omitting_completion_guard_can_expose_future_payload(self):
        # Negative control: the model must detect the race it claims to guard.
        self.assertTrue(explore(ranks=2, slots=1, rounds=3,
                                guard_reuse=False)['corruption'])


class ActualNativeAddressing(unittest.TestCase):
    def test_every_fragment_producer_matches_all_sixteen_consumer_rows(self):
        compiler = os.environ.get('NVCC') or shutil.which('nvcc')
        if compiler is None:
            candidate = Path('/data/apps/cuda/12.8/bin/nvcc')
            if candidate.is_file():
                compiler = str(candidate)
        self.assertIsNotNone(compiler, 'CUDA compiler required for native host-address audit')
        header = Path(__file__).with_name('ma12_tile_ready.cuh').resolve()
        with tempfile.TemporaryDirectory(prefix='ma12-addressing-') as temp:
            root = Path(temp)
            (root / 'gemm_rs').mkdir()
            # Only the struct fields used by the header are needed to compile
            # its host helpers. No device function is launched or simulated.
            (root / 'gemm_rs/taco_runtime.h').write_text(
                '#include <cuda_runtime.h>\n'
                'struct FluxTacoConfig { int* ma12_ready[8]; int ma12_epoch; int rank; };\n')
            source = root / 'addresses.cu'
            source.write_text('#include "' + str(header) + '"\n' + r'''
#include <array>
#include <cstdio>
int main() {
  if (taco_ma12::kWords != 4097 || taco_ma12::kReadyWords != 4096) return 1;
  for (int destination = 0; destination < 8; ++destination) {
    std::array<int, 4096> coverage{};
    for (int source = 0; source < 8; ++source)
      for (int local_tile = 0; local_tile < 64; ++local_tile)
        for (int fragment = 0; fragment < 8; ++fragment)
          for (int compact = 0; compact < 16; ++compact) {
            const int physical_tile = destination * 64 + local_tile;
            const int tile_m = physical_tile / 32, tile_n = physical_tile % 32;
            // Original encoder fragment owns [8*f,8*f+7] and that strip+64.
            const int row = fragment * 8 + compact % 8 + (compact / 8) * 64;
            const int owned_row = tile_m * 128 + row - destination * 256;
            const int group = owned_row * 32 + tile_n;
            const int producer = taco_ma12::ready_index(source, local_tile, fragment);
            const int consumer = taco_ma12::consumer_ready_index(source, group);
            if (producer != consumer || consumer < 0 || consumer >= 4096) return 2;
            ++coverage[consumer];
          }
    for (int count : coverage) if (count != 16) return 3;
  }
  std::puts("8 receivers x 4096 flags x 16 rows: full bijective fragment coverage");
}
''')
            binary = root / 'addresses'
            subprocess.run([compiler, '-std=c++17', '-arch=sm_90', '-I', str(root),
                            str(source), '-o', str(binary)], check=True,
                           text=True, capture_output=True)
            result = subprocess.run([str(binary)], check=True, text=True, capture_output=True)
            self.assertIn('full bijective fragment coverage', result.stdout)


if __name__ == '__main__':
    unittest.main()
