"""Check GPU coordinates, destination/window preservation and physical masks on TP8."""
import csv
import io
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
info = json.loads((root/'submission.json').read_text())
plan = json.loads((root/'scripts/selection-plan.json').read_text())
config = json.loads((root/'scripts/config.json').read_text())
case = config['cases'][0]
m, n, k_global = case['sequence']*config['common']['micro_batch'], case['hidden'], case['ffn']
tiles = (m//128)*(n//128)
per = tiles//8
assert plan['world'] == 8
build = Path(info['artifact_root'])/'build'
out = root/'results/mapping'
checks = []
for policy in info['mappings']:
    base = 'remote_first' if policy == 'remote_arrival' else 'interleaved'
    entry = next(e for e in plan['policies'][base] if e['shape'] == [m,n,k_global])
    for k, mlp in [(n, False), (k_global, True)]:
        run = subprocess.run([str(build/'mapping'/policy), str(m), str(n), str(k), '8', str(int(mlp))], capture_output=True, text=True, check=True)
        (out/f'{policy}-k{k}.csv').write_text(run.stdout)
        (out/f'{policy}-k{k}.log').write_text(run.stderr)
        rows = [{key: int(value) for key, value in row.items()} for row in csv.DictReader(io.StringIO(run.stdout))]
        assert len(rows) == 8*tiles
        for rank in range(8):
            rr = [r for r in rows if r['rank'] == rank]
            assert [r['index'] for r in rr] == list(range(tiles))
            coords = [r['m']*(n//128)+r['n'] for r in rr]
            assert sorted(coords) == list(range(tiles))
            if mlp:
                assert coords == entry['coords'][rank]
                previous = entry['base_coords'][rank]
                mapped = entry['maps'][rank]
                assert coords == [previous[j] for j in mapped]
                slots = {dst: [j for j, t in enumerate(previous) if t//per == dst] for dst in range(8)}
                positions = {slot: (dst, j//plan['window']) for dst, indices in slots.items() for j, slot in enumerate(indices)}
                assert all(positions[i] == positions[j] for i, j in enumerate(mapped))
                mask = entry['mask'][rank]
                assert len(mask) == tiles and not any(mask[rank*per:(rank+1)*per])
                assert sum(mask) <= int((tiles-per)*plan['budget_fraction'])
                assert {t for t in previous if mask[t]} == {t for t in coords if mask[t]}
        checks.append(dict(policy=policy, shape=[m, n, k], mlp=mlp, passed=True))
(out/'verification.json').write_text(json.dumps(dict(completed=True, world=8, checks=checks), indent=2)+'\n')
