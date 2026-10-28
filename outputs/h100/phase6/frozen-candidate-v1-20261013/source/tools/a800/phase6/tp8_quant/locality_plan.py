"""Restrict the frozen tail priority to smaller legal destination windows.

The TP8 source plan sorted all 32 slots of each destination (window64).
Its inverse permutation therefore gives the same stable priority order inside
any smaller window, without accessing the held-out samples or changing masks.
"""
import argparse
import hashlib
import json
from pathlib import Path

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main(args):
    source, out = args.source.resolve(), args.out.resolve()
    plan = json.loads(source.read_text())
    assert plan['world'] == 8 and plan['window'] == 64 and plan['schema'] == 2
    assert args.window in (8,16)
    for name, digest in plan['inputs'].items():
        assert sha(Path(name)) == digest, name
    for entries in plan['policies'].values():
        for entry in entries:
            assert entry['tiles'] == 256
            per = entry['tiles']//8
            for rank in range(8):
                old = entry['maps'][rank]
                inverse = {tile:slot for slot,tile in enumerate(old)}
                assert set(inverse) == set(range(entry['tiles']))
                base = entry['base_coords'][rank]
                order = list(range(entry['tiles']))
                for dst in range(8):
                    slots = [i for i,tile in enumerate(base) if tile//per == dst]
                    assert len(slots) == per
                    for start in range(0,len(slots),args.window):
                        window = slots[start:start+args.window]
                        ranked = sorted(window,key=inverse.__getitem__)
                        for slot,tile in zip(window,ranked):
                            order[slot] = tile
                assert sorted(order) == list(range(entry['tiles']))
                entry['maps'][rank] = order
                entry['coords'][rank] = [base[t] for t in order]
                entry['changed_per_rank'][rank] = sum(i != t for i,t in enumerate(order))
    plan.update(window=args.window,
        limitation=f'Frozen TP8 BF16 tail priority restricted to {args.window} destination slots; identical physical quantization mask. No new calibration, no held-out tuning, no post-quantization recalibration.',
        locality_parent=dict(path=str(source),sha256=sha(source)))
    plan['inputs'][str(source)] = sha(source)
    plan['inputs'][str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    out.parent.mkdir(parents=True,exist_ok=True)
    assert not out.exists()
    out.write_text(json.dumps(plan,indent=2)+'\n')
    print(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('--window',type=int,choices=[8,16],default=16)
    main(p.parse_args())
