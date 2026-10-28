"""CPU-only protocol for node-local TP4 groups with DP across two nodes."""
import hashlib
import json
from pathlib import Path

POLICIES = ['native', 'original', 'native_taco', 'taco_fused',
            'remote_arrival_selective', 'interleaved_arrival_selective']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def topology(world, local, tp, dp):
    if (world, local, tp, dp) != (8, 4, 4, 2):
        raise ValueError('This runner requires two complete node-local TP4 groups, DP2')
    return dict(world=world, local=local, tp=tp, dp=dp,
                tp_groups=[[0, 1, 2, 3], [4, 5, 6, 7]],
                dp_groups=[[i, i + 4] for i in range(4)])


def orders(round_number):
    rows = [POLICIES[i:] + POLICIES[:i] for i in range(len(POLICIES))]
    if round_number == 2:
        rows = [list(reversed(row)) for row in reversed(rows)]
    elif round_number != 1:
        raise ValueError('Expected round 1 or 2')
    return rows


def replica_checks(rows):
    assert [r['rank'] for r in rows] == list(range(8))
    assert len({r['gpu_uuid'] for r in rows}) == 8
    assert len({r['host'] for r in rows}) == 2
    assert len({r['global_tokens_sha256'] for r in rows}) == 1
    for rank, r in enumerate(rows):
        assert r['local_rank'] == r['tp_rank'] == rank % 4
        assert r['dp_rank'] == rank // 4
        assert r['tp_group_ranks'] == list(range((rank // 4)*4, (rank // 4 + 1)*4))
        assert r['dp_group_ranks'] == [rank % 4, rank % 4 + 4]
    for node in range(2):
        rr = rows[node*4:(node+1)*4]
        assert len({r['host'] for r in rr}) == 1
        assert len({r['tokens_sha256'] for r in rr}) == 1
    assert rows[0]['tokens_sha256'] != rows[4]['tokens_sha256'], 'DP replicas must see different data'
    for tp in range(4):
        for key in ('initial_parameters_sha256', 'final_parameters_sha256'):
            assert rows[tp][key] == rows[tp+4][key], (tp, key, 'DP replicas diverged')
        if rows[tp]['mode'] == 'smoke':
            assert rows[tp]['synced_gradients_sha256'] == rows[tp+4]['synced_gradients_sha256']
