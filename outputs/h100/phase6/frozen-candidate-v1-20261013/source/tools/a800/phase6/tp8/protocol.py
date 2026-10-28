"""Fixed two-node TP8 BF16 measurement protocol (not quantization acceptance)."""
import hashlib
import json
from pathlib import Path

POLICIES=['native','original','remote_first','interleaved']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def orders(repetition):
    rows=[POLICIES[i:]+POLICIES[:i] for i in range(4)]
    assert repetition in (1,2)
    return rows if repetition==1 else [list(reversed(r)) for r in reversed(rows)]


def validate_topology(rows):
    assert [r['rank'] for r in rows]==list(range(8))
    assert len({r['host'] for r in rows})==2 and len({r['gpu_uuid'] for r in rows})==8
    assert len({r['tokens_sha256'] for r in rows})==len({r['global_tokens_sha256'] for r in rows})==1
    for rank,r in enumerate(rows):
        assert r['local_rank']==rank%4 and r['tp_rank']==rank and r['dp_rank']==0
        assert r['tp_group_ranks']==list(range(8)) and r['dp_group_ranks']==[rank]
        if r['policy']!='native':assert r['local_group_ranks']==list(range(rank//4*4,rank//4*4+4))
    for node in range(2):assert len({r['host'] for r in rows[node*4:node*4+4]})==1
