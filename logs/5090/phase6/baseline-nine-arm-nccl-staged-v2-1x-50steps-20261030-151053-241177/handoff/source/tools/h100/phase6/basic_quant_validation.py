"""Smoke-only receiver fusion check against the original CUDA decode + BF16 adds."""
import baseline_codec_validation as upstream

CASES = upstream.CASES


def validate(codec, reference, rank):
    import torch
    record = upstream.validate(codec, reference)
    gen = torch.Generator(device='cuda').manual_seed(20261025 + rank)
    checks, retained = [], []
    for case in CASES:
        rows, n = (256, 4096) if case == 'full_shape' else (8, 128)
        partials = torch.randn((8, rows, n), generator=gen, device='cuda', dtype=torch.bfloat16)
        if case == 'zeros':
            partials.zero_()
        elif case == 'small':
            partials.mul_(1e-6)
        elif case == 'outliers':
            partials[:, :, ::128] = 64
        packets = codec.pack_rows(torch.cat([partials[s] for s in range(8) if s != rank]), 7)
        decoded = codec.unpack_rows(packets, rows, n)
        expected = torch.zeros_like(partials[rank])
        for offset in range(1, 9):
            src = (rank + offset) % 8
            value = partials[src] if src == rank else decoded[src - int(src > rank)]
            expected = (expected + value).to(torch.bfloat16)
        actual = codec.decode_reduce(packets, partials[rank], rank)
        equal = torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        if not equal:
            raise RuntimeError('Receiver fusion changed CUDA baseline rounding/order: ' + case)
        checks.append(dict(case=case, rank=rank, shape=[rows, n], bitwise_equal=True))
        retained.append((actual, actual.clone()))
    if not all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in retained):
        raise RuntimeError('Receiver fusion overwrote a retained output')
    record['receiver_fusion'] = dict(passed=True, rank=rank, cases=checks,
                                    retained_outputs_unchanged=True, synthetic=False)
    return record


def verify(record, library, rank):
    from common import require
    upstream.verify(record, library)
    check = record.get('receiver_fusion', {})
    require(check.get('passed') is True and check.get('synthetic') is False and
            check.get('rank') == rank and check.get('retained_outputs_unchanged') is True and
            check.get('cases') == [dict(case=c, rank=rank, shape=[256, 4096] if c == 'full_shape' else [8, 128],
                                       bitwise_equal=True) for c in CASES],
            'Missing/failed receiver fusion equivalence precheck')
