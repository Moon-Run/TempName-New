"""GPU codec preflight outside timing, using an independent tensor reference."""

CASES = ('zeros', 'small', 'outliers', 'full_shape')


def validate(codec, reference):
    import torch
    gen = torch.Generator(device='cuda').manual_seed(20261021)
    rows, retained, snapshots = [], [], []
    for case in CASES:
        # The final case uses exactly the seven remote packets of each projection.
        chunks, count, n = (7, 256, 4096) if case == 'full_shape' else (7, 8, 128)
        x = torch.randn((chunks*count, n), generator=gen, device='cuda', dtype=torch.bfloat16)
        if case == 'zeros': x.zero_()
        elif case == 'small': x.mul_(1e-6)
        elif case == 'outliers': x[:, ::128] = 64
        packet = codec.pack_rows(x, chunks)
        actual = codec.unpack_rows(packet, count, n).reshape_as(x)
        q, scale, adaptive, _ = reference.encode(x)
        oracle = reference.decode(q, scale, adaptive, n)
        denominator = x.float().norm().clamp_min(1e-20)
        error = float((actual.float()-x.float()).norm()/denominator)
        protocol = float((actual.float()-oracle.float()).norm()/denominator)
        # Independently interpret actual wire bytes, including both scale arrays.
        elements = count*n; groups = elements//128
        pq = packet[:, :elements].contiguous().view(torch.float8_e4m3fn).reshape(chunks*count, n//128, 128)
        ps = packet[:, elements:elements+groups*4].contiguous().view(torch.float32).reshape(chunks*count, n//128)
        pa = packet[:, elements+groups*4:].contiguous().view(torch.float32).reshape(chunks*count, n//128)
        wire_oracle = reference.decode(pq, ps, pa, n)
        wire_error = float((actual.float()-wire_oracle.float()).norm()/denominator)
        finite = bool(torch.isfinite(actual).all() and torch.isfinite(ps).all() and torch.isfinite(pa).all())
        row = dict(case=case, shape=list(x.shape), chunks=chunks, relative_l2=error,
                   protocol_relative_l2=protocol, packet_relative_l2=wire_error, finite=finite,
                   budget_vs_bf16=.05, budget_vs_codec=.01, budget_vs_packet=.005)
        row['passed'] = finite and error <= .05 and protocol <= .01 and wire_error <= .005
        if not row['passed']: raise RuntimeError('Upstream codec preflight failed: '+repr(row))
        rows.append(row); retained.append(actual); snapshots.append(actual.clone())
    # Fresh allocations must not alias output kept from previous calls.
    preservation = all(torch.equal(a, b) for a, b in zip(retained, snapshots))
    if not preservation: raise RuntimeError('Codec overwrote a retained output')
    return dict(passed=True, cases=rows, retained_outputs_unchanged=True, synthetic=False,
                library=codec.loaded_library(), denominator='BF16 input norm; minimum 1e-20')


def verify(record, library):
    """CPU-side admission; reject missing, relabelled or failed GPU evidence."""
    import math
    from common import require
    require(record and record['passed'] and record['synthetic'] is False and
            record['retained_outputs_unchanged'] and record['library'] == library and
            [r['case'] for r in record['cases']] == list(CASES), 'Missing codec GPU preflight')
    for row in record['cases']:
        expected = [1792, 4096] if row['case'] == 'full_shape' else [56, 128]
        require(row['passed'] and row['finite'] and row['shape'] == expected and row['chunks'] == 7,
                'Incomplete codec preflight cases')
        for key, budget_key, limit in (('relative_l2', 'budget_vs_bf16', .05),
                                       ('protocol_relative_l2', 'budget_vs_codec', .01),
                                       ('packet_relative_l2', 'budget_vs_packet', .005)):
            require(row[budget_key] == limit and math.isfinite(row[key]) and 0 <= row[key] <= limit,
                    'Codec preflight exceeded its fixed numerical limit')
