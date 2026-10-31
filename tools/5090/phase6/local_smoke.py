#!/usr/bin/env python3
"""Eight-rank GPU correctness probe for the non-P2P local TACO transport.

Preview is CPU-only. Launch each arm in a fresh torchrun process with --execute.
This probe produces correctness evidence, never a performance measurement.
"""
import argparse
from datetime import timedelta
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
from rtx5090_loader import REPO, load

PROTOCOL = 'rtx5090-local-transport-smoke-v1'
TRANSPORT_PROTOCOL = 'rtx5090-nccl-staged-v2'
ARMS = ('ma11', 'ma89', 'ma8', 'ma9', 'cr3', 'r', 'c', 'a')
FUSED = ARMS[:5]
ROLES = {'attn': 512, 'mlp': 2048}
CASES = ('zero', 'random', 'outlier')
PACKET_BYTES = 8192*136
CHUNK_BYTES = 256*4096*2
SCRATCH_POLICY = 'fixed-stream-shared-scratch-private-packets-v1'
SHARED_BUFFERS = ('tx_bf16', 'tx_packets', 'rx_bf16', 'tx_wire', 'rx_wire')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    args.build, args.out = args.build.resolve(), args.out.resolve()
    return args


def tile_bytes(bf16, packets, peer, tile, quantized):
    """Independent CPU gather of the physical rows represented by one tile."""
    row0, column = (tile//32)*128, tile % 32
    if not quantized:
        return b''.join(bf16[peer*CHUNK_BYTES+((row0+row)*4096+column*128)*2:
                            peer*CHUNK_BYTES+((row0+row)*4096+column*128)*2+256]
                        for row in range(128))
    base = peer*PACKET_BYTES
    payload = b''.join(packets[base+((row0+row)*32+column)*128:
                              base+((row0+row)*32+column+1)*128] for row in range(128))
    scales = []
    for scale_base in (base+8192*128, base+8192*132):
        scales.append(b''.join(packets[scale_base+((row0+row)*32+column)*4:
                                      scale_base+((row0+row)*32+column+1)*4] for row in range(128)))
    return payload+b''.join(scales)


def check_wire_bytes(bf16, packets, wire, offsets, splits, mask, rank, receive=False):
    require(len(bf16) == 8*CHUNK_BYTES and len(packets) == 8*PACKET_BYTES, 'Wrong staging allocation size')
    require(len(wire) == sum(splits) and len(offsets) == 512 and splits[rank] == 0,
            'Wrong compact wire shape or self transfer')
    tiles = 0
    for peer in range(8):
        if peer == rank:
            require(offsets[peer*64:(peer+1)*64] == [-1]*64, 'Self tile entered the wire')
            continue
        for tile in range(64):
            quantized = mask is None or bool(mask[peer][rank*64+tile] if receive else mask[rank][peer*64+tile])
            expected = tile_bytes(bf16, packets, peer, tile, quantized)
            offset = offsets[peer*64+tile]
            require(len(expected) == (17408 if quantized else 32768)
                    and wire[offset:offset+len(expected)] == expected,
                    f'Lossless packet copy mismatch: rank={rank}, peer={peer}, tile={tile}, receive={receive}')
            tiles += 1
    return dict(passed=True, physical_tiles=tiles, bytes=len(wire))


def wire_audit(slot, torch, dist):
    """Compare actual pack/unpack bytes and every directed NCCL payload hash."""
    torch.cuda.synchronize()
    buffers = slot.buffers
    snapshot = lambda tensor: tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    layout = slot._exchange.layout
    tx, rx = snapshot(buffers['tx_wire']), snapshot(buffers['rx_wire'])
    packed = check_wire_bytes(snapshot(buffers['tx_bf16']), snapshot(buffers['tx_packets']), tx,
                              layout['tx_offsets'], layout['tx_splits'], slot.mask, slot.rank)
    unpacked = check_wire_bytes(snapshot(buffers['rx_bf16']), snapshot(buffers['rx_packets']), rx,
                                layout['rx_offsets'], layout['rx_splits'], slot.mask, slot.rank, True)
    sent, cursor = {}, 0
    for peer, size in enumerate(layout['tx_splits']):
        if peer != slot.rank:
            sent[str(peer)] = dict(bytes=size, sha256=hashlib.sha256(tx[cursor:cursor+size]).hexdigest())
        cursor += size
    all_sent = [None]*8
    dist.all_gather_object(all_sent, sent, group=slot.group)
    received, cursor = {}, 0
    for peer, size in enumerate(layout['rx_splits']):
        if peer != slot.rank:
            item = dict(bytes=size, sha256=hashlib.sha256(rx[cursor:cursor+size]).hexdigest())
            require(item == all_sent[peer][str(slot.rank)], 'NCCL payload differs from source staging bytes')
            received[str(peer)] = item
        cursor += size
    return dict(passed=True, pack=packed, unpack=unpacked, sent=sent, received=received,
                cross_rank_payload_identity=True)


def reference_check(torch, dist, reference, input_, weight, actual, mask):
    partial = input_.matmul(weight.t())
    parts = [torch.empty_like(partial) for _ in range(8)]
    dist.all_gather(parts, partial)
    rank = dist.get_rank()
    bf16, coded = torch.zeros_like(actual), torch.zeros_like(actual)
    for offset in range(1, 9):
        source = (rank+offset) % 8
        piece = parts[source][rank*256:(rank+1)*256]
        bf16 = (bf16+piece).to(torch.bfloat16)
        if source != rank:
            quantized, _ = reference.roundtrip(piece)
            if mask is not None:
                tiles = torch.tensor(mask[source], dtype=torch.bool).reshape(16, 32)[rank*2:(rank+1)*2]
                selected = tiles.repeat_interleave(128, 0).repeat_interleave(128, 1).to(piece.device)
                quantized = torch.where(selected, quantized, piece)
            piece = quantized
        coded = (coded+piece).to(torch.bfloat16)
    scale = max(bf16.float().norm().item(), 1e-20)
    result = dict(relative_l2=(actual.float()-bf16.float()).norm().item()/scale,
                  protocol_relative_l2=(actual.float()-coded.float()).norm().item()/scale,
                  max_abs=(actual.float()-bf16.float()).abs().max().item(),
                  budget_vs_bf16=.05, budget_vs_codec=.01)
    require(bool(torch.isfinite(actual).all()) and all(math.isfinite(value) for value in result.values())
            and result['relative_l2'] <= .05 and result['protocol_relative_l2'] <= .01,
            'Independent PyTorch TACO reference mismatch: '+repr(result))
    return dict(result, passed=True)


def current_slot(op):
    return op.slots[op.next_slot ^ 1] if hasattr(op, 'slots') else op


def shared_scratch_probe(torch, dist, frontend, reference, arm, role, selected):
    """Exercise two live model-like instances, including reuse and owner cleanup."""
    constructor = frontend.GemmRSTacoDoubleBuffered if selected is not None else frontend.GemmRSTaco
    options = dict(placement='fused')
    if selected is not None:
        options.update(selected=selected, quantization_mode='targeted')
    operators = []
    try:
        for _ in range(2):
            operators.append(constructor(dist.group.WORLD, 2048, 4096, ROLES[role], **options))
        slots = [op.slots if hasattr(op, 'slots') else [op] for op in operators]
        pointers = [[dict(shared={key: slot.buffers[key].data_ptr() for key in SHARED_BUFFERS},
                          packet=slot.buffers['rx_packets'].data_ptr(),
                          mask=slot.selected.data_ptr() if slot.selected is not None else None,
                          native=slot._exchange.native.handle.value)
                     for slot in instance] for instance in slots]
        require(all(pointers[0][index]['shared'] == pointers[1][index]['shared']
                    for index in range(len(slots[0]))), 'Model instances did not share staging scratch')
        for key in SHARED_BUFFERS:
            require(len({item['shared'][key] for item in pointers[0]}) == len(slots[0]),
                    'Rotating staging slots unexpectedly share scratch')
        for key in ('packet', 'native') + (('mask',) if selected is not None else ()):
            require(len({item[key] for instance in pointers for item in instance}) == 2*len(slots[0]),
                    'Per-instance packet, mask or native state was aliased')
        inputs, retained, expected, references = [], [], [], []
        for index in range(2):
            torch.manual_seed(20261101+dist.get_rank()*97+index)
            inputs.append((torch.randn((2048, ROLES[role]), device='cuda', dtype=torch.bfloat16)*.03,
                           torch.randn((4096, ROLES[role]), device='cuda', dtype=torch.bfloat16)*.03))
        for iteration in range(4):
            for index, op in enumerate(operators):
                x, weight = inputs[index]
                output = op.forward(x, weight)
                if iteration == 0:
                    expected.append(output.clone())
                    references.append(reference_check(torch, dist, reference, x, weight, output, selected))
                # After the reference iteration, enqueue six alternating calls
                # without a host read/synchronize. Per-call equality would hide
                # missing stream dependencies in actual model execution.
                retained.append((output, output.clone()))
        torch.cuda.synchronize()
        require(all(torch.equal(snapshot.view(torch.uint8), expected[index % 2].view(torch.uint8))
                    for index, (_, snapshot) in enumerate(retained)),
                'Shared scratch changed repeated projection bits')
        before_close = [op.transport_state() for op in operators]
        operators[0].close()
        output = operators[1].forward(*inputs[1])
        torch.cuda.synchronize()
        require(torch.equal(output.view(torch.uint8), expected[1].view(torch.uint8)),
                'Closing one instance invalidated another shared scratch user')
        require(len({value.data_ptr() for value, _ in retained} | {output.data_ptr()}) == 9
                and all(torch.equal(value.view(torch.uint8), snapshot.view(torch.uint8))
                        for value, snapshot in retained),
                'Shared scratch reuse overwrote a retained projection output')
        result = dict(passed=True, scratch_policy=SCRATCH_POLICY, instances=2,
                      slots_per_instance=len(slots[0]), pointers=pointers,
                      calls_before_close=[4, 4], async_burst_calls=6, repeated_bitwise=True,
                      retained_outputs=8, retained_bitwise=True, output_storage_independent=True,
                      survivor_after_close_bitwise=True, reference_checks=references,
                      before_close=before_close, survivor_after_close=operators[1].transport_state())
        validate_shared_scratch(result, arm)
        return result
    finally:
        for op in operators:
            slots = op.slots if hasattr(op, 'slots') else [op]
            if all(not slot._exchange.failed for slot in slots):
                op.close()


def validate_shared_scratch(record, arm):
    require(isinstance(record, dict), 'Missing shared scratch GPU evidence')
    count = 1 if arm in ('a', 'r') else 2
    require(record['passed'] and record['scratch_policy'] == SCRATCH_POLICY
            and record['instances'] == 2 and record['slots_per_instance'] == count
            and record['calls_before_close'] == [4, 4] and record['async_burst_calls'] == 6
            and record['repeated_bitwise']
            and record['retained_outputs'] == 8 and record['retained_bitwise']
            and record['output_storage_independent'] and record['survivor_after_close_bitwise'],
            'Shared scratch reuse/retention/cleanup evidence failed')
    pointers = record['pointers']
    require(len(pointers) == 2 and all(len(instance) == count for instance in pointers),
            'Missing shared scratch pointer evidence')
    for instance in pointers:
        for item in instance:
            require(set(item['shared']) == set(SHARED_BUFFERS)
                    and all(type(value) is int and value > 0 for value in item['shared'].values()),
                    'Invalid shared scratch pointer')
    require(all(pointers[0][index]['shared'] == pointers[1][index]['shared'] for index in range(count)),
            'Model instances did not share staging scratch')
    require(len({value for item in pointers[0] for value in item['shared'].values()}) == 5*count,
            'Staging buffers or rotating slots alias')
    for key in ('packet', 'native', 'mask'):
        values = [item[key] for instance in pointers for item in instance]
        if key == 'mask' and count == 1:
            require(values == [None, None], 'All-remote instances have unexpected masks')
        else:
            require(all(type(value) is int and value > 0 for value in values)
                    and len(set(values)) == 2*count, 'Per-instance packet, mask or native state was aliased')
    require(len(record['reference_checks']) == 2, 'Missing shared scratch numerical references')
    for check in record['reference_checks']:
        require(check['passed'] and check['budget_vs_bf16'] == .05 and check['budget_vs_codec'] == .01
                and all(math.isfinite(check[key]) for key in ('relative_l2', 'protocol_relative_l2', 'max_abs'))
                and 0 <= check['relative_l2'] <= .05 and 0 <= check['protocol_relative_l2'] <= .01,
                'Shared scratch numerical reference failed')
    states = record['before_close']
    require(len(states) == 2, 'Missing shared scratch lifecycle state')
    for state, calls, instance in ((states[0], 4, 0), (states[1], 4, 1),
                                    (record['survivor_after_close'], 5, 1)):
        slots = state.get('slots', [state])
        require(state['protocol'] == TRANSPORT_PROTOCOL and not state['failed'] and len(slots) == count,
                'Wrong shared scratch transport family')
        for index, slot in enumerate(slots):
            expected = calls if count == 1 else (calls+1-index)//2
            require(slot['protocol'] == TRANSPORT_PROTOCOL and slot['scratch_policy'] == SCRATCH_POLICY
                    and not slot['failed'] and not slot['closed'] and not slot['pending_exchange']
                    and not slot['peer_memory'] and slot['native_phase'] == 0
                    and slot['calls'] == slot['producer_calls'] == slot['pack_calls'] == slot['exchange_calls']
                    == slot['wait_calls'] == slot['unpack_calls'] == slot['decode_calls'] == expected
                    and slot['consumer_calls'] == 0
                    and slot['initializations'] == (1 if arm in FUSED else expected)
                    and slot['reconfigure_calls'] == (expected if arm == 'a' else 0)
                    and slot['shared_scratch_ptrs'] == pointers[instance][index]['shared']
                    and type(slot['shared_scratch_bytes']) is int and slot['shared_scratch_bytes'] > 0,
                    'Shared scratch lifecycle or completion evidence failed')


def role_probe(torch, dist, frontend, reference, arm, role, selected):
    rank, k = dist.get_rank(), ROLES[role]
    constructor = frontend.GemmRSTacoDoubleBuffered if selected is not None else frontend.GemmRSTaco
    options = dict(placement='fused')
    if selected is not None:
        options.update(selected=selected, quantization_mode='targeted')
    op = constructor(dist.group.WORLD, 2048, 4096, k, **options)
    checks = {}
    try:
        for case_index, case in enumerate(CASES):
            torch.manual_seed(20261030+rank*97+case_index)
            x = torch.randn((2048, k), device='cuda', dtype=torch.bfloat16)*.03
            weight = torch.randn((4096, k), device='cuda', dtype=torch.bfloat16)*.03
            if case == 'zero':
                x.zero_()
            elif case == 'outlier':
                x[::128, 0] = 8
                weight[::128, 0] = 4
            output = op.forward(x, weight)
            require(output._base is None and output.is_contiguous(), 'Projection output must own complete storage')
            checks[case] = reference_check(torch, dist, reference, x, weight, output, selected)
        wire = wire_audit(current_slot(op), torch, dist)
        retained = []
        for iteration in range(7):
            # One rank arrives later each time. Every slot must remain valid
            # even when host enqueue progress differs across ranks.
            if rank == iteration % 8:
                time.sleep(.025)
            x = torch.full((2048, k), (iteration+1)/128, device='cuda', dtype=torch.bfloat16)
            weight = torch.full((4096, k), (rank+1)/128, device='cuda', dtype=torch.bfloat16)
            output = op.forward(x, weight)
            if iteration < 6:
                retained.append((output, output.clone()))
        torch.cuda.synchronize()
        pointers = [value.data_ptr() for value, _ in retained]
        require(len(set(pointers)) == 6 and all(torch.equal(value, saved) for value, saved in retained),
                'A later submission overwrote a retained projection output')
        checks['after_delayed_reuse'] = reference_check(torch, dist, reference, x, weight, output, selected)
        consumer = None
        if arm in FUSED:
            plain = op.forward(x, weight)
            bias = torch.randn((4096,), device='cuda', dtype=torch.bfloat16)*.02
            residual = torch.randn((256, 4096), device='cuda', dtype=torch.bfloat16)*.02
            expected = residual+(plain+bias)
            actual = op.forward_consumer(x, weight, bias, residual)
            torch.cuda.synchronize()
            require(actual._base is None and actual.data_ptr() != residual.data_ptr()
                    and torch.equal(actual, expected), 'Consumer changed ordered BF16 bias/residual arithmetic')
            consumer = dict(passed=True, bitwise_equal=True, independent_storage=True)
        state = op.transport_state()
        slots = state.get('slots', [state])
        expected_calls = 12 if arm in FUSED else 10
        require(sum(item['calls'] for item in slots) == expected_calls and
                all(item['calls'] == item['producer_calls'] == item['pack_calls'] == item['exchange_calls']
                    == item['wait_calls'] == item['unpack_calls'] == item['decode_calls']
                    and item['native_phase'] == 0 and not item['failed'] and not item['pending_exchange']
                    for item in slots), 'Native/exchange completion counters diverged')
        require(sum(item['consumer_calls'] for item in slots) == int(arm in FUSED), 'Consumer fallback detected')
        return dict(passed=True, k_local=k, cases=checks, wire=wire, consumer=consumer,
                    retained_outputs=dict(passed=True, count=6, reuse_calls=7, independent_storage=True,
                                          unchanged_after_subsequent_calls=True, rank_delay_seconds=.025),
                    transport=state,
                    shared_scratch=shared_scratch_probe(torch, dist, frontend, reference, arm, role, selected))
    finally:
        # Normal cleanup synchronizes only here, outside the measured campaign.
        # Failed NCCL submissions deliberately retain their buffers until the
        # isolated worker is terminated; do not block GC on a poisoned group.
        slots = op.slots if hasattr(op, 'slots') else [op]
        if all(not slot._exchange.failed for slot in slots):
            op.close()


def validate_reports(root, arm, manifest_sha256):
    root = Path(root)
    rows = [read(root/f'rank{rank}.json') for rank in range(8)]
    require({row['rank'] for row in rows} == set(range(8)) and len({row['gpu_uuid'] for row in rows}) == 8,
            'Missing/duplicate smoke rank or GPU')
    for rank, row in enumerate(rows):
        require(row['protocol'] == PROTOCOL and row['transport_protocol'] == TRANSPORT_PROTOCOL
                and row['completed'] is True and row['arm'] == arm and row['rank'] == rank
                and row['build_manifest_sha256'] == manifest_sha256 and set(row['roles']) == set(ROLES),
                'Local transport smoke identity/completion mismatch')
        for role, record in row['roles'].items():
            validate_shared_scratch(record.get('shared_scratch'), arm)
            require(record['passed'] and record['k_local'] == ROLES[role]
                    and set(record['cases']) == set(CASES) | {'after_delayed_reuse'}, 'Missing reference cases')
            for check in record['cases'].values():
                require(check['passed'] is True and check['budget_vs_bf16'] == .05 and check['budget_vs_codec'] == .01
                        and all(math.isfinite(check[key]) for key in ('relative_l2', 'protocol_relative_l2', 'max_abs'))
                        and 0 <= check['relative_l2'] <= .05 and 0 <= check['protocol_relative_l2'] <= .01,
                        'Numerical reference gate failed')
            wire = record['wire']
            require(wire['passed'] and wire['cross_rank_payload_identity'] and
                    set(wire['sent']) == set(wire['received']) == {str(peer) for peer in range(8) if peer != rank},
                    'Missing directed payload audit')
            require(all(wire[direction]['passed'] and wire[direction]['physical_tiles'] == 448
                        for direction in ('pack', 'unpack')), 'Missing lossless packet audit')
            require(wire['pack']['bytes'] == sum(item['bytes'] for item in wire['sent'].values())
                    and wire['unpack']['bytes'] == sum(item['bytes'] for item in wire['received'].values()),
                    'Payload audit byte counts changed')
            retention = record['retained_outputs']
            require(retention['passed'] and retention['count'] == 6 and retention['reuse_calls'] == 7
                    and retention['independent_storage'] and retention['unchanged_after_subsequent_calls'],
                    'Output retention failed')
            require(record['consumer'] == dict(passed=True, bitwise_equal=True, independent_storage=True)
                    if arm in FUSED else record['consumer'] is None, 'Consumer equivalence missing')
            state = record['transport']
            slots = state.get('slots', [state])
            require(state['protocol'] == TRANSPORT_PROTOCOL and not state['failed']
                    and len(slots) == (2 if arm not in ('a', 'r') else 1), 'Wrong transport slot family')
            require(sum(slot['calls'] for slot in slots) == (12 if arm in FUSED else 10), 'Missing native transactions')
            require(sum(slot['consumer_calls'] for slot in slots) == int(arm in FUSED), 'Missing native consumer call')
            for slot in slots:
                require(slot['protocol'] == TRANSPORT_PROTOCOL and not slot['failed'] and not slot['peer_memory']
                        and not slot['pending_exchange'] and slot['native_phase'] == 0
                        and slot['calls'] == slot['producer_calls'] == slot['pack_calls'] == slot['exchange_calls']
                        == slot['wait_calls'] == slot['unpack_calls'] == slot['decode_calls'],
                        'Unfinished or substituted transport')
                require(slot['reconfigure_calls'] == (slot['calls'] if arm == 'a' else 0)
                        and slot['initializations'] == (int(slot['calls'] > 0) if arm in FUSED else slot['calls'])
                        and slot['send_bytes'] == wire['pack']['bytes']
                        and slot['receive_bytes'] == wire['unpack']['bytes']
                        and type(slot['native_workspace_bytes']) is int and slot['native_workspace_bytes'] >= 0,
                        'Submission lifecycle or native workspace evidence changed')
        for peer in range(8):
            if peer == rank:
                continue
            for role in ROLES:
                require(row['roles'][role]['wire']['received'][str(peer)] ==
                        rows[peer]['roles'][role]['wire']['sent'][str(rank)], 'Cross-rank wire hash mismatch')
    return dict(protocol=PROTOCOL, transport_protocol=TRANSPORT_PROTOCOL, completed=True, arm=arm,
                build_manifest_sha256=manifest_sha256, gpu_validated=True, performance_measured=False,
                rank_records={f'rank{rank}.json': sha(root/f'rank{rank}.json') for rank in range(8)})


def run(args):
    require(os.environ.get('WORLD_SIZE') == '8' and os.environ.get('LOCAL_WORLD_SIZE') == '8',
            'Launch this probe through torchrun with eight local ranks')
    require(os.environ.get('NCCL_P2P_DISABLE') == '1', 'Set NCCL_P2P_DISABLE=1 for this transport probe')
    require(os.environ.get('FLUX_GEMM_RS_GEMM_ONLY') == '1', 'Set FLUX_GEMM_RS_GEMM_ONLY=1 for local producer kernels')
    rank = int(os.environ['LOCAL_RANK'])
    args.out.mkdir(parents=True, exist_ok=True)
    result_path = args.out/f'rank{rank}.json'
    require(not result_path.exists(), 'Use a fresh operator smoke directory')
    report = dict(protocol=PROTOCOL, transport_protocol=TRANSPORT_PROTOCOL, arm=args.arm, rank=rank,
                  completed=False, performance_measured=False, roles={})
    initialized = False
    try:
        builder = load('build')
        if rank == 0:
            builder.verify(args.build, built=True)
        manifest_sha256 = sha(args.build/'baseline-build.json')
        native = builder.native_root(args.build, args.arm)
        sys.path.insert(0, str(native/'python'))
        import torch
        import torch.distributed as dist
        torch.cuda.set_device(rank)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        dist.init_process_group('nccl', timeout=timedelta(seconds=180))
        initialized = True
        require(dist.get_rank() == rank and dist.get_world_size() == 8, 'Expected one complete TP8 group')
        import flux.gemm_rs_taco as frontend
        require(frontend.PROTOCOL == TRANSPORT_PROTOCOL and
                frontend.NATIVE_ARM == builder.ALIASES.get(args.arm, args.arm), 'Wrong installed transport frontend')
        reference_path = REPO/'tools/a800/phase4/reference.py'
        spec = importlib.util.spec_from_file_location('rtx5090_smoke_tensor_reference', reference_path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        selection = read(args.build/'selective/selection-plan.json')
        policy = 'interleaved_arrival_selective_attn_mlp'
        with torch.no_grad():
            for role in ROLES:
                selected = None if args.arm in ('a', 'r') else selection['variants'][policy]['roles'][role]['entry']['mask']
                report['roles'][role] = role_probe(torch, dist, frontend, reference, args.arm, role, selected)
        torch.cuda.synchronize()
        require(sha(args.build/'baseline-build.json') == manifest_sha256, 'Native build changed during smoke')
        report.update(completed=True, gpu_uuid=str(torch.cuda.get_device_properties(rank).uuid),
                      build_manifest_sha256=manifest_sha256,
                      reference_sha256=sha(reference_path), entrypoint_sha256=sha(__file__),
                      native_libraries=builder.libraries(args.build, args.arm), torch=str(torch.__version__),
                      cuda=torch.version.cuda, nccl=list(torch.cuda.nccl.version()))
        write(result_path, report)
        dist.barrier(device_ids=[rank])
        if rank == 0:
            summary = validate_reports(args.out, args.arm, report['build_manifest_sha256'])
            write(args.out/'report.json', summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    except BaseException as exc:
        report.update(completed=False, error=f'{type(exc).__name__}: {exc}')
        write(result_path, report)
        raise
    finally:
        if initialized:
            dist.destroy_process_group()


def main(argv=None):
    args = arguments(argv)
    if not args.execute:
        print(json.dumps(dict(protocol=PROTOCOL, preview=True, arm=args.arm, build=str(args.build),
                              out=str(args.out), roles=ROLES, cases=CASES, retained_outputs=6,
                              NCCL_P2P_DISABLE='1', performance_measured=False), indent=2))
        return
    run(args)


if __name__ == '__main__':
    main()
