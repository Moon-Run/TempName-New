#!/usr/bin/env python3
"""Independent old/new A packet and output oracle, run in separate eight-rank processes."""
import argparse
from datetime import timedelta
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
from rtx5090_loader import REPO, load
import op1_a_protocol as p

PROTOCOL = 'rtx5090-op1-a-bitwise-probe-v1'
PACKET_BYTES = 8192*136
ROLES = {'attn': 512, 'mlp': 2048}
CASES = ('zero', 'random', 'outlier')


def tensor_sha(value):
    import torch
    return hashlib.sha256(value.detach().contiguous().view(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def packet_hashes(op, rank, variant):
    p.require(variant in ('old-a', 'a'), 'Unknown native packet layout')
    result = {}
    for direction in ('tx', 'rx'):
        raw = op.buffers[direction+'_packets'].reshape(-1)
        p.require(raw.numel() == (8 if variant == 'old-a' else 7)*PACKET_BYTES,
                  'Wrong native packet allocation size')
        result[direction] = {str(peer): tensor_sha(raw[
            (peer if variant == 'old-a' else peer-int(peer > rank))*PACKET_BYTES:
            (peer if variant == 'old-a' else peer-int(peer > rank))*PACKET_BYTES+PACKET_BYTES])
            for peer in range(8) if peer != rank}
    return result


def is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(char in '0123456789abcdef' for char in value)


def shared_addresses(first, second, variant):
    keys = ('tx_wire', 'rx_bf16') if variant == 'a' else (
        'tx_bf16', 'tx_packets', 'rx_bf16', 'tx_wire', 'rx_wire')
    record = dict(shared_keys=list(keys), instances=[dict(
        shared={key: op.buffers[key].data_ptr() for key in keys},
        rx_packet=op.buffers['rx_packets'].data_ptr(), native=op._exchange.native.handle.value,
        aliases={key: op.buffers[key].data_ptr() for key in
                 ('tx_bf16', 'rx_bf16', 'tx_packets', 'rx_packets', 'tx_wire', 'rx_wire')})
        for op in (first, second)])
    validate_shared_addresses(record, variant)
    return record


def validate_shared_addresses(record, variant):
    keys = ['tx_wire', 'rx_bf16'] if variant == 'a' else [
        'tx_bf16', 'tx_packets', 'rx_bf16', 'tx_wire', 'rx_wire']
    rows = record['instances']
    p.require(record['shared_keys'] == keys and len(rows) == 2,
              'Wrong native shared buffer evidence')
    p.require(all(set(row['shared']) == set(keys) and
                  all(type(value) is int and value > 0 for value in row['shared'].values())
                  and type(row['rx_packet']) is int and row['rx_packet'] > 0
                  and type(row['native']) is int and row['native'] > 0 for row in rows),
              'Invalid native storage/handle pointers')
    p.require(rows[0]['shared'] == rows[1]['shared']
              and rows[0]['rx_packet'] != rows[1]['rx_packet']
              and rows[0]['native'] != rows[1]['native'], 'Scratch not shared or receive/handle state aliased')
    for row in rows:
        aliases = row['aliases']
        p.require(set(aliases) == {'tx_bf16', 'rx_bf16', 'tx_packets', 'rx_packets', 'tx_wire', 'rx_wire'}
                  and all(type(value) is int and value > 0 for value in aliases.values())
                  and row['rx_packet'] == aliases['rx_packets']
                  and all(row['shared'][key] == aliases[key] for key in keys), 'Inconsistent pointer evidence')
        if variant == 'a':
            p.require(aliases['tx_bf16'] == aliases['rx_bf16']
                      and aliases['tx_packets'] == aliases['tx_wire']
                      and aliases['rx_packets'] == aliases['rx_wire']
                      and len({aliases['tx_wire'], aliases['rx_wire'], aliases['rx_bf16']}) == 3,
                      'Direct packet aliases do not match the native ABI')
        else:
            p.require(len(set(aliases.values())) == len(aliases), 'Legacy A staging buffers unexpectedly alias')


def loaded_libraries(native):
    expected = {name: (Path(native)/name).resolve() for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')}
    maps = Path('/proc/self/maps').read_text().splitlines()
    result = {}
    for name, target in expected.items():
        loaded = {Path(line.split()[-1]).resolve() for line in maps if line.endswith('/'+name)}
        p.require(loaded == {target}, 'Wrong or multiple native library loaded: ' + name)
        result[name] = dict(path=str(target), sha256=p.sha(target))
    return result


def check_records(root, hardware, manifest_sha256, build=None):
    root = Path(root)
    libraries = None
    if build is not None:
        import op1_a_build as builder
        p.require(p.sha(Path(build)/builder.MANIFEST) == manifest_sha256, 'Probe build manifest changed')
        old = load('build').native_root(builder.base_root(build), 'a')
        libraries = {variant: builder.libraries(build, variant) for variant in ('a', 'flux-basic-interleaved')}
        libraries['old-a'] = {name: dict(path=str((old/name).resolve()), sha256=p.sha(old/name))
                             for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')}

    def identity(row):
        p.require(type(row['probe_stream']) is int and row['probe_stream'] > 0
                  and row['probe_stream'] != row['default_stream'], 'Probe did not use a nondefault CUDA stream')
        if libraries is not None:
            p.require(row['libraries'] == libraries[row['variant']], 'Probe loaded foreign native binaries')

    groups = {}
    for variant in ('old-a', 'a'):
        rows = [p.read(root/variant/f'rank{rank}.json') for rank in range(8)]
        for rank, row in enumerate(rows):
            p.require(row['protocol'] == PROTOCOL and row['completed'] is True and row['rank'] == rank
                      and row['variant'] == variant and row['build_manifest_sha256'] == manifest_sha256
                      and row['entrypoint_sha256'] == p.sha(__file__)
                      and row['gpu_uuid'] == hardware['devices'][rank]['uuid']
                      and all(row[key] == hardware['environment'][key] for key in ('torch', 'cuda', 'nccl'))
                      and set(row['roles']) == set(ROLES), 'Incomplete/foreign OP1 native probe')
            identity(row)
            for role, record in row['roles'].items():
                p.require(record['k_local'] == ROLES[role] and set(record['cases']) == set(CASES)
                          and record['retained_outputs'] == 6 and record['reuse_calls'] == 7
                          and record['retention_passed'] is True and record['shared_instances_passed'] is True,
                          'Missing native shape/case/retention/shared-instance checks')
                validate_shared_addresses(record['shared_addresses'], variant)
                for case, values in record['cases'].items():
                    check = values['reference']
                    p.require(check['passed'] and check['budget_vs_bf16'] == .05 and check['budget_vs_codec'] == .01
                              and all(math.isfinite(check[k]) for k in ('relative_l2', 'protocol_relative_l2', 'max_abs'))
                              and 0 <= check['relative_l2'] <= .05 and 0 <= check['protocol_relative_l2'] <= .01
                              and check['max_abs'] >= 0,
                              'Native numerical reference failed')
                    p.require(all(is_digest(values[key]) for key in ('output_sha256', 'input_sha256', 'weight_sha256')),
                              'Missing input/output tensor identity')
                    for direction in ('tx', 'rx'):
                        p.require(set(values[direction]) == {str(peer) for peer in range(8) if peer != rank},
                                  'Missing directed packet hash')
                        p.require(all(is_digest(h) for h in values[direction].values()),
                                  'Invalid packet digest')
                    for peer in range(8):
                        if peer != rank:
                            remote = rows[peer]['roles'][role]['cases'][case]
                            p.require(values['tx'][str(peer)] == remote['rx'][str(rank)],
                                      'NCCL directed packet bytes changed')
                state = record['transport']
                p.require(not state['failed'] and not state['pending_exchange'] and state['native_phase'] == 0
                          and all(state[key] == 10 for key in
                                  ('calls', 'producer_calls', 'exchange_calls', 'wait_calls', 'decode_calls')),
                          'Native transport did not complete every transaction')
                expected = 10 if variant == 'old-a' else 0
                p.require(state['pack_calls'] == state['unpack_calls'] == state['reconfigure_calls'] == expected,
                          'Wrong direct-wire/legacy execution path')
                p.require(state['stream'] == row['probe_stream'] and state['device'] == rank
                          and state['native']['protocol_version'] == (1 if variant == 'old-a' else 2),
                          'Native stream/device/ABI evidence changed')
                if variant == 'a':
                    import op1_a_native
                    p.require(state['protocol'] == op1_a_native.PROTOCOL
                              and state['native_api'] == op1_a_native.NATIVE_API
                              and state['configuration_scope'] == 'instance'
                              and state['output_ownership'] == 'independent'
                              and state['gemm_submission_cache'] is False
                              and state['initializations'] == 10, 'Wrong OP1 A runtime policy')
        groups[variant] = rows
    for rank in range(8):
        for role in ROLES:
            for case in CASES:
                old, new = (groups[variant][rank]['roles'][role]['cases'][case] for variant in ('old-a', 'a'))
                p.require(all(old[key] == new[key] for key in ('input_sha256', 'weight_sha256', 'output_sha256', 'tx', 'rx')),
                          f'A changed packet/output bits: rank={rank}, role={role}, case={case}')
    for rank in range(8):
        row = p.read(root/'flux-basic-interleaved'/f'rank{rank}.json')
        p.require(row['protocol'] == PROTOCOL and row['completed'] is True and row['rank'] == rank
                  and row['variant'] == 'flux-basic-interleaved' and row['build_manifest_sha256'] == manifest_sha256
                  and row['entrypoint_sha256'] == p.sha(__file__)
                  and row['gpu_uuid'] == hardware['devices'][rank]['uuid']
                  and all(row[key] == hardware['environment'][key] for key in ('torch', 'cuda', 'nccl'))
                  and set(row['mapping']) == set(ROLES), 'Incomplete sorted Flux mapping probe')
        identity(row)
        for role in ROLES:
            validate_mapping(row['mapping'][role], rank)
    return dict(protocol=PROTOCOL, completed=True, old_new_output_bitwise=True,
                old_new_packets_bitwise=True, directed_links=56, roles=list(ROLES), cases=list(CASES),
                actual_sorted_gemm_mapping_passed=True,
                build_manifest_sha256=manifest_sha256,
                evidence_files={str(path.relative_to(root)): p.sha(path)
                    for variant in ('old-a', 'a', 'flux-basic-interleaved')
                    for path in sorted((root/variant).glob('rank*.json'))})


def validate_mapping(rows, rank):
    expected = [[((index % 8 + rank) % 8)*2+(index//8)//32, (index//8) % 32]
                for index in range(512)]
    p.require(type(rank) is int and 0 <= rank < 8 and len(rows) == 512
              and all(len(row) == 4 and all(type(value) is int for value in row) for row in rows)
              and [row[:2] for row in rows] == expected
              and len({tuple(row[2:]) for row in rows}) == 512
              and all(0 <= row[2] < 16 and 0 <= row[3] < 32 for row in rows)
              and any(row[:2] != row[2:] for row in rows), 'Actual GEMM CTA mapping is not the required permutation')


def probe(args):
    import torch
    import torch.distributed as dist
    import op1_a_build as builder
    import local_smoke
    rank = int(os.environ['LOCAL_RANK'])
    record = dict(protocol=PROTOCOL, completed=False, rank=rank, variant=args.variant,
                  build_manifest_sha256=p.sha(args.build/'op1-build.json'), entrypoint_sha256=p.sha(__file__))
    try:
        torch.cuda.set_device(rank)
        p.require(torch.cuda.device_count() == 8 and int(os.environ['LOCAL_WORLD_SIZE']) == 8,
                  'OP1 probe requires eight visible GPUs')
        prop = torch.cuda.get_device_properties(rank)
        p.require('5090' in prop.name and (prop.major, prop.minor, prop.multi_processor_count) == (12, 0, 170),
                  'OP1 probe requires RTX5090 SM120/170 SM')
        record.update(gpu_uuid=str(prop.uuid), torch=str(torch.__version__), cuda=torch.version.cuda,
                      nccl=list(torch.cuda.nccl.version()))
        dist.init_process_group('nccl', timeout=timedelta(seconds=300))
        native = (load('build').native_root(builder.base_root(args.build), 'a') if args.variant == 'old-a'
                  else builder.native_root(args.build, args.variant))
        sys.path.insert(0, str(native/'python'))
        import flux
        record['libraries'] = loaded_libraries(native)
        if args.variant == 'flux-basic-interleaved':
            import ctypes
            lib = ctypes.CDLL(str(native/'libflux_cuda.so'))
            lib.op1_a_sorted_version.restype = ctypes.c_int
            p.require(lib.op1_a_sorted_version() == 1, 'Wrong sorted mapping ABI')
            lib.op1_a_sorted_mapping.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
            lib.op1_a_sorted_mapping.restype = ctypes.c_int
            mapping = {}
            stream = torch.cuda.Stream()
            record.update(probe_stream=int(stream.cuda_stream), default_stream=int(torch.cuda.default_stream().cuda_stream))
            with torch.cuda.stream(stream):
                for role, k in ROLES.items():
                    output = torch.empty((512, 4), dtype=torch.int32, device='cuda')
                    p.require(lib.op1_a_sorted_mapping(output.data_ptr(), k, stream.cuda_stream) == 0,
                              'Sorted mapping CUDA probe failed')
                    rows = output.cpu().tolist()
                    validate_mapping(rows, rank)
                    mapping[role] = rows
            record.update(completed=True, mapping=mapping)
            p.write(args.out/f'rank{rank}.json', record)
            dist.destroy_process_group()
            return
        from flux import gemm_rs_taco as frontend
        spec = importlib.util.spec_from_file_location('op1_codec_reference', REPO/'tools/a800/phase4/reference.py')
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        record['roles'] = {}
        # A nondefault stream catches accidental dependencies on the default stream.
        stream = torch.cuda.Stream()
        record.update(probe_stream=int(stream.cuda_stream), default_stream=int(torch.cuda.default_stream().cuda_stream))
        with torch.cuda.stream(stream), torch.no_grad():
            for role, k in ROLES.items():
                op = frontend.GemmRSTaco(dist.group.WORLD, 2048, 4096, k, placement='fused')
                cases = {}
                for index, case in enumerate(CASES):
                    torch.manual_seed(20261101+rank*97+index)
                    x = torch.randn((2048, k), dtype=torch.bfloat16, device='cuda')*.03
                    w = torch.randn((4096, k), dtype=torch.bfloat16, device='cuda')*.03
                    if case == 'zero':
                        x.zero_()
                    elif case == 'outlier':
                        x[::128, 0] = 8
                        w[::128, 0] = 4
                    output = op.forward(x, w)
                    torch.cuda.synchronize()
                    cases[case] = dict(input_sha256=tensor_sha(x), weight_sha256=tensor_sha(w),
                        output_sha256=tensor_sha(output), **packet_hashes(op, rank, args.variant),
                        reference=local_smoke.reference_check(torch, dist, reference, x, w, output, None))
                retained = []
                for iteration in range(7):
                    if rank == iteration % 8:
                        time.sleep(.025)
                    output = op.forward(x*(iteration+1), w)
                    if iteration < 6:
                        retained.append((output, output.clone()))
                torch.cuda.synchronize()
                p.require(len({value.data_ptr() for value, _ in retained}) == 6
                          and all(torch.equal(value.view(torch.uint8), saved.view(torch.uint8))
                                  for value, saved in retained), 'Retained native output overwritten')
                state = op.transport_state()
                # Two module instances share scratch but own distinct receive/output storage.
                other = frontend.GemmRSTaco(dist.group.WORLD, 2048, 4096, k, placement='fused')
                addresses = shared_addresses(op, other, args.variant)
                first = op.forward(x, w)
                saved_first = first.clone()
                second = other.forward(x*2, w)
                saved_second = second.clone()
                p.require(first.data_ptr() != second.data_ptr(), 'Different instances returned aliased output')
                for _ in range(3):
                    op.forward(x, w)
                    other.forward(x*2, w)
                torch.cuda.synchronize()
                p.require(torch.equal(first, saved_first) and torch.equal(second, saved_second),
                          'Shared scratch corrupted another instance output')
                op.close()
                survivor = other.forward(x*2, w)
                torch.cuda.synchronize()
                p.require(torch.equal(survivor, saved_second), 'Closing one instance corrupted its neighbor')
                other.close()
                record['roles'][role] = dict(k_local=k, cases=cases, retained_outputs=6, reuse_calls=7,
                    retention_passed=True, shared_instances_passed=True, shared_addresses=addresses, transport=state)
        record['completed'] = True
        p.write(args.out/f'rank{rank}.json', record)
        dist.destroy_process_group()
    except BaseException as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
        p.write(args.out/f'rank{rank}.json', record)
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--variant', choices=('old-a', 'a', 'flux-basic-interleaved'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('PLAN ONLY: eight-rank OP1 old/new native correctness probe; add --execute.')
        return
    probe(args)


if __name__ == '__main__':
    main()
