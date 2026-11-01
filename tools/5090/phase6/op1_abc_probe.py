#!/usr/bin/env python3
"""Independent eight-rank packet/reference/retention gates for the ABC experiment."""
import argparse
import ctypes
from datetime import timedelta
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace

sys.dont_write_bytecode = True
from rtx5090_loader import HERE, REPO
import op1_abc_protocol as p
from op1_a_probe import is_digest, tensor_sha, validate_mapping, loaded_libraries

PROTOCOL = 'rtx5090-op1-abc-distributed-probe-v1'
ROLES = {'attn': 512, 'mlp': 2048}
CASES = ('zero', 'random', 'outlier')
require, read, write, sha = p.require, p.read, p.write, p.sha


def improved_probe():
    """Private old-C probe with a composite build adapter; no legacy globals change."""
    import flux_baseline_probe
    import op1_abc_build as builder
    source = (HERE/'flux_baseline_probe.py').read_text()
    source = source[:source.index("\nif __name__ == '__main__':")]
    source = source.replace("'        import flux_baseline_build as builder'",
                            "'        builder = _abc_builder()'")
    source = source.replace('"args.build/\'flux-build.json\'"',
                            '"args.build/\'op1-abc-build.json\'"')
    module = ModuleType('op1_abc_private_improved_probe')
    module.__file__ = str(HERE/'op1_abc_probe.py')
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    result = module.probe
    result._abc_builder = lambda: SimpleNamespace(
        verify=builder.verify, ALIASES={'c': 'c'},
        native_root=lambda root, arm: builder.native_root(root, 'flux-improved'),
        libraries=lambda root, arm: builder.libraries(root, 'flux-improved'))
    return result


def check_reference(value):
    require(value['passed'] is True and value['budget_vs_bf16'] == .05
            and value['budget_vs_codec'] == .01
            and all(math.isfinite(value[key]) and value[key] >= 0
                    for key in ('relative_l2', 'protocol_relative_l2', 'max_abs'))
            and value['relative_l2'] <= .05 and value['protocol_relative_l2'] <= .01,
            'Independent projection reference failed')


def reference_check(torch, dist, reference, partial, output, selective):
    """Gather independent matmul partials; decode tensor reference and fold BF16."""
    rank = dist.get_rank()
    parts = [torch.empty_like(partial) for _ in range(8)]
    dist.all_gather(parts, partial)
    bf16, coded = torch.zeros_like(output), torch.zeros_like(output)
    for offset in range(1, 9):
        source = (rank + offset) % 8
        piece = parts[source][rank*256:(rank+1)*256]
        bf16 = (bf16 + piece).to(torch.bfloat16)
        if source != rank:
            quantized, _ = reference.roundtrip(piece)
            if selective:
                # New 5090 policy: only the last 1/64 H128 groups are quantized.
                quantized = quantized.clone()
                count = piece.numel() // 128
                quantized.view(-1, 128)[:count-count//64].copy_(piece.view(-1, 128)[:count-count//64])
            piece = quantized
        coded = (coded + piece).to(torch.bfloat16)
    denominator = max(bf16.float().norm().item(), 1e-20)
    result = dict(passed=bool(torch.isfinite(output).all()), budget_vs_bf16=.05,
        budget_vs_codec=.01, relative_l2=(output.float()-bf16.float()).norm().item()/denominator,
        protocol_relative_l2=(output.float()-coded.float()).norm().item()/denominator,
        max_abs=(output.float()-bf16.float()).abs().max().item())
    check_reference(result)
    return result


def mapping_probe(torch, native, stream, rank):
    lib = ctypes.CDLL(str(native/'libflux_cuda.so'))
    lib.op1_a_sorted_version.restype = ctypes.c_int
    require(lib.op1_a_sorted_version() == 1, 'Wrong actual sorted GEMM mapping ABI')
    lib.op1_a_sorted_mapping.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    lib.op1_a_sorted_mapping.restype = ctypes.c_int
    result = {}
    for role, k in ROLES.items():
        tensor = torch.empty((512, 4), dtype=torch.int32, device='cuda')
        require(lib.op1_a_sorted_mapping(tensor.data_ptr(), k, stream.cuda_stream) == 0,
                'Actual sorted GEMM mapping failed')
        result[role] = tensor.cpu().tolist()
        validate_mapping(result[role], rank)
    return result


def wire_audit(codec, partial, output, upstream):
    """Check actual AoS wire bits and decoder against unchanged upstream exports."""
    import torch
    chunks = codec.debug_packets()
    sent, received = {}, {}
    total_groups = codec.rows*codec.n//128
    raw_groups = total_groups-total_groups//64 if codec.arm == p.BASIC_ARMS[2] else 0
    quant_groups = total_groups-raw_groups
    expected_output = torch.zeros_like(output)
    decoded = {}
    for peer in range(8):
        if peer == codec.rank:
            continue
        slot = peer - int(peer > codec.rank)
        wires = {direction: torch.cat([item[direction][slot] for item in chunks])
                 for direction in ('tx', 'rx')}
        source = partial.reshape(8, codec.rows, codec.n)[peer]
        # Independent layout conversion only in the untimed probe.
        quantized = upstream.pack_rows(source.reshape(-1, 128)[raw_groups:].reshape(-1, codec.n).contiguous(), 1)[0]
        elems = quant_groups*128
        aos = torch.cat((quantized[:elems].reshape(quant_groups, 128),
                         quantized[elems:elems+quant_groups*4].reshape(quant_groups, 4),
                         quantized[elems+quant_groups*4:].reshape(quant_groups, 4)), dim=1).reshape(-1)
        raw = source.reshape(-1, 128)[:raw_groups].contiguous().view(torch.uint8).reshape(-1)
        expected = torch.cat((raw, aos))
        require(torch.equal(wires['tx'], expected), 'Direct packet differs from upstream codec bits')
        remote = wires['rx'][raw_groups*256:].reshape(quant_groups, 136)
        soa = torch.cat((remote[:, :128].reshape(-1), remote[:, 128:132].reshape(-1),
                         remote[:, 132:].reshape(-1))).reshape(1, -1)
        quant = upstream.unpack_rows(soa, quant_groups*128//codec.n, codec.n).reshape(-1)
        decoded[peer] = torch.cat((wires['rx'][:raw_groups*256].view(torch.bfloat16), quant)).reshape_as(output)
        for direction, result in (('tx', sent), ('rx', received)):
            data = wires[direction].cpu().numpy().tobytes()
            result[str(peer)] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    for offset in range(1, 9):
        source = (codec.rank+offset) % 8
        contribution = partial.reshape(8, codec.rows, codec.n)[source] if source == codec.rank else decoded[source]
        expected_output = (expected_output+contribution).to(torch.bfloat16)
    require(torch.equal(output.view(torch.uint8), expected_output.view(torch.uint8)),
            'Direct decoder changed upstream BF16 decode/fold bits')
    return dict(sent=sent, received=received, upstream_packet_bitwise=True, upstream_decode_fold_bitwise=True)


def check_basic_rows(rows, arm, manifest, hardware=None, libraries=None, codec_library=None):
    from op1_abc_validation import validate_codec_state
    require(len(rows) == 8, 'Eight probe records required')
    for rank, row in enumerate(rows):
        require(row['protocol'] == PROTOCOL and row['completed'] is True and row['rank'] == rank
                and row['arm'] == arm and row['build_manifest_sha256'] == manifest
                and row['entrypoint_sha256'] == sha(__file__)
                and row['reference_sha256'] == sha(REPO/'tools/a800/phase4/reference.py')
                and set(row['roles']) == set(ROLES) and set(row['mapping']) == set(ROLES)
                and type(row['probe_stream']) is int and row['probe_stream'] > 0
                and row['probe_stream'] != row['default_stream'], 'Incomplete/foreign ABC probe')
        if hardware is not None:
            require(row['gpu_uuid'] == hardware['devices'][rank]['uuid']
                    and all(row[key] == hardware['environment'][key] for key in ('torch', 'cuda', 'nccl')),
                    'Probe hardware/software differs from preflight')
        if libraries is not None:
            require(row['libraries'] == libraries and row['codec_library'] == codec_library,
                    'Probe loaded a foreign library')
        for role, record in row['roles'].items():
            validate_mapping(row['mapping'][role], rank)
            require(record['k_local'] == ROLES[role] and set(record['cases']) == set(CASES)
                    and record['retained_outputs'] == 6 and record['reuse_calls'] == 7
                    and record['retention_passed'] is True and record['independent_storage'] is True,
                    'Missing shape/case/output ownership evidence')
            validate_codec_state(record['state'], arm, 10, rank)
            check_reference(record['delayed_reference'])
            for case, values in record['cases'].items():
                check_reference(values['reference'])
                require(all(is_digest(values[key]) for key in ('input_sha256', 'weight_sha256', 'output_sha256')),
                        'Missing native input/output fingerprint')
                wire = values['wire']
                require(wire['upstream_packet_bitwise'] is True and wire['upstream_decode_fold_bitwise'] is True,
                        'Direct codec lacks independent upstream bitwise oracle')
                peers = {str(peer) for peer in range(8) if peer != rank}
                require(set(wire['sent']) == set(wire['received']) == peers, 'Missing directed packet')
                for direction in ('sent', 'received'):
                    require(all(type(item['bytes']) is int and item['bytes'] > 0 and is_digest(item['sha256'])
                                for item in wire[direction].values()), 'Invalid directed packet evidence')
                require(sum(item['bytes'] for item in wire['sent'].values()) == record['state']['wire_bytes_per_rank']
                        == sum(item['bytes'] for item in wire['received'].values()), 'Wire byte budget differs')
                for peer in range(8):
                    if peer != rank:
                        remote = rows[peer]['roles'][role]['cases'][case]['wire']
                        require(wire['sent'][str(peer)] == remote['received'][str(rank)],
                                'NCCL directed payload mismatch')
    return dict(completed=True, arm=arm, protocol=PROTOCOL, directed_links=56,
                actual_sorted_gemm_mapping_passed=True, performance_measured=False)


def check_records(root, hardware, manifest_sha256, build=None):
    import op1_abc_build as builder
    root = Path(root)
    build = Path(build) if build is not None else None
    groups, reports = {}, {}
    for arm in p.BASIC_ARMS:
        groups[arm] = [read(root/arm/f'rank{rank}.json') for rank in range(8)]
        reports[arm] = check_basic_rows(groups[arm], arm, manifest_sha256, hardware,
            builder.libraries(build, arm) if build else None, builder.codec_library(build) if build else None)
    # Chunking/arrival changes transport scheduling only, preserving each group's fold.
    for rank in range(8):
        for role in ROLES:
            for case in CASES:
                a, b = [groups[arm][rank]['roles'][role]['cases'][case] for arm in p.BASIC_ARMS[:2]]
                require(all(a[key] == b[key] for key in ('input_sha256', 'weight_sha256', 'output_sha256', 'wire')),
                        'Arrival pipeline changed all-quantized arithmetic or packet bytes')
    legacy = improved_probe()
    directory = root/'flux-improved'
    reports['flux-improved'] = legacy.validate_reports(directory, 'c', manifest_sha256)
    for rank in range(8):
        row = read(directory/f'rank{rank}.json')
        require(row['gpu_uuid'] == hardware['devices'][rank]['uuid']
                and all(row[key] == hardware['environment'][key] for key in ('torch', 'cuda', 'nccl'))
                and row['entrypoint_sha256'] == sha(__file__)
                and row['reference_sha256'] == sha(REPO/'tools/a800/phase4/reference.py'),
                'Improved comparator probe identity changed')
        if build is not None:
            require(row['native_libraries'] == builder.libraries(build, 'flux-improved'),
                    'Improved comparator loaded foreign libraries')
    return dict(protocol=PROTOCOL, completed=True, arms=reports,
                arrival_packet_output_bitwise=True, build_manifest_sha256=manifest_sha256,
                evidence_files={str(path.relative_to(root)): sha(path) for arm in p.ARMS
                                for path in sorted((root/arm).glob('rank*.json'))})


def native_evidence(root, plan, hardware):
    import op1_abc as controller
    root = Path(root)
    result = check_records(root/'native-probe', hardware, plan['build']['manifest_sha256'],
                           build=plan['build']['root'])
    require(result == read(root/'native-probe/report.json'), 'Native ABC evidence changed')
    mapping = read(root/'mapping-summary.json')
    require(mapping['completed'] is True and mapping['build_manifest_sha256'] == plan['build']['manifest_sha256']
            and set(mapping['evidence_files']) == {f'mapping/flux-improved-{role}.log' for role in ROLES},
            'Improved comparator actual mapping missing')
    controller.base.verify_files(root, mapping['evidence_files'])


def native_checks(args, build, plan):
    import op1_abc as controller
    import op1_abc_build as builder
    from tp8_dp2_train import run_process
    mapping = {}
    for role, k in (('attn', 4096), ('mlp', 16384)):
        native = builder.native_root(build, 'flux-improved')
        job = dict(arm='flux-improved', mode='smoke', trial=0, block=-1, index=-1)
        log = args.out/'mapping'/f'flux-improved-{role}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        env = controller.worker_environment(args, build, args.out/'workers/flux-improved/scripts', job, log.parent, plan)
        run_process([native/'check_mapping', '2048', '4096', str(k), '8', native/'libflux_cuda.so'],
                    REPO, env, log, 180, controller.base.Peer())
        mapping[str(log.relative_to(args.out))] = sha(log)
    write(args.out/'mapping-summary.json', dict(completed=True, evidence_files=mapping,
          build_manifest_sha256=plan['build']['manifest_sha256']))
    for arm in p.ARMS:
        controller.base.verify_controllers(args.out)
        controller.base.gpu_idle()
        dest = args.out/'native-probe'/arm
        dest.mkdir(parents=True)
        job = dict(arm=arm, mode='smoke', trial=0, block=-1, index=-1)
        scripts = args.out/'workers'/arm/'scripts'
        env = controller.worker_environment(args, build, scripts, job, dest, plan)
        command = [args.python, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
                   '--max-restarts=0', HERE/'op1_abc_probe.py', '--build', build, '--arm', arm,
                   '--out', dest, '--execute']
        write(dest/'command.json', dict(command=list(map(str, command))))
        print('NATIVE PROBE', arm, flush=True)
        try:
            run_process(command, REPO, env, dest/'process.log', max(900, args.worker_timeout), controller.base.Peer())
        except Exception:
            controller.base.print_smoke_failures(dest)
            raise
    write(args.out/'native-probe/report.json', check_records(args.out/'native-probe',
          read(args.out/'environment-preflight.json'), plan['build']['manifest_sha256'], build=build))
    native_evidence(args.out, plan, read(args.out/'environment-preflight.json'))


def run(args):
    if args.arm == 'flux-improved':
        return improved_probe().run(SimpleNamespace(**(vars(args) | {'arm': 'c'})))
    import torch
    import torch.distributed as dist
    import op1_abc_build as builder
    rank = int(os.environ['LOCAL_RANK'])
    require(os.environ['WORLD_SIZE'] == os.environ['LOCAL_WORLD_SIZE'] == '8', 'Eight local ranks required')
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    record = dict(protocol=PROTOCOL, completed=False, arm=args.arm, rank=rank,
                  build_manifest_sha256=sha(args.build/builder.MANIFEST), entrypoint_sha256=sha(__file__),
                  reference_sha256=sha(REPO/'tools/a800/phase4/reference.py'))
    initialized = False
    try:
        dist.init_process_group('nccl', timeout=timedelta(seconds=300))
        initialized = True
        native = builder.native_root(args.build, args.arm)
        scripts = args.out.parent.parent/'workers'/args.arm/'scripts'
        sys.path.insert(0, str(native/'python'))
        sys.path.insert(0, str(scripts))
        import flux
        import megatron_taco as codec
        from flux_tensor_taco import FluxTensorTaco
        spec = importlib.util.spec_from_file_location('abc_independent_reference', REPO/'tools/a800/phase4/reference.py')
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        stream = torch.cuda.Stream()
        prop = torch.cuda.get_device_properties(rank)
        require('5090' in prop.name and (prop.major, prop.minor, prop.multi_processor_count) == (12, 0, 170),
                'Expected full RTX 5090 SM120 GPU')
        record.update(gpu_uuid=str(prop.uuid), torch=str(torch.__version__), cuda=torch.version.cuda,
            nccl=list(torch.cuda.nccl.version()), libraries=loaded_libraries(native),
            probe_stream=int(stream.cuda_stream),
            default_stream=int(torch.cuda.default_stream().cuda_stream), roles={})
        with torch.no_grad(), torch.cuda.stream(stream):
            record['mapping'] = mapping_probe(torch, native, stream, rank)
            for role, k in ROLES.items():
                op = FluxTensorTaco(dist.group.WORLD, 2048, 4096, k)
                record['codec_library'] = codec.loaded_library()
                require(record['codec_library'] == builder.codec_library(args.build),
                        'Native probe loaded a foreign standalone codec')
                checks = {}
                for index, case in enumerate(CASES):
                    generator = torch.Generator(device='cuda').manual_seed(20261101+rank*97+index)
                    x = torch.randn((2048, k), generator=generator, dtype=torch.bfloat16, device='cuda')*.03
                    weight = torch.randn((4096, k), generator=generator, dtype=torch.bfloat16, device='cuda')*.03
                    if case == 'zero':
                        x.zero_()
                    elif case == 'outlier':
                        x[::128, 0], weight[::128, 0] = 8, 4
                    output = op.forward(x, weight)
                    checks[case] = dict(input_sha256=tensor_sha(x), weight_sha256=tensor_sha(weight),
                        output_sha256=tensor_sha(output), wire=wire_audit(op.tensor_codec, op.partial, output, codec),
                        reference=reference_check(torch, dist, reference, x.matmul(weight.t()), output,
                                                  args.arm == 'flux-basic-selective'))
                retained = []
                for iteration in range(7):
                    if rank == iteration % 8:
                        time.sleep(.025)
                    delayed_x = torch.full((2048, k), (iteration+1)/128, device='cuda', dtype=torch.bfloat16)
                    delayed_weight = torch.full((4096, k), (rank+1)/128, device='cuda', dtype=torch.bfloat16)
                    output = op.forward(delayed_x, delayed_weight)
                    if iteration < 6:
                        retained.append((output, output.clone()))
                stream.synchronize()
                require(len({value.data_ptr() for value, _ in retained}) == 6
                        and all(torch.equal(value, snapshot) for value, snapshot in retained),
                        'Workspace reuse changed retained outputs')
                record['roles'][role] = dict(k_local=k, cases=checks, retained_outputs=6, reuse_calls=7,
                    retention_passed=True, independent_storage=True, state=op.tensor_codec.state(),
                    delayed_reference=reference_check(torch, dist, reference, delayed_x.matmul(delayed_weight.t()),
                        output, args.arm == 'flux-basic-selective'))
                op.tensor_codec.close()
        record['completed'] = True
        write(args.out/f'rank{rank}.json', record)
        dist.barrier(device_ids=[rank])
    except BaseException as exc:
        record.update(completed=False, error=f'{type(exc).__name__}: {exc}')
        write(args.out/f'rank{rank}.json', record)
        raise
    finally:
        if initialized:
            dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--arm', choices=p.ARMS, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    args.build, args.out = args.build.resolve(), args.out.resolve()
    if not args.execute:
        print(json.dumps(dict(protocol=PROTOCOL, preview=True, arm=args.arm, roles=ROLES,
                              cases=CASES, performance_measured=False)))
        return
    run(args)


if __name__ == '__main__':
    main()
