#!/usr/bin/env python3
"""Independent TP8 OP3 packet, decoder, consumer rounding and ownership gates."""
import argparse
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
from rtx5090_loader import HERE, REPO
import op1_abc_probe as original
import op3_protocol as p

PROTOCOL = 'rtx5090-op3-distributed-probe-v1'
ROLES, CASES = original.ROLES, original.CASES
FUSION_CASES = ('zero', 'half_ulp', 'signed_zero', 'subnormal', 'cancellation', 'random')
read, write, require, sha = p.read, p.write, p.require, p.sha
tensor_sha, is_digest = original.tensor_sha, original.is_digest


def check_state(state, arm, calls, rank, fused_calls=0):
    """Use the same transport contract as full training with explicit probe counts."""
    from op3_validation import validate_codec_state
    validate_codec_state(state, arm, calls, rank, fused_calls=fused_calls)


def _check_wire(value, rank, peers):
    require(value['upstream_packet_bitwise'] is True and value['upstream_decode_fold_bitwise'] is True,
            'OP3 packet or decoder lacks the original upstream bitwise oracle')
    for direction in ('sent', 'received'):
        require(set(value[direction]) == peers and all(item['bytes'] == 8192*136
                and is_digest(item['sha256']) for item in value[direction].values()),
                'Invalid OP3 directed packet evidence')


def check_rows(rows, arm, manifest, hardware=None, libraries=None, codec_library=None):
    require(arm in p.ARMS and len(rows) == 8, 'Eight OP3 probe ranks required')
    for rank, row in enumerate(rows):
        require(row['protocol'] == PROTOCOL and row['completed'] is True and row['rank'] == rank
                and row['arm'] == arm and row['build_manifest_sha256'] == manifest
                and row['entrypoint_sha256'] == sha(__file__)
                and row['reference_sha256'] == sha(REPO/'tools/a800/phase4/reference.py')
                and row['wire_oracle_sha256'] == sha(HERE/'op1_abc_probe.py')
                and set(row['roles']) == set(ROLES) and set(row['mapping']) == set(ROLES)
                and type(row['probe_stream']) is int and row['probe_stream'] > 0
                and row['probe_stream'] != row['default_stream'], 'Foreign/incomplete OP3 native evidence')
        if hardware is not None:
            require(row['gpu_uuid'] == hardware['devices'][rank]['uuid'] and all(
                row[key] == hardware['environment'][key] for key in ('torch', 'cuda', 'nccl')),
                'OP3 probe environment differs from preflight')
        if libraries is not None:
            require(row['libraries'] == libraries and row['codec_library'] == codec_library,
                    'OP3 probe loaded a foreign library')
        peers = {str(peer) for peer in range(8) if peer != rank}
        for role, item in row['roles'].items():
            original.validate_mapping(row['mapping'][role], rank)
            require(item['k_local'] == ROLES[role] and set(item['cases']) == set(CASES)
                    and item['retained_outputs'] == 6 and item['reuse_calls'] == 7
                    and item['retention_passed'] is True and item['independent_storage'] is True,
                    'Missing OP3 full-shape/reference/output-ownership cases')
            check_state(item['state'], arm, 10, rank)
            original.check_reference(item['delayed_reference'])
            for case, value in item['cases'].items():
                original.check_reference(value['reference'])
                require(all(is_digest(value[key]) for key in
                    ('input_sha256', 'weight_sha256', 'output_sha256')), 'Invalid OP3 projection fingerprint')
                _check_wire(value['wire'], rank, peers)
                for peer in range(8):
                    if peer != rank:
                        other = rows[peer]['roles'][role]['cases'][case]['wire']
                        require(value['wire']['sent'][str(peer)] == other['received'][str(rank)],
                                'OP3 NCCL directed packet mismatch')
        if arm == 'op3-r':
            fusion = row['consumer_fusion']
            require(set(fusion['cases']) == set(FUSION_CASES) and fusion['retained_outputs'] == 6
                    and fusion['reuse_calls'] == 7 and fusion['retention_passed'] is True
                    and fusion['independent_storage'] is True
                    and fusion['dynamic_bias_residual_storage'] is True,
                    'Missing fused-consumer edge cases or lifetime proof')
            check_state(fusion['state'], arm, 13, rank, fused_calls=7)
            for case, value in fusion['cases'].items():
                require(value['two_rounding_bitwise'] is True and value['inputs_unchanged'] is True
                    and all(is_digest(value[key]) for key in
                    ('partial_sha256', 'bias_sha256', 'residual_sha256', 'raw_sha256',
                     'expected_sha256', 'output_sha256'))
                    and value['output_sha256'] == value['expected_sha256'],
                    'R consumer differs from two separately rounded BF16 adds')
                _check_wire(value['wire'], rank, peers)
                for peer in range(8):
                    if peer != rank:
                        other = rows[peer]['consumer_fusion']['cases'][case]['wire']
                        require(value['wire']['sent'][str(peer)] == other['received'][str(rank)],
                                'Fusion input packets disagree across NCCL ranks')
                if case == 'half_ulp':
                    require(value['single_rounding_differs'] is True,
                            'Half-ULP sentinel did not distinguish fused FP32 addition')
        else:
            require(row['consumer_fusion'] is None, 'Candidate silently enabled CR3 consumer fusion')
    return dict(completed=True, arm=arm, protocol=PROTOCOL, directed_links=56,
        actual_sorted_gemm_mapping_passed=True, upstream_packet_decode_bitwise=True,
        fused_two_rounding_bitwise=arm == 'op3-r', performance_measured=False)


def check_records(root, hardware, plan):
    import op3_build as builder
    root, build = Path(root), Path(plan['build']['root'])
    groups, summaries = {}, {}
    for arm in p.ARMS:
        groups[arm] = [read(root/arm/f'rank{rank}.json') for rank in range(8)]
        summaries[arm] = check_rows(groups[arm], arm, plan['build']['manifest_sha256'],
            hardware, builder.libraries(build, arm), builder.codec_library(build))
    for rank in range(8):
        for role in ROLES:
            for case in CASES:
                left, right = [groups[arm][rank]['roles'][role]['cases'][case] for arm in p.ARMS]
                require(all(left[key] == right[key] for key in
                    ('input_sha256', 'weight_sha256', 'output_sha256', 'wire')),
                    'OP3 arms changed raw projection arithmetic or wire bytes')
    return dict(protocol=PROTOCOL, completed=True, arms=summaries,
        raw_projection_packet_output_bitwise=True, fused_two_rounding_bitwise=True,
        build_manifest_sha256=plan['build']['manifest_sha256'],
        evidence_files={str(path.relative_to(root)): sha(path) for arm in p.ARMS
                        for path in sorted((root/arm).glob('rank*.json'))})


def native_evidence(root, plan, hardware):
    root = Path(root)
    require(check_records(root/'native-probe', hardware, plan) == read(root/'native-probe/report.json'),
            'OP3 native probe evidence changed')


def native_checks(args, build, plan):
    import op3 as controller
    from tp8_dp2_train import run_process
    for arm in p.ARMS:
        controller.base.verify_controllers(args.out)
        controller.base.gpu_idle()
        dest = args.out/'native-probe'/arm
        dest.mkdir(parents=True)
        job = dict(arm=arm, mode='smoke', trial=0, block=-1, index=-1)
        env = controller.worker_environment(args, build, args.out/'workers'/arm/'scripts', job, dest, plan)
        command = [args.python, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
                   '--max-restarts=0', HERE/'op3_probe.py', '--build', build,
                   '--arm', arm, '--out', dest, '--execute']
        write(dest/'command.json', dict(command=list(map(str, command))))
        print('NATIVE PROBE', arm, flush=True)
        try:
            run_process(command, REPO, env, dest/'process.log', max(900, args.worker_timeout), controller.base.Peer())
        except Exception:
            controller.base.print_smoke_failures(dest)
            raise
    write(args.out/'native-probe/report.json', check_records(args.out/'native-probe',
          read(args.out/'environment-preflight.json'), plan))
    native_evidence(args.out, plan, read(args.out/'environment-preflight.json'))


def fusion_inputs(torch, rank, case, index):
    """Local-only sentinels avoid quantization obscuring exact BF16 boundaries."""
    partial = torch.zeros((2048, 4096), device='cuda', dtype=torch.bfloat16)
    local = partial.reshape(8, 256, 4096)[rank]
    bias = torch.zeros(4096, device='cuda', dtype=torch.bfloat16)
    residual = torch.zeros((256, 4096), device='cuda', dtype=torch.bfloat16)
    if case == 'half_ulp':
        local.fill_(1)
        bias.fill_(1/256)
        residual.fill_(-1)
    elif case == 'signed_zero':
        partial[:, ::2] = -0.0
        bias[::2], residual[:, 1::2] = -0.0, -0.0
    elif case == 'subnormal':
        local[:, ::2], local[:, 1::2] = 2.0**-133, -2.0**-133
        bias.fill_(2.0**-133)
        residual.fill_(-2.0**-132)
    elif case == 'cancellation':
        local[:, ::2], local[:, 1::2] = 128., -128.
        bias[::2], bias[1::2] = .5, -.5
        residual[:, ::2], residual[:, 1::2] = -128., 128.
    elif case == 'random':
        generator = torch.Generator(device='cuda').manual_seed(20261102+rank*97+index)
        partial = torch.randn(partial.shape, generator=generator, dtype=torch.bfloat16, device='cuda')*.03
        bias = torch.randn(bias.shape, generator=generator, dtype=torch.bfloat16, device='cuda')*.03
        residual = torch.randn(residual.shape, generator=generator, dtype=torch.bfloat16, device='cuda')*.03
    elif case != 'zero':
        raise ValueError('Unknown fusion sentinel')
    return partial, bias, residual


def fusion_probe(torch, dist, codec, stream, rank):
    operator = codec.TensorTacoRS(dist.group.WORLD, 2048, 4096)
    checks, retained, inputs = {}, [], []
    try:
        for index, case in enumerate(FUSION_CASES):
            partial, bias, residual = fusion_inputs(torch, rank, case, index)
            # Keep inputs alive to prove changing storage rather than allocator reuse.
            inputs.append((bias, residual))
            raw = operator.reduce(partial)
            wire = original.wire_audit(operator, partial, raw, codec)
            expected = residual + (raw+bias)
            bias_before, residual_before = tensor_sha(bias), tensor_sha(residual)
            output = operator.reduce(partial, bias=bias, residual=residual)
            same = torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
            require(same, 'Fused R decoder changed bias/residual BF16 rounding in '+case)
            require(tensor_sha(bias) == bias_before and tensor_sha(residual) == residual_before,
                    'Fused R decoder modified consumer inputs in '+case)
            checks[case] = dict(partial_sha256=tensor_sha(partial), bias_sha256=tensor_sha(bias),
                residual_sha256=tensor_sha(residual), raw_sha256=tensor_sha(raw),
                expected_sha256=tensor_sha(expected), output_sha256=tensor_sha(output),
                two_rounding_bitwise=True, inputs_unchanged=True, wire=wire)
            if case == 'half_ulp':
                single = (raw.float()+bias.float()+residual.float()).to(torch.bfloat16)
                require(not torch.equal(single.view(torch.uint8), expected.view(torch.uint8)),
                        'Half-ULP sentinel failed to distinguish the rounding contract')
                checks[case]['single_rounding_differs'] = True
            retained.append((output, output.clone()))
        # One further submission with live earlier outputs catches workspace aliasing.
        if rank == 3:
            time.sleep(.025)
        partial, bias, residual = fusion_inputs(torch, rank, 'random', 99)
        operator.reduce(partial, bias=bias, residual=residual)
        stream.synchronize()
        require(len({value.data_ptr() for value, _ in retained}) == 6 and all(
                torch.equal(value.view(torch.uint8), snapshot.view(torch.uint8)) for value, snapshot in retained),
                'Fused R output storage was overwritten by a later call')
        require(len({bias.data_ptr() for bias, _ in inputs}) == 6 and
                len({residual.data_ptr() for _, residual in inputs}) == 6,
                'Fused consumer probe did not vary input storage')
        return dict(cases=checks, retained_outputs=6, reuse_calls=7, retention_passed=True,
            independent_storage=True, dynamic_bias_residual_storage=True, state=operator.state())
    finally:
        operator.close()


def run(args):
    import torch
    import torch.distributed as dist
    import op3_build as builder
    rank = int(os.environ['LOCAL_RANK'])
    require(os.environ['WORLD_SIZE'] == os.environ['LOCAL_WORLD_SIZE'] == '8', 'Eight local ranks required')
    require(os.environ['E2E_POLICY'] == args.arm, 'Probe policy differs from requested arm')
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    record = dict(protocol=PROTOCOL, completed=False, arm=args.arm, rank=rank,
        build_manifest_sha256=sha(args.build/builder.MANIFEST), entrypoint_sha256=sha(__file__),
        reference_sha256=sha(REPO/'tools/a800/phase4/reference.py'),
        wire_oracle_sha256=sha(HERE/'op1_abc_probe.py'))
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
        spec = importlib.util.spec_from_file_location('op3_independent_reference', REPO/'tools/a800/phase4/reference.py')
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        stream = torch.cuda.Stream()
        prop = torch.cuda.get_device_properties(rank)
        require('5090' in prop.name and (prop.major, prop.minor, prop.multi_processor_count) == (12, 0, 170),
                'Expected full RTX 5090 SM120 GPU')
        record.update(gpu_uuid=str(prop.uuid), torch=str(torch.__version__), cuda=torch.version.cuda,
            nccl=list(torch.cuda.nccl.version()), libraries=original.loaded_libraries(native),
            probe_stream=int(stream.cuda_stream), default_stream=int(torch.cuda.default_stream().cuda_stream), roles={})
        with torch.no_grad(), torch.cuda.stream(stream):
            record['mapping'] = original.mapping_probe(torch, native, stream, rank)
            for role, k in ROLES.items():
                op = FluxTensorTaco(dist.group.WORLD, 2048, 4096, k)
                record['codec_library'] = codec.loaded_library()
                require(record['codec_library'] == builder.codec_library(args.build), 'Foreign OP3 codec library')
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
                        output_sha256=tensor_sha(output), wire=original.wire_audit(op.tensor_codec, op.partial, output, codec),
                        reference=original.reference_check(torch, dist, reference, x.matmul(weight.t()), output, False))
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
                require(len({value.data_ptr() for value, _ in retained}) == 6 and all(
                    torch.equal(value.view(torch.uint8), snapshot.view(torch.uint8)) for value, snapshot in retained),
                    'OP3 projection workspace reuse changed retained outputs')
                record['roles'][role] = dict(k_local=k, cases=checks, retained_outputs=6, reuse_calls=7,
                    retention_passed=True, independent_storage=True, state=op.tensor_codec.state(),
                    delayed_reference=original.reference_check(torch, dist, reference,
                        delayed_x.matmul(delayed_weight.t()), output, False))
                op.tensor_codec.close()
            record['consumer_fusion'] = fusion_probe(torch, dist, codec, stream, rank) if args.arm == 'op3-r' else None
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
            cases=CASES, fusion_cases=FUSION_CASES if args.arm == 'op3-r' else [], performance_measured=False)))
        return
    run(args)


if __name__ == '__main__':
    main()
