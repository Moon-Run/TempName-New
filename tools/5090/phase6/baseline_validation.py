"""Validate complete RTX 5090 worker evidence without accepting historical results."""
import math

from rtx5090_loader import load

p = load('protocol')
from common import require
from ma4_f_fullstep import validate_parameter_trajectory
from ma4_f_bench import validate_consumer
import ma8_fullstep
import ma9_fullstep
import ma11_fullstep
import basic_quant_validation
from nccl_transport import wire_layout
from tp8_dp2_protocol import validate_projection_checks

PREFIX = {'ma11': 'ma11', 'ma89': 'ma11', 'ma8': 'ma8', 'ma9': 'ma9', 'cr3': 'ma4_f'}


def identity(rows):
    keys = ('rank', 'gpu_uuid', 'torch', 'cuda', 'nccl', 'python', 'hardware', 'model', 'transport',
            'initial_parameters_sha256', 'initial_rng_sha256', 'tokens_sha256',
            'parameter_count_local', 'padded_vocab_size', 'communication')
    return [{key: row[key] for key in keys} for row in rows]


def equivalent(rows, other, smoke=False):
    keys = ['losses', 'grad_norms', 'final_parameters_sha256']
    if smoke:
        keys += ['parameter_trajectory', 'trainable_parameters']
    require([{key: row[key] for key in keys} for row in rows]
            == [{key: row[key] for key in keys} for row in other],
            'Equivalent policy family changed loss, gradients or parameter updates')


def validate_hardware(row, rank):
    devices = row['hardware']['devices']
    require(len(devices) == 8 and len({d['uuid'] for d in devices}) == 8
            and all('5090' in d['name'] and d['capability'] == [12, 0]
                    and d['sm_count'] == 170 for d in devices), 'Expected eight full RTX 5090 SM120 devices')
    require(row['hardware']['local_rank'] == rank and row['gpu_uuid'] == devices[rank]['uuid'],
            'GPU rank/device order changed')
    require(row['hardware']['transport'] == p.TRANSPORT, 'Wrong worker hardware transport identity')
    peers = row['hardware']['peer_access']
    require(len(peers) == 8 and all(len(values) == 8 and all(type(x) is bool for x in values)
                                    for values in peers), 'Incomplete peer-access diagnostics')


def validate_transport(row, arm, calls, names, layouts):
    require(row['transport'] == p.TRANSPORT, 'Wrong transport protocol')
    records = row['rtx5090_transport']
    if arm == 'flux-basic':
        require(records == {}, 'Basic codec cannot masquerade as the staged TACO producer')
        require(row['native_workspace_bytes'] == 0, 'Basic worker has a TACO native workspace')
        return
    require(set(records) == names, 'Missing per-module local transport evidence')
    double = arm in p.QUANT_SELECTIVE
    workspace_bytes = 0
    for name, record in records.items():
        require(record['protocol'] == p.TRANSPORT and record['failed'] is False, 'Failed/foreign transport')
        if double:
            require(record['next_slot'] == calls % 2 and len(record['slots']) == 2,
                    'Missing double staging slots or reuse order')
            slots = record['slots']
        else:
            require('slots' not in record, 'Unexpected extra staging slot')
            slots = [record]
        role = 'attn' if name.endswith('.self_attention.linear_proj') else 'mlp'
        layout = layouts[(role, row['rank'])]
        for index, slot in enumerate(slots):
            expected_calls = (calls + 1 - index) // 2 if double else calls
            require(not double or slot['slot'] == index, 'Staging slot identity changed')
            require(slot['protocol'] == p.TRANSPORT and slot['native_api'] == 'rtx5090_local_v1'
                    and slot['peer_memory'] is False and slot['failed'] is False
                    and slot['pending_exchange'] is False and slot['closed'] is False
                    and slot['native_phase'] == 0 and slot['device'] == row['rank']
                    and type(slot['stream']) is int and slot['stream'] >= 0
                    and slot['synchronization'] == 'fixed_stream_producer_NCCL_work_wait_decode_reuse',
                    'Incomplete local producer/exchange/decode lifecycle')
            require(all(slot[key] == expected_calls for key in
                        ('calls', 'producer_calls', 'pack_calls', 'exchange_calls',
                         'wait_calls', 'unpack_calls', 'decode_calls')),
                    'Transport stage missing, duplicated or not waited')
            initializations = int(expected_calls > 0) if arm in p.CR3_FAMILY else expected_calls
            consumers = expected_calls if arm in p.CR3_FAMILY else 0
            reconfigures = expected_calls if arm == 'a' else 0
            require(slot['initializations'] == initializations and slot['consumer_calls'] == consumers
                    and slot['reconfigure_calls'] == reconfigures
                    and slot['configuration_scope'] == ('per_call' if arm == 'a' else 'instance'),
                    'Native submission/consumer/configuration variant changed')
            require(slot['send_split_bytes'] == layout['tx_splits']
                    and slot['receive_split_bytes'] == layout['rx_splits']
                    and slot['send_bytes'] == layout['tx_bytes'] and slot['receive_bytes'] == layout['rx_bytes'],
                    'Compact transport byte budget or self exclusion changed')
            allocated = slot['native_workspace_bytes']
            require(type(allocated) is int and allocated >= 0, 'Missing raw CUDA TACO workspace accounting')
            workspace_bytes += allocated
            expected_native = dict(protocol_version=1, initializations=initializations,
                                   producer_calls=expected_calls, decode_calls=expected_calls,
                                   bias_residual_calls=consumers, phase=0, failed=False,
                                   bound_k=512 if role == 'attn' else 2048, reconfigure_calls=reconfigures,
                                   workspace_bytes=allocated)
            require(slot['native'] == expected_native, 'Native local execution evidence disagrees with transport')
    require(type(row['native_workspace_bytes']) is int and row['native_workspace_bytes'] == workspace_bytes,
            'Native TACO workspace total differs from per-slot allocations')


def validate_rca_state(row, arm, worker, names):
    states, workspaces = row['ma2c_codec_state'], row['workspace_checks']
    slots = 2 if arm == 'c' else 1
    require(row['ma2c_arm'] == p.NATIVE_ARM[arm] and set(states) == names == set(workspaces),
            'R/C/A native identity or workspace coverage changed')
    expected = dict(owns_output=arm != 'a', owns_configuration=arm != 'a', workspaces=slots)
    require(all(state == expected for state in states.values()), 'R/C/A ownership or workspace count changed')
    pointers = []
    for name, values in workspaces.items():
        role = 'attn' if name.endswith('.self_attention.linear_proj') else 'mlp'
        shape = [2048, 4096, 4096 if role == 'attn' else 16384]
        keys = {'shape', 'packet_ptr', 'mask_ptr', 'mask_sha256'} if arm == 'c' else {'shape', 'packet_ptr'}
        require(len(values) == slots and all(set(value) == keys and value['shape'] == shape
                                            for value in values), 'R/C/A workspace layout changed')
        for value in values:
            pointers.append(value['packet_ptr'])
            if arm == 'c':
                mask = worker['selection']['variants'][p.POLICY[arm]]['roles'][role]['mask_sha256']
                require(value['mask_sha256'] == mask and value['mask_ptr'] > 0, 'Frozen C mask changed')
    require(all(type(value) is int and value > 0 for value in pointers)
            and len(set(pointers)) == len(pointers), 'Aliased R/C/A communication workspaces')


def validate_mechanisms(row, arm, mode, warmup, count, names, worker):
    calls = warmup + count
    if arm in p.CR3_FAMILY:
        prefix = PREFIX[arm]
        consumer = row[prefix+'_consumer_state']
        require(set(consumer) == names == set(row[prefix+'_submission_state']), 'Missing CR3 consumer modules')
        for name, state in consumer.items():
            require(state['projection_calls'] == state['bda_calls'] == calls*4
                    and state['bias_parameter_id'] > 0
                    and sum(x['successful_calls'] for x in state['native']) == calls*4,
                    'Consumer calls missing')
            validate_consumer(state['native'], row[prefix+'_submission_state'][name], 'F4')
        if arm in ('ma11', 'ma89'):
            ma11_fullstep.validate_activation_setup(row['ma11_activation_setup'], names)
            ma11_fullstep.validate_zero_count_state(row['ma11_zero_count_state'], p.NATIVE_ARM[arm],
                                                   calls, mode, row['rank'])
            ma11_fullstep.validate_softmax_state(row['ma11_softmax_state'], p.NATIVE_ARM[arm], calls, mode)
        elif arm == 'ma8':
            ma8_fullstep.validate_activation_setup(row['ma8_activation_setup'], 'F4-cache', names)
        elif arm == 'ma9':
            ma9_fullstep.validate_zero_count_state(row['ma9_zero_count_state'], 'F4-zcount', calls, mode, row['rank'])
    elif arm in ('r', 'c', 'a'):
        validate_rca_state(row, arm, worker, names)
    else:
        cfg = worker['config']
        require(row['codec_identity'] == cfg['codec_identity'] and row['codec_library'] == cfg['codec_library']
                and set(row['codec_calls']) == names and all(v == calls*4 for v in row['codec_calls'].values()),
                'Wrong standalone basic codec')
        if mode == 'smoke':
            basic_quant_validation.verify(row['codec_validation'], cfg['codec_library'], row['rank'])
            require(set(row['retained_output_checks']) == {'attn', 'mlp'}
                    and all(value['passed'] and value['independent_storage']
                            and value['unchanged_after_subsequent_call']
                            for value in row['retained_output_checks'].values()), 'Basic output ownership failed')
        else:
            require(row['codec_validation'] is None, 'Codec audit leaked into timing/profile')
    allowed = {'ma11': {'ma11_activation_setup', 'ma11_zero_count_state', 'ma11_softmax_state'},
               'ma89': {'ma11_activation_setup', 'ma11_zero_count_state', 'ma11_softmax_state'},
               'ma8': {'ma8_activation_setup'}, 'ma9': {'ma9_zero_count_state'}}.get(arm, set())
    mechanism_fields = {'ma11_activation_setup', 'ma11_zero_count_state', 'ma11_softmax_state',
                        'ma8_activation_setup', 'ma9_zero_count_state'}
    require(set(row).intersection(mechanism_fields) == allowed, 'Arm has missing or unintended mechanisms')


def validate_rows(rows, job, plan, worker):
    require(len(rows) == 8 and {row['rank'] for row in rows} == set(range(8)), 'Missing/duplicate rank')
    rows = sorted(rows, key=lambda row: row['rank'])
    require(len({row['gpu_uuid'] for row in rows}) == 8, 'Eight distinct GPUs required')
    arm, mode = job['arm'], job['mode']
    require(arm in p.ARMS and mode in ('smoke', 'timing', 'profile'), 'Unknown arm or mode')
    warmup, count = (2, 3) if mode == 'smoke' else (10, 1 if mode == 'profile' else 50)
    names = {f'decoder.layers.{i}.{suffix}' for i in range(32)
             for suffix in ('self_attention.linear_proj', 'mlp.linear_fc2')}
    cfg = worker['config']
    require(cfg['transport'] == plan['transport'] == p.TRANSPORT, 'Configuration is from a foreign transport')
    require(all(plan['communication']['environment'].get(key) == value
                for key, value in p.TRANSPORT_ENV.items()), 'NCCL staging settings changed')
    layouts = {}
    if arm != 'flux-basic':
        for role in ('attn', 'mlp'):
            mask = (worker['selection']['variants'][p.POLICY[arm]]['roles'][role]['entry']['mask']
                    if arm in p.QUANT_SELECTIVE else None)
            for rank in range(8):
                layouts[(role, rank)] = wire_layout(mask, rank)
    for rank, row in enumerate(rows):
        if arm == 'ma11' and plan.get('ma11_compat'):
            compat = plan['ma11_compat']
            require(row.get('rtx5090_ma11_softmax') ==
                    dict(path=compat['library'], sha256=compat['library_sha256']),
                    'MA11 compatibility library identity changed')
        require(row['protocol'] == p.PROTOCOL and row['run_id'] == plan['run_id']
                and row['arm'] == arm and row['mode'] == mode and row['block'] == job['block']
                and row['window'] == job['index'] and row['trial'] == job['trial']
                and row['worker_config_sha256'] == worker['files']['config.json']
                and row['completed'] is True and row['passed'] is True and row['skips'] == 0,
                'Worker identity/completion changed')
        require(row['world_size'] == 8 and row['local_rank'] == row['tp_rank'] == rank
                and row['dp_rank'] == 0 and row['tp_group_ranks'] == list(range(8))
                and row['dp_group_ranks'] == [rank], 'Expected single-node TP8 DP1')
        validate_hardware(row, rank)
        require(all(row['model'][key] == value for key, value in p.MODEL.items() if key != 'sm_count')
                and row['precision'] == 'bf16'
                and all(row[key] == plan['environment'][key] for key in ('torch', 'cuda', 'nccl', 'python'))
                and row['policy'] == p.POLICY[arm] and row['communication'] == plan['communication']
                and row['scope_definition'] == worker['definition']
                and row['role_plan_sha256'] == cfg['role_plan_sha256'], 'Model/software/route changed')
        require(row['sampling'] is False and row['profiling'] == (mode == 'profile')
                and row['tokens_per_optimizer_step'] == 8192 and row['warmup_steps'] == warmup
                and row['timing_protocol'] == 'full-step-one-clear-v2'
                and len(row['losses']) == len(row['grad_norms']) == count
                and all(math.isfinite(value) for value in row['losses'] + row['grad_norms'])
                and row['losses'] == rows[0]['losses'], 'Workload/timing or finite TP loss checks failed')
        require(set(row['target_modules']) == names == set(row['timed_calls']) == set(row['target_calls'])
                == set(row['enabled_modules']) and row['independent_scope_instances'] == row['replaced_modules'] == 64
                and all(value == count*4 for value in row['timed_calls'].values())
                and row['warmup_calls'] == row['target_calls'] == dict.fromkeys(names, warmup*4)
                and row['fallback_calls'] == 0, 'Missing projection coverage or fallback')
        route = 'flux_tensor' if arm == 'flux-basic' else 'candidate'
        require(row['module_policies'] == row['post_audit_routes'] == dict.fromkeys(names, route),
                'Measured module route changed')
        validate_projection_checks(row, mode)
        if mode == 'smoke':
            for check in row['forward_checks'].values():
                require(check['budget_vs_bf16'] == .05 and check['budget_vs_codec'] == .01
                        and all(math.isfinite(check[key]) for key in
                                ('max_abs', 'relative_l2', 'protocol_relative_l2'))
                        and 0 <= check['relative_l2'] <= .05
                        and 0 <= check['protocol_relative_l2'] <= .01,
                        'Projection numerical budget changed')
            require(len(row['trainable_parameters']) == 389, 'Incomplete parameter universe')
            require({name+suffix for name in names for suffix in ('.weight', '.bias')}
                    <= set(row['trainable_parameters']), 'Missing target weight/bias gradients')
            validate_parameter_trajectory(row['parameter_trajectory'], row['trainable_parameters'])
        else:
            require(row['parameter_trajectory'] == [], 'Parameter audit leaked into timing/profile')
        fingerprint = row['final_parameters_sha256']
        require(isinstance(fingerprint, str) and len(fingerprint) == 64
                and all(c in '0123456789abcdef' for c in fingerprint), 'Missing final parameter fingerprint')
        for key in ('peak_allocated_gib', 'peak_reserved_gib'):
            require(math.isfinite(row[key]) and row[key] > 0, 'Invalid memory peak')
        if mode != 'profile':
            elapsed = row['elapsed_seconds']
            require(row['timed_steps'] == count and math.isfinite(elapsed) and elapsed > 0
                    and math.isclose(row['ms_per_step'], elapsed*1000/count, rel_tol=1e-10),
                    'Invalid wall-clock sample')
        for name, expected in worker['libraries'].items():
            require(row[name] == expected, 'Wrong loaded native binary: '+name)
        validate_mechanisms(row, arm, mode, warmup, count, names, worker)
        validate_transport(row, arm, (warmup+count)*4, names, layouts)
    return rows
