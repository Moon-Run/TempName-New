"""Launch-correlated OP1 path audit, including absence of packet transpose work."""
import inspect
from types import ModuleType, SimpleNamespace

from rtx5090_loader import load
import op1_a_protocol as p
from op1_a_sources import BASIC_ARMS, replace

base = load('profile')
require = p.require
SORTED_SWIZZLE = 'Op1AInterleavedSwizzle'


def is_basic_sk_gemm(name):
    """Keep the original Flux StreamkSK registration for both basic arms."""
    return (base.is_basic_gemm(name) and '_streamksk_' in name.lower()
            and '_streamkdp_' not in name.lower())


def _direct_projection_analyzer():
    # Derive only the changed transport stages from the already audited 5090
    # analyzer. Launch ownership, NCCL stream dependencies, all GPU work and
    # exact downstream BF16-add/LayerNorm checks remain in this private copy.
    source = inspect.getsource(base.staged_projection_records)
    source = replace(source, "require(arm in p.ARMS[:-1], 'Unknown staged projection arm')",
                     "require(arm == 'a', 'Unknown direct projection arm')")
    source = source.replace("if arm == 'a':", 'if False:')
    source = source.replace("int(arm == 'a')", '0')
    source = replace(source, """        pack = _one([event for event in producer_kernels if PACK_KERNEL in event['name']],
                    'Missing local packet pack kernel')
        require(len(producer_kernels) == 2, 'Unexpected producer kernel or unfused encoding fallback')""",
        "        require(len(producer_kernels) == 1, 'Direct producer must only launch its fused GEMM')")
    source = replace(source, "        _before(gemm, pack, 'Packet packing started before local GEMM finished')\n", '')
    source = replace(source, """        require(len(decoded) == 2 and all(event['cat'] == 'kernel' for event in decoded),
                'Expected one unpack kernel and one original decoder')
        unpack = _one([event for event in decoded if UNPACK_KERNEL in event['name']],
                      'Missing local packet unpack kernel')""",
        """        require(len(decoded) == 1 and decoded[0]['cat'] == 'kernel',
                'Direct packets must feed exactly one original A decoder')""")
    source = replace(source, "        _before(unpack, decoder, 'Decoder consumed packets before unpack completed')\n", '')
    source = replace(source, '        compute = (gemm, pack, unpack, decoder)', '        compute = (gemm, decoder)')
    source = replace(source, "            _before(pack, event, 'NCCL exchange consumed unfinished packed data')",
                     "            _before(gemm, event, 'NCCL consumed an unfinished direct packet')")
    source = replace(source, "            _before(event, unpack, 'Unpack started before NCCL exchange completed')",
                     "            _before(event, decoder, 'Direct decoder started before NCCL completion')")
    source = replace(source, "require(actual_packets <= claimed and len(actual_packets) == 512,\n            'Unowned or missing packet pack/unpack kernels')",
                     "require(not actual_packets, 'OP1 A must eliminate every packet pack/unpack kernel')")
    source = replace(source, 'packet_pack_kernels=128, packet_unpack_kernels=128, decoder_kernels=128,',
                     'packet_pack_kernels=0, packet_unpack_kernels=0, decoder_kernels=128,')
    module = ModuleType('rtx5090_op1_direct_profile')
    module.__dict__.update(base.__dict__)
    module.PRODUCE = 'rtx5090.transport.produce_direct'
    module.DECODE = 'rtx5090.transport.decode_direct'
    module.p = SimpleNamespace(ARMS=p.ARMS, CR3_FAMILY=(), NATIVE_ARM={'a': 'A'},
                              TRANSPORT='rtx5090-op1-a-direct-packet-v1')
    exec(compile(source, __file__, 'exec'), module.__dict__)
    return module.staged_projection_records


direct_projection_records = _direct_projection_analyzer()


def basic_projection_records(events, arm):
    require(arm in BASIC_ARMS, 'Unknown basic OP1 arm')
    result = base.basic_projection_records(events)
    index = base.Index(events)
    sorted_arm = arm == 'flux-basic-interleaved'
    for scope in events:
        if (scope.get('cat') != 'user_annotation' or scope.get('ph') != 'X'
                or scope.get('name') not in ('scope.attn.flux_tensor', 'scope.mlp.flux_tensor')):
            continue
        work, _ = index.work(scope)
        gemms = [event for event in work if event.get('cat') == 'kernel'
                 and base.is_basic_gemm(event.get('name', ''))]
        require(len(gemms) == 1 and is_basic_sk_gemm(gemms[0]['name']),
                'OP1 Flux must retain the original StreamkSK registration')
        require((SORTED_SWIZZLE in gemms[0]['name']) == sorted_arm,
                'Actual GemmOnly kernel swizzle does not match the selected arm')
    for value in result.values():
        value.update(cta_order='rank-rotated-destination-interleaved' if sorted_arm else 'original-flux',
                     gemm_schedule='StreamkSK', may_change_streamk_reduction_assignment=sorted_arm,
                     swizzle_verified_from_kernel_type=True)
    return result


def analyze(events, arm, rank):
    require(arm in p.ARMS and 0 <= rank < 8, 'Unknown OP1 profile identity')
    require(not any(token in event.get('name', '') for event in events
                    for token in ('_MA8CachedRawLinear', 'ma11.attention.softmax', 'ma9.optimizer.zero_count')),
            'OP1 unexpectedly enabled a later optimization')
    if arm == 'a':
        result = direct_projection_records(events, arm, rank)
    else:
        result = basic_projection_records(events, arm)
    return dict(protocol=p.PROTOCOL, arm=arm, rank=rank,
                execution_validated=True, row_forward=result)
