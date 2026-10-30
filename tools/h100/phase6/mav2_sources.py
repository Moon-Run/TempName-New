"""MAV2 worker overlays on the registered Flux basic-quantization v2 path."""
import hashlib

import basic_quant_sources as baseline
import mav2_build as codec_build
import mav2_protocol as routing
from common import HERE, REPO, replace_once, sha


def codec_source():
    return (HERE/'mav2_codec.py').read_text()


def codec_identity():
    return dict(implementation=routing.CODEC,
                source_sha256=hashlib.sha256(codec_source().encode()).hexdigest(),
                reference_sha256=sha(REPO/'tools/a800/phase4/reference.py'),
                selective_reference_sha256=sha(HERE/'mav2_reference.py'),
                cuda_source_identity=codec_build.source_identity(),
                transform=dict(routing.TRANSFORM), quality_budget=dict(routing.QUALITY_BUDGET),
                receiver_implementation='basic-ordered-bf16-fold-h8-or-d8-v1')


def taco_source():
    text = baseline.taco_source()
    text = replace_once(text, 'from reference import roundtrip',
                        'import os\nfrom reference import roundtrip\n'
                        'from mav2_reference import roundtrip as selective_roundtrip, direct_roundtrip')
    # Each source contribution uses the same independent per-group mode decision
    # as the candidate. The H8 control continues to use the frozen H8 reference.
    return replace_once(text, '    coded, _ = roundtrip(piece)',
        "    oracle = {'basic': roundtrip, 'mav2': selective_roundtrip, 'd8': direct_roundtrip}[os.environ['E2E_POLICY']]\n"
        "    coded, _ = oracle(piece)")


def adapter_source():
    return baseline.adapter_source()


def worker_source(source_only=False):
    text = baseline.worker_source(source_only)
    text = text.replace('H100_BASIC_QUANT_EXECUTE', 'H100_MAV2_EXECUTE')
    text = text.replace('GPU worker requires basic-quant --execute', 'GPU worker requires MAV2 --execute')
    text = replace_once(text, 'from basic_quant_validation import validate as validate_codec',
                        'from mav2_validation import validate as validate_codec')
    anchor = "report['codec_calls'] = adapter.codec_calls()"
    return replace_once(text, anchor, anchor+'''
# Read the first real packet sample only after the measured/profile region.
# No mode-count copies, reductions or synchronization enter formal timing.
report['transform_diagnostics'] = ({name: op.codec.tensor_codec.transform_diagnostics()
    for name, op in adapter.scope_ops.items()} if MODE in ('smoke', 'profile') else {})
''')
