"""Reuse ablation6 A/B/C adapters; replace only the shared baseline codec."""
import hashlib
import ablation6_sources as parent
import baseline_codec_build as codec_build
import baseline_protocol as routing
from common import HERE, REPO, replace_once, sha


def codec_source():
    return (HERE/'baseline_codec.py').read_text()


def codec_identity():
    return dict(implementation=routing.CODEC,
                source_sha256=hashlib.sha256(codec_source().encode()).hexdigest(),
                reference_sha256=sha(REPO/'tools/a800/phase4/reference.py'),
                cuda_source_identity=codec_build.source_identity())


taco_source = parent.taco_source


def adapter_source():
    return parent.adapter_source().replace('naive_codec_calls', 'codec_calls')


def worker_source(source_only=False):
    text = parent.worker_source(source_only).replace('H100_ABLATION6_EXECUTE', 'H100_BASELINE_EXECUTE')
    text = text.replace('GPU worker requires ablation6 --execute', 'GPU worker requires baseline --execute')
    text = text.replace('naive_codec_identity', 'codec_identity').replace('naive_codec_calls', 'codec_calls')
    text = replace_once(text, "report['codec_identity'] = dict(implementation=PLAN['codec_identity']['implementation'],",
                        "report['codec_identity'] = dict(PLAN['codec_identity'],")
    anchor = 'torch.cuda.set_device(local_rank)'
    text = replace_once(text, anchor, anchor+'''
from routing import is_tensor
import megatron_taco as standalone_codec
codec_validation = None
if is_tensor(POLICY) and MODE == 'smoke':
    from baseline_codec_validation import validate as validate_codec
    import reference as codec_reference
    codec_validation = validate_codec(standalone_codec, codec_reference)
''')
    anchor = "report['codec_calls'] = adapter.codec_calls()"
    return replace_once(text, anchor, anchor+'''
report['codec_library'] = standalone_codec.loaded_library()
report['codec_validation'] = codec_validation
''')
