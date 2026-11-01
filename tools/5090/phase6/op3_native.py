"""OP3 direct-wire R decoder and warp-owned, upstream-exact H128 encoder."""
import hashlib
from pathlib import Path

from rtx5090_loader import HERE, H100
import op1_abc_native as original

EXPORTS = ('baseline_taco_encode', 'baseline_taco_decode', 'basic_quant_decode_reduce',
           'baseline_taco_error', 'op1_abc_encode', 'op1_abc_decode',
           'op3_r_decode', 'op3_encode_warp')


def source_files():
    sources = original.source_files()
    sources.update({
        'op3_codec.cu': (HERE/'op3_codec.cu').read_text(),
        'op3.cu': '#include "op1_abc.cu"\n#include "op3_codec.cu"\n',
    })
    return sources


def identity():
    return dict(protocol='rtx5090-op3-direct-h128-v1', upstream=original.identity(),
        r_reference_sha256=hashlib.sha256((H100/'ma2_decode.cuh').read_bytes()).hexdigest(),
        consumer_reference_sha256=hashlib.sha256((H100/'ma4_f_sources.py').read_bytes()).hexdigest(),
        source_sha256={name: hashlib.sha256(text.encode()).hexdigest()
                       for name, text in source_files().items()},
        recipe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
