"""MA7 keeps every native F4 source byte and its decoder geometry unchanged.

The candidate changes the backward schedule using an explicit deferred-wgrad queue. Its native
implementation is independently copied and verified against the registered F4
source, so a receiver experiment cannot accidentally enter the comparison.
"""
from common import require
import ma4_f_sources as f4

RUNTIME, HEADER, WRAPPER, DECODER = f4.RUNTIME, f4.HEADER, f4.WRAPPER, f4.DECODER
VARIANTS = ('F4', 'F4-wgrad')
BASELINE_TUNING = dict(decoder_threads=128, decoder_blocks=2048, decoder_rows=1)
ROLES = ('attn', 'mlp')
ROLE_K = dict(attn=512, mlp=2048)
DEFAULT_TUNING = {role: dict(BASELINE_TUNING) for role in ROLES}


def tuning_config(tuning=None):
    tuning = DEFAULT_TUNING if tuning is None else tuning
    require(isinstance(tuning, dict) and set(tuning) == set(ROLES),
            'MA7 tuning requires attn and mlp')
    for role in ROLES:
        item = tuning[role]
        require(isinstance(item, dict) and set(item) == set(BASELINE_TUNING)
                and all(type(value) is int for value in item.values()) and item == BASELINE_TUNING,
                'MA7 fixes the original F4 decoder geometry: ' + role)
    return {role: dict(BASELINE_TUNING) for role in ROLES}


def kernel_name(role, tuning=None, arm='F4'):
    tuning_config(tuning)
    require(role in ROLES and arm in VARIANTS, 'Unknown MA7 role or arm')
    return 'taco_ma4f_decode_consumer_kernel'
