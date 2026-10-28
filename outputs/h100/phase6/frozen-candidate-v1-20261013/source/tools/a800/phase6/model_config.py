"""Read a model profile without importing Torch or starting experiments."""
import json
from pathlib import Path

DEFAULT_MODEL = Path(__file__).with_name('model-6.7b.json')


def read_model(path, tp):
    model = json.loads(Path(path).read_text())
    required = {'layers', 'hidden', 'ffn', 'heads', 'sequence', 'micro_batch',
                'global_batch', 'seed', 'token_seed', 'vocab'}
    if set(model) != required:
        raise ValueError(f'Model profile must contain exactly {sorted(required)}')
    if any(type(v) is not int or v < (0 if k in ('seed', 'token_seed') else 1)
           for k, v in model.items()):
        raise ValueError('Invalid model integer')
    if tp not in (4, 8):
        raise ValueError('Single-node model profile requires TP4 or TP8')
    m, n, k = mlp_shape(model)
    if (model['hidden'] % model['heads'] or model['heads'] % tp or
            model['sequence'] % tp or model['global_batch'] % model['micro_batch']):
        raise ValueError('Invalid attention, sequence or batch divisibility')
    if m % (128*tp) or n % (32*tp) or n % 128 or k % (32*tp):
        raise ValueError('Unsupported GEMM-RS shape')
    tiles = (m//128)*(n//128)
    if tiles > 2048 or (tiles//tp) % 4:
        raise ValueError('Model exceeds the calibrated sampler tile domain')
    return model


def mlp_shape(model):
    return [model['sequence']*model['micro_batch'], model['hidden'], model['ffn']]


def padded_vocab(model, tp):
    # NullTokenizer adds an EOD token; Megatron's default padding is 128*TP.
    multiple = 128*tp
    return ((model['vocab']+1+multiple-1)//multiple)*multiple
