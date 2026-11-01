"""Old A/C with original Flux ordering, no arrival table and all-remote quantization.

This deliberately retains the old staged transport, decoder and ownership
differences. It does not apply the separate OP1 direct-wire experiment.
"""
import hashlib
from pathlib import Path
from rtx5090_loader import HERE, REPO
import nccl_transport

SWIZZLE = 'src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp'
MARKER = 'FluxBaselineOriginalOrder'


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Flux baseline source anchor changed: '+old[:180])
    return text.replace(old, new, 1)


def identity():
    return dict(protocol='rtx5090-flux-quant-native-v1', old_arms=['A', 'C'],
        interleaved=False, arrival=False, selective=False, quantization='all_remote',
        ordering='original-Flux-StreamKRankOffset', op1_direct_wire=False,
        sources={str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in (Path(__file__), HERE/'nccl_transport.py', REPO/SWIZZLE)})


def frontend_source(arm):
    if arm not in ('a', 'c'):
        raise ValueError('Expected old A/C')
    source = nccl_transport.frontend_source(arm)
    source = replace(source,
        "require((self.mask is None) == (NATIVE_ARM in ('a', 'r')), 'Native arm/selection family mismatch')",
        "require(self.mask is None, 'Flux quant baseline forbids selective masks')")
    # Null selected means ALL remote contributions, never no quantization.
    return source


def overrides(root, arm):
    root = Path(root)
    if arm not in ('a', 'c'):
        raise ValueError('Expected old A/C')
    native = (root/'src/gemm_rs/local_transport.cu').read_text()
    if arm == 'c':
        native = replace(native, '#define RTX5090_LOCAL_SELECTIVE 1\n',
                         '#define RTX5090_LOCAL_SELECTIVE 0\n')
    mapping = (root/'check_mapping.cu').read_text()
    mapping = replace(mapping, '#include "phase6_host.hpp"\n', '')
    mapping = replace(mapping,
        '      int expected=phase6_host_lookup(tm,tn,k/tp,rank,i);\n'
        '      if(expected>=0 && x*tn+y!=expected) return 69;',
        '      if(x!=(bx+tm/tp*rank)%tm || y!=by) return 69;')
    return {SWIZZLE: (REPO/SWIZZLE).read_text(),
        'phase6_host.hpp': '#pragma once\n// No arrival/interleaved lookup table in this experiment.\n',
        'check_mapping.cu': mapping, 'src/gemm_rs/local_transport.cu': native,
        'python/flux/gemm_rs_taco.py': frontend_source(arm)}


def registration_source(original):
    import re
    names = re.findall(r'struct (flux_\w+): public GemmKernel_0 \{\};', original)
    if len(names) != 1 or original.count(names[0]) != 2 or MARKER in original:
        raise ValueError('Expected one untouched native registration')
    return original.replace(names[0], names[0]+'_'+MARKER)


def original_registration(transformed):
    original = transformed.replace('_'+MARKER, '')
    if registration_source(original) != transformed:
        raise ValueError('Unregistered Flux original-order kernel')
    return original
