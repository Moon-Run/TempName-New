"""Pure producer-order candidates using the unchanged ablation-7 A implementation.

Within each destination, pair its two M rows in small N-column spans.  Every
destination slot and physical quantization decision remains unchanged.  Smaller
spans shorten B-column reuse distance while increasing the live A-row set; only
same-codec GPU timing can resolve that tradeoff.
"""
import re


METHOD = 'destination-preserving-row-pair-v1'
SPANS = (1, 4, 8, 16, 32)
CANDIDATE_SPANS = SPANS[:-1]
IDENTITY_SPAN = 32
ROLES = ('attn', 'mlp')
ENV = {'attn': 'H100_OP2_ATTN_SPAN', 'mlp': 'H100_OP2_MLP_SPAN'}
SWIZZLE = 'src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp'
HOST = 'phase6_host.hpp'
ALLOWED_SOURCE_CHANGES = (SWIZZLE, HOST)
WORLD = 8
TILES = 512
WINDOW = 64


def validate_span(span):
    if type(span) is not int or span not in SPANS:
        raise ValueError('Pair span must be one of 1, 4, 8, 16, 32')
    return span


def validate_spans(spans):
    if not isinstance(spans, dict) or set(spans) != set(ROLES):
        raise ValueError('Pair spans require exactly attn and mlp')
    return {role: validate_span(spans[role]) for role in ROLES}


def indices(span):
    """Return the 512-entry new logical slot -> original A logical slot map."""
    span = validate_span(span)
    result = []
    for slot in range(TILES):
        within = slot // WORLD
        paired = ((within // (2 * span)) * span + within % span +
                  32 * ((within // span) % 2))
        result.append(paired * WORLD + slot % WORLD)
    return result


def role_mapping(span):
    order = indices(span)
    base = [[((slot % WORLD + rank) % WORLD) * WINDOW + slot // WORLD
             for slot in range(TILES)] for rank in range(WORLD)]
    return dict(maps=[list(order) for _ in range(WORLD)],
                coords=[[row[slot] for slot in order] for row in base],
                base_coords=base)


def mapping(spans):
    return {role: role_mapping(span) for role, span in validate_spans(spans).items()}


def diagnostics(span):
    order = indices(span)
    changed = sum(slot != old for slot, old in enumerate(order))
    return dict(method=METHOD, span=span, candidate_kind='producer_locality',
                changed_per_rank=[changed] * WORLD, identity_fallback=(changed == 0),
                destination_slots_preserved=all(slot % WORLD == old % WORLD
                                                for slot, old in enumerate(order)),
                destination_window=WINDOW, permutation=sorted(order) == list(range(TILES)),
                b_row_pair_slot_gap=WORLD * span, baseline_b_row_pair_slot_gap=256,
                gpu_validated=False, timing_prediction=False)


def _replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Expected exactly one scheduler source anchor: ' + old[:100])
    return source.replace(old, new, 1)


def _array(source, name):
    matches = re.findall(r'\b' + re.escape(name) + r'\[\]\s*=\s*\{([^}]*)\};', source)
    if len(matches) != 1:
        raise ValueError('Missing/duplicate baseline A table: ' + name)
    try:
        return [int(value.strip()) for value in matches[0].split(',')]
    except ValueError as error:
        raise ValueError('Invalid baseline A integer table: ' + name) from error


def _validate_base_tables(swizzle, host):
    expected = role_mapping(IDENTITY_SPAN)
    flat_maps = [value for row in expected['maps'] for value in row]
    flat_coords = [value for row in expected['coords'] for value in row]
    for role in ROLES:
        if _array(swizzle, 'phase6_device_' + role + '_data') != flat_maps:
            raise ValueError('Scheduler source is not the identity-order A baseline: ' + role)
        if _array(host, 'phase6_host_' + role + '_data') != flat_coords:
            raise ValueError('Host source is not the interleaved A baseline: ' + role)


def helper_source():
    # Both transformed headers may appear in one translation unit.  Keep the
    # environment parser host-only and the arithmetic usable by host and device.
    return '''
#ifndef FLUX_H100_OP2_PAIR_ORDER_HELPERS
#define FLUX_H100_OP2_PAIR_ORDER_HELPERS
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#ifdef __CUDACC__
#define PHASE6_OP2_HOST __host__
#define PHASE6_OP2_HOST_DEVICE __host__ __device__
#else
#define PHASE6_OP2_HOST
#define PHASE6_OP2_HOST_DEVICE
#endif
PHASE6_OP2_HOST inline int phase6_op2_pair_span(int k) {
  const char *name = k == 512 ? "H100_OP2_ATTN_SPAN" :
                     k == 2048 ? "H100_OP2_MLP_SPAN" : nullptr;
  if (!name) throw std::invalid_argument("Unsupported H100 op2 pair-order K");
  const char *value = std::getenv(name);
  if (value) {
    if (std::strcmp(value, "1") == 0) return 1;
    if (std::strcmp(value, "4") == 0) return 4;
    if (std::strcmp(value, "8") == 0) return 8;
    if (std::strcmp(value, "16") == 0) return 16;
    if (std::strcmp(value, "32") == 0) return 32;
  }
  throw std::invalid_argument(name);
}
PHASE6_OP2_HOST_DEVICE inline int phase6_op2_pair_slot(int slot, int span) {
  const int within = slot / 8;
  // Allowed spans are powers of two. Move the span bit to the row bit
  // without runtime integer division in every GEMM thread.
  const int paired = ((within & ~(2 * span - 1)) >> 1) +
                     (within & (span - 1)) + (within & span ? 32 : 0);
  return paired * 8 + slot % 8;
}
#undef PHASE6_OP2_HOST
#undef PHASE6_OP2_HOST_DEVICE
#endif
'''


def sources(base_swizzle, base_host):
    """Transform exactly two A scheduler files, keeping all A tables verbatim.

    Spans are mandatory for the supported exact shapes and captured in the host
    swizzle constructor.  The kernel performs arithmetic only; it never reads
    environment variables.  Non-target shapes retain the original A behavior.
    """
    _validate_base_tables(base_swizzle, base_host)
    helper = helper_source()
    swizzle = _replace(base_swizzle, '#pragma once', '#pragma once\n' + helper)
    swizzle = _replace(swizzle, '  bool scope_shape = false;',
                       '  bool scope_shape = false;\n  int scope_pair_span = 32;')
    swizzle = _replace(
        swizzle, '                    (scope_k==512 || scope_k==2048);',
        '                    (scope_k==512 || scope_k==2048);\n'
        '      if (scope_shape) scope_pair_span = phase6_op2_pair_span(scope_k);')
    swizzle = _replace(
        swizzle,
        '    if(scope_shape && tile_idx>=0 && tile_idx<512)\n'
        '      tile_idx=phase6_device_lookup(16,32,scope_k,local_rank,tile_idx);',
        '    if(scope_shape && tile_idx>=0 && tile_idx<512) {\n'
        '      tile_idx=phase6_op2_pair_slot(tile_idx,scope_pair_span);\n'
        '      tile_idx=phase6_device_lookup(16,32,scope_k,local_rank,tile_idx);\n'
        '    }')
    host = _replace(base_host, '#pragma once', '#pragma once\n' + helper)
    host = _replace(host, 'inline int phase6_host_lookup(',
                    'inline int phase6_host_lookup_base(')
    host += '''
inline int phase6_host_lookup(int tm,int tn,int k,int rank,int i) {
  if(tm!=16 || tn!=32 || (k!=512 && k!=2048) ||
     rank<0 || rank>=8 || i<0 || i>=512) return -1;
  const int span=phase6_op2_pair_span(k);
  return phase6_host_lookup_base(tm,tn,k,rank,phase6_op2_pair_slot(i,span));
}
'''
    return {SWIZZLE: swizzle, HOST: host}
