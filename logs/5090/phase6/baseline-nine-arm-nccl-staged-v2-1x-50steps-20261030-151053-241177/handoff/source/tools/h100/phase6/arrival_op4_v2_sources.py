"""Reduction-closure factorial from pinned fused-v2, without lifecycle changes.

SP/CP: dispersed/closed masks with per-source inverses.
SM/CM: the same masks with one inverse per nonempty receiver group.
All four keep the same codec, producer schedule, workspace and two barriers.
"""
from pathlib import Path

from common import HERE, replace_once, require

PROTOCOL = 'h100-reduction-closure-v1'
VARIANTS = ('SP', 'CP', 'SM', 'CM')
HEADER = 'src/gemm_rs/taco_runtime.h'
RUNTIME = 'src/gemm_rs/taco_runtime.cu'
EPILOGUE = 'src/gemm_rs/epilogue_evt.hpp'
WRAPPER = 'src/gemm_rs/ths_op/gemm_reduce_scatter.cc'
FRONTEND = 'python/flux/gemm_rs_taco.py'
DECODER = 'src/gemm_rs/arrival_op4_v2_decode.cuh'


def header(source, variant):
    require(variant in VARIANTS, 'Expected SP, CP, SM or CM')
    text = replace_once(source, '  int n = 0;\n};',
                        '  int n = 0;\n  int closure_phase = 0;\n};')
    text = replace_once(text,
        'extern "C" int taco_configure(void** peers, int rank, int world, int m, int n);',
        'extern "C" int taco_configure(void** peers, int rank, int world, int m, int n, int phase);')
    return text + f'''
// GF(8), irreducible polynomial x^3+x+1. Shared by sender and receiver.
#define FLUX_CLOSURE_CLOSED {int(variant[0] == 'C')}
#define FLUX_CLOSURE_MERGED {int(variant[1] == 'M')}
__host__ __device__ inline int taco_closure_mul2(int value) {{
  return ((value << 1) ^ ((value & 4) ? 11 : 0)) & 7;
}}
__host__ __device__ inline bool taco_closure_selected(
    const FluxTacoConfig& c, int source, int destination, int physical_tile) {{
  if (source == destination) return false;
  const int residue = ((physical_tile % 64) + c.closure_phase) % 8;
#if FLUX_CLOSURE_CLOSED
  return residue == destination;
#else
  return residue == (taco_closure_mul2(source) ^
                     taco_closure_mul2(destination) ^ destination);
#endif
}}
extern "C" const char* taco_closure_protocol();
extern "C" const char* taco_closure_arm();
'''


def epilogue(source):
    return replace_once(source,
        '        if (params.taco.enabled && dst_rank != params.rank) {',
        '''        if (params.taco.enabled && taco_closure_selected(
                params.taco, params.rank, dst_rank, tile_idx)) {''')


def runtime(source, variant):
    kernel = 'taco_closure_decode_merged' if variant[1] == 'M' else 'taco_closure_decode_per_source'
    text = replace_once(source, '#include "gemm_rs/taco_codec.cuh"',
        '#include "gemm_rs/taco_codec.cuh"\n#include "gemm_rs/arrival_op4_v2_decode.cuh"')
    text = replace_once(text,
        'extern "C" int taco_configure(void** peers, int rank, int world, int m, int n) {',
        'extern "C" int taco_configure(void** peers, int rank, int world, int m, int n, int phase) {')
    text = replace_once(text,
        '''  if (active.enabled || !peers || (world != 2 && world != 4 && world != 8) ||
      rank < 0 || rank >= world || m <= 0 || m % (128 * world) || n <= 0 || n % 8)''',
        '''  if (active.enabled || !peers || world != 8 || m != 2048 || n != 4096 ||
      rank < 0 || rank >= world || phase < 0 || phase > 7 || taco_placement() != 1)''')
    text = replace_once(text, '  c.rank = rank; c.world = world; c.m = m; c.n = n;',
                        '  c.rank = rank; c.world = world; c.m = m; c.n = n; c.closure_phase = phase;')
    start = text.index('__global__ void taco_decode_ring_kernel(')
    end = text.index('extern "C" int taco_decode_reduce(', start)
    text = text[:start] + f'''__global__ void {kernel}(FluxTacoConfig c,
    const __nv_bfloat16* local, __nv_bfloat16* output) {{
  const int64_t group = int64_t(blockIdx.x) * 4 + threadIdx.x / 32;
  taco_closure_decode_warp(c, local, output, group);
}}

''' + text[end:]
    text = replace_once(text,
        '''  if (!c.enabled || !local || !output) return int(cudaErrorInvalidValue);
  taco_decode_ring_kernel<<<taco_groups(c), 128, 0, stream>>>(''',
        f'''  if (!c.enabled || !local || !output || c.world != 8 || c.m != 2048 ||
      c.n != 4096 || c.closure_phase < 0 || c.closure_phase > 7)
    return int(cudaErrorInvalidValue);
  {kernel}<<<taco_groups(c) / 4, 128, 0, stream>>>(''')
    return text + f'''
extern "C" const char* taco_closure_protocol() {{ return "{PROTOCOL}"; }}
extern "C" const char* taco_closure_arm() {{ return "{variant}"; }}
'''


def frontend(source, variant):
    text = replace_once(source, 'import ctypes\n', 'import ctypes\nimport hashlib\n')
    text = replace_once(text,
        "        if placement not in (None, 'fused', 'separate'):",
        "        if placement not in (None, 'fused'):")
    text = replace_once(text, '        self._lock = threading.Lock()', f'''        self._lock = threading.Lock()
        if (self.world, m, n) != (8, 2048, 4096) or k_local not in (512, 2048):
            raise ValueError('Reduction closure requires exact TP8 attention/MLP shapes')
        if os.environ.get('H100_CLOSURE_PROTOCOL') != '{PROTOCOL}':
            raise RuntimeError('Reduction closure requires its independent protocol')
        if os.environ.get('H100_CLOSURE_ARM') != '{variant}':
            raise RuntimeError('Reduction closure arm does not match this isolated library')
        self.closure_phase = int(os.environ.get('H100_CLOSURE_PHASE', '0'))
        if self.closure_phase not in range(8):
            raise ValueError('H100_CLOSURE_PHASE must be in [0, 7]')
        self.closure_arm = '{variant}'
        def mul2(value):
            return ((value << 1) ^ (11 if value & 4 else 0)) & 7
        def selected(source, tile):
            destination = tile // 64
            target = destination if self.closure_arm[0] == 'C' else (
                mul2(source) ^ mul2(destination) ^ destination)
            return int(source != destination and
                       (tile % 64 + self.closure_phase) % 8 == target)
        mask = [[selected(source, tile) for tile in range(512)] for source in range(8)]
        self.selection_cpu = torch.tensor(mask, dtype=torch.uint8)
        self.selection_sha256 = hashlib.sha256(bytes(value for row in mask for value in row)).hexdigest()''')
    text = replace_once(text,
        '            self.lib.taco_configure.argtypes = [ctypes.POINTER(ctypes.c_void_p)] + [ctypes.c_int]*4',
        '            self.lib.taco_configure.argtypes = [ctypes.POINTER(ctypes.c_void_p)] + [ctypes.c_int]*5')
    text = replace_once(text, '            self.lib.taco_reset.argtypes = []', f'''            self.lib.taco_closure_protocol.argtypes = []
            self.lib.taco_closure_protocol.restype = ctypes.c_char_p
            self.lib.taco_closure_arm.argtypes = []
            self.lib.taco_closure_arm.restype = ctypes.c_char_p
            if (self.lib.taco_closure_protocol() != b'{PROTOCOL}' or
                    self.lib.taco_closure_arm() != b'{variant}'):
                raise RuntimeError('Wrong reduction-closure library protocol or arm')
            self.lib.taco_reset.argtypes = []''')
    text = replace_once(text,
        'self.lib.taco_configure(self.pointers, self.rank, self.world, m, n)',
        'self.lib.taco_configure(self.pointers, self.rank, self.world, m, n, self.closure_phase)')
    text = replace_once(text,
        'self.lib.taco_configure(self.pointers, self.rank, self.world, self.m, self.n)',
        'self.lib.taco_configure(self.pointers, self.rank, self.world, self.m, self.n, self.closure_phase)')
    text = replace_once(text,
        '        return (self.world-1)*self.packet_bytes_per_source',
        '''        # 1/8 of every remote link's 64 physical tiles uses 136 B/group;
        # the rest uses BF16 (256 B/group). Allocated packet capacity is unchanged.
        return 7 * (8 * 128 * 136 + 56 * 128 * 256)''')
    anchor = '    @property\n    def wire_bytes_per_rank(self):'
    text = replace_once(text, anchor, f'''    def closure_state(self):
        return dict(protocol='{PROTOCOL}', arm=self.closure_arm,
                    phase=self.closure_phase,
                    mask_sha256=self.selection_sha256,
                    mask_kind={'"closed"' if variant[0] == 'C' else '"dispersed"'},
                    reduction={'"merged"' if variant[1] == 'M' else '"per_source"'},
                    inverse_h128_per_rank={1024 if variant == 'CM' else 7168},
                    selected_per_source=56, selected_per_link=8,
                    wire_bytes_per_rank=self.wire_bytes_per_rank)

''' + anchor)
    return text


def sources(root, variant):
    require(variant in VARIANTS, 'Expected SP, CP, SM or CM')
    root = Path(root)
    return {HEADER: header((root / HEADER).read_text(), variant),
            RUNTIME: runtime((root / RUNTIME).read_text(), variant),
            EPILOGUE: epilogue((root / EPILOGUE).read_text()),
            FRONTEND: frontend((root / FRONTEND).read_text(), variant),
            DECODER: (HERE / 'arrival_op4_v2_decode.cuh').read_text()}
