"""A/B retain fused-v2 compute; C uses its separately selected double buffer mask."""
import ablation4_sources
from common import replace_once


def adapter_source():
    # Keep the audited autograd, role dispatch and output ownership checks intact.
    return ablation4_sources.adapter_source()


def taco_source():
    source = ablation4_sources.taco_source()
    source = replace_once(source, 'Original/A/B retain the same output clone.',
                          'A/B retain the same fused-v2 output clone.')
    anchor = "        from flux.gemm_rs_taco import GemmRSTacoDoubleBuffered"
    return replace_once(source, anchor, '''        # Reject an oversized or cross-role selection before allocating CUDA buffers.
        mask = record['entry']['mask']
        world, tiles = dist.get_world_size(group), (m//128)*((n+127)//128)
        budget = definition(policy)['max_tiles_per_source']
        if (world != 8 or len(mask) != world or any(len(row) != tiles for row in mask) or
                any(type(value) is not int or value not in (0, 1) for row in mask for value in row) or
                any(sum(row) > budget or any(row[src*tiles//world:(src+1)*tiles//world])
                    for src, row in enumerate(mask))):
            raise ValueError('Ablation7 C requires at most 28/448 remote tiles per source')
'''+anchor)


def worker_source(source_only=False):
    source = ablation4_sources.worker_source(source_only)
    source = replace_once(source, "os.environ.get('H100_ABLATION4_EXECUTE')",
                          "os.environ.get('H100_ABLATION7_EXECUTE')")
    source = replace_once(source, 'GPU worker requires ablation4 --execute',
                          'GPU worker requires ablation7 --execute')
    if source_only:
        source = replace_once(source, 'Source-only scope worker is permanently non-executable',
                              'Source-only ablation7 worker is permanently non-executable')
    anchor = "PLAN = json.loads((HERE / 'config.json').read_text())"
    source = replace_once(source, anchor, anchor+'''
from routing import PROTOCOL, execution_timing
execution = execution_timing()
if (PLAN.get('protocol') != PROTOCOL or PLAN.get('execution_timing') != execution or
        any(type(PLAN.get(key)) is not int or PLAN[key] != value for key, value in execution.items())):
    raise ValueError('Ablation7 requires frozen 10-warmup / 50-timed-step execution timing')
'''.rstrip())
    return replace_once(source, "report['protocol'] = PLAN['protocol']",
                         "report['execution_timing'] = execution\nreport['protocol'] = PLAN['protocol']")
