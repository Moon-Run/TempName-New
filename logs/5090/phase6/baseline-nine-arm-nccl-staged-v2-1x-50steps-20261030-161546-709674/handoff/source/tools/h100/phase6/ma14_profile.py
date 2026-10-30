"""MA14 preserves the complete MA11 compute path in both arms.

Publication kernel identity is verified by ma14_bench/ma14_fullstep. These
unchanged MA11 checks verify attention, backward, cache and optimizer paths.
"""
import ma11_profile

ARMS = ('F4-cache-zcount-csoftmax', 'F4-cache-zcount-csoftmax-epoch')
SUBMISSION_FIELDS = ('cpu_marker_to_gemm_launch_us', 'gemm_to_barrier_api_gap_us')
SUBMISSION_LIMITATION = (
    'CPU intervals use correlated launch API timestamps inside the existing projection scope; '
    'they include profiler overhead and other host submission work. They neither isolate guard '
    'cost nor measure GPU execution time or unprofiled training savings.')


def submission_intervals(marker, path, events):
    """Attribute host submission gaps without adding measured-path markers."""
    require = ma11_profile.require
    require(len(path) == 3, 'Submission attribution requires GEMM/barrier/decoder')
    launches = []
    for kernel in path[:2]:
        correlation = kernel.get('args', {}).get('correlation')
        require(correlation is not None, 'Kernel lacks launch correlation')
        matches = [event for event in events
                   if event.get('cat') in ('cuda_runtime', 'cuda_driver') and
                   event.get('args', {}).get('correlation') == correlation and
                   event.get('pid') == marker.get('pid') and
                   event.get('tid') == marker.get('tid') and
                   marker['ts'] <= event['ts'] and
                   ma11_profile.end(event) <= ma11_profile.end(marker) + .01 and
                   'launch' in event.get('name', '').lower()]
        # Kineto can record a runtime launch and its nested driver API. Prefer
        # the outer runtime call so the gap excludes all GEMM launch work.
        runtime = [event for event in matches if event['cat'] == 'cuda_runtime']
        matches = runtime or matches
        require(len(matches) == 1, 'Missing or ambiguous correlated launch API')
        launches.append(matches[0])
    gemm, barrier = launches
    require(ma11_profile.end(gemm) <= barrier['ts'] + .01,
            'Publication API precedes completion of GEMM launch API')
    return dict(zip(SUBMISSION_FIELDS,
                    (gemm['ts'] - marker['ts'],
                     max(0.0, barrier['ts'] - ma11_profile.end(gemm)))))


def trace_composed_records(events, arm, rank=None):
    ma11_profile.require(arm in ARMS, 'Unknown MA14 arm')
    result = ma11_profile.trace_composed_records(events, ma11_profile.ARMS[1], rank)
    result['attention_softmax']['source_arm'] = ma11_profile.ARMS[1]
    return result


def validate_projection_work(work, arm):
    """Epoch counters initialize once; retain only CUTLASS's 128-byte clear."""
    require = ma11_profile.require
    require(arm in ARMS, 'Unknown MA14 arm')
    candidate = arm == ARMS[1]
    clears = [e for e in work if e.get('cat') == 'gpu_memset']
    require(len(clears) == 1 and clears[0].get('args', {}).get('bytes') == 128,
            'Original workspace clear changed or epoch counters cleared per call')
    token = 'taco_ma14_epoch_barrier_kernel' if candidate else 'CudaIpcBarrierAllKernel'
    barriers = [e for e in work if e.get('cat') == 'kernel' and token in e.get('name', '')]
    require(len(barriers) == 1, 'Missing unique publication barrier')
    if candidate:
        require(barriers[0]['args'].get('grid') == [1, 1, 1] and
                barriers[0]['args'].get('block') == [32, 1, 1], 'Epoch barrier launch changed')
    kernels = sorted((e for e in work if e.get('cat') == 'kernel'), key=lambda e:e['ts'])
    require(len(kernels) == 3 and kernels[1] is barriers[0], 'Unexpected publication helper kernels')
    require(clears[0]['args'].get('stream') == kernels[0]['args'].get('stream') and
            ma11_profile.end(clears[0]) <= kernels[0]['ts'] + .01,
            'CUTLASS workspace initialization moved after GEMM')
