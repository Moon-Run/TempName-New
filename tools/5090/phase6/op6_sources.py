"""OP4/OP6 worker overlays for local graphs and output-filesystem trace export.

Both variants use identical generated training code and original OP3 native
artifacts. The explicit configuration selects candidate-only graph setup.
"""
from pathlib import Path
import shutil

from rtx5090_loader import HERE
import op3_protocol as p
import op3_sources as original

PROTOCOL = 'rtx5090-phase6-op6-local-attention-graph-paired-v1'
VARIANTS = ('op4', 'op6')
ADAPTER = HERE/'op6_adapter.py'
PROFILE_IO = HERE/'op6_profile_io.py'
LEGACY_PROFILE_EXPORT = """    with tempfile.TemporaryDirectory(prefix='flux-e2e-scale-trace-',dir='/tmp') as temp:
        path=Path(temp)/'trace.json'
        profiler.export_chrome_trace(str(path))
        with gzip.open(OUT/f'rank{rank}-profile.json.gz','wb') as target:target.write(path.read_bytes())
"""
PROFILE_EXPORT = """    from op6_profile_io import export_profile
    export_profile(profiler, OUT, rank)
"""
_worker_source = original.worker_source
_prepare_workers = original.prepare_workers


def implementation(variant):
    p.require(variant in VARIANTS, 'Unknown OP6 variant')
    return dict(protocol=PROTOCOL, variant=variant,
                attention_graphs=variant == 'op6',
                adapter_sha256=p.sha(ADAPTER),
                profile_exporter_sha256=p.sha(PROFILE_IO),
                scope='local_attention_only', communication_captured=False,
                optimizer_captured=False, original_op3_native=True)


def replace_once(source, old, new):
    p.require(source.count(old) == 1, 'OP6 worker source anchor changed: '+old)
    return source.replace(old, new)


def validate_worker(source):
    # Normalize only the registered post-capture file I/O replacement. The
    # original training functions, capture region and timed loop remain under
    # their original exact AST fingerprints; no global validator is relaxed.
    normalized = replace_once(source, PROFILE_EXPORT, LEGACY_PROFILE_EXPORT)
    original.base.validate_worker(normalized)
    original.previous.validate_installation(source)


def worker_source(arm):
    source = _worker_source(arm)
    setup = "ma11_softmax_tracker = ma11_adapter.install_softmax(raw, op3_native['ma11'], MODE)"
    source = replace_once(source, setup, setup+'''
assert PLAN['op6_protocol'] == 'rtx5090-phase6-op6-local-attention-graph-paired-v1'
assert PLAN['op6_variant'] in ('op4', 'op6')
op6_attention_graph_tracker = None
if PLAN['op6_variant'] == 'op6':
    import op6_adapter
    op6_attention_graph_tracker = op6_adapter.install_attention_graphs(raw, op3_native['ma11'], MODE)
''')
    report = "report['ma11_softmax_state'] = ma11_softmax_tracker.state()"
    source = replace_once(source, report, report+'''
report['op6_variant'] = PLAN['op6_variant']
report['op6_implementation'] = PLAN['op6_implementation']
report['op6_attention_graph_state'] = (op6_attention_graph_tracker.state()
    if op6_attention_graph_tracker is not None else {'installed': False})
''')
    source = replace_once(source, LEGACY_PROFILE_EXPORT, PROFILE_EXPORT)
    validate_worker(source)
    return source


def prepare_workers(build, out, plan, variant):
    p.require(plan.get('op6_variant') == variant and plan.get('op6_protocol') == PROTOCOL,
              'OP6 child plan lacks its variant/protocol')
    workers = _prepare_workers(build, out, plan)
    for worker in workers.values():
        scripts = Path(worker['root'])
        shutil.copy2(ADAPTER, scripts/ADAPTER.name)
        shutil.copy2(PROFILE_IO, scripts/PROFILE_IO.name)
        cfg = worker['config']
        cfg.update(op6_protocol=PROTOCOL, op6_variant=variant,
                   op6_implementation=implementation(variant))
        p.write(scripts/'config.json', cfg)
        worker['files'] = {path.name: p.sha(path) for path in scripts.iterdir() if path.is_file()}
    return workers


def install(variant):
    """Scope the overlay to one OP6 child execution or read-only replay."""
    p.require(variant in VARIANTS, 'Unknown OP6 variant')
    previous_worker, previous_prepare = original.worker_source, original.prepare_workers
    original.worker_source = worker_source
    original.prepare_workers = lambda build, out, plan: prepare_workers(build, out, plan, variant)

    def restore():
        original.worker_source, original.prepare_workers = previous_worker, previous_prepare
    return restore
