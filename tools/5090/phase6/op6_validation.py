"""Retain every OP3 check and require actual OP6 graph ownership/audits."""
from rtx5090_loader import HERE
import op3_validation as original
import op3_protocol as p
import op6_adapter as adapter
import op6_sources as sources


def validate_graph_rows(rows, job, plan, worker, variant):
    p.require(plan.get('op6_protocol') == sources.PROTOCOL and plan.get('op6_variant') == variant,
              'OP6 child protocol/variant changed')
    expected = sources.implementation(variant)
    cfg = worker['config']
    p.require(cfg.get('op6_protocol') == sources.PROTOCOL and cfg.get('op6_variant') == variant
              and cfg.get('op6_implementation') == expected
              and worker['files'].get('op6_adapter.py') == p.sha(HERE/'op6_adapter.py'),
              'OP6 worker graph implementation changed')
    mode = job['mode']
    warmup, count = (2, 3) if mode == 'smoke' else (10, 1 if mode == 'profile' else 50)
    for row in rows:
        p.require(row.get('op6_variant') == variant and row.get('op6_implementation') == expected,
                  'Missing or changed OP6 worker variant/implementation')
        state = row.get('op6_attention_graph_state')
        if variant == 'op4':
            p.require(state == {'installed': False}, 'OP4 control unexpectedly installed attention graphs')
        else:
            p.require(isinstance(state, dict), 'Missing OP6 graph execution state')
            adapter.validate_state(state, warmup+count, mode)
    return rows


def install(variant):
    p.require(variant in sources.VARIANTS, 'Unknown OP6 variant')
    previous = original.validate_rows

    def validate_rows(rows, job, plan, worker):
        # In smoke the original MA11 wrapper runs inside each eager graph oracle;
        # all its exact softmax checks/counts therefore remain applicable. The
        # capture-only quiet wrapper has a private counter. No original check is
        # synthesized or disabled to accommodate graph execution.
        rows = previous(rows, job, plan, worker)
        return validate_graph_rows(rows, job, plan, worker, variant)

    original.validate_rows = validate_rows

    def restore():
        original.validate_rows = previous
    return restore
