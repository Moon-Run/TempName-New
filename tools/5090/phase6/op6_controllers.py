"""Freeze OP6 dependencies without binding the independent OP5 workflow.

Keep the original collector and SHA256 verifier. Only these exact OP5 paths
are omitted from NEW snapshots; saved manifests are never filtered on replay.
The offline analyze_op6_baseline utility imports op5_evidence, but is not used
by OP6 execution, build verification, worker validation or report replay.
"""

EXCLUDED = frozenset('tools/5090/phase6/'+name for name in (
    'op5.py', 'op5_evidence.py', 'op5_probe.py', 'run_op5.sh',
    'analyze_op5_baseline.py', 'test_op5.py', 'test_op5_packet_probe.py',
))


def collect(original):
    return {name: digest for name, digest in original().items() if name not in EXCLUDED}


def install(base):
    """Scope snapshot creation to an OP6 child; verification stays unchanged."""
    previous = base.controller_files
    replacement = lambda: collect(previous)
    base.controller_files = replacement

    def restore():
        if base.controller_files is replacement:
            base.controller_files = previous
    return restore
