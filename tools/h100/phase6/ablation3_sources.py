"""Use the audited scope adapter; select an explicit codec mode per policy."""
import scope_sources
from common import replace_once


def taco_source():
    source = scope_sources.taco_source()
    source = replace_once(source, 'from routing import is_native, selection_entry',
                          'from routing import is_native, selection_entry, definition')
    return replace_once(source, "quantization_mode='targeted'", "quantization_mode=definition(policy)['quantization']")


def worker_source(source_only=False):
    source = scope_sources.worker_source(source_only)
    source = replace_once(source, "os.environ.get('H100_SCOPE_EXECUTE')", "os.environ.get('H100_ABLATION3_EXECUTE')")
    return replace_once(source, 'GPU worker requires scope --execute', 'GPU worker requires ablation3 --execute')
