"""Keep the RTX5090 campaign separate from similarly named H100 modules."""
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
H100 = REPO / 'tools/h100/phase6'

# Legacy overlays import one another by basename, including baseline_protocol.
# Only the explicitly named rtx5090_* modules below load files in this directory.
if str(H100) not in sys.path:
    sys.path.insert(0, str(H100))
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))


def load(part):
    name = 'rtx5090_' + part
    if name in sys.modules:
        return sys.modules[name]
    path = HERE / ('baseline_' + part + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module
