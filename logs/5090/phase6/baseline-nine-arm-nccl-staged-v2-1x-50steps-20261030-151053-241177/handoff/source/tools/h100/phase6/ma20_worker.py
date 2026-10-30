"""Bounded MA11 worker overlay; mask readback happens after measured work."""
import hashlib
from pathlib import Path


def device_masks(adapter):
    """Read the actual uint8 device buffers consumed by both native slots."""
    result = {}
    for name, op in adapter.scope_ops.items():
        slots = []
        for index, slot in enumerate(op.codec.slots):
            if not slot.selected.is_cuda:
                raise ValueError('MA20 selection must be the actual CUDA mask buffer')
            value = slot.selected.detach().cpu().contiguous()
            if tuple(value.shape) != (8, 512) or str(value.dtype) != 'torch.uint8':
                raise ValueError('MA20 requires physical [source, tile] uint8 masks')
            rows = value.tolist()
            if any(x not in (0, 1) for row in rows for x in row):
                raise ValueError('Invalid device mask')
            if any(rows[s][t] for s in range(8) for t in range(s * 64, (s + 1) * 64)):
                raise ValueError('Local contributions must remain BF16')
            digest = hashlib.sha256(bytes(x for row in rows for x in row)).hexdigest()
            if digest != slot.selection_sha256:
                raise ValueError('The actual device mask differs from the frontend mask')
            counts = list(map(sum, rows))
            slots.append(dict(slot=index, mask_sha256=digest,
                              selected_per_source=counts,
                              wire_bytes_per_source=[14680064 - 15360 * q for q in counts]))
        result[name] = slots
    return result


def worker_source():
    import ma11_fullstep
    from common import replace_once
    source = ma11_fullstep.worker_source()
    # Leave all original training functions and the measured/profile loop intact.
    # The internal MA11 mechanism arm is the complete MA11 candidate in BOTH arms.
    anchor = "report['ma11_activation_setup'] = ma11_activation_setup"
    source = replace_once(source, anchor,
        "from ma20_worker import device_masks\n"
        "report['ma20_arm'] = PLAN['ma20_arm']\n"
        "report['ma20_python_version'] = sys.version\n"
        "report['ma20_bundle_sha256'] = PLAN['ma20_bundle_sha256']\n"
        "report['ma20_device_masks'] = device_masks(adapter)\n" + anchor)
    return source


def validate_sources(source):
    """Original training functions and timing AST must remain registered MA11."""
    import ast
    import ma11_fullstep
    tree = ast.parse(source)
    functions = {n.name: hashlib.sha256(ast.dump(n).encode()).hexdigest()
                 for n in tree.body if isinstance(n, ast.FunctionDef)}
    if functions != ma11_fullstep.BASELINE_FUNCTIONS:
        raise ValueError('MA20 changed the original MA11 training functions')
    loops = [n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "MODE == 'profile'"]
    if len(loops) != 1 or hashlib.sha256(ast.dump(loops[0]).encode()).hexdigest() != ma11_fullstep.BASELINE_TIMING_SHA256:
        raise ValueError('MA20 changed the original MA11 timing loop')
    assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                   and ast.unparse(n).startswith("report['ma20_device_masks'] =")]
    if len(assignments) != 1 or tree.body.index(assignments[0]) <= tree.body.index(loops[0]):
        raise ValueError('Device mask readback must occur after the measured/profile loop')
    return True
