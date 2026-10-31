"""Migrate only legacy startup pinning when preparing a new worker snapshot."""
import ast
import hashlib
from pathlib import Path


def _os_call(node, name):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == 'os'
            and node.func.attr == name)


def _environment_int(node, key):
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == 'int' and len(node.args) == 1 and not node.keywords):
        return False
    value = node.args[0]
    return (isinstance(value, ast.Subscript) and isinstance(value.value, ast.Attribute)
            and isinstance(value.value.value, ast.Name) and value.value.value.id == 'os'
            and value.value.attr == 'environ' and isinstance(value.slice, ast.Constant)
            and value.slice.value == key)


def _legacy_rank_index(index, cpu_name):
    if not (isinstance(index, ast.BinOp) and isinstance(index.op, ast.FloorDiv)
            and isinstance(index.left, ast.BinOp) and isinstance(index.left.op, ast.Mult)):
        return False
    rank, count, divisor = index.left.left, index.left.right, index.right
    rank_ok = (isinstance(rank, ast.Name) and rank.id in {'rank', 'local_rank', 'local', '_rank'}
               or _environment_int(rank, 'LOCAL_RANK'))
    count_ok = (isinstance(count, ast.Call) and isinstance(count.func, ast.Name)
                and count.func.id == 'len' and len(count.args) == 1 and not count.keywords
                and isinstance(count.args[0], ast.Name) and count.args[0].id == cpu_name)
    divisor_ok = (isinstance(divisor, ast.Name) and divisor.id in {'WORLD', 'world'}
                  or isinstance(divisor, ast.Constant) and type(divisor.value) is int
                  and divisor.value in {2, 4, 8}
                  or _environment_int(divisor, 'LOCAL_WORLD_SIZE'))
    return rank_ok and count_ok and divisor_ok


def inherited_worker_source(source):
    """Remove recognized pre-Torch singleton pinning; reject unfamiliar setters.

    Historical inputs remain immutable. No training or timing statements are
    removed, and a changed legacy startup shape fails preparation explicitly.
    """
    tree = ast.parse(source)
    setters = [node for node in ast.walk(tree) if _os_call(node, 'sched_setaffinity')]
    removed = []
    for call in setters:
        statements = [node for node in tree.body if isinstance(node, ast.Expr) and node.value is call]
        if (len(statements) != 1 or len(call.args) != 2 or call.keywords
                or not isinstance(call.args[0], ast.Constant) or call.args[0].value != 0
                or not isinstance(call.args[1], ast.Set) or len(call.args[1].elts) != 1):
            raise ValueError('Unrecognized legacy CPU binding; prepare from a current worker template')
        setter = statements[0]
        imports = [node.lineno for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
                   and (getattr(node, 'module', '') or '').startswith('torch')]
        imports += [node.lineno for node in tree.body if isinstance(node, ast.Import)
                    and any(alias.name.startswith('torch') for alias in node.names)]
        if imports and setter.lineno >= min(imports):
            raise ValueError('Legacy CPU binding must precede Torch initialization')
        selection = call.args[1].elts[0]
        assignment = None
        if isinstance(selection, ast.Name):
            candidates = [node for node in tree.body if isinstance(node, ast.Assign)
                          and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                          and node.targets[0].id == selection.id and node.lineno < setter.lineno]
            if len(candidates) == 1:
                assignment = candidates[0]
                selection = assignment.value
        if not isinstance(selection, ast.Subscript) or not isinstance(selection.value, ast.Name):
            raise ValueError('Legacy CPU binding is not an indexed inherited CPU selection')
        cpu_name = selection.value.id
        if not _legacy_rank_index(selection.slice, cpu_name):
            raise ValueError('Unrecognized legacy rank-to-CPU formula')
        candidates = [node for node in tree.body if isinstance(node, ast.Assign)
                      and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                      and node.targets[0].id == cpu_name and node.lineno <= setter.lineno]
        if len(candidates) != 1:
            raise ValueError('Legacy CPU selection source is ambiguous')
        getter = candidates[0]
        value = getter.value
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and value.func.id == 'sorted' and len(value.args) == 1 and not value.keywords
                and _os_call(value.args[0], 'sched_getaffinity')
                and len(value.args[0].args) == 1 and not value.args[0].keywords
                and ast.dump(value.args[0].args[0]) == ast.dump(ast.Constant(value=0))):
            raise ValueError('Legacy CPU selection does not use the inherited mask')
        removed.extend([getter, setter] + ([assignment] if assignment else []))
    if removed:
        line_numbers = {line for node in removed for line in range(node.lineno, node.end_lineno + 1)}
        if any(node not in removed and any(line in line_numbers for line in
               range(node.lineno, node.end_lineno + 1)) for node in tree.body):
            raise ValueError('Legacy CPU binding shares a line with unrelated startup code')
        removed_names = {node.targets[0].id for node in removed if isinstance(node, ast.Assign)}
        if any(isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
               and child.id in removed_names for node in tree.body if node not in removed
               for child in ast.walk(node)):
            raise ValueError('Legacy CPU selection variable is used outside its startup block')
        source = ''.join(line for number, line in enumerate(source.splitlines(keepends=True), 1)
                         if number not in line_numbers)
    tree = ast.parse(source)
    if 'cpu_affinity_policy' not in source:
        reports = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'report' for target in node.targets)]
        if len(reports) != 1:
            raise ValueError('Expected one report constructor in the worker snapshot')
        report = reports[0]
        lines = source.splitlines(keepends=True)
        if not lines[report.end_lineno - 1].endswith('\n'):
            lines[report.end_lineno - 1] += '\n'
        lines.insert(report.end_lineno, ' ' * report.col_offset
                     + "report.update(cpu_affinity_policy='inherit', cpu_affinity=sorted(os.sched_getaffinity(0)))\n")
        source = ''.join(lines)
    ast.parse(source)
    return source


def prepare_worker_affinity(path):
    """Rewrite a newly copied snapshot and return both identities for its plan."""
    path = Path(path)
    before = path.read_text()
    after = inherited_worker_source(before)
    path.write_text(after)
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    return dict(policy='inherit', source_sha256=digest(before), worker_sha256=digest(after),
                changed=before != after, transformation='remove-legacy-startup-singleton-v1')
