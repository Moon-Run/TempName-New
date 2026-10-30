"""Two isolated dual-shape libraries: base order and arrival order; no GPU use."""
import copy
from pathlib import Path
import shutil

import scope_build
import ablation3_plan
from ablation_plan import FREEZE
from ablation3_protocol import PROTOCOL, digest
from common import HERE, read, require, sha, verify_files, write

ORDERS = ('base', 'arrival')


def order_plans(parent, order):
    require(order in ORDERS, 'Unknown ablation3 order')
    plans = copy.deepcopy(parent)
    if order == 'base':
        for p in plans['roles'].values():
            e = p['entry']
            e['maps'] = [list(range(512)) for _ in range(8)]
            e['coords'] = copy.deepcopy(e['base_coords'])
            e['changed_per_rank'] = [0]*8
            p['entry_sha256'] = digest(e)
    return plans


def prepare(out, attn=None, mlp=None, source_only=False):
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    arrival = scope_build.prepare(out/'arrival', attn, mlp, source_only)
    parent = read(arrival/'plans.json')
    plans = ablation3_plan.bundle(parent)
    write(out/'plans.json', plans)
    # Copy registered sources plus construction metadata, preserving the shared
    # CUTLASS symlink. No compiled artifacts exist at this point.
    shutil.copytree(arrival, out/'base', symlinks=True)
    base = out/'base'; m = read(base/'manifest.json'); v = m['variants']['scope-candidate']
    derived = order_plans(parent, 'base'); write(base/'plans.json', derived)
    entries = {role: p['entry'] for role, p in derived['roles'].items()}
    source = base/'scope-candidate'
    upstream = (FREEZE/'source'/scope_build.SWIZZLE).read_text()
    (source/scope_build.SWIZZLE).write_text(scope_build.swizzle(upstream, entries))
    (source/'phase6_host.hpp').write_text('#pragma once\n'+scope_build.tables(entries))
    for name in (scope_build.SWIZZLE, 'phase6_host.hpp'):
        v['source_sha256'][name] = sha(source/name)
        before = v['source_changes'].get(name, {}).get('before',
                    read(arrival/'manifest.json')['variants']['scope-candidate']['source_sha256'][name])
        v['source_changes'][name] = dict(before=before, after=v['source_sha256'][name])
    m['files']['plans.json'] = sha(base/'plans.json')
    m['inputs'][str(HERE/'ablation3_build.py')] = sha(HERE/'ablation3_build.py')
    write(base/'manifest.json', m)
    builds = {}
    for order in ORDERS:
        child = scope_build.verify(out/order)
        builds[order] = dict(path=order, source_sha256=child['variants']['scope-candidate']['source_sha256'])
    record = dict(protocol=PROTOCOL, source_only=source_only, diagnostic_only=source_only,
                  gpu_validated=False, config=m['config'], parent_source_commit=m['parent_source_commit'],
                  builds=builds, files={'plans.json': sha(out/'plans.json')},
                  inputs={str(HERE/n): sha(HERE/n) for n in
                          ('ablation3_build.py', 'ablation3_plan.py', 'ablation3_protocol.py')})
    write(out/'ablation3-manifest.json', record)
    verify(out)
    return out


def validate_snapshots(record, plans, children, built=False, verify_evidence=True):
    require(record['protocol'] == PROTOCOL and set(record['builds']) == set(ORDERS), 'Wrong ablation3 build')
    ablation3_plan.validate(plans, verify_evidence)
    require(record['source_only'] == record['diagnostic_only'] == plans['source_only'], 'Build rehearsal flags changed')
    require(record['config'] == plans['parent']['roles']['attn']['config'], 'Model changed')
    common = None
    for order in ORDERS:
        child = children[order]
        require(child['protocol'] == scope_build.PROTOCOL and child['kind'] == 'scope_candidate' and
                set(child['variants']) == {'scope-candidate'} and child['config'] == record['config'] and
                child['source_only'] == child['diagnostic_only'] == record['source_only'] and
                child['parent_source_commit'] == record['parent_source_commit'], 'Child build identity changed')
        v = child['variants']['scope-candidate']
        require(v['role'] == 'candidate' and not v['sampling'] and
                v['source_sha256'] == record['builds'][order]['source_sha256'], 'Compute sources changed')
        compute = {k: h for k, h in v['source_sha256'].items() if k not in (scope_build.SWIZZLE, 'phase6_host.hpp')}
        if common is None:
            common = compute
        require(compute == common, 'Base/arrival must share compute, codec and wrappers')
        if built:
            require(child['status'] == 'built' and not record['source_only'], 'Unbuilt/diagnostic build')


def verify(out, built=False):
    out = Path(out).resolve(); m = read(out/'ablation3-manifest.json')
    verify_files(out, m['files']); plans = read(out/'plans.json')
    children = {order: scope_build.verify(out/order, built=built, kind='scope_candidate') for order in ORDERS}
    validate_snapshots(m, plans, children, built)
    for order in ORDERS:
        expected = order_plans(plans['parent'], order)
        require(read(out/order/'plans.json') == expected, 'Child build order differs from shared parent')
        entries = {r: p['entry'] for r, p in expected['roles'].items()}
        src = out/order/'scope-candidate'
        require((src/'phase6_host.hpp').read_text() == '#pragma once\n'+scope_build.tables(entries), 'Wrong host mapping')
        upstream = (FREEZE/'source'/scope_build.SWIZZLE).read_text()
        require((src/scope_build.SWIZZLE).read_text() == scope_build.swizzle(upstream, entries), 'Wrong device mapping')
    return m
