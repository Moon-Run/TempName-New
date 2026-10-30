"""Freeze shared naive-codec provenance and checked ablation4 source overlays."""
import hashlib

import ablation4_sources
from common import HERE, REPO, replace_once


def naive_source():
    text = (REPO/'tools/a800/phase5/e2e/megatron_taco.py').read_text()
    text = replace_once(text, 'self.world==4', 'self.world in (4,8)')
    text = replace_once(text, '        self.selection_sha256=None', '        self.calls=0\n        self.selection_sha256=None')
    return replace_once(text, '    def reduce(self,partial):\n', '    def reduce(self,partial):\n        self.calls += 1\n')


def naive_identity():
    import ablation6_protocol as routing
    return dict(implementation=routing.NAIVE_CODEC,
                source_sha256=hashlib.sha256(naive_source().encode()).hexdigest(),
                reference_sha256=hashlib.sha256((REPO/'tools/a800/phase4/reference.py').read_bytes()).hexdigest())


def taco_source():
    text = ablation4_sources.taco_source()
    text = replace_once(text, 'from routing import is_native, is_fused_v2, selection_entry, definition',
                        'from routing import is_native, is_fused_v2, is_flux_tensor, selection_entry, definition')
    return replace_once(text, '        if is_fused_v2(policy):', '''        if is_flux_tensor(policy):
            from flux_tensor_taco import FluxTensorTaco
            self.role, self.plan_sha256 = role, None
            self.codec = FluxTensorTaco(group, m, n, k_local)
            self.owns_output = True
            self.workspace_info = self.codec.workspace_info
            return
        if is_fused_v2(policy):''')


def adapter_source():
    text = ablation4_sources.adapter_source()
    text = replace_once(text, 'is_fused_v2, module_role, route', 'is_fused_v2, is_flux_tensor, is_tensor, module_role, route')
    start = text.index('            flux.init_flux_shm(group)') if '            flux.init_flux_shm(group)' in text else text.index('            flux.init_flux_shm(parallel_state')
    end = text.index('        for name, module in model.named_modules():', start)
    text = text[:start]+'            if not is_flux_tensor(policy):\n'+''.join('    '+line+'\n' for line in text[start:end].rstrip().splitlines())+text[end:]
    text = replace_once(text, "                    adapter.actual_routes[name] = 'candidate'", "                    adapter.actual_routes[name] = route(adapter.policy, name)")
    text = replace_once(text, "                actual = 'candidate'", "                actual = route(self.policy, name)")
    text = replace_once(text, "(actual != 'candidate')", "(actual not in ('candidate', 'flux_tensor'))")
    text = replace_once(text, "adapter.placements[name] in ('fused','separate')",
                        "adapter.placements[name] in ('fused','separate','tensor')")
    text = replace_once(text, "for field in ('packet_ptr','mask_ptr'):", "for field in ('packet_ptr','mask_ptr','gemm_output_ptr'):")
    anchor = "        assert self.post_audit_routes == expected\n"
    return replace_once(text, anchor, anchor+'''
    def naive_codec_calls(self):
        if not is_tensor(self.policy):
            return {}
        codecs = {name: op.codec.tensor_codec if is_flux_tensor(self.policy) else op
                  for name, op in self.scope_ops.items()}
        assert all(type(codec) is TensorTacoRS for codec in codecs.values())
        return {name: codec.calls for name, codec in codecs.items()}
''')


def worker_source(source_only=False):
    text = ablation4_sources.worker_source(source_only)
    text = replace_once(text, "os.environ.get('H100_ABLATION4_EXECUTE')", "os.environ.get('H100_ABLATION6_EXECUTE')")
    text = replace_once(text, 'GPU worker requires ablation4 --execute', 'GPU worker requires ablation6 --execute')
    return replace_once(text, "report['communication'] = communication", '''report['communication'] = communication
report['naive_codec_identity'] = dict(implementation=PLAN['naive_codec_identity']['implementation'],
    source_sha256=hashlib.sha256((HERE/'megatron_taco.py').read_bytes()).hexdigest(),
    reference_sha256=hashlib.sha256((HERE/'reference.py').read_bytes()).hexdigest())''').replace(
        "report['post_audit_routes'] = adapter.post_audit_routes",
        "report['naive_codec_calls'] = adapter.naive_codec_calls()\nreport['post_audit_routes'] = adapter.post_audit_routes")
