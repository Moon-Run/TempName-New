"""Original/A/B share pinned fused v2; only C uses ablation3 double buffers."""
import ablation3_sources
import scope_sources
from common import replace_once


def taco_source():
    source = ablation3_sources.taco_source()
    source = replace_once(source, 'from routing import is_native, selection_entry, definition',
                          'from routing import is_native, is_fused_v2, selection_entry, definition')
    anchor = "        plan = json.loads((Path(__file__).parent/'selection-plan.json').read_text())"
    return replace_once(source, anchor, '''        if is_fused_v2(policy):
            from flux.gemm_rs_taco import GemmRSTaco
            self.role = role
            self.plan_sha256 = None
            self.codec = GemmRSTaco(group, m, n, k_local, placement='fused')
            self.owns_output = False  # Original/A/B retain the same output clone.
            self.workspace_info = [dict(shape=[m,n,k_local*dist.get_world_size(group)],
                packet_ptr=self.codec.peers[self.codec.rank].data_ptr())]
            return
'''+anchor)


def adapter_source():
    source = scope_sources.adapter_source()
    source = replace_once(source, 'from routing import codec_placement, enabled, is_native, module_role, route',
                          'from routing import codec_placement, enabled, is_native, is_fused_v2, module_role, route')
    source = replace_once(source, "pointers = [slot[field] for slots in self.workspace_checks.values() for slot in slots]",
                          "pointers = [slot[field] for slots in self.workspace_checks.values() for slot in slots if field in slot]")
    return replace_once(source, '                assert self.scope_ops[name].owns_output',
                         '                assert self.scope_ops[name].owns_output == (not is_fused_v2(self.policy))')


def worker_source(source_only=False):
    source = scope_sources.worker_source(source_only)
    source = replace_once(source, "os.environ.get('H100_SCOPE_EXECUTE')", "os.environ.get('H100_ABLATION4_EXECUTE')")
    source = replace_once(source, 'GPU worker requires scope --execute', 'GPU worker requires ablation4 --execute')
    # Reject configuration drift before CUDA initialization or Megatron setup.
    source = replace_once(source, 'torch.cuda.set_device(local_rank)',
                          "from communication import verify as verify_communication\n"
                          "communication = verify_communication(PLAN['communication'])\n"
                          'torch.cuda.set_device(local_rank)')
    return replace_once(source, "report['protocol'] = PLAN['protocol']",
                         "report['communication'] = communication\nreport['protocol'] = PLAN['protocol']")
