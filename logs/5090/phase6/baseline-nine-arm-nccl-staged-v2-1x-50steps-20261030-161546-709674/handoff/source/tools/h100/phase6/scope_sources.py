"""Checked overlays on the existing worker/adapter; never edit shared A800 code."""
from common import REPO, replace_once
from prepare import model_worker


def taco_source():
    text = (REPO/'tools/a800/phase5/e2e/taco_support.py').read_text()
    text = replace_once(text, 'from routing import base_policy, codec_placement, selection_entry',
                        'from routing import is_native, selection_entry')
    start = text.index('    def __init__(self, group, m, n, k_local, policy):')
    end = text.index('\n    def forward(', start)
    text = text[:start]+'''    def __init__(self, group, m, n, k_local, policy, role):
        if is_native(policy):
            raise ValueError('Native scope cannot construct a Flux candidate')
        plan = json.loads((Path(__file__).parent/'selection-plan.json').read_text())
        record = selection_entry(plan, policy, role, m, n, k_local*dist.get_world_size(group))
        from flux.gemm_rs_taco import GemmRSTacoDoubleBuffered
        self.role = role
        self.plan_sha256 = record['entry_sha256']
        self.codec = GemmRSTacoDoubleBuffered(group,m,n,k_local,placement='fused',
            selected=record['entry']['mask'],quantization_mode='targeted')
        self.owns_output = self.codec.owns_configuration
        if not self.owns_output:
            raise RuntimeError('Scope requires the frozen instance-owned V2 path')
        self.workspace_info = [dict(shape=[slot.m,slot.n,slot.k_local*dist.get_world_size(group)],
            mask_sha256=slot.selection_sha256, packet_ptr=slot.peers[slot.rank].data_ptr(),
            mask_ptr=slot.selected.data_ptr()) for slot in self.codec.slots]
        if len(self.workspace_info) != 2 or any(s['shape'] != [m,n,k_local*dist.get_world_size(group)]
                                               for s in self.workspace_info):
            raise RuntimeError('Double workspace/full-shape mismatch')
'''+text[end:]
    return text


def adapter_source():
    text = (REPO/'tools/a800/phase5/e2e/adapter.py').read_text()
    text = replace_once(text, 'from routing import codec_placement',
                        'from routing import codec_placement, enabled, is_native, module_role, route')
    text = replace_once(text, "if policy not in ('native','native_taco'):", 'if not is_native(policy):')
    text = replace_once(text, '        self.flux = None',
                        '        self.scope_ops = {}\n        self.actual_routes = {}\n'
                        "        self.profile = os.environ.get('E2E_MODE') == 'profile'\n        self.flux = None")
    text = replace_once(text, "if adapter.policy == 'native' and not adapter.control_backward:\n                return native(input_)",
                        "if not enabled(adapter.policy, name):\n"
                        "                adapter.actual_routes[name] = 'native_bf16'\n"
                        "                adapter.placements[name] = 'native'\n                return native(input_)")
    text = replace_once(text, "            if adapter.policy == 'native_taco':\n"
                        "                if name.endswith('.self_attention.linear_proj'):\n"
                        "                    adapter.placements[name]='native'\n"
                        "                    return native(input_)", '            if is_native(adapter.policy):')
    text = replace_once(text, '                    def quantized_rs(partial):',
                        "                    adapter.scope_ops[name] = op\n"
                        "                    adapter.actual_routes[name] = 'tensor'\n                    def quantized_rs(partial):")
    text = replace_once(text, "if codec_placement(adapter.policy) in ('fused','separate') and name.endswith('.mlp.linear_fc2'):",
                        "if enabled(adapter.policy, name) and not is_native(adapter.policy):")
    text = replace_once(text, 'input_.shape[-1], adapter.policy)', 'input_.shape[-1], adapter.policy, module_role(name))')
    text = replace_once(text, '                    adapter.placements[name] = op.codec.placement',
                        "                    adapter.scope_ops[name] = op\n"
                        "                    adapter.actual_routes[name] = 'candidate'\n"
                        '                    adapter.placements[name] = op.codec.placement')
    # This branch must be unreachable: all unselected projections returned native above.
    text = replace_once(text, '                    op = adapter.flux.GemmRS(',
                        "                    raise RuntimeError('Unexpected scope fallback')\n                    op = adapter.flux.GemmRS(")
    start = text.index('    def stop_audit(self):')
    text = text[:start]+'''    def stop_audit(self):
        self.audit = False
        expected = {name: route(self.policy,name) for name in self.modules}
        assert self.actual_routes == expected
        assert set(self.scope_ops) == {name for name in self.modules if enabled(self.policy,name)}
        assert len({id(op) for op in self.scope_ops.values()}) == len(self.scope_ops)
        self.post_audit_routes = {}
        self.forward_preserved = {}
        self.timed_calls = {name:0 for name in self.modules}
        self.workspace_checks = {} if is_native(self.policy) else {name:op.workspace_info for name,op in self.scope_ops.items()}
        for field in ('packet_ptr','mask_ptr'):
            pointers = [slot[field] for slots in self.workspace_checks.values() for slot in slots]
            assert all(p > 0 for p in pointers) and len(pointers) == len(set(pointers))
        for name,module in self.modules.items():
            if not enabled(self.policy,name):
                target = types.MethodType(RowParallelLinear.forward,module)
                actual = 'native_bf16'
            elif is_native(self.policy):
                target = self.native_quant_forwards[name]
                actual = 'tensor'
            else:
                target = module.forward
                assert self.scope_ops[name].owns_output
                actual = 'candidate'
            self.forward_preserved[name] = target.__func__.__code__ is RowParallelLinear.forward.__code__
            assert self.forward_preserved[name] == (actual != 'candidate')
            self.post_audit_routes[name] = actual
            if enabled(self.policy,name):
                op = self.scope_ops[name]
                self.selection_checks[name].update(role=module_role(name),
                    shape=[self.observed[name][k] for k in ('M','N','K_global')],
                    plan_sha256=None if is_native(self.policy) else op.plan_sha256)
            def timed(module,input_,_target=target,_name=name,_role=module_role(name),_route=actual):
                self.timed_calls[_name] += 1
                if self.profile:
                    with torch.profiler.record_function('scope.'+_role+'.'+_route):
                        return _target(input_)
                return _target(input_)
            module.forward = types.MethodType(timed,module)
        assert self.post_audit_routes == expected
'''
    return text


def worker_source(source_only=False):
    text = model_worker()
    text = replace_once(text, "local_rank = int(os.environ['LOCAL_RANK'])",
                        "if os.environ.get('H100_SCOPE_EXECUTE') != '1':\n"
                        "    raise RuntimeError('GPU worker requires scope --execute')\n"
                        "local_rank = int(os.environ['LOCAL_RANK'])")
    text = replace_once(text, 'from routing import module_routes',
                        'from routing import module_routes, definition, is_native, enabled')
    text = replace_once(text, "if POLICY not in ('native','native_taco'):", 'if not is_native(POLICY):')
    text = replace_once(text, "elif POLICY in ('native', 'native_taco'):", 'else:')
    text = replace_once(text, "report['initial_parameters_sha256'] = fingerprint(raw.named_parameters())",
                        "report['protocol'] = PLAN['protocol']\n"
                        "report['scope_definition'] = definition(POLICY)\n"
                        "report['role_plan_sha256'] = PLAN['role_plan_sha256']\n"
                        "report['initial_parameters_sha256'] = fingerprint(raw.named_parameters())")
    text = replace_once(text, "if POLICY=='native_taco':", 'if is_native(POLICY):')
    text = replace_once(text, "assert len(adapter.native_quant_forwards)==SPEC['model']['layers']",
                        "assert len(adapter.native_quant_forwards)==SPEC['model']['layers']*len(definition(POLICY)['roles'])")
    text = replace_once(text, "report['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30", '''report['post_audit_routes'] = adapter.post_audit_routes
report['timed_calls'] = adapter.timed_calls
report['forward_preserved'] = adapter.forward_preserved
report['enabled_modules'] = sorted(adapter.scope_ops)
report['independent_scope_instances'] = len({id(op) for op in adapter.scope_ops.values()})
report['workspace_checks'] = adapter.workspace_checks
assert all(n == (1 if MODE=='profile' else count)*microsteps for n in adapter.timed_calls.values())
assert adapter.post_audit_routes == report['module_policies'] == adapter.actual_routes
report['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30''')
    text = replace_once(text, "replaced_modules=0 if POLICY=='native' else len(adapter.native_quant_forwards) if POLICY=='native_taco' else len(adapter.modules)",
                        'replaced_modules=len(adapter.scope_ops)')
    if source_only:
        text = "raise RuntimeError('Source-only scope worker is permanently non-executable')\n"+text
    return text
