"""Frozen training overlays; no changes to the existing baseline worker files."""
import hashlib

import baseline_sources as parent
import a_baseline_protocol as routing
from common import HERE, REPO, replace_once, sha


def codec_source():
    return parent.codec_source()


def codec_identity():
    identity = parent.codec_identity()
    assert identity['implementation'] == routing.CODEC
    return identity


def taco_source():
    return parent.taco_source()


def adapter_source():
    text = parent.adapter_source()
    text = replace_once(text, 'import types\n', 'import types\nimport hashlib\n')
    text = replace_once(text, '        self.scope_ops = {}', '''        self.smoke_outputs = {}
        self.retained_outputs = {}
        self.retention_targets = {}
        self.retention_checks = {}
        self.scope_ops = {}''')
    text = replace_once(text, '            if module.skip_bias_add:', '''            if os.environ.get('E2E_MODE') == 'smoke':
                adapter.record_smoke_output(name, output)
            if module.skip_bias_add:''')
    anchor = '    def stop_audit(self):\n'
    methods = '''    def record_smoke_output(self, name, output):
        # Only smoke workers hash or retain tensors. There are no extra forward calls.
        if name not in self.smoke_outputs:
            data = output.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            self.smoke_outputs[name] = hashlib.sha256(data).hexdigest()
        role = module_role(name)
        if role not in self.retention_targets:
            self.retention_targets[role] = name
        if self.retention_targets[role] != name or role in self.retention_checks:
            return
        retained = self.retained_outputs.setdefault(role, [])
        if len(retained) < 6:
            retained.append((output.detach(), output.detach().clone()))
            return
        # A seventh real model call has now reused the internal workspace.
        pointers = [value.data_ptr() for value, _ in retained]
        assert len(set(pointers)) == 6
        assert all(torch.equal(value, reference) for value, reference in retained)
        self.retention_checks[role] = dict(module=name, outputs=6,
            independent_storage=True, unchanged_after_subsequent_call=True, passed=True)
        del self.retained_outputs[role]

'''
    return replace_once(text, anchor, methods+anchor)


def worker_source(source_only=False):
    text = parent.worker_source(source_only)
    text = text.replace('H100_BASELINE_EXECUTE', 'H100_A_BASELINE_EXECUTE')
    text = text.replace('GPU worker requires baseline --execute', 'GPU worker requires a-baseline --execute')
    anchor = "report['codec_validation'] = codec_validation"
    return replace_once(text, anchor, anchor+'''
report['smoke_output_sha256'] = adapter.smoke_outputs
report['retained_output_checks'] = adapter.retention_checks
if MODE == 'smoke':
    assert set(adapter.smoke_outputs) == set(adapter.scope_ops)
    assert set(adapter.retention_checks) == {'attn', 'mlp'}
else:
    assert not adapter.smoke_outputs and not adapter.retention_checks
report['shared_environment'] = {key: os.environ[key] for key in
    ('CUDA_DEVICE_MAX_CONNECTIONS', 'FLUX_GEMM_RS_GEMM_ONLY')}
''')
