"""Frozen training overlays; no changes to the existing baseline worker files."""
import hashlib

import baseline_sources as parent
import basic_quant_protocol as routing
import basic_quant_build as codec_build
from common import HERE, REPO, replace_once, sha


def codec_source():
    return (HERE/'basic_quant_codec.py').read_text()


def codec_identity():
    identity = parent.codec_identity()
    assert identity['implementation'] == routing.CODEC
    identity.update(source_sha256=hashlib.sha256(codec_source().encode()).hexdigest(),
                    cuda_source_identity=codec_build.source_identity(),
                    receiver_implementation='basic-serial-h128-decode-bf16-fold-v2')
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
    text = text.replace('H100_BASELINE_EXECUTE', 'H100_BASIC_QUANT_EXECUTE')
    text = text.replace('GPU worker requires baseline --execute', 'GPU worker requires basic-quant --execute')
    text = replace_once(text, 'from baseline_codec_validation import validate as validate_codec',
                        'from basic_quant_validation import validate as validate_codec')
    text = replace_once(text, 'codec_validation = validate_codec(standalone_codec, codec_reference)',
        'codec_validation = validate_codec(standalone_codec, codec_reference, local_rank)')
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
