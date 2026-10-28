"""Independent adversarial checks for MA4-F's layer-owned consumer context.

These exercise the generated adapter methods without CUDA or PyTorch; actual
numerical/gradient validation remains the eight-GPU operator preflight.
"""
import types
import unittest

import ma4_f_adapter


class FakeTorch:
    bfloat16 = object()
    grad = True
    inference = False

    def is_grad_enabled(self):
        return self.grad

    def is_inference_mode_enabled(self):
        return self.inference


class Tensor:
    _version = 0


class ConsumerContextReview(unittest.TestCase):
    def setUp(self):
        self.torch = FakeTorch()
        namespace = dict(torch=self.torch, types=types)
        exec('class Adapter:\n'+ma4_f_adapter.ADAPTER_METHODS, namespace)
        self.adapter = namespace['Adapter']()
        self.adapter._consumer_enabled = True
        self.adapter._consumer_failed = False
        self.state = dict(residual=None, output=None, residual_version=None,
                          projection_calls=0, bda_calls=0, bias_parameter_id=123)
        self.layer = types.SimpleNamespace(training=True)
        self.residual = Tensor()
        self.output = Tensor()
        self.consume = self.adapter._consumer_bda(self.state)(True, False)

    def project(self):
        self.state['projection_calls'] += 1
        self.state['output'] = self.output

    def run_phase(self, function, *, mlp=False):
        phase = self.adapter._consumer_phase(function, self.state,
                    1 if mlp else 0, 'residual' if mlp else 'hidden_states')
        return phase(self.layer, **({'pre_mlp_layernorm_output': Tensor(), 'residual': self.residual}
                                    if mlp else {'hidden_states': self.residual}))

    def assert_cleared(self):
        self.assertIsNone(self.state['residual'])
        self.assertIsNone(self.state['residual_version'])
        self.assertIsNone(self.state['output'])

    def test_identity_binding_and_keyword_mlp_argument_are_preserved(self):
        def original(**kwargs):
            self.assertIs(kwargs['residual'], self.residual)
            self.assertIs(self.state['residual'], self.residual)
            self.project()
            return self.consume((self.output, None), kwargs['residual'], 0.)
        self.assertIs(self.run_phase(original, mlp=True), self.output)
        self.assert_cleared()
        self.assertEqual(self.state['projection_calls'], 1)
        self.assertEqual(self.state['bda_calls'], 1)

    def test_equal_looking_different_residual_cannot_be_consumed(self):
        def original(**kwargs):
            self.project()
            return self.consume((self.output, None), Tensor(), 0.)
        with self.assertRaises(RuntimeError):
            self.run_phase(original)
        self.assertTrue(self.adapter._consumer_failed)
        self.assert_cleared()

    def test_inplace_residual_change_after_projection_is_rejected(self):
        def original(**kwargs):
            self.project()
            self.residual._version += 1
            return self.consume((self.output, None), self.residual, 0.)
        with self.assertRaises(RuntimeError):
            self.run_phase(original)
        self.assertTrue(self.adapter._consumer_failed)
        self.assert_cleared()

    def test_second_bda_consumption_poison_and_cleanup(self):
        def original(**kwargs):
            self.project()
            self.consume((self.output, None), self.residual, 0.)
            return self.consume((self.output, None), self.residual, 0.)
        with self.assertRaises(RuntimeError):
            self.run_phase(original)
        self.assertTrue(self.adapter._consumer_failed)
        self.assert_cleared()

    def test_exception_after_projection_cannot_leave_live_output_or_retry(self):
        def original(**kwargs):
            self.project()
            raise RuntimeError('synthetic post-projection exception')
        with self.assertRaisesRegex(RuntimeError, 'synthetic'):
            self.run_phase(original)
        self.assert_cleared()
        with self.assertRaises(RuntimeError):
            self.run_phase(original)
        self.assertEqual(self.state['projection_calls'], 1)

    def test_no_grad_or_eval_cannot_execute_projection(self):
        for field, value in (('grad', False), ('inference', True), ('training', False)):
            self.setUp()
            setattr(self.layer if field=='training' else self.torch, field, value)
            executed = []
            def original(**kwargs):
                executed.append(True)
                self.project()
                return self.consume((self.output, None), self.residual, 0.)
            with self.subTest(field=field):
                with self.assertRaises(RuntimeError):
                    self.run_phase(original)
                self.assertEqual(executed, [])
                self.assert_cleared()

    def test_version_read_exception_cannot_leak_phase_residual(self):
        class InferenceLikeTensor:
            @property
            def _version(self):
                raise RuntimeError('no version counter')
        self.residual = InferenceLikeTensor()
        with self.assertRaisesRegex(RuntimeError, 'version'):
            self.run_phase(lambda **kwargs: self.fail('phase executed after invalid version read'))
        self.assert_cleared()


if __name__ == '__main__':
    unittest.main()
