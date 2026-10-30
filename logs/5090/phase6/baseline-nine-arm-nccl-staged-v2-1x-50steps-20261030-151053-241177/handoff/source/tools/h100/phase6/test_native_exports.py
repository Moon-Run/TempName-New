"""Real host ELF linkage tests; these libraries contain no GPU implementation."""
import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

from native_exports import verify_native_exports


def compile_cpu_runtime(library, version_script, symbols):
    """Build a labeled host-only stub with the production export policy."""
    library = Path(library)
    library.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='native-export-link-') as tmp:
        root = Path(tmp)
        source = root / 'cpu_fixture.cc'
        source.write_text('// CPU LINK FIXTURE ONLY: NOT A GPU RUNTIME\n' + '\n'.join(
            f'extern "C" int {name}() {{ return {value}; }}' for name, value in symbols.items()))
        script = root / 'version.ld'
        script.write_text(version_script)
        subprocess.run(['g++', '-shared', '-fPIC', str(source),
                        '-Wl,--version-script=' + str(script), '-o', str(library)],
                       capture_output=True, text=True, check=True)


class NativeExportTests(unittest.TestCase):
    def test_local_function_is_rejected_even_when_present_in_library(self):
        with tempfile.TemporaryDirectory(prefix='native-export-test-') as tmp:
            library = Path(tmp) / 'hidden.so'
            compile_cpu_runtime(library, '{global: taco_*; local: *;};',
                                {'taco_reference': 1, 'candidate_protocol': 9})
            loaded = ctypes.CDLL(str(library))
            self.assertEqual(loaded.taco_reference(), 1)
            with self.assertRaises(AttributeError):
                getattr(loaded, 'candidate_protocol')
            with self.assertRaisesRegex(ValueError, 'Missing exported native functions.*candidate_protocol'):
                verify_native_exports(library, required=('candidate_protocol',))

    def test_exported_function_resolves_and_control_rejects_it(self):
        with tempfile.TemporaryDirectory(prefix='native-export-test-') as tmp:
            library = Path(tmp) / 'exported.so'
            compile_cpu_runtime(library, '{global: candidate_protocol; local: *;};',
                                {'candidate_protocol': 9})
            self.assertEqual(ctypes.CDLL(str(library)).candidate_protocol(), 9)
            verify_native_exports(library, required=('candidate_protocol',))
            with self.assertRaisesRegex(ValueError, 'Unexpected candidate exports'):
                verify_native_exports(library, forbidden=('candidate_protocol',))

    def test_placeholder_is_not_a_valid_linked_runtime(self):
        with tempfile.TemporaryDirectory(prefix='native-export-test-') as tmp:
            library = Path(tmp) / 'placeholder.so'
            library.write_text('CPU placeholder, not an ELF shared library\n')
            with self.assertRaisesRegex(ValueError, 'Cannot inspect native exports'):
                verify_native_exports(library, required=('candidate_protocol',))


if __name__ == '__main__':
    unittest.main()
