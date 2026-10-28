"""Warmup-only control after the normal worker has initialized CUDA/Flux."""
def configure_decoder_variant():
    import ctypes
    import os
    from pathlib import Path
    import flux
    variant = int(os.environ['E2E_DECODE_VARIANT'])
    assert variant in (0, 3, 4, 5, 6, 10, 11, 12, 13)
    library = ctypes.CDLL(str((Path(flux.__file__).parent/'lib/libflux_cuda.so').resolve()))
    library.taco_set_decode_variant.argtypes = [ctypes.c_int]
    library.taco_set_decode_variant.restype = ctypes.c_int
    assert library.taco_set_decode_variant(variant) == 0
