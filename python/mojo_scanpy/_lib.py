"""ctypes boundary for the Mojo kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_LIBRARY = _ROOT / "dist" / "libmojo-scanpy.so"
_I64 = ctypes.c_int64
_F64 = ctypes.c_double
_lib: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        if not _LIBRARY.exists():
            raise RuntimeError("Mojo library missing; run `pixi run build` first")
        _lib = ctypes.CDLL(str(_LIBRARY))
        _lib.msp_knn.argtypes = [_I64] * 9
        _lib.msp_knn.restype = _I64
        _lib.msp_pca_fit.argtypes = [_I64] * 9
        _lib.msp_pca_fit.restype = _F64
        _lib.msp_pca_transform.argtypes = [_I64] * 7
        _lib.msp_pca_transform.restype = _I64
    return _lib


def f64(value, *, name: str = "input") -> np.ndarray:
    """Make a checked, C-contiguous float64 array safe to pass to Mojo."""
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise TypeError(f"{name} must be real-valued; complex input would lose data")
    if array.dtype.kind not in "biuf":
        raise TypeError(f"{name} must have a numeric dtype")
    array = np.ascontiguousarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def addr(value: np.ndarray) -> int:
    if value.dtype not in (np.dtype(np.float64), np.dtype(np.int64)) or not value.flags.c_contiguous:
        raise TypeError("native buffers must be C-contiguous float64 or int64 arrays")
    if value.size == 0:
        raise ValueError("native buffers must not be empty")
    return int(value.ctypes.data)


def check_status(status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"Mojo {operation} rejected invalid native buffer arguments")
