"""Dense PCA backed by the Mojo covariance/eigensolver kernel."""

from __future__ import annotations

import numpy as np

from ._lib import addr, check_status, f64, lib


def pca_array(X, n_comps: int | None = None):
    """Return scores, loadings, variances and variance ratios for dense X."""
    data = f64(X, name="X")
    if data.ndim != 2 or min(data.shape) < 2:
        raise ValueError("X must be a two-dimensional array with at least two rows and columns")
    n_obs, n_vars = data.shape
    if n_comps is None:
        n_comps = min(n_obs, n_vars) - 1
    elif isinstance(n_comps, (bool, np.bool_)) or int(n_comps) != n_comps:
        raise TypeError("n_comps must be an integer")
    else:
        n_comps = int(n_comps)
    if not 1 <= n_comps <= min(n_obs, n_vars):
        raise ValueError("n_comps must be between 1 and min(X.shape)")
    mean = np.empty(n_vars, dtype=np.float64)
    components = np.empty((n_comps, n_vars), dtype=np.float64)
    variance = np.empty(n_comps, dtype=np.float64)
    matrix = np.empty((n_vars, n_vars), dtype=np.float64)
    vectors = np.empty((n_vars, n_vars), dtype=np.float64)
    total = lib().msp_pca_fit(
        addr(data), addr(mean), addr(components), addr(variance), addr(matrix), addr(vectors),
        n_obs, n_vars, n_comps,
    )
    if total < 0:
        raise RuntimeError("Mojo PCA fit rejected invalid native buffer arguments")
    scores = np.empty((n_obs, n_comps), dtype=np.float64)
    check_status(
        lib().msp_pca_transform(addr(data), addr(mean), addr(components), addr(scores), n_obs, n_vars, n_comps),
        "PCA transform call",
    )
    ratio = variance / total if total > 0 else np.zeros_like(variance)
    return scores, components, variance, ratio
