"""Preprocessing API compatible with the covered Scanpy neighbours operation."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ._lib import addr, check_status, f64, lib


def neighbors(
    adata, n_neighbors: int = 15, n_pcs: int | None = None, *, use_rep: str | None = None,
    knn: bool = True, method: str = "umap", metric: str = "euclidean", metric_kwds=None,
    key_added: str | None = None, copy: bool = False, **kwargs,
):
    """Compute exact Euclidean k-nearest neighbours and store Scanpy graph slots."""
    if copy:
        adata = adata.copy()
    if kwargs:
        raise NotImplementedError(f"unsupported neighbors options: {', '.join(sorted(kwargs))}")
    if not knn:
        raise NotImplementedError("only knn=True is supported")
    if metric != "euclidean" or metric_kwds:
        raise NotImplementedError("only Euclidean distance is supported")
    if method not in {"umap", "gauss"}:
        raise NotImplementedError("method must be 'umap' or 'gauss'")
    if use_rep is None:
        values = adata.obsm["X_pca"][:, :n_pcs] if n_pcs is not None and "X_pca" in adata.obsm else adata.X
        rep_name = "X_pca" if n_pcs is not None and "X_pca" in adata.obsm else "X"
    else:
        values = adata.obsm[use_rep]
        rep_name = use_rep
    if sparse.issparse(values):
        values = values.toarray()
    values = f64(values, name="representation")
    if values.ndim != 2:
        raise ValueError("representation must be a two-dimensional matrix")
    n_obs, n_vars = values.shape
    if n_obs < 2 or n_vars < 1:
        raise ValueError("neighbors requires at least two observations and one feature")
    if isinstance(n_neighbors, (bool, np.bool_)) or int(n_neighbors) != n_neighbors:
        raise TypeError("n_neighbors must be an integer")
    requested = int(n_neighbors)
    if requested < 2:
        raise ValueError("n_neighbors must be at least 2 (including self)")
    count = min(requested - 1, n_obs - 1)
    indices = np.empty((n_obs, count), dtype=np.float64)
    squared = np.empty((n_obs, count), dtype=np.float64)
    check_status(
        lib().msp_knn(addr(values), addr(values), addr(indices), addr(squared), n_obs, n_vars, n_obs, count, 1),
        "k-nearest-neighbor call",
    )
    indices_i = indices.astype(np.int64)
    distances = np.sqrt(squared)
    rows = np.repeat(np.arange(n_obs), count)
    cols = indices_i.ravel()
    distance_graph = sparse.csr_matrix((distances.ravel(), (rows, cols)), shape=(n_obs, n_obs))
    full_indices = np.column_stack((np.arange(n_obs), indices_i))
    full_distances = np.column_stack((np.zeros(n_obs), distances))
    if method == "umap":
        from scanpy.neighbors._connectivity import umap

        connectivities = umap(full_indices, full_distances, n_obs=n_obs, n_neighbors=count + 1)
    else:
        from scanpy.neighbors._connectivity import gauss

        connectivities = gauss(distance_graph, count + 1, knn=True)
    if key_added is None:
        prefix = "neighbors"
        distances_key, connectivities_key = "distances", "connectivities"
    else:
        prefix = key_added
        distances_key, connectivities_key = f"{prefix}_distances", f"{prefix}_connectivities"
    adata.obsp[distances_key] = distance_graph
    adata.obsp[connectivities_key] = connectivities
    adata.uns[prefix] = {
        "connectivities_key": connectivities_key,
        "distances_key": distances_key,
        "params": {"n_neighbors": count, "method": method, "metric": metric, "use_rep": rep_name},
    }
    return adata if copy else None
