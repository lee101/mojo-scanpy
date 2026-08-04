"""Tools API for PCA and graph community clustering."""

from __future__ import annotations

import numpy as np
import warnings

from ._pca import pca_array


def pca(
    adata, n_comps: int | None = None, *, layer: str | None = None, zero_center: bool = True,
    svd_solver: str | None = None, random_state=0, return_info: bool = False, **kwargs,
):
    """Run centered dense PCA and place results in Scanpy-compatible slots."""
    if not zero_center:
        raise NotImplementedError("only zero_center=True is supported")
    if kwargs:
        raise NotImplementedError(f"unsupported pca options: {', '.join(sorted(kwargs))}")
    if svd_solver not in (None, "full"):
        raise NotImplementedError("only the full dense solver is supported")
    if layer is not None:
        values = adata.layers[layer]
    else:
        values = adata.X
    scores, components, variance, ratio = pca_array(values, n_comps)
    adata.obsm["X_pca"] = scores
    adata.varm["PCs"] = components.T
    adata.uns["pca"] = {
        "variance": variance,
        "variance_ratio": ratio,
        "params": {"zero_center": True, "use_highly_variable": False},
    }
    if return_info:
        return scores, components, variance, ratio
    return None


def _community(adata, *, tool: str, resolution: float = 1.0, key_added: str | None = None,
               neighbors_key: str | None = None, random_state=0, directed: bool = True, **kwargs):
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("community clustering requires the scanpy reference backend") from exc
    fn = getattr(sc.tl, tool)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"The `igraph` implementation of leiden clustering")
        return fn(adata, resolution=resolution, key_added=key_added, neighbors_key=neighbors_key,
                  random_state=random_state, directed=directed, **kwargs)


def leiden(adata, resolution: float = 1.0, *, key_added: str = "leiden", neighbors_key: str | None = None,
           random_state=0, directed: bool = True, **kwargs):
    """Run Scanpy's established Leiden optimizer over a mojo-scanpy graph."""
    kwargs.setdefault("flavor", "leidenalg")
    return _community(adata, tool="leiden", resolution=resolution, key_added=key_added,
                      neighbors_key=neighbors_key, random_state=random_state, directed=directed, **kwargs)


def louvain(adata, resolution: float | None = None, *, key_added: str = "louvain", neighbors_key: str | None = None,
            random_state=0, directed: bool = True, **kwargs):
    """Run Scanpy's Louvain backend when its optional package is installed."""
    options = dict(key_added=key_added, neighbors_key=neighbors_key, random_state=random_state, directed=directed)
    if resolution is not None:
        options["resolution"] = resolution
    options.update(kwargs)
    try:
        return _community(adata, tool="louvain", **options)
    except ImportError as exc:
        raise ImportError("tl.louvain needs Scanpy's optional `louvain` backend; use tl.leiden on Python 3.13") from exc
