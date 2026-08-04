# mojo-scanpy

`mojo-scanpy` is a focused, standalone Mojo port of the dense, CPU-bound part
of a single-cell analysis workflow. It provides exact Euclidean neighbours and
centered PCA through a compiled Mojo library, with an AnnData-facing Python API
that follows the covered Scanpy names.

## Covered subset

`mojo_scanpy.pp.neighbors` computes an exact dense Euclidean k-neighbour graph
and writes Scanpy-compatible `obsp`/`uns` slots. `mojo_scanpy.tl.pca` performs
centered dense PCA and writes float64 `obsm['X_pca']`, `varm['PCs']`, and `uns['pca']`.
`mojo_scanpy.tl.leiden` preserves Scanpy's graph clustering API by calling the
established Leiden optimizer on the stored graph.

Sparse input, approximate neighbour search, incremental/randomized PCA, Louvain,
and the rest of Scanpy are out of scope. UMAP/gaussian graph calibration uses Scanpy's
well-tested sparse post-processing after Mojo has generated the exact kNN lists.
Community optimization is intentionally not reimplemented: its correctness
depends on the mature igraph/Leiden implementations, while distance scans and
the dense covariance/eigensolve are the compute-bound kernels ported to Mojo.

## Install and use

```bash
pixi install
pixi run build
```

```python
import anndata as ad
import numpy as np
import mojo_scanpy as msc

adata = ad.AnnData(np.random.default_rng(0).normal(size=(100, 20)))
msc.tl.pca(adata, n_comps=10)
msc.pp.neighbors(adata, n_neighbors=12, use_rep="X_pca")
msc.tl.leiden(adata, random_state=0)
print(adata.obsm["X_pca"].shape, adata.obs["leiden"].nunique())
```

Run the complete checks with `pixi run build && pixi run test && pixi run bench`;
the benchmark uses a machine-wide lock so concurrent jobs do not alter measurements.

## How it works

The Python package validates finite, real dense inputs and converts them to C-contiguous `float64` NumPy arrays
and passes their integer data addresses to a single Mojo shared library through
`ctypes`. Mojo owns no Python memory or allocations. Matrices are row-major:
the neighbour kernel scans each query against all observations and keeps a
sorted k-element frontier; PCA forms the sample covariance and diagonalizes it
with cyclic Jacobi rotations.

## Benchmark

Run `pixi run bench` to regenerate this table on the recorded machine. Times
are best of three and include AnnData result construction. The values below are
measured after the test suite in this checkout.

| case | mojo-scanpy | scanpy | scanpy/mojo | result |
|---|---:|---:|---:|---|
| PCA (1,800 x 28, 15 PCs) | 3.7 ms | 4.7 ms | 1.28x | faster |
| Exact neighbors (1,800 x 28, k=15) | 58.6 ms | 146.7 ms | 2.50x | faster |

Measured on `x86_64`, Python 3.13.14. The benchmark script prints the processor
and Python version with every run. Neighbour search uses SIMD distance reductions.

No GPU path is included. This project targets dense CPU kernels.
