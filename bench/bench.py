"""Measured mojo-scanpy benchmarks against Scanpy's exact CPU paths."""

from __future__ import annotations

import math
import platform
import time

import anndata as ad
import numpy as np
import scanpy as sc

import mojo_scanpy as msc


def timed(fn, repeat=3):
    values = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        values.append(time.perf_counter() - start)
    return min(values)


def main():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(1800, 28))
    print(f"machine: {platform.processor() or platform.machine()}, Python {platform.python_version()}")
    print("| case | mojo-scanpy | scanpy | scanpy/mojo | result |")
    print("|---|---:|---:|---:|---|")
    cases = [
        ("PCA (1,800 x 28, 15 PCs)", lambda: msc.tl.pca(ad.AnnData(data), n_comps=15),
         lambda: sc.tl.pca(ad.AnnData(data), n_comps=15, svd_solver="full")),
        ("Exact neighbors (1,800 x 28, k=15)", lambda: msc.pp.neighbors(ad.AnnData(data), n_neighbors=15),
         lambda: sc.pp.neighbors(ad.AnnData(data), n_neighbors=15, transformer="sklearn")),
    ]
    for name, ours, theirs in cases:
        ours(); theirs()
        a, b = timed(ours), timed(theirs)
        print(f"| {name} | {a * 1e3:.1f} ms | {b * 1e3:.1f} ms | {b / a:.2f}x | {'faster' if a < b else 'slower'} |")


if __name__ == "__main__":
    main()
