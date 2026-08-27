import numpy as np
import pytest
import warnings
from types import SimpleNamespace

ad = pytest.importorskip("anndata")
sc = pytest.importorskip("scanpy")

import mojo_scanpy as msc
from mojo_scanpy._lib import addr, lib
from mojo_scanpy._pca import pca_array


def test_pca_array_matches_scanpy_scores_and_variance():
    rng = np.random.default_rng(41)
    data = rng.normal(size=(80, 9))
    ours, components, variance, ratio = pca_array(data, 5)
    reference = ad.AnnData(data)
    sc.tl.pca(reference, n_comps=5, svd_solver="full")
    assert np.allclose(variance, reference.uns["pca"]["variance"], rtol=2e-6, atol=1e-9)
    assert np.allclose(ratio, reference.uns["pca"]["variance_ratio"], rtol=2e-6, atol=1e-9)
    assert np.allclose(np.abs(ours), np.abs(reference.obsm["X_pca"]), rtol=2e-5, atol=2e-6)
    assert components.shape == (5, 9)


def test_tl_pca_populates_scanpy_slots():
    rng = np.random.default_rng(2)
    ours = ad.AnnData(rng.normal(size=(50, 7)))
    reference = ours.copy()
    msc.tl.pca(ours, n_comps=4)
    sc.tl.pca(reference, n_comps=4, svd_solver="full")
    assert set(("X_pca",)) <= set(ours.obsm)
    assert ours.varm["PCs"].shape == (7, 4)
    assert ours.obsm["X_pca"].dtype == np.float64
    assert np.allclose(ours.uns["pca"]["variance"], reference.uns["pca"]["variance"], rtol=2e-6)


def test_neighbors_match_scanpy_exact_neighbor_distances():
    rng = np.random.default_rng(9)
    data = rng.normal(size=(64, 6))
    ours = ad.AnnData(data)
    reference = ad.AnnData(data.copy())
    msc.pp.neighbors(ours, n_neighbors=8)
    sc.pp.neighbors(reference, n_neighbors=8, transformer="sklearn")
    a = ours.obsp["distances"].toarray()
    b = reference.obsp["distances"].toarray()
    assert np.allclose(a, b, atol=1e-10)
    assert np.allclose(ours.obsp["connectivities"].toarray(), reference.obsp["connectivities"].toarray(), atol=1e-6)


def test_knn_simd_tail_matches_scalar_reference():
    values = np.ascontiguousarray(np.random.default_rng(21).normal(size=(19, 33)))
    indices = np.empty((19, 4), dtype=np.int64)
    squared = np.empty((19, 4), dtype=np.float64)
    lib().msp_knn(addr(values), addr(values), addr(indices), addr(squared), 19, 33, 19, 4, 1)
    reference = ((values[:, None] - values[None, :]) ** 2).sum(axis=2)
    np.fill_diagonal(reference, np.inf)
    expected_indices = np.argsort(reference, axis=1)[:, :4]
    expected_squared = np.take_along_axis(reference, expected_indices, axis=1)
    assert np.array_equal(indices, expected_indices)
    assert np.allclose(squared, expected_squared, rtol=1e-12, atol=1e-12)


def test_knn_parallel_threshold_with_simd_tail_matches_scalar_reference():
    values = np.ascontiguousarray(np.random.default_rng(22).normal(size=(180, 33)))
    indices = np.empty((180, 5), dtype=np.int64)
    squared = np.empty((180, 5), dtype=np.float64)
    lib().msp_knn(addr(values), addr(values), addr(indices), addr(squared), 180, 33, 180, 5, 1)
    reference = ((values[:, None] - values[None, :]) ** 2).sum(axis=2)
    np.fill_diagonal(reference, np.inf)
    expected_indices = np.argsort(reference, axis=1)[:, :5]
    expected_squared = np.take_along_axis(reference, expected_indices, axis=1)
    assert np.array_equal(indices, expected_indices)
    assert np.allclose(squared, expected_squared, rtol=1e-12, atol=1e-12)


def test_neighbors_copy_and_custom_key():
    data = np.arange(40, dtype=float).reshape(10, 4)
    source = ad.AnnData(data)
    copied = msc.pp.neighbors(source, n_neighbors=3, key_added="rna", copy=True)
    assert "rna_distances" in copied.obsp and "rna_distances" not in source.obsp


def test_neighbors_gauss_populates_scanpy_graph_slots():
    adata = ad.AnnData(np.random.default_rng(6).normal(size=(12, 3)))
    msc.pp.neighbors(adata, n_neighbors=4, method="gauss")
    assert adata.obsp["distances"].shape == (12, 12)
    assert adata.obsp["connectivities"].shape == (12, 12)


def test_leiden_delegates_to_scanpy_backend():
    rng = np.random.default_rng(11)
    data = np.vstack([rng.normal(-3, 0.2, (12, 4)), rng.normal(3, 0.2, (12, 4))])
    adata = ad.AnnData(data)
    msc.pp.neighbors(adata, n_neighbors=5)
    reference = adata.copy()
    msc.tl.leiden(adata, random_state=0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"The `igraph` implementation of leiden clustering")
        sc.tl.leiden(reference, random_state=0, flavor="leidenalg")
    assert np.array_equal(adata.obs["leiden"].to_numpy(), reference.obs["leiden"].to_numpy())


@pytest.mark.parametrize("bad", [np.array([[1 + 2j, 3]]), np.array([[1.0, np.nan]])])
def test_pca_rejects_inputs_that_are_unsafe_or_lossy_at_the_ffi_boundary(bad):
    with pytest.raises((TypeError, ValueError)):
        pca_array(bad, 1)


def test_neighbors_rejects_non_matrix_and_fractional_neighbor_count():
    with pytest.raises(ValueError, match="two-dimensional"):
        msc.pp.neighbors(SimpleNamespace(X=np.ones(4), obsm={}), n_neighbors=2)
    with pytest.raises(TypeError, match="integer"):
        msc.pp.neighbors(ad.AnnData(np.ones((4, 2))), n_neighbors=2.5)
