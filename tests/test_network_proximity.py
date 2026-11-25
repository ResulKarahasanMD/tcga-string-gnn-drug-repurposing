"""Network-proximity tests: distances, disconnected graphs, degree bins, null, BH (Section 32)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from repurpose_gnn.network_proximity import (
    benjamini_hochberg,
    build_degree_bins,
    build_sparse_graph,
    degree_matched_sample,
    distance_to_module,
    proximity_zscore,
)


def _line_graph_edges(n: int, weight: float = 1.0):
    """Undirected path 0-1-2-...-(n-1) as a bidirectional edge_index + weights."""
    src = list(range(n - 1))
    dst = list(range(1, n))
    ei = np.array([src + dst, dst + src])
    ew = np.full(ei.shape[1], weight)
    return ei, ew


def test_build_sparse_graph_transforms():
    ei, ew = _line_graph_edges(3, weight=1.0)
    g = build_sparse_graph(ei, ew, 3, transform="neglog")
    # conf=1 -> distance ~0 (clamped to 1e-9)
    assert g.max() < 1e-6
    g2 = build_sparse_graph(ei, np.full(ei.shape[1], 0.5), 3, transform="inverse")
    assert np.isclose(g2.data.max(), 2.0)  # 1/0.5


def test_distance_to_module_line():
    ei, ew = _line_graph_edges(5, weight=0.5)  # neglog dist per edge = -log(0.5)=0.693
    g = build_sparse_graph(ei, ew, 5, transform="neglog")
    d = distance_to_module(g, [0])
    step = -np.log(0.5)
    assert np.isclose(d[0], 0.0, atol=1e-6)
    assert np.isclose(d[4], 4 * step, atol=1e-6)


def test_disconnected_graph_unreachable_is_inf():
    # two components: 0-1 and 2-3
    ei = np.array([[0, 1, 2, 3], [1, 0, 3, 2]])
    ew = np.ones(4)
    g = build_sparse_graph(ei, ew, 4, transform="inverse")
    d = distance_to_module(g, [0])
    assert np.isfinite(d[1])
    assert not np.isfinite(d[2])  # different component -> inf


def test_proximity_zscore_excludes_unreachable_and_handles_zero_std():
    # module reachable only from node 1; node 3 unreachable
    dist = np.array([0.0, 1.0, 2.0, np.inf])
    degrees = np.array([1, 1, 1, 1])
    giant = np.array([True, True, True, False])
    bins = build_degree_bins(degrees, giant, n_bins=2)
    # targets include an unreachable one -> n_reachable counts only finite
    res = proximity_zscore(dist, [1, 3], bins, n_random=50, rng=np.random.default_rng(0))
    assert res["n_targets"] == 2 and res["n_reachable"] == 1
    assert np.isfinite(res["zscore"])  # zero-std guarded -> not nan/inf


def test_all_unreachable_returns_nan():
    dist = np.array([0.0, np.inf, np.inf])
    bins = build_degree_bins(np.array([1, 1, 1]), np.array([True, True, True]), n_bins=2)
    res = proximity_zscore(dist, [1, 2], bins, n_random=10)
    assert np.isnan(res["d_obs"]) and res["n_reachable"] == 0


def test_degree_matched_sample_respects_bins():
    degrees = np.array([1, 1, 1, 100, 100, 100])
    giant = np.ones(6, dtype=bool)
    bins = build_degree_bins(degrees, giant, n_bins=2)
    rng = np.random.default_rng(0)
    # sampling for a low-degree target should draw low-degree nodes
    sample = degree_matched_sample([0], bins, 100, rng)
    drawn_degrees = degrees[sample.ravel()]
    assert (drawn_degrees < 50).mean() > 0.9


def test_benjamini_hochberg_monotone():
    p = np.array([0.001, 0.01, 0.5, np.nan])
    q = benjamini_hochberg(p)
    assert np.isnan(q[3])
    assert q[0] <= q[1] <= q[2]
    assert (q[np.isfinite(q)] <= 1).all()
