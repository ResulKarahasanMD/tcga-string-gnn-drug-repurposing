"""Tests for PyG graph assembly: bidirectional edges, edge-index bounds, splits (Section 32)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from repurpose_gnn.graph_builder import (
    _validate_data,
    build_bidirectional_edge_index,
    make_splits,
)
from torch_geometric.data import Data


def test_build_bidirectional_edge_index():
    edges = pd.DataFrame(
        {"protein1": ["A", "B"], "protein2": ["B", "C"], "edge_weight": [0.9, 0.7]}
    )
    node_index = {"A": 0, "B": 1, "C": 2}
    ei, ea = build_bidirectional_edge_index(edges, node_index)
    assert ei.shape == (2, 4)  # 2 undirected edges -> 4 directed
    assert ei.dtype == torch.long
    # every edge appears in both directions
    pairs = set(map(tuple, ei.t().tolist()))
    assert (0, 1) in pairs and (1, 0) in pairs
    assert (1, 2) in pairs and (2, 1) in pairs
    # edge_attr duplicated for both directions
    assert ea.shape == (4,)
    assert torch.allclose(ea.sort().values, torch.tensor([0.7, 0.7, 0.9, 0.9]))


def test_edge_index_bounds_respected():
    edges = pd.DataFrame({"protein1": ["A"], "protein2": ["B"], "edge_weight": [0.8]})
    ei, _ = build_bidirectional_edge_index(edges, {"A": 0, "B": 1})
    assert int(ei.max()) < 2
    assert int(ei.min()) >= 0


def test_build_edge_index_rejects_unknown_node():
    edges = pd.DataFrame({"protein1": ["A"], "protein2": ["Z"], "edge_weight": [0.8]})
    with pytest.raises(ValueError):
        build_bidirectional_edge_index(edges, {"A": 0, "B": 1})


def test_make_splits_disjoint_and_cover_labeled():
    rng = np.random.default_rng(0)
    y = np.array([-1] * 100)
    labeled = rng.choice(100, size=40, replace=False)
    y[labeled[:20]] = 1
    y[labeled[20:]] = 0
    degrees = rng.integers(1, 50, size=100)
    masks = make_splits(y, degrees, strategy="stratified_random",
                        val_frac=0.2, test_frac=0.2, seed=1)
    tr, va, te = masks["train"], masks["val"], masks["test"]
    # disjoint
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    # only labeled nodes are in any split, and all labeled are covered
    union = tr | va | te
    assert set(np.where(union)[0]) == set(np.where(y >= 0)[0])
    # unlabeled never in a split
    assert not union[y < 0].any()


def test_make_splits_degree_aware_runs():
    rng = np.random.default_rng(0)
    y = np.full(200, -1)
    y[:50] = 1
    y[50:120] = 0
    degrees = rng.integers(1, 100, size=200)
    masks = make_splits(y, degrees, strategy="degree_aware",
                        val_frac=0.15, test_frac=0.15, seed=2)
    assert masks["train"].sum() + masks["val"].sum() + masks["test"].sum() == 120


def test_validate_data_catches_out_of_bounds():
    x = torch.zeros((3, 2))
    bad_ei = torch.tensor([[0, 5], [1, 2]], dtype=torch.long)  # 5 >= n_nodes
    data = Data(x=x, edge_index=bad_ei, y=torch.zeros(3, dtype=torch.long))
    with pytest.raises(AssertionError):
        _validate_data(data, 3)


def test_validate_data_catches_nan():
    x = torch.tensor([[0.0, float("nan")], [1.0, 2.0], [3.0, 4.0]])
    ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    data = Data(x=x, edge_index=ei, y=torch.zeros(3, dtype=torch.long))
    with pytest.raises(AssertionError):
        _validate_data(data, 3)
