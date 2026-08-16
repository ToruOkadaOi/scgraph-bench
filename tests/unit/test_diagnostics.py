"""Unit tests for Phase 7 Graph Diagnostics Suite."""

import inspect

import numpy as np
import torch

from scgraph_bench.config.graph import PCAkNNConfig
from scgraph_bench.diagnostics.homophily import compute_label_diagnostics
from scgraph_bench.diagnostics.metadata_mixing import compute_metadata_diagnostics
from scgraph_bench.diagnostics.runner import run_graph_diagnostics
from scgraph_bench.graph.pca_knn import PCAkNNGraphBuilder
from scgraph_bench.graph.schema import GraphBundle, GraphManifest


def _create_mock_graph_bundle(
    edge_index_np: np.ndarray,
    n_tr: int,
    n_va: int,
    n_te: int,
    graph_name: str = "mock_graph",
) -> GraphBundle:
    n_total = n_tr + n_va + n_te
    edge_index = torch.from_numpy(edge_index_np).to(torch.long)

    train_mask = torch.zeros(n_total, dtype=torch.bool)
    val_mask = torch.zeros(n_total, dtype=torch.bool)
    test_mask = torch.zeros(n_total, dtype=torch.bool)

    train_mask[:n_tr] = True
    val_mask[n_tr : n_tr + n_va] = True
    test_mask[n_tr + n_va :] = True

    node_cell_ids = [f"cell_{i}" for i in range(n_total)]

    manifest = GraphManifest(
        graph_name=graph_name,
        builder_type="mock",
        dataset_name="synthetic",
        split_id="mock_split",
        k=2,
        metric="euclidean",
        weighting="unweighted",
        num_nodes=n_total,
        num_edges=edge_index_np.shape[1],
        num_train_nodes=n_tr,
        num_val_nodes=n_va,
        num_test_nodes=n_te,
        num_train_train_edges=0,
        num_train_to_val_edges=0,
        num_train_to_test_edges=0,
        num_disallowed_edges=0,
        edge_index_hash="mock_hash",
        feature_manifest_hash="mock_feat_hash",
    )

    return GraphBundle(
        edge_index=edge_index,
        num_nodes=n_total,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        node_cell_ids=node_cell_ids,
        manifest=manifest,
    )


def test_pure_same_label_graph_homophily():
    """Verify that a perfectly homogeneous graph yields homophily and purity 1.0."""
    # 4 train nodes, complete graph (all directed pairs)
    edges = []
    for i in range(4):
        for j in range(4):
            if i != j:
                edges.append((i, j))
    edge_index = np.array(edges, dtype=np.int64).T

    bundle = _create_mock_graph_bundle(edge_index, n_tr=4, n_va=0, n_te=0)
    y = np.array([0, 0, 0, 0], dtype=np.int64)

    diag = compute_label_diagnostics(bundle, y_all=y, label_names=["ClassA"])

    assert diag.overall_edge_homophily == 1.0
    assert diag.overall_node_homophily == 1.0
    assert diag.train_train_edge_homophily == 1.0
    assert diag.macro_average_class_purity == 1.0
    assert diag.per_class_neighborhood_purity["ClassA"] == 1.0


def test_known_mixed_label_graph_homophily():
    """Verify exact expected homophily on a calibrated 50% same-label graph."""
    # 4 nodes: nodes 0,1 are class A (0), nodes 2,3 are class B (1)
    # Directed edges:
    # 0->1 (same), 1->0 (same)
    # 2->3 (same), 3->2 (same)
    # 0->2 (cross), 2->0 (cross)
    # 1->3 (cross), 3->1 (cross)
    # Total: 4 same, 4 cross = 8 edges -> 50% homophily
    edges = [
        (0, 1),
        (1, 0),
        (2, 3),
        (3, 2),
        (0, 2),
        (2, 0),
        (1, 3),
        (3, 1),
    ]
    edge_index = np.array(edges, dtype=np.int64).T
    bundle = _create_mock_graph_bundle(edge_index, n_tr=4, n_va=0, n_te=0)
    y = np.array([0, 0, 1, 1], dtype=np.int64)

    diag = compute_label_diagnostics(bundle, y_all=y, label_names=["ClassA", "ClassB"])

    assert np.isclose(diag.overall_edge_homophily, 0.5000, atol=1e-5)
    assert np.isclose(diag.overall_node_homophily, 0.5000, atol=1e-5)
    assert np.isclose(diag.train_train_edge_homophily, 0.5000, atol=1e-5)
    assert np.isclose(diag.per_class_neighborhood_purity["ClassA"], 0.5000, atol=1e-5)
    assert np.isclose(diag.per_class_neighborhood_purity["ClassB"], 0.5000, atol=1e-5)


def test_diagnostics_leaves_graph_tensors_and_hashes_unaltered():
    """Verify that running diagnostics is strictly read-only with zero mutation."""
    rng = np.random.default_rng(42)
    n_tr, n_va, n_te, d = 40, 10, 10, 6

    X_tr = rng.normal(size=(n_tr, d)).astype(np.float32)
    X_va = rng.normal(size=(n_va, d)).astype(np.float32)
    X_te = rng.normal(size=(n_te, d)).astype(np.float32)

    builder = PCAkNNGraphBuilder(PCAkNNConfig(k=3, metric="euclidean"))
    bundle = builder.build(
        X_pca_train=X_tr,
        X_pca_val=X_va,
        X_pca_test=X_te,
        train_cell_ids=[f"tr_{i}" for i in range(n_tr)],
        val_cell_ids=[f"va_{i}" for i in range(n_va)],
        test_cell_ids=[f"te_{i}" for i in range(n_te)],
        feature_manifest_hash="feat_hash",
        dataset_name="synthetic",
        split_id="test_split",
    )

    edge_index_before = bundle.edge_index.clone()
    manifest_hash_before = bundle.manifest.compute_manifest_hash()

    y_all = rng.integers(0, 3, size=bundle.num_nodes)
    donor_ids = [f"donor_{i % 3}" for i in range(bundle.num_nodes)]
    site_ids = [f"site_{i % 2}" for i in range(bundle.num_nodes)]

    report = run_graph_diagnostics(
        graph_bundle=bundle,
        y_all=y_all,
        donor_ids=donor_ids,
        site_ids=site_ids,
    )

    assert torch.equal(bundle.edge_index, edge_index_before)
    assert bundle.manifest.compute_manifest_hash() == manifest_hash_before
    assert report.edge_index_hash == bundle.manifest.edge_index_hash


def test_label_isolation_architecture_guard():
    """Verify that graph builder modules do not import label diagnostics or reference labels."""
    import scgraph_bench.graph.base as graph_base
    import scgraph_bench.graph.pca_knn as graph_pca_knn

    for mod in [graph_base, graph_pca_knn]:
        source = inspect.getsource(mod)
        assert "compute_label_diagnostics" not in source
        assert "LabelDiagnostics" not in source


def test_donor_and_site_mixing_properties():
    """Verify donor entropy and intra-donor fractions on controlled configurations."""
    # 2 train nodes from donor A, 2 train nodes from donor B
    # Node 0 connected to 0 and 1 (intra-donor only) -> entropy 0
    # Node 2 connected to 0 and 2 (50% A, 50% B) -> entropy 1.0 bit
    edges = [
        (0, 0),
        (1, 0),  # target 0 gets sources 0, 1 (both donor A)
        (0, 2),
        (2, 2),  # target 2 gets sources 0 (donor A), 2 (donor B)
    ]
    edge_index = np.array(edges, dtype=np.int64).T
    bundle = _create_mock_graph_bundle(edge_index, n_tr=4, n_va=0, n_te=0)

    donor_ids = ["DonorA", "DonorA", "DonorB", "DonorB"]
    site_ids = ["Site1", "Site1", "Site2", "Site2"]

    diag = compute_metadata_diagnostics(bundle, donor_ids=donor_ids, site_ids=site_ids)

    # 3 intra-donor edges (0->0, 1->0, 2->2) out of 4 edges = 0.75
    assert np.isclose(diag.train_intra_donor_edge_fraction, 0.75, atol=1e-5)
    # Node 0 entropy = 0.0, Node 2 entropy = 1.0 bit
    assert diag.mean_train_donor_entropy > 0.0
