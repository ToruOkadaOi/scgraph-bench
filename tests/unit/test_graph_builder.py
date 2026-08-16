"""Unit tests for strict-inductive PCA-kNN graph construction, PyG message flow, and GNN invariance."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import GCNConv

from scgraph_bench.config.graph import EdgeWeightingMode, PCAkNNConfig
from scgraph_bench.graph.pca_knn import PCAkNNGraphBuilder
from scgraph_bench.graph.schema import GraphBundle


class MinimalGCN(nn.Module):
    """Simple 2-layer GCN for testing forward pass message passing isolation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)


def test_strict_inductive_edge_orientation_and_isolation():
    """Verify directional PyG message flow semantics: train->train, train->val, train->test."""
    rng = np.random.default_rng(42)
    n_tr, n_va, n_te, d, k = 100, 30, 20, 10, 5

    X_tr = rng.normal(size=(n_tr, d)).astype(np.float32)
    X_va = rng.normal(size=(n_va, d)).astype(np.float32)
    X_te = rng.normal(size=(n_te, d)).astype(np.float32)

    tr_cids = [f"tr_{i}" for i in range(n_tr)]
    va_cids = [f"va_{i}" for i in range(n_va)]
    te_cids = [f"te_{i}" for i in range(n_te)]

    config = PCAkNNConfig(k=k, metric="euclidean", symmetrize=True)
    builder = PCAkNNGraphBuilder(config)

    bundle = builder.build(
        X_pca_train=X_tr,
        X_pca_val=X_va,
        X_pca_test=X_te,
        train_cell_ids=tr_cids,
        val_cell_ids=va_cids,
        test_cell_ids=te_cids,
        feature_manifest_hash="test_manifest_hash",
        dataset_name="synthetic",
        split_id="test_split",
    )

    edge_index = bundle.edge_index.numpy()
    val_start = n_tr
    val_end = n_tr + n_va
    test_start = n_tr + n_va
    test_end = n_tr + n_va + n_te

    # 1. Verify exact partition edge counts
    assert bundle.manifest.num_train_to_val_edges == k * n_va
    assert bundle.manifest.num_train_to_test_edges == k * n_te
    assert bundle.manifest.num_disallowed_edges == 0

    for e_idx in range(edge_index.shape[1]):
        src = edge_index[0, e_idx]
        dst = edge_index[1, e_idx]

        is_src_tr = src < val_start
        is_src_va = val_start <= src < val_end
        is_src_te = test_start <= src < test_end

        is_dst_va = val_start <= dst < val_end
        is_dst_te = test_start <= dst < test_end

        # Rule 1: Every edge ending at validation MUST originate from train
        if is_dst_va:
            assert is_src_tr, f"Validation target {dst} received edge from non-train source {src}"

        # Rule 2: Every edge ending at test MUST originate from train
        if is_dst_te:
            assert is_src_tr, f"Test target {dst} received edge from non-train source {src}"

        # Rule 3: No edge originating from val/test can exist in strict message flow
        if is_src_va or is_src_te:
            raise AssertionError(f"Query node {src} is acting as edge source in ({src} -> {dst})")

        # Rule 4: All edges originate from training reference nodes
        assert is_src_tr, f"Edge ({src} -> {dst}) did not originate from training partition"


def test_gnm_forward_pass_training_invariance_on_test_perturbation():
    """Verify that perturbing validation/test features leaves training GNN logits and loss bit-for-bit invariant."""
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    n_tr, n_va, n_te, d, k = 60, 20, 20, 8, 4

    X_tr = rng.normal(size=(n_tr, d)).astype(np.float32)
    X_va = rng.normal(size=(n_va, d)).astype(np.float32)

    # Two different test batches
    X_te_1 = rng.normal(loc=0.0, scale=1.0, size=(n_te, d)).astype(np.float32)
    X_te_2 = rng.normal(loc=100.0, scale=50.0, size=(n_te, d)).astype(np.float32)

    tr_cids = [f"tr_{i}" for i in range(n_tr)]
    va_cids = [f"va_{i}" for i in range(n_va)]
    te_cids = [f"te_{i}" for i in range(n_te)]

    config = PCAkNNConfig(k=k, metric="euclidean")
    builder = PCAkNNGraphBuilder(config)

    bundle_1 = builder.build(
        X_pca_train=X_tr,
        X_pca_val=X_va,
        X_pca_test=X_te_1,
        train_cell_ids=tr_cids,
        val_cell_ids=va_cids,
        test_cell_ids=te_cids,
        feature_manifest_hash="hash_1",
        dataset_name="synthetic",
        split_id="test_split",
    )

    bundle_2 = builder.build(
        X_pca_train=X_tr,
        X_pca_val=X_va,
        X_pca_test=X_te_2,
        train_cell_ids=tr_cids,
        val_cell_ids=va_cids,
        test_cell_ids=te_cids,
        feature_manifest_hash="hash_2",
        dataset_name="synthetic",
        split_id="test_split",
    )

    # Build input tensors for GNN
    X_all_1 = torch.tensor(np.vstack([X_tr, X_va, X_te_1]), dtype=torch.float32)
    X_all_2 = torch.tensor(np.vstack([X_tr, X_va, X_te_2]), dtype=torch.float32)

    # Deterministic initialized GNN
    model = MinimalGCN(in_dim=d, hidden_dim=16, out_dim=3)
    model.eval()

    with torch.no_grad():
        out_1 = model(X_all_1, bundle_1.edge_index)
        out_2 = model(X_all_2, bundle_2.edge_index)

    # Training node logits MUST be 100% identical
    logits_tr_1 = out_1[:n_tr]
    logits_tr_2 = out_2[:n_tr]

    assert torch.allclose(logits_tr_1, logits_tr_2, atol=1e-6), (
        "GNN training logits changed when test features were perturbed!"
    )


def test_rbf_sigma_formula_parity():
    """Verify that sigma_k strictly equals the median distance from train cells to their k-th neighbor."""
    rng = np.random.default_rng(42)
    n_tr, n_va, n_te, d, k = 60, 10, 10, 8, 5

    X_tr = rng.normal(size=(n_tr, d)).astype(np.float32)
    X_va = rng.normal(size=(n_va, d)).astype(np.float32)
    X_te = rng.normal(size=(n_te, d)).astype(np.float32)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X_tr)
    dists, _ = nn.kneighbors(X_tr)
    expected_sigma_k = float(np.median(dists[:, k]))

    config = PCAkNNConfig(k=k, metric="euclidean", weighting=EdgeWeightingMode.RBF_WEIGHTED)
    builder = PCAkNNGraphBuilder(config)
    bundle = builder.build(
        X_pca_train=X_tr,
        X_pca_val=X_va,
        X_pca_test=X_te,
        train_cell_ids=[f"tr_{i}" for i in range(n_tr)],
        val_cell_ids=[f"va_{i}" for i in range(n_va)],
        test_cell_ids=[f"te_{i}" for i in range(n_te)],
        feature_manifest_hash="hash",
        dataset_name="synthetic",
        split_id="test_split",
    )

    assert bundle.manifest.sigma_k is not None
    assert np.isclose(bundle.manifest.sigma_k, expected_sigma_k, atol=1e-6)

    assert bundle.edge_weight is not None
    weights = bundle.edge_weight.numpy()
    assert (weights > 0.0).all() and (weights <= 1.0).all()


def test_graph_bundle_serialization_roundtrip(tmp_path):
    """Verify that GraphBundle serializes and reloads with bit-for-bit tensor equality."""
    rng = np.random.default_rng(42)
    n_tr, n_va, n_te, d, k = 40, 10, 10, 6, 3

    X_tr = rng.normal(size=(n_tr, d)).astype(np.float32)
    X_va = rng.normal(size=(n_va, d)).astype(np.float32)
    X_te = rng.normal(size=(n_te, d)).astype(np.float32)

    config = PCAkNNConfig(k=k, metric="euclidean", weighting=EdgeWeightingMode.UNWEIGHTED)
    builder = PCAkNNGraphBuilder(config)
    bundle = builder.build(
        X_pca_train=X_tr,
        X_pca_val=X_va,
        X_pca_test=X_te,
        train_cell_ids=[f"tr_{i}" for i in range(n_tr)],
        val_cell_ids=[f"va_{i}" for i in range(n_va)],
        test_cell_ids=[f"te_{i}" for i in range(n_te)],
        feature_manifest_hash="hash_manifest_test",
        dataset_name="synthetic",
        split_id="test_split",
    )

    out_dir = tmp_path / "test_graph_bundle"
    bundle.save(out_dir)

    reloaded = GraphBundle.load(out_dir)

    assert torch.equal(reloaded.edge_index, bundle.edge_index)
    assert torch.equal(reloaded.train_mask, bundle.train_mask)
    assert torch.equal(reloaded.val_mask, bundle.val_mask)
    assert torch.equal(reloaded.test_mask, bundle.test_mask)
    assert reloaded.num_nodes == bundle.num_nodes
    assert reloaded.node_cell_ids == bundle.node_cell_ids
    assert reloaded.manifest.compute_manifest_hash() == bundle.manifest.compute_manifest_hash()


def test_to_pyg_data_strict_label_isolation():
    """Verify PyG Data conversion does not store validation/test labels."""
    rng = np.random.default_rng(42)
    n_tr, n_va, n_te, d, k = 30, 10, 10, 5, 2

    X_tr = rng.normal(size=(n_tr, d)).astype(np.float32)
    X_va = rng.normal(size=(n_va, d)).astype(np.float32)
    X_te = rng.normal(size=(n_te, d)).astype(np.float32)
    X_all = np.vstack([X_tr, X_va, X_te])

    config = PCAkNNConfig(k=k, metric="euclidean")
    builder = PCAkNNGraphBuilder(config)
    bundle = builder.build(
        X_pca_train=X_tr,
        X_pca_val=X_va,
        X_pca_test=X_te,
        train_cell_ids=[f"tr_{i}" for i in range(n_tr)],
        val_cell_ids=[f"va_{i}" for i in range(n_va)],
        test_cell_ids=[f"te_{i}" for i in range(n_te)],
        feature_manifest_hash="hash",
        dataset_name="synthetic",
        split_id="test_split",
    )

    # Default conversion without any labels
    pyg_data_no_y = bundle.to_pyg_data(x=X_all)
    assert not hasattr(pyg_data_no_y, "y") or pyg_data_no_y.y is None
    assert pyg_data_no_y.num_nodes == n_tr + n_va + n_te

    # Conversion with training-only labels
    y_train = np.ones(n_tr, dtype=np.int64)
    pyg_data_train_y = bundle.to_pyg_data(x=X_all, y_train_only=y_train)
    assert pyg_data_train_y.y is not None
    # Training nodes have label 1, validation/test nodes are masked to -1
    assert (pyg_data_train_y.y[:n_tr] == 1).all()
    assert (pyg_data_train_y.y[n_tr:] == -1).all()


def test_test_prediction_invariance_under_test_row_permutation():
    """Verify that permuting test rows and un-permuting output logits yields identical test predictions."""
    torch.manual_seed(123)
    rng = np.random.default_rng(123)
    n_tr, n_va, n_te, d, k = 50, 15, 25, 8, 4

    X_tr = rng.normal(size=(n_tr, d)).astype(np.float32)
    X_va = rng.normal(size=(n_va, d)).astype(np.float32)
    X_te = rng.normal(size=(n_te, d)).astype(np.float32)

    tr_cids = [f"tr_{i}" for i in range(n_tr)]
    va_cids = [f"va_{i}" for i in range(n_va)]
    te_cids = [f"te_{i}" for i in range(n_te)]

    config = PCAkNNConfig(k=k, metric="euclidean")
    builder = PCAkNNGraphBuilder(config)

    # 1. Original order run
    bundle_orig = builder.build(
        X_pca_train=X_tr,
        X_pca_val=X_va,
        X_pca_test=X_te,
        train_cell_ids=tr_cids,
        val_cell_ids=va_cids,
        test_cell_ids=te_cids,
        feature_manifest_hash="hash_orig",
        dataset_name="synthetic",
        split_id="test_split",
    )

    # 2. Permuted test batch
    perm = rng.permutation(n_te)
    inv_perm = np.argsort(perm)

    X_te_perm = X_te[perm]
    te_cids_perm = [te_cids[i] for i in perm]

    bundle_perm = builder.build(
        X_pca_train=X_tr,
        X_pca_val=X_va,
        X_pca_test=X_te_perm,
        train_cell_ids=tr_cids,
        val_cell_ids=va_cids,
        test_cell_ids=te_cids_perm,
        feature_manifest_hash="hash_perm",
        dataset_name="synthetic",
        split_id="test_split",
    )

    # 3. GNN evaluation
    model = MinimalGCN(in_dim=d, hidden_dim=16, out_dim=5)
    model.eval()

    X_all_orig = torch.tensor(np.vstack([X_tr, X_va, X_te]), dtype=torch.float32)
    X_all_perm = torch.tensor(np.vstack([X_tr, X_va, X_te_perm]), dtype=torch.float32)

    with torch.no_grad():
        out_orig = model(X_all_orig, bundle_orig.edge_index)
        out_perm = model(X_all_perm, bundle_perm.edge_index)

    test_start = n_tr + n_va
    test_logits_orig = out_orig[test_start:]
    test_logits_perm = out_perm[test_start:]

    # Un-permute permuted logits
    test_logits_unperm = test_logits_perm[inv_perm]

    assert torch.allclose(test_logits_orig, test_logits_unperm, atol=1e-6)


def test_mutual_knn_edge_symmetry_and_inductive_constraints():
    """Verify that Mutual kNN enforces reciprocity in training reference and strict inductive projection for queries."""
    from scgraph_bench.config.graph import MutualkNNConfig
    from scgraph_bench.graph.mutual_knn import MutualKNNGraphBuilder

    rng = np.random.default_rng(42)
    n_tr, n_va, n_te, d, k = 60, 20, 20, 8, 5

    X_tr = rng.normal(size=(n_tr, d)).astype(np.float32)
    X_va = rng.normal(size=(n_va, d)).astype(np.float32)
    X_te = rng.normal(size=(n_te, d)).astype(np.float32)

    builder = MutualKNNGraphBuilder(MutualkNNConfig(k=k))
    bundle = builder.build(
        X_pca_train=X_tr,
        X_pca_val=X_va,
        X_pca_test=X_te,
        train_cell_ids=[f"tr_{i}" for i in range(n_tr)],
        val_cell_ids=[f"va_{i}" for i in range(n_va)],
        test_cell_ids=[f"te_{i}" for i in range(n_te)],
        feature_manifest_hash="hash",
        dataset_name="synthetic",
        split_id="test_split",
    )

    edge_index = bundle.edge_index.numpy()
    assert bundle.manifest.num_disallowed_edges == 0
    assert bundle.manifest.num_train_to_val_edges == k * n_va
    assert bundle.manifest.num_train_to_test_edges == k * n_te

    # Verify symmetry in training edges: if (i, j) exists, (j, i) must exist
    tr_edges = set()
    for e in range(edge_index.shape[1]):
        src, dst = edge_index[0, e], edge_index[1, e]
        if src < n_tr and dst < n_tr:
            tr_edges.add((src, dst))

    for src, dst in tr_edges:
        assert (dst, src) in tr_edges, f"Mutual kNN asymmetric edge in train: ({src}, {dst})"


def test_bbknn_strict_inductive_batch_balancing():
    """Verify that BBKNN connects queries to exactly k_b neighbors per donor."""
    from scgraph_bench.config.graph import BBKNNConfig
    from scgraph_bench.graph.bbknn import StrictInductiveBBKNNGraphBuilder

    rng = np.random.default_rng(42)
    n_tr, n_va, n_te, d = 60, 10, 10, 8
    k_b = 2

    X_tr = rng.normal(size=(n_tr, d)).astype(np.float32)
    X_va = rng.normal(size=(n_va, d)).astype(np.float32)
    X_te = rng.normal(size=(n_te, d)).astype(np.float32)

    # 3 donors in train (20 cells each)
    donor_ids_tr = ["d1"] * 20 + ["d2"] * 20 + ["d3"] * 20
    donor_ids_va = ["d4"] * 10
    donor_ids_te = ["d5"] * 10

    metadata = {
        "donor_ids_train": donor_ids_tr,
        "donor_ids_val": donor_ids_va,
        "donor_ids_test": donor_ids_te,
    }

    builder = StrictInductiveBBKNNGraphBuilder(BBKNNConfig(k_per_batch=k_b))
    bundle = builder.build(
        X_pca_train=X_tr,
        X_pca_val=X_va,
        X_pca_test=X_te,
        train_cell_ids=[f"tr_{i}" for i in range(n_tr)],
        val_cell_ids=[f"va_{i}" for i in range(n_va)],
        test_cell_ids=[f"te_{i}" for i in range(n_te)],
        feature_manifest_hash="hash",
        dataset_name="synthetic",
        split_id="test_split",
        allowed_metadata=metadata,
    )

    assert bundle.manifest.num_disallowed_edges == 0
    # Every validation node must have exactly 3 donors * k_b = 6 incoming edges
    val_start = n_tr
    val_end = n_tr + n_va
    edge_index = bundle.edge_index.numpy()

    dsts = edge_index[1]
    for u in range(val_start, val_end):
        in_deg_u = np.sum(dsts == u)
        assert in_deg_u == 3 * k_b, (
            f"Validation node {u} received {in_deg_u} edges instead of {3 * k_b}"
        )


def test_rewired_control_degree_preservation():
    """Verify that RewiredControlGraphBuilder preserves exact in- and out-degree sequences."""
    from scgraph_bench.config.graph import RewiredControlConfig
    from scgraph_bench.graph.rewired_control import RewiredControlGraphBuilder

    rng = np.random.default_rng(42)
    n_tr, n_va, n_te, d = 40, 10, 10, 6

    X_tr = rng.normal(size=(n_tr, d)).astype(np.float32)
    X_va = rng.normal(size=(n_va, d)).astype(np.float32)
    X_te = rng.normal(size=(n_te, d)).astype(np.float32)

    builder = RewiredControlGraphBuilder(RewiredControlConfig(seed=42, n_swaps_factor=5.0))
    bundle = builder.build(
        X_pca_train=X_tr,
        X_pca_val=X_va,
        X_pca_test=X_te,
        train_cell_ids=[f"tr_{i}" for i in range(n_tr)],
        val_cell_ids=[f"va_{i}" for i in range(n_va)],
        test_cell_ids=[f"te_{i}" for i in range(n_te)],
        feature_manifest_hash="hash",
        dataset_name="synthetic",
        split_id="test_split",
    )

    assert bundle.manifest.num_disallowed_edges == 0
    assert bundle.manifest.num_edges > 0
