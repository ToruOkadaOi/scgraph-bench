"""Unit tests for configuration parsing, validation, and serialization."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from scgraph_bench.config import (
    BenchmarkConfig,
    DatasetConfig,
    GraphBuilderType,
    GraphConfig,
    HVGFlavor,
    LogisticRegressionConfig,
    ModelConfig,
    ModelType,
    PCAkNNConfig,
    PreprocessingConfig,
    RareClassAction,
    RareClassConfig,
    SplitConfig,
    SplitType,
)


def test_preprocessing_config_defaults_and_explicit_hvg():
    """Verify default preprocessing configuration specifies explicit seurat HVG flavor."""
    cfg = PreprocessingConfig()
    assert cfg.hvg_flavor == HVGFlavor.SEURAT
    assert cfg.n_top_genes == 2000
    assert cfg.n_comps == 50
    assert cfg.target_sum == 10000.0
    assert cfg.scale_data is True


def test_split_config_validation():
    """Verify split fraction validation summing to 1.0."""
    valid_cfg = SplitConfig(train_fraction=0.7, val_fraction=0.15, test_fraction=0.15)
    assert valid_cfg.split_type == SplitType.DONOR_HELD_OUT

    with pytest.raises(ValidationError):
        # Invalid fractions not summing to 1.0
        SplitConfig(train_fraction=0.6, val_fraction=0.3, test_fraction=0.3)


def test_rare_class_config():
    """Verify configurable rare-class thresholds and actions."""
    cfg = RareClassConfig(
        min_train_cells_per_class=15,
        min_test_cells_per_class=8,
        action=RareClassAction.EXCLUDE,
    )
    assert cfg.min_train_cells_per_class == 15
    assert cfg.action == RareClassAction.EXCLUDE


def test_graph_config_modes():
    """Verify graph config initialization for different builders."""
    pca_cfg = GraphConfig(
        builder_type=GraphBuilderType.PCA_KNN,
        pca_knn=PCAkNNConfig(k=10, symmetrize=True),
    )
    assert pca_cfg.pca_knn is not None
    assert pca_cfg.pca_knn.k == 10


def test_yaml_serialization_and_deserialization(tmp_path):
    """Verify round-trip YAML export and load for master BenchmarkConfig."""
    bench_cfg = BenchmarkConfig(
        experiment_id="test_exp",
        seed=100,
        dataset=DatasetConfig(name="test_data"),
        split=SplitConfig(split_id="test_split_01"),
        preprocessing=PreprocessingConfig(n_top_genes=1500),
        graph=GraphConfig(builder_type=GraphBuilderType.PCA_KNN),
        model=ModelConfig(
            model_type=ModelType.LOGISTIC_REGRESSION,
            logistic_regression=LogisticRegressionConfig(max_iter=500),
        ),
    )

    yaml_file = tmp_path / "benchmark.yaml"
    bench_cfg.to_yaml(yaml_file)

    loaded_cfg = BenchmarkConfig.from_yaml(yaml_file)
    assert loaded_cfg.experiment_id == "test_exp"
    assert loaded_cfg.preprocessing.n_top_genes == 1500
    assert loaded_cfg.compute_hash() == bench_cfg.compute_hash()


def test_load_default_project_configs():
    """Verify that all default YAML configurations in configs/ parse cleanly."""
    root = Path(__file__).parents[2]
    config_dir = root / "configs"

    # Dataset
    ds = DatasetConfig.from_yaml(config_dir / "dataset" / "development_pbmc.yaml")
    assert ds.name == "kang_pbmc"

    # Split
    sp = SplitConfig.from_yaml(config_dir / "split" / "donor_held_out.yaml")
    assert sp.split_type == SplitType.DONOR_HELD_OUT

    # Preprocessing
    pp = PreprocessingConfig.from_yaml(config_dir / "preprocessing" / "standard_pca50.yaml")
    assert pp.hvg_flavor == HVGFlavor.SEURAT

    # Graphs
    g_pca = GraphConfig.from_yaml(config_dir / "graph" / "pca_knn.yaml")
    assert g_pca.builder_type == GraphBuilderType.PCA_KNN

    g_bbknn = GraphConfig.from_yaml(config_dir / "graph" / "bbknn_inductive.yaml")
    assert g_bbknn.builder_type == GraphBuilderType.BBKNN

    # Models
    m_lr = ModelConfig.from_yaml(config_dir / "model" / "logistic_regression.yaml")
    assert m_lr.model_type == ModelType.LOGISTIC_REGRESSION

    m_mlp = ModelConfig.from_yaml(config_dir / "model" / "mlp_smoke.yaml")
    assert m_mlp.model_type == ModelType.MLP
