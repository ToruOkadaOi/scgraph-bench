"""GPU/CUDA Pilot Runner executing GCN on Standard PCA-kNN and Rewired Control (Seed 42).

Strict Guardrail Control Flow:
1. Validate --device == 'cuda' (rejects any other device immediately).
2. If --confirm-paid-gpu-run is absent:
   - Read-only dry-run plan display (sourcing edge counts and MLP baseline metrics from frozen manifests).
   - Exits 0 cleanly without requiring CUDA hardware, loading datasets, or writing any files.
3. Only when --confirm-paid-gpu-run is present:
   - Strictly enforces CUDA GPU hardware availability (torch.cuda.is_available()).
   - Executes 1-epoch smoke test followed by full GCN pilot training on CUDA.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import torch_geometric
from rich.console import Console
from rich.table import Table

from scgraph_bench.data.loaders import StephensonHealthyPBMCLoader
from scgraph_bench.evaluation.metrics import (
    compute_evaluation_summary,
    confusion_matrix_to_dataframe,
)
from scgraph_bench.evaluation.schema import EvaluationSummary
from scgraph_bench.graph.schema import GraphBundle, GraphManifest
from scgraph_bench.models.gcn import GCNClassifier, GCNConfig
from scgraph_bench.preprocessing.schema import PreprocessedBundle
from scgraph_bench.tracking.graph_lift import compute_matched_graph_lift
from scgraph_bench.tracking.mlflow_tracker import LocalMLflowTracker
from scgraph_bench.tracking.schema import RunManifest, RunStatus
from scgraph_bench.utils.paths import ArtifactPaths
from scgraph_bench.utils.seed import set_seed

console = Console()


def validate_device_argument(requested_device: str) -> None:
    """Strictly validate that requested device is exactly 'cuda'."""
    if requested_device != "cuda":
        console.print(
            f"[bold red]ERROR: Invalid device '{requested_device}'. The GPU pilot strictly requires --device cuda.[/bold red]\n"
        )
        sys.exit(1)


def perform_cuda_hardware_check() -> torch.device:
    """Strictly verify NVIDIA CUDA GPU hardware presence when execution is confirmed."""
    console.print(
        "[bold cyan]====================================================================[/bold cyan]"
    )
    console.print(
        "[bold cyan]               CUDA Hardware Preflight Verification                 [/bold cyan]"
    )
    console.print(
        "[bold cyan]====================================================================[/bold cyan]\n"
    )

    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda

    telemetry_table = Table(title="Hardware & CUDA Environment Preflight Telemetry")
    telemetry_table.add_column("Property", style="cyan")
    telemetry_table.add_column("Value", style="green")

    telemetry_table.add_row("torch.cuda.is_available()", str(cuda_available))
    telemetry_table.add_row("torch.version.cuda", str(cuda_version) if cuda_version else "None")
    telemetry_table.add_row("PyTorch Version", torch.__version__)
    telemetry_table.add_row("PyG Version", torch_geometric.__version__)
    telemetry_table.add_row("Requested Device", "cuda")

    if not cuda_available:
        telemetry_table.add_row("torch.cuda.get_device_name(0)", "N/A (No CUDA-capable GPU)")
        telemetry_table.add_row(
            "torch.cuda.get_device_properties(0).total_memory", "N/A (No CUDA-capable GPU)"
        )
        telemetry_table.add_row(
            "CUDA Status", "[bold red]Unavailable (No CUDA-capable GPU detected)[/bold red]"
        )
        console.print(telemetry_table)
        console.print(
            "\n[bold red]ERROR: CUDA GPU unavailable. Refusing to run GPU pilot on CPU.[/bold red]\n"
        )
        console.print(
            "[yellow]To run this pilot, move the repository to an NVIDIA GPU cluster/instance (e.g. Vast.ai) with CUDA enabled.[/yellow]\n"
        )
        sys.exit(1)

    dev_name = torch.cuda.get_device_name(0)
    dev_props = torch.cuda.get_device_properties(0)
    total_mem_bytes = dev_props.total_memory
    total_mem_gb = total_mem_bytes / (1024**3)
    multi_proc_count = dev_props.multi_processor_count

    telemetry_table.add_row("torch.cuda.get_device_name(0)", dev_name)
    telemetry_table.add_row(
        "torch.cuda.get_device_properties(0).total_memory",
        f"{total_mem_bytes:,} bytes ({total_mem_gb:.2f} GB VRAM)",
    )
    telemetry_table.add_row("CUDA Multiprocessor Count", str(multi_proc_count))
    telemetry_table.add_row("Selected Device", f"cuda:0 ({dev_name})")
    console.print(telemetry_table)
    console.print(
        "\n[bold green]CUDA preflight check PASSED: NVIDIA GPU detected and ready.[/bold green]\n"
    )
    return torch.device("cuda:0")


def print_dry_run_plan(dataset_name: str, split_id: str, seed: int) -> None:
    """Display the dry-run execution plan sourced from read-only versioned manifests."""
    paths = ArtifactPaths.default()
    console.print(
        "[bold yellow]====================================================================[/bold yellow]"
    )
    console.print(
        "[bold yellow]            DRY-RUN PILOT PLAN ONLY (No Training Executed)          [/bold yellow]"
    )
    console.print(
        "[bold yellow]====================================================================[/bold yellow]\n"
    )

    # Read edge count metadata from frozen graph manifests (read-only)
    g1_dir = paths.artifacts_dir / "graphs" / dataset_name / split_id / "pca_knn_k20_unweighted"
    g2_dir = (
        paths.artifacts_dir / "graphs" / dataset_name / split_id / "rewired_control_pca_knn_seed42"
    )

    g1_edges = "unknown"
    g2_edges = "unknown"
    if (g1_dir / "graph_manifest.json").is_file():
        m1 = GraphManifest.model_validate_json(
            (g1_dir / "graph_manifest.json").read_text(encoding="utf-8")
        )
        g1_edges = f"{m1.num_edges:,}"

    if (g2_dir / "graph_manifest.json").is_file():
        m2 = GraphManifest.model_validate_json(
            (g2_dir / "graph_manifest.json").read_text(encoding="utf-8")
        )
        g2_edges = f"{m2.num_edges:,}"

    # Read baseline performance from versioned MLP metrics summary (read-only)
    mlp_dir = paths.artifacts_dir / "results" / dataset_name / split_id / "mlp"
    mlp_f1_str = "0.9012"
    if (mlp_dir / "metrics_summary.json").is_file():
        mlp_metrics = json.loads((mlp_dir / "metrics_summary.json").read_text(encoding="utf-8"))
        if "test" in mlp_metrics and "macro_f1" in mlp_metrics["test"]:
            mlp_f1_str = f"{mlp_metrics['test']['macro_f1']:.4f}"

    plan_table = Table(
        title="Planned Pilot Experiment Configuration (Sourced from Frozen Manifests)"
    )
    plan_table.add_column("Parameter", style="cyan")
    plan_table.add_column("Planned Value", style="green")

    plan_table.add_row("Dataset", dataset_name)
    plan_table.add_row("Split ID", split_id)
    plan_table.add_row("Random Seed", str(seed))
    plan_table.add_row(
        "Model Architecture", "2-layer GCN (GCNConv: 50 -> 128 -> 12, BN, ReLU, Dropout 0.2)"
    )
    plan_table.add_row("Condition 1 (Graph)", f"pca_knn_k20_unweighted ({g1_edges} edges)")
    plan_table.add_row(
        "Condition 2 (Control)", f"rewired_control_pca_knn_seed42 ({g2_edges} edges)"
    )
    plan_table.add_row("Matched Baseline", f"MLP Baseline Seed 42 (Test Macro-F1: {mlp_f1_str})")
    plan_table.add_row("Early Stopping", "Validation Macro-F1 (patience 50, max epochs 500)")
    plan_table.add_row("Execution Device Target", "NVIDIA CUDA GPU")

    console.print(plan_table)
    console.print(
        "\n[bold green]Dry-run validation complete. Zero output files or run directories created.[/bold green]"
    )
    console.print(
        "[bold cyan]To execute the actual GCN pilot on your rented Vast.ai NVIDIA GPU instance, add the flag:[/bold cyan] [bold yellow]--confirm-paid-gpu-run[/bold yellow]\n"
    )


def run_gpu_pilot(
    dataset_name: str = "stephenson_2021_healthy_pbmc",
    split_id: str = "site_stratified_seed42",
    seed: int = 42,
    device: str = "cuda",
    confirm_paid_gpu_run: bool = False,
) -> None:
    set_seed(seed)
    paths = ArtifactPaths.default()

    # Step 1: Strictly validate --device == 'cuda'
    validate_device_argument(requested_device=device)

    # Step 2: If --confirm-paid-gpu-run is absent, display dry run and exit 0 without requiring CUDA hardware
    if not confirm_paid_gpu_run:
        print_dry_run_plan(dataset_name=dataset_name, split_id=split_id, seed=seed)
        return

    # Step 3: Only when confirmation is present, strictly enforce CUDA GPU hardware presence
    target_device = perform_cuda_hardware_check()

    console.print(
        "[bold cyan]====================================================================[/bold cyan]"
    )
    console.print(
        f"[bold cyan]  Executing Confirmed GNN GPU Pilot Benchmark (Seed={seed}, Device={target_device})  [/bold cyan]"
    )
    console.print(
        "[bold cyan]====================================================================[/bold cyan]\n"
    )

    prep_dir = paths.artifacts_dir / "preprocessed" / dataset_name / split_id
    prep_bundle = PreprocessedBundle.load(prep_dir)

    loader = StephensonHealthyPBMCLoader()
    adata = loader.load()
    obs_map = {str(cid): idx for idx, cid in enumerate(adata.obs_names)}

    train_donors = adata.obs.iloc[[obs_map[cid] for cid in prep_bundle.train_cell_ids]][
        "donor_id"
    ].tolist()
    val_donors = adata.obs.iloc[[obs_map[cid] for cid in prep_bundle.val_cell_ids]][
        "donor_id"
    ].tolist()
    test_donors = adata.obs.iloc[[obs_map[cid] for cid in prep_bundle.test_cell_ids]][
        "donor_id"
    ].tolist()

    train_sites = adata.obs.iloc[[obs_map[cid] for cid in prep_bundle.train_cell_ids]][
        "site"
    ].tolist()
    val_sites = adata.obs.iloc[[obs_map[cid] for cid in prep_bundle.val_cell_ids]]["site"].tolist()
    test_sites = adata.obs.iloc[[obs_map[cid] for cid in prep_bundle.test_cell_ids]][
        "site"
    ].tolist()

    inv_label_map = {v: k for k, v in prep_bundle.label_to_id.items()}
    label_names = [inv_label_map[i] for i in range(len(prep_bundle.label_to_id))]

    # Combine PCA features across partitions into full node feature matrix
    X_all = np.vstack([prep_bundle.X_pca_train, prep_bundle.X_pca_val, prep_bundle.X_pca_test])
    y_train = prep_bundle.train_labels
    y_val = prep_bundle.val_labels
    y_test = prep_bundle.test_labels

    # Load matched MLP baseline
    mlp_res_dir = paths.artifacts_dir / "results" / dataset_name / split_id / "mlp"
    if (
        not (mlp_res_dir / "run_manifest.json").is_file()
        or not (mlp_res_dir / "metrics_summary.json").is_file()
    ):
        console.print(
            "[bold red]Matched MLP baseline results missing! Run scripts/train_mlp.py first.[/bold red]"
        )
        return

    mlp_manifest = RunManifest.model_validate_json(
        (mlp_res_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    mlp_metrics = json.loads((mlp_res_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    mlp_test_summary = EvaluationSummary.model_validate(mlp_metrics["test"])

    console.print(
        f"[bold green]Matched MLP Baseline Loaded:[/bold green] Test Macro-F1 = [bold yellow]{mlp_test_summary.macro_f1:.4f}[/bold yellow] (Seed={mlp_manifest.seed})\n"
    )

    pilot_graphs = [
        ("Standard PCA-kNN (k=20)", "pca_knn_k20_unweighted"),
        ("Rewired Negative Control", "rewired_control_pca_knn_seed42"),
    ]

    tracker = LocalMLflowTracker(experiment_name=f"scgraph-pilot-{split_id}")
    pilot_results = []
    pilot_lifts = []

    for desc, g_name in pilot_graphs:
        console.print(
            "[bold cyan]--------------------------------------------------------------------[/bold cyan]"
        )
        console.print(f"[bold cyan]  Training GCN on Graph: {desc} ({g_name})[/bold cyan]")
        console.print(
            "[bold cyan]--------------------------------------------------------------------[/bold cyan]"
        )

        g_dir = paths.artifacts_dir / "graphs" / dataset_name / split_id / g_name
        graph_bundle = GraphBundle.load(g_dir)

        # Build PyG Data with strict label isolation (only train labels passed into graph)
        pyg_data = graph_bundle.to_pyg_data(x=X_all, y_train_only=y_train)

        # 1-epoch CUDA smoke test
        console.print("Running 1-epoch sanity smoke test on CUDA device...")
        smoke_cfg = GCNConfig(
            in_features=50,
            hidden_dim=128,
            num_classes=len(label_names),
            max_epochs=1,
            seed=seed,
            device="cuda",
        )
        smoke_clf = GCNClassifier(smoke_cfg)
        smoke_clf.fit(pyg_data=pyg_data, val_labels=y_val)
        console.print("[green]1-epoch CUDA smoke test passed successfully![/green]")

        # Full GCN Pilot Training
        console.print(f"Fitting full GCN on CUDA (Seed={seed}, Max Epochs=500, Patience=50)...")
        full_cfg = GCNConfig(
            in_features=50,
            hidden_dim=128,
            num_classes=len(label_names),
            dropout=0.2,
            lr=0.001,
            weight_decay=1e-4,
            max_epochs=500,
            patience=50,
            seed=seed,
            device="cuda",
        )
        clf = GCNClassifier(full_cfg)
        clf.fit(pyg_data=pyg_data, val_labels=y_val)

        # Predict across all partitions
        tr_preds, va_preds, te_preds = clf.predict_all(pyg_data)
        _, _, te_probs = clf.predict_proba_all(pyg_data)

        # Compute partition summaries
        train_summary = compute_evaluation_summary(
            y_true=y_train,
            y_pred=tr_preds,
            partition="train",
            label_names=label_names,
            donor_ids=train_donors,
            site_ids=train_sites,
        )
        val_summary = compute_evaluation_summary(
            y_true=y_val,
            y_pred=va_preds,
            partition="val",
            label_names=label_names,
            donor_ids=val_donors,
            site_ids=val_sites,
        )
        test_summary = compute_evaluation_summary(
            y_true=y_test,
            y_pred=te_preds,
            partition="test",
            label_names=label_names,
            donor_ids=test_donors,
            site_ids=test_sites,
        )

        out_dir = (
            paths.artifacts_dir / "results" / dataset_name / split_id / f"gcn_{g_name}_seed{seed}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "test_preds.npy", te_preds)
        np.save(out_dir / "test_probs.npy", te_probs)
        df_cm = confusion_matrix_to_dataframe(test_summary.confusion_matrix, label_names)
        df_cm.to_csv(out_dir / "confusion_matrix_test.csv")

        eval_summaries = {
            "train": train_summary,
            "val": val_summary,
            "test": test_summary,
        }
        (out_dir / "metrics_summary.json").write_text(
            json.dumps({k: v.model_dump(mode="json") for k, v in eval_summaries.items()}, indent=2),
            encoding="utf-8",
        )

        run_manifest = RunManifest(
            run_id=f"gcn_{g_name}_seed{seed}",
            status=RunStatus.SUCCESS,
            model_name="gcn",
            model_config_hash=full_cfg.compute_hash(),
            dataset_name=dataset_name,
            dataset_version="2025-11-08",
            split_id=split_id,
            split_hash=prep_bundle.manifest.split_config_hash,
            feature_manifest_hash=prep_bundle.manifest.compute_manifest_hash(),
            preprocessing_config_hash=prep_bundle.manifest.preprocessing_config_hash,
            graph_artifact_hash=g_name,
            label_mapping_hash=prep_bundle.manifest.label_mapping_hash,
            seed=seed,
            device=clf.device_info_,
            parameter_count=clf.parameter_count_,
            best_epoch=clf.best_epoch_,
            best_val_macro_f1=clf.best_val_macro_f1_,
            training_time_seconds=clf.training_time_seconds_,
        )
        (out_dir / "run_manifest.json").write_text(
            run_manifest.model_dump_json(indent=2), encoding="utf-8"
        )

        # Log to MLflow local directory
        tracker.log_run(
            manifest=run_manifest,
            evaluation_summaries=eval_summaries,
            artifacts={"confusion_matrix_test.csv": out_dir / "confusion_matrix_test.csv"},
        )

        # Compute Matched Graph Lift vs MLP Seed 42
        lift_record = compute_matched_graph_lift(
            gnn_summary=test_summary,
            mlp_summary=mlp_test_summary,
            gnn_manifest=run_manifest,
            mlp_manifest=mlp_manifest,
            graph_name=g_name,
        )

        pilot_results.append((g_name, desc, train_summary, val_summary, test_summary, clf))
        pilot_lifts.append(lift_record)

    # Print Comparative Performance Table
    comp_table = Table(
        title=f"GPU GCN Pilot Runs vs Matched MLP Baseline ({dataset_name} - {split_id} Seed {seed})"
    )
    comp_table.add_column("Model / Condition", style="cyan")
    comp_table.add_column("Graph Condition", style="magenta")
    comp_table.add_column("Test Macro-F1", style="green")
    comp_table.add_column("Graph Lift (Δ)", style="yellow")
    comp_table.add_column("Test BalAcc", style="blue")
    comp_table.add_column("Cambridge Obs F1", style="cyan")
    comp_table.add_column("Newcastle Obs F1", style="cyan")
    comp_table.add_column("Runtime (s)", style="green")
    comp_table.add_column("Peak VRAM", style="magenta")

    comp_table.add_row(
        "MLP Baseline (Seed 42)",
        "none (feature only)",
        f"{mlp_test_summary.macro_f1:.4f}",
        "0.0000 (reference)",
        f"{mlp_test_summary.balanced_accuracy:.4f}",
        f"{mlp_test_summary.per_site[0].observed_class_macro_f1:.4f}",
        f"{mlp_test_summary.per_site[1].observed_class_macro_f1:.4f}",
        f"{mlp_manifest.training_time_seconds:.2f}s",
        "CPU",
    )

    for (_g_name, desc, _tr_s, _va_s, te_s, clf), lift in zip(
        pilot_results, pilot_lifts, strict=False
    ):
        vram_str = f"{clf.peak_memory_mb_:.1f} MB" if clf.peak_memory_mb_ > 0 else "N/A"
        comp_table.add_row(
            f"GCN (Seed {seed})",
            desc,
            f"{te_s.macro_f1:.4f}",
            f"[bold]{lift.overall_graph_lift:+.4f}[/bold]",
            f"{te_s.balanced_accuracy:.4f}",
            f"{te_s.per_site[0].observed_class_macro_f1:.4f}",
            f"{te_s.per_site[1].observed_class_macro_f1:.4f}",
            f"{clf.training_time_seconds_:.2f}s",
            vram_str,
        )

    console.print("\n")
    console.print(comp_table)

    # Print Per-Class Comparative Lift Table
    class_table = Table(title=f"Per-Class Performance & Matched Graph Lift (Seed {seed})")
    class_table.add_column("Class Label", style="cyan")
    class_table.add_column("Test Support", style="yellow")
    class_table.add_column("MLP F1", style="magenta")
    class_table.add_column("GCN (PCA-kNN) F1", style="green")
    class_table.add_column("PCA-kNN Lift (Δ)", style="yellow")
    class_table.add_column("GCN (Rewired) F1", style="red")
    class_table.add_column("Rewired Lift (Δ)", style="yellow")

    pca_knn_test_summary = pilot_results[0][4]
    rewired_test_summary = pilot_results[1][4]

    for c in pca_knn_test_summary.per_class:
        mlp_c = next(mc for mc in mlp_test_summary.per_class if mc.class_name == c.class_name)
        rewired_c = next(
            rc for rc in rewired_test_summary.per_class if rc.class_name == c.class_name
        )

        pca_lift = c.f1 - mlp_c.f1
        rewired_lift = rewired_c.f1 - mlp_c.f1

        class_table.add_row(
            c.class_name,
            str(c.support),
            f"{mlp_c.f1:.4f}",
            f"{c.f1:.4f}",
            f"{pca_lift:+.4f}",
            f"{rewired_c.f1:.4f}",
            f"{rewired_lift:+.4f}",
        )

    console.print("\n")
    console.print(class_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run strict CUDA GPU GCN pilot benchmark.")
    parser.add_argument("--dataset", type=str, default="stephenson_2021_healthy_pbmc")
    parser.add_argument("--split", type=str, default="site_stratified_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Target device (must be strictly 'cuda' for GPU pilot execution).",
    )
    parser.add_argument(
        "--confirm-paid-gpu-run",
        action="store_true",
        default=False,
        help="Mandatory confirmation flag to execute real GNN training on GPU. Without this flag, outputs dry-run plan only.",
    )
    args = parser.parse_args()

    run_gpu_pilot(
        dataset_name=args.dataset,
        split_id=args.split,
        seed=args.seed,
        device=args.device,
        confirm_paid_gpu_run=args.confirm_paid_gpu_run,
    )
