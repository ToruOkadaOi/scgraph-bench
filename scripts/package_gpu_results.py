"""Package benchmark results into a cryptographically verified delivery bundle.

Run this on the training machine (e.g. a GPU instance) immediately before teardown.
Every result file is SHA-256 indexed, every run is audited against the canonical
artifacts available on the producing machine, and the resulting batch manifest is
bound to a single fingerprint that can be re-verified after transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import tempfile
from pathlib import Path

from rich.console import Console
from rich.table import Table

from scgraph_bench.analysis.delivery import (
    AuditVerdict,
    RunAuditReport,
    audit_run_dir,
    build_batch_manifest,
)
from scgraph_bench.preprocessing.schema import PreprocessedBundle
from scgraph_bench.utils.paths import ArtifactPaths
from scgraph_bench.utils.seed import set_seed

console = Console()


def _label_names(prep_dir: Path) -> list[str] | None:
    try:
        bundle = PreprocessedBundle.load(prep_dir)
    except Exception:
        return None
    inv = {v: k for k, v in bundle.label_to_id.items()}
    return [inv[i] for i in range(len(inv))]


def run_packaging(
    dataset_name: str,
    split_id: str,
    output: Path | None,
) -> Path:
    set_seed(42)
    paths = ArtifactPaths.default()
    results_root = paths.artifacts_dir / "results" / dataset_name / split_id
    prep_dir = paths.artifacts_dir / "preprocessed" / dataset_name / split_id
    graphs_root = paths.artifacts_dir / "graphs" / dataset_name / split_id

    if not results_root.is_dir():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    prep_bundle = None
    label_names = None
    if (prep_dir / "feature_manifest.json").is_file():
        prep_bundle = PreprocessedBundle.load(prep_dir)
        inv = {v: k for k, v in prep_bundle.label_to_id.items()}
        label_names = [inv[i] for i in range(len(inv))]

    run_dirs = sorted(d for d in results_root.iterdir() if d.is_dir())
    if not run_dirs:
        raise RuntimeError(f"No run directories under {results_root}")

    console.print(f"[bold cyan]Auditing {len(run_dirs)} runs under[/bold cyan] {results_root}")
    reports: list[RunAuditReport] = []
    for run_dir in run_dirs:
        report = audit_run_dir(
            run_dir, prep_bundle=prep_bundle, label_names=label_names, graphs_root=graphs_root
        )
        reports.append(report)
        icon = {
            "pass": "[green]PASS[/green]",
            "warn": "[yellow]WARN[/yellow]",
            "fail": "[red]FAIL[/red]",
        }[report.verdict.value]
        failed = report.failed_hard_checks + report.failed_soft_checks
        detail = f" ({', '.join(failed)})" if failed else ""
        console.print(f"  {icon} {run_dir.name}{detail}")

    verdict_table = Table(title="Pack-Time Audit Summary")
    verdict_table.add_column("Verdict", style="bold")
    verdict_table.add_column("Count", justify="right")
    for v in AuditVerdict:
        n = sum(1 for r in reports if r.verdict == v)
        if n:
            verdict_table.add_row(v.value.upper(), str(n))
    console.print(verdict_table)

    with tempfile.TemporaryDirectory(prefix="scgraph_pack_") as tmp:
        staging = Path(tmp) / "batch"
        staging.mkdir()
        for run_dir in run_dirs:
            shutil.copytree(run_dir, staging / run_dir.name)

        manifest = build_batch_manifest(staging, dataset_name, split_id, reports)
        manifest_path = staging / "batch_manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

        batch_hash = manifest.compute_batch_hash()

        out_path = output or Path(f"gpu_results_{split_id}_{batch_hash[:12]}.tar.gz")
        with tarfile.open(out_path, "w:gz") as tar:
            tar.add(staging, arcname=".")

        size_mb = out_path.stat().st_size / (1024 * 1024)
        console.print("\n[bold green]Delivery package written:[/bold green]", out_path)
        console.print(
            f"  Runs: [bold]{len(run_dirs)}[/bold] | Files: {len(manifest.files)} | Size: {size_mb:.1f} MB"
        )

    console.rule("[bold yellow]BATCH FINGERPRINT")
    console.print(
        f"[bold]{hashlib.sha256(batch_hash.encode()).hexdigest()[:32]}  ({batch_hash})[/bold]"
    )
    console.print("Verify this fingerprint after rsync on the receiving machine.")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="stephenson_2021_healthy_pbmc")
    parser.add_argument("--split", type=str, default="site_stratified_seed42")
    parser.add_argument("--output", type=Path, default=None, help="Output tarball path.")
    args = parser.parse_args()

    run_packaging(dataset_name=args.dataset, split_id=args.split, output=args.output)
