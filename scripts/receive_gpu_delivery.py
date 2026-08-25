"""Receive, verify, and ingest a packaged GPU results delivery.

Verification layers:
  1. Per-file SHA-256 against the pack-time batch manifest (transfer corruption).
  2. Batch aggregate hash over the manifest index (tampering).
  3. Provenance hash-chain match against local canonical artifacts.
  4. Independent semantic recomputation of reported metrics from frozen labels.

Passing runs are ingested into artifacts/results/<dataset>/<split>/; failing runs
are quarantined under audits/gpu_runs/<batch>/. Every delivery is recorded in an
append-only JSONL ledger and a human-readable audit report.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from scgraph_bench.analysis.delivery import (
    AuditVerdict,
    BatchManifest,
    RunAuditReport,
    append_ingestion_log,
    audit_run_dir,
    verify_batch_files,
)
from scgraph_bench.preprocessing.schema import PreprocessedBundle
from scgraph_bench.utils.paths import ArtifactPaths

console = Console()


def receive_delivery(
    source: Path,
    dataset_name: str | None,
    split_id: str | None,
    dry_run: bool,
    force: bool,
    legacy: bool = False,
) -> int:
    paths = ArtifactPaths.default()
    staging_root = Path(tempfile.mkdtemp(prefix="scgraph_recv_"))
    batch_dir = staging_root / "batch"

    try:
        if source.is_file() and source.suffix == ".gz":
            console.print(f"[cyan]Extracting {source.name} ...[/cyan]")
            with tarfile.open(source, "r:gz") as tar:
                tar.extractall(staging_root, filter="data")
            if (staging_root / "batch_manifest.json").is_file():
                batch_dir = staging_root
            else:
                candidates = [p for p in staging_root.iterdir() if p.is_dir()]
                batch_dir = staging_root if len(candidates) != 1 else candidates[0]
        elif source.is_dir():
            batch_dir = source
        else:
            raise FileNotFoundError(f"Source must be a .tar.gz or directory: {source}")

        manifest: BatchManifest | None = None
        manifest_path = batch_dir / "batch_manifest.json"
        if manifest_path.is_file():
            manifest = BatchManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        elif not legacy:
            raise RuntimeError(
                "No batch_manifest.json found. This looks like a legacy delivery created "
                "before the packaging pipeline existed; re-run with --legacy to audit it "
                "without transfer-integrity layers."
            )
        else:
            console.print(
                "[yellow]Legacy delivery (no batch manifest); layers 1-2 skipped, "
                "auditing runs directly.[/yellow]"
            )

        ds = dataset_name or (manifest.dataset_name if manifest else None)
        sp = split_id or (manifest.split_id if manifest else None)
        if not ds or not sp:
            raise RuntimeError("Dataset/split could not be inferred; pass --dataset and --split.")
        console.print(
            f"[bold cyan]Delivery:[/bold cyan] "
            f"{(manifest.hostname if manifest else '') or 'unknown host'} | "
            f"code={(manifest.code_version if manifest else None) or 'n/a'} | dataset={ds}/{sp}"
        )

        problems: list[str] = []
        extras: list[str] = []
        computed_hash = ""
        if manifest is not None:
            problems, extras = verify_batch_files(batch_dir, manifest)
            computed_hash = manifest.compute_batch_hash()
            layer1_ok = not problems
            console.print(
                f"  Layer 1 file hashes: {'[green]OK[/green]' if layer1_ok else '[red]FAILED[/red]'} "
                f"({len(manifest.files)} files)"
            )
            for p in problems[:10]:
                console.print(f"    [red]{p}[/red]")
            if extras:
                console.print(f"    [yellow]unexpected extra files: {extras[:5]}[/yellow]")
            if not layer1_ok:
                console.print("[bold red]Transfer integrity failed; aborting.[/bold red]")
                return 2
        else:
            console.print("  [yellow]Layer 1 file hashes: SKIPPED (legacy)[/yellow]")

        prep_dir = paths.artifacts_dir / "preprocessed" / ds / sp
        graphs_root = paths.artifacts_dir / "graphs" / ds / sp
        prep_bundle = None
        label_names = None
        if (prep_dir / "feature_manifest.json").is_file():
            prep_bundle = PreprocessedBundle.load(prep_dir)
            inv = {v: k for k, v in prep_bundle.label_to_id.items()}
            label_names = [inv[i] for i in range(len(inv))]
        else:
            console.print(
                "[yellow]Local preprocessed bundle missing; provenance checks limited.[/yellow]"
            )

        results_root = paths.artifacts_dir / "results" / ds / sp
        run_dirs = sorted(d for d in batch_dir.iterdir() if d.is_dir())
        reports: list[RunAuditReport] = []
        for run_dir in run_dirs:
            report = audit_run_dir(
                run_dir,
                prep_bundle=prep_bundle,
                label_names=label_names,
                graphs_root=graphs_root,
            )
            reports.append(report)

        table = Table(title=f"Delivery Audit ({len(reports)} runs)")
        table.add_column("Run", style="cyan")
        table.add_column("Pack", justify="center")
        table.add_column("Receive", justify="center")
        table.add_column("Failed Checks")
        pack_by_id = {r.run_id: r.verdict for r in manifest.runs} if manifest else {}
        for report in sorted(reports, key=lambda r: r.verdict.value):
            failed = report.failed_hard_checks + report.failed_soft_checks
            table.add_row(
                report.run_id,
                pack_by_id.get(report.run_id, "?").upper(),
                report.verdict.value.upper(),
                ", ".join(failed) if failed else "-",
            )
        console.print(table)

        passing = [r for r in reports if r.verdict != AuditVerdict.FAIL]
        failing = [r for r in reports if r.verdict == AuditVerdict.FAIL]

        batch_label = (
            f"{sp}_{computed_hash[:12]}" if computed_hash else f"{sp}_legacy_{source.stem}"
        )
        audit_out = paths.root_dir / "audits" / "gpu_runs" / batch_label
        log_path = paths.root_dir / "audits" / "gpu_runs" / "ingestion_log.jsonl"

        ingested: list[str] = []
        skipped: list[str] = []
        quarantined: list[str] = []
        if not dry_run:
            audit_out.mkdir(parents=True, exist_ok=True)
            for report in passing:
                src_dir = batch_dir / report.dir_name
                dest = results_root / report.dir_name
                if dest.exists() and not force:
                    console.print(
                        f"[yellow]Refusing to overwrite existing {dest.name}; use --force.[/yellow]"
                    )
                    skipped.append(report.dir_name)
                    continue
                if dest.exists() and force:
                    shutil.rmtree(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src_dir, dest)
                ingested.append(report.dir_name)
            for report in failing:
                qdir = audit_out / "quarantine" / report.dir_name
                qdir.parent.mkdir(parents=True, exist_ok=True)
                if qdir.exists():
                    shutil.rmtree(qdir)
                shutil.copytree(batch_dir / report.dir_name, qdir)
                quarantined.append(report.dir_name)
            write_audit_report_md_batch(audit_out, reports, computed_hash, ingested, quarantined)
            append_ingestion_log(
                log_path,
                {
                    "timestamp": pd.Timestamp.utcnow().isoformat(),
                    "batch_hash": computed_hash,
                    "batch_label": batch_label,
                    "source": str(source),
                    "n_runs": len(reports),
                    "ingested": ingested,
                    "skipped_existing": skipped,
                    "quarantined": quarantined,
                },
            )
        else:
            console.print("[yellow]--dry-run: no files written.[/yellow]")

        summary = Table(title="Disposition")
        summary.add_column("Action", style="bold")
        summary.add_column("Runs")
        summary.add_row(
            "Ingested" if not dry_run else "Would ingest (pass/warn)",
            str(len(passing) if dry_run else len(ingested)),
        )
        summary.add_row(
            "Quarantined" if not dry_run else "Would quarantine (fail)",
            str(len(failing) if dry_run else len(quarantined)),
        )
        summary.add_row("Skipped (already present)", str(len(skipped)))
        console.print(summary)
        if not dry_run:
            console.print(f"[green]Audit report:[/green] {audit_out}")
            console.print(f"[green]Ledger:[/green] {log_path}")
        return 0
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def write_audit_report_md_batch(
    out_dir: Path,
    reports: list[RunAuditReport],
    batch_hash: str,
    ingested: list[str],
    quarantined: list[str],
) -> None:
    lines = [
        "# GPU Delivery Audit Report",
        "",
        f"- Batch hash: `{batch_hash}`",
        f"- Runs audited: {len(reports)}",
        f"- Ingested: {len(ingested)}",
        f"- Quarantined: {len(quarantined)}",
        "",
    ]
    for report in sorted(reports, key=lambda r: (r.verdict.value, r.run_id)):
        lines.append(f"## {report.run_id} — {report.verdict.value.upper()}")
        lines.append("")
        lines.append("| Check | Result | Detail |")
        lines.append("|---|---|---|")
        for c in report.checks:
            status = "✅" if c.passed else ("⚠️" if c.severity == "soft" else "❌")
            detail = c.detail.replace("\n", " ")
            lines.append(f"| {c.name} | {status} | {detail} |")
        lines.append("")
    (out_dir / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to gpu_results_*.tar.gz or extracted dir.")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Audit a pre-pipeline delivery without a batch manifest (skips transfer-integrity layers).",
    )
    args = parser.parse_args()

    sys.exit(
        receive_delivery(
            source=args.source,
            dataset_name=args.dataset,
            split_id=args.split,
            dry_run=args.dry_run,
            force=args.force,
            legacy=args.legacy,
        )
    )
