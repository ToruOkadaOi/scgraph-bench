"""Schemas and helpers for cryptographically verified GPU result deliveries."""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from scgraph_bench.analysis.flatten import describe_run
from scgraph_bench.evaluation.schema import EvaluationSummary
from scgraph_bench.tracking.schema import RunManifest
from scgraph_bench.utils.hashing import hash_dict, hash_file
from scgraph_bench.utils.logging import get_logger
from scgraph_bench.utils.versioning import get_code_version, get_torch_geometric_version

logger = get_logger("analysis.delivery")

BATCH_SCHEMA_VERSION = "1.0"
MACRO_F1_TOLERANCE = 1e-6


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class AuditVerdict(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class AuditCheck(BaseModel):
    """Outcome of one named integrity or consistency check."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    passed: bool
    severity: Literal["hard", "soft"] = "hard"
    detail: str = ""


class RunAuditReport(BaseModel):
    """Aggregated audit outcome for a single benchmark run directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    dir_name: str
    verdict: AuditVerdict
    checks: list[AuditCheck]
    manifest_hash: str = ""

    @property
    def failed_hard_checks(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed and c.severity == "hard"]

    @property
    def failed_soft_checks(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed and c.severity == "soft"]


class BatchFileEntry(BaseModel):
    """Per-file record inside a packaged result batch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str
    size_bytes: int


class BatchRunEntry(BaseModel):
    """Per-run audit summary recorded at pack time."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    verdict: AuditVerdict
    failed_checks: list[str] = Field(default_factory=list)


class BatchManifest(BaseModel):
    """Cryptographic delivery manifest created on the producing machine."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = BATCH_SCHEMA_VERSION
    created_at_utc: str = Field(default_factory=_utc_now)
    hostname: str = ""
    platform_info: str = ""
    code_version: str | None = None
    python_version: str = ""
    torch_version: str = ""
    torch_geometric_version: str | None = None
    cuda_available: bool = False
    dataset_name: str = ""
    split_id: str = ""
    files: list[BatchFileEntry] = Field(default_factory=list)
    runs: list[BatchRunEntry] = Field(default_factory=list)

    def compute_batch_hash(self) -> str:
        """Deterministic SHA-256 over the sorted file index and run verdicts."""
        payload = {
            "schema_version": self.schema_version,
            "files": [f.model_dump() for f in sorted(self.files, key=lambda x: x.path)],
            "runs": [
                r.model_dump() for r in sorted(self.runs, key=lambda x: (x.run_id, x.verdict))
            ],
        }
        return hash_dict(payload)


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes."""
    return hash_file(path)


def collect_files(root: Path) -> list[Path]:
    """All regular files under root, as stable relative paths."""
    return sorted(p for p in root.rglob("*") if p.is_file())


def build_batch_manifest(
    results_root: Path,
    dataset_name: str,
    split_id: str,
    audit_reports: list[RunAuditReport],
) -> BatchManifest:
    """Hash every file under results_root and attach pack-time audit verdicts."""
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_available = bool(torch.cuda.is_available())
    except ImportError:
        torch_version = ""
        cuda_available = False

    entries = [
        BatchFileEntry(
            path=str(p.relative_to(results_root)),
            sha256=sha256_file(p),
            size_bytes=p.stat().st_size,
        )
        for p in collect_files(results_root)
    ]
    runs = [
        BatchRunEntry(
            run_id=r.run_id,
            verdict=r.verdict,
            failed_checks=r.failed_hard_checks + r.failed_soft_checks,
        )
        for r in audit_reports
    ]
    return BatchManifest(
        hostname=platform.node(),
        platform_info=platform.platform(),
        code_version=get_code_version(),
        python_version=platform.python_version(),
        torch_version=torch_version,
        torch_geometric_version=get_torch_geometric_version(),
        cuda_available=cuda_available,
        dataset_name=dataset_name,
        split_id=split_id,
        files=entries,
        runs=runs,
    )


def verify_batch_files(batch_dir: Path, manifest: BatchManifest) -> tuple[list[str], list[str]]:
    """Verify every indexed file exists with matching SHA-256.

    Returns (missing_or_corrupt, unexpected_extra_files) relative paths.
    """
    problems: list[str] = []
    expected: set[str] = set()
    for entry in manifest.files:
        expected.add(entry.path)
        local = batch_dir / entry.path
        if not local.is_file():
            problems.append(f"missing: {entry.path}")
            continue
        actual = sha256_file(local)
        if actual != entry.sha256:
            problems.append(f"hash-mismatch: {entry.path}")
    extras = [
        str(p.relative_to(batch_dir))
        for p in collect_files(batch_dir)
        if p.relative_to(batch_dir).as_posix() not in expected and p.name != "batch_manifest.json"
    ]
    return problems, extras


def recompute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def audit_run_dir(
    run_dir: Path,
    prep_bundle: Any,
    label_names: list[str] | None = None,
    graphs_root: Path | None = None,
) -> RunAuditReport:
    """Full provenance and consistency audit of one run directory against canonical artifacts."""
    checks: list[AuditCheck] = []

    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "metrics_summary.json"

    manifest: RunManifest | None = None
    if manifest_path.is_file():
        try:
            manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
            checks.append(AuditCheck(name="manifest_schema", passed=True))
        except Exception as err:
            checks.append(AuditCheck(name="manifest_schema", passed=False, detail=str(err)))
    else:
        checks.append(
            AuditCheck(name="manifest_schema", passed=False, detail="run_manifest.json absent")
        )

    summaries: dict[str, EvaluationSummary] = {}
    if metrics_path.is_file():
        try:
            raw = json.loads(metrics_path.read_text(encoding="utf-8"))
            summaries = {part: EvaluationSummary.model_validate(v) for part, v in raw.items()}
            checks.append(AuditCheck(name="metrics_schema", passed=True))
        except Exception as err:
            checks.append(AuditCheck(name="metrics_schema", passed=False, detail=str(err)))
    else:
        checks.append(
            AuditCheck(name="metrics_schema", passed=False, detail="metrics_summary.json absent")
        )

    if manifest is not None and prep_bundle is not None:
        expected_feature_hash = prep_bundle.manifest.compute_manifest_hash()
        checks.append(
            AuditCheck(
                name="feature_manifest_hash_match",
                passed=manifest.feature_manifest_hash == expected_feature_hash,
                detail=(
                    "ok"
                    if manifest.feature_manifest_hash == expected_feature_hash
                    else f"run={manifest.feature_manifest_hash[:12]}… "
                    f"local={expected_feature_hash[:12]}…"
                ),
            )
        )
        if manifest.split_hash:
            checks.append(
                AuditCheck(
                    name="split_hash_match",
                    passed=manifest.split_hash == prep_bundle.manifest.split_config_hash,
                    detail=(
                        "ok"
                        if manifest.split_hash == prep_bundle.manifest.split_config_hash
                        else f"run={manifest.split_hash[:12]}… "
                        f"local={prep_bundle.manifest.split_config_hash[:12]}…"
                    ),
                )
            )
        checks.append(
            AuditCheck(
                name="label_mapping_hash_match",
                passed=manifest.label_mapping_hash == prep_bundle.manifest.label_mapping_hash,
                detail=(
                    "ok"
                    if manifest.label_mapping_hash == prep_bundle.manifest.label_mapping_hash
                    else f"run={manifest.label_mapping_hash[:12]}… "
                    f"local={prep_bundle.manifest.label_mapping_hash[:12]}…"
                ),
            )
        )
        if graphs_root is not None:
            _, graph_name = describe_run(manifest.run_id, manifest.model_name)
            if graph_name != "none":
                graph_exists = (graphs_root / graph_name / "graph_manifest.json").is_file()
                checks.append(
                    AuditCheck(
                        name="graph_artifact_present",
                        passed=graph_exists,
                        severity="soft",
                        detail=graph_name if graph_exists else f"{graph_name} not found locally",
                    )
                )
    elif manifest is not None:
        checks.append(
            AuditCheck(
                name="feature_manifest_hash_match",
                passed=False,
                detail="local preprocessed bundle unavailable",
            )
        )

    y_test = getattr(prep_bundle, "test_labels", None) if prep_bundle is not None else None
    preds_path = run_dir / "test_preds.npy"
    probs_path = run_dir / "test_probs.npy"

    y_pred: np.ndarray | None = None
    if preds_path.is_file() and y_test is not None:
        try:
            y_pred = np.load(preds_path)
            if len(y_pred) != len(y_test):
                checks.append(
                    AuditCheck(
                        name="preds_length",
                        passed=False,
                        detail=f"preds={len(y_pred)} labels={len(y_test)}",
                    )
                )
                y_pred = None
            else:
                checks.append(AuditCheck(name="preds_length", passed=True))
        except Exception as err:
            checks.append(AuditCheck(name="preds_length", passed=False, detail=str(err)))
    elif not preds_path.is_file():
        checks.append(AuditCheck(name="preds_length", passed=False, detail="test_preds.npy absent"))

    if probs_path.is_file() and y_pred is not None:
        try:
            probs = np.load(probs_path)
            sane_shape = probs.shape == (len(y_pred), probs.shape[-1])
            sums_ok = bool(np.allclose(probs.sum(axis=1), 1.0, atol=1e-3))
            argmax_ok = bool(np.all(probs.argmax(axis=1) == y_pred))
            checks.append(
                AuditCheck(
                    name="probs_sanity",
                    passed=sane_shape and sums_ok and argmax_ok,
                    detail=(
                        f"shape_ok={sane_shape} rows_sum_to_1={sums_ok} "
                        f"argmax_matches_preds={argmax_ok}"
                    ),
                )
            )
        except Exception as err:
            checks.append(AuditCheck(name="probs_sanity", passed=False, detail=str(err)))
    elif probs_path.is_file() and y_pred is None:
        checks.append(
            AuditCheck(name="probs_sanity", passed=False, detail="preds unavailable to cross-check")
        )
    else:
        checks.append(AuditCheck(name="probs_sanity", passed=False, detail="test_probs.npy absent"))

    reported_f1: float | None = None
    test_summary = summaries.get("test")
    if test_summary is not None:
        reported_f1 = test_summary.macro_f1
        if y_test is not None and y_pred is not None and label_names is not None:
            recomputed_cm = recompute_confusion_matrix(y_test, y_pred, len(label_names))
            cm_ok = test_summary.confusion_matrix and np.array_equal(
                np.array(test_summary.confusion_matrix), recomputed_cm
            )
            checks.append(
                AuditCheck(
                    name="confusion_matrix_consistency",
                    passed=bool(cm_ok),
                    detail="ok" if cm_ok else "stored confusion matrix differs from recomputation",
                )
            )
        else:
            checks.append(
                AuditCheck(
                    name="confusion_matrix_consistency",
                    passed=False,
                    severity="soft",
                    detail="insufficient inputs for recomputation",
                )
            )

    if reported_f1 is not None and y_test is not None and y_pred is not None:
        from sklearn.metrics import f1_score

        n_classes = len(label_names) if label_names is not None else int(y_pred.max()) + 1
        recomputed = float(
            f1_score(
                y_test,
                y_pred,
                labels=list(range(n_classes)),
                average="macro",
                zero_division=0.0,
            )
        )
        delta = abs(recomputed - reported_f1)
        checks.append(
            AuditCheck(
                name="macro_f1_recomputation",
                passed=bool(delta <= MACRO_F1_TOLERANCE),
                detail=f"reported={reported_f1:.6f} recomputed={recomputed:.6f} Δ={delta:.2e}",
            )
        )
    else:
        checks.append(
            AuditCheck(
                name="macro_f1_recomputation",
                passed=False,
                detail="preds/labels/report missing",
            )
        )

    for optional in ("training_history.csv", "embeddings_test.npy"):
        checks.append(
            AuditCheck(
                name=f"optional_{optional}",
                passed=(run_dir / optional).is_file(),
                severity="soft",
                detail="present" if (run_dir / optional).is_file() else "absent",
            )
        )

    has_hard_fail = any(not c.passed and c.severity == "hard" for c in checks)
    has_soft_fail = any(not c.passed and c.severity == "soft" for c in checks)
    verdict = (
        AuditVerdict.FAIL
        if has_hard_fail
        else (AuditVerdict.WARN if has_soft_fail else AuditVerdict.PASS)
    )

    manifest_hash = manifest.compute_manifest_hash() if manifest is not None else ""
    return RunAuditReport(
        run_id=manifest.run_id if manifest is not None else run_dir.name,
        dir_name=run_dir.name,
        verdict=verdict,
        checks=checks,
        manifest_hash=manifest_hash,
    )


def write_audit_report_md(report: RunAuditReport, out_path: Path) -> None:
    lines = [f"# Run Audit: {report.run_id}", "", f"- Verdict: **{report.verdict.value.upper()}**"]
    if report.manifest_hash:
        lines.append(f"- Manifest hash: `{report.manifest_hash}`")
    lines.extend(["", "| Check | Result | Detail |", "|---|---|---|"])
    for c in report.checks:
        status = "✅" if c.passed else ("⚠️" if c.severity == "soft" else "❌")
        lines.append(f"| {c.name} | {status} | {c.detail} |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_ingestion_log(log_path: Path, entry: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def summarize_run_metrics(summaries: dict[str, EvaluationSummary]) -> pd.DataFrame:
    rows = []
    for part, s in summaries.items():
        rows.append({"partition": part, "macro_f1": s.macro_f1, "n": s.num_samples})
    return pd.DataFrame(rows)
