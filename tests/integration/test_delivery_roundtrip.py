"""Integration tests for the GPU result audit and delivery pipeline."""

from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from scgraph_bench.analysis.delivery import (
    AuditVerdict,
    BatchManifest,
    RunAuditReport,
    append_ingestion_log,
    audit_run_dir,
    build_batch_manifest,
    verify_batch_files,
)
from scgraph_bench.evaluation.schema import EvaluationSummary, PerClassMetric
from scgraph_bench.tracking.schema import RunManifest

N_TEST = 120
N_CLASSES = 4


def _fake_prep_bundle(seed: int = 42):
    rng = np.random.default_rng(seed)
    y_test = rng.integers(0, N_CLASSES, size=N_TEST)
    hashes = dict.fromkeys(range(6), "0" * 64)
    manifest = SimpleNamespace(
        split_config_hash="split" + "a" * 59,
        label_mapping_hash="labels" + "b" * 56,
        compute_manifest_hash=lambda: "feat" + "c" * 60,
    )
    return SimpleNamespace(test_labels=y_test, manifest=manifest, _hashes=hashes)


def _write_consistent_run(run_dir, y_test, seed: int = 7, corrupt_f1: bool = False):
    run_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    logits = rng.normal(scale=3.0, size=(N_TEST, N_CLASSES))
    logits[np.arange(N_TEST), y_test] += 2.5
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)

    reported_f1 = float(
        f1_score(y_test, preds, labels=list(range(N_CLASSES)), average="macro", zero_division=0.0)
    )
    if corrupt_f1:
        reported_f1 = min(1.0, reported_f1 + 0.05)

    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(y_test, preds, strict=True):
        cm[t, p] += 1

    per_class = [
        PerClassMetric(
            class_index=c,
            class_name=f"c{c}",
            precision=0.9,
            recall=0.9,
            f1=0.9,
            support=int((y_test == c).sum()),
        )
        for c in range(N_CLASSES)
    ]
    summary = EvaluationSummary(
        partition="test",
        num_samples=N_TEST,
        macro_f1=reported_f1,
        weighted_f1=reported_f1,
        balanced_accuracy=0.9,
        overall_accuracy=0.9,
        macro_precision=0.9,
        macro_recall=0.9,
        per_class=per_class,
        confusion_matrix=cm.tolist(),
    )
    manifest = RunManifest(
        run_id=f"gcn_k20_seed{seed}",
        model_name="gcn",
        model_config_hash="d" * 64,
        dataset_name="synthetic",
        split_id="s42",
        feature_manifest_hash="feat" + "c" * 60,
        split_hash="split" + "a" * 59,
        label_mapping_hash="labels" + "b" * 56,
        seed=seed,
    )
    np.save(run_dir / "test_preds.npy", preds)
    np.save(run_dir / "test_probs.npy", probs)
    np.save(run_dir / "embeddings_test.npy", rng.normal(size=(N_TEST, 16)))
    pd.DataFrame([{"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_macro_f1": 0.5}]).to_csv(
        run_dir / "training_history.csv", index=False
    )
    (run_dir / "run_manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    (run_dir / "metrics_summary.json").write_text(
        json.dumps({"test": summary.model_dump(mode="json")}), encoding="utf-8"
    )


def _audit(report: RunAuditReport, name: str) -> bool:
    check = next(c for c in report.checks if c.name == name)
    return check.passed


def test_audit_passes_on_consistent_run(tmp_path):
    bundle = _fake_prep_bundle()
    run_dir = tmp_path / "gcn_k20_seed7"
    _write_consistent_run(run_dir, bundle.test_labels, seed=7)
    report = audit_run_dir(
        run_dir, prep_bundle=bundle, label_names=[f"c{c}" for c in range(N_CLASSES)]
    )
    assert report.verdict == AuditVerdict.PASS
    assert _audit(report, "feature_manifest_hash_match")
    assert _audit(report, "macro_f1_recomputation")
    assert _audit(report, "probs_sanity")
    assert _audit(report, "confusion_matrix_consistency")


def test_audit_flags_reported_metric_mismatch(tmp_path):
    bundle = _fake_prep_bundle()
    run_dir = tmp_path / "gcn_k20_seed8"
    _write_consistent_run(run_dir, bundle.test_labels, seed=8, corrupt_f1=True)
    report = audit_run_dir(
        run_dir, prep_bundle=bundle, label_names=[f"c{c}" for c in range(N_CLASSES)]
    )
    assert report.verdict == AuditVerdict.FAIL
    assert not _audit(report, "macro_f1_recomputation")


def test_audit_flags_stale_feature_hash(tmp_path):
    bundle = _fake_prep_bundle()
    run_dir = tmp_path / "gcn_k20_seed9"
    _write_consistent_run(run_dir, bundle.test_labels, seed=9)
    stale = json.loads((run_dir / "run_manifest.json").read_text())
    stale["feature_manifest_hash"] = "stale" + "e" * 59
    (run_dir / "run_manifest.json").write_text(json.dumps(stale))
    report = audit_run_dir(
        run_dir, prep_bundle=bundle, label_names=[f"c{c}" for c in range(N_CLASSES)]
    )
    assert report.verdict == AuditVerdict.FAIL
    assert not _audit(report, "feature_manifest_hash_match")


def test_full_pack_verify_roundtrip_with_tampering(tmp_path):
    bundle = _fake_prep_bundle()
    results_root = tmp_path / "results"
    good_dir = results_root / "gcn_k20_seed7"
    bad_dir = results_root / "gcn_k20_seed8"
    _write_consistent_run(good_dir, bundle.test_labels, seed=7)
    _write_consistent_run(bad_dir, bundle.test_labels, seed=8, corrupt_f1=True)

    reports = [
        audit_run_dir(d, prep_bundle=bundle, label_names=[f"c{c}" for c in range(N_CLASSES)])
        for d in sorted(p for p in results_root.iterdir() if p.is_dir())
    ]
    assert [r.verdict for r in reports] == [AuditVerdict.PASS, AuditVerdict.FAIL]

    manifest = build_batch_manifest(results_root, "synthetic", "s42", reports)
    batch_hash = manifest.compute_batch_hash()

    problems, extras = verify_batch_files(results_root, manifest)
    assert not problems and not extras

    pack_verdicts = {r.run_id: r.verdict for r in manifest.runs}
    assert pack_verdicts["gcn_k20_seed7"] == AuditVerdict.PASS
    assert pack_verdicts["gcn_k20_seed8"] == AuditVerdict.FAIL

    victim = results_root / "gcn_k20_seed7" / "test_preds.npy"
    original = np.load(victim)
    np.save(victim, original + 1)
    problems, _ = verify_batch_files(results_root, manifest)
    assert any("hash-mismatch" in p for p in problems)
    assert batch_hash != ""


def test_ingestion_log_append(tmp_path):
    log = tmp_path / "ingestion_log.jsonl"
    append_ingestion_log(log, {"batch": "a", "n": 1})
    append_ingestion_log(log, {"batch": "b", "n": 2})
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["batch"] == "b"


def test_missing_optional_files_produce_warn_not_fail(tmp_path):
    bundle = _fake_prep_bundle()
    run_dir = tmp_path / "gcn_k20_seed10"
    _write_consistent_run(run_dir, bundle.test_labels, seed=10)
    (run_dir / "training_history.csv").unlink()
    (run_dir / "embeddings_test.npy").unlink()
    report = audit_run_dir(
        run_dir, prep_bundle=bundle, label_names=[f"c{c}" for c in range(N_CLASSES)]
    )
    soft_names = report.failed_soft_checks
    assert any("optional_" in s for s in soft_names)
    assert report.verdict == AuditVerdict.WARN


@pytest.fixture(autouse=True)
def _clean(tmp_path):
    yield
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_batch_manifest_deterministic(tmp_path):
    bundle = _fake_prep_bundle()
    run_dir = tmp_path / "gcn_k20_seed11"
    _write_consistent_run(run_dir, bundle.test_labels, seed=11)
    m1 = build_batch_manifest(tmp_path, "synthetic", "s42", [])
    m2 = build_batch_manifest(tmp_path, "synthetic", "s42", [])
    assert isinstance(m1, BatchManifest)
    assert m1.compute_batch_hash() == m2.compute_batch_hash()
