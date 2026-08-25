"""Export canonical benchmark tables as markdown fragments for the results document.

Reads only audited artifacts: the canonical results tree (delivered via the GPU audit
pipeline), the archived pre-instrumented GNN ablation runs (GNN-side predictions are
genuine; lifts are recomputed against canonical baselines), and aggregated CSVs under
artifacts/aggregated/. Writes results/reports/table_data.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scgraph_bench.utils.paths import ArtifactPaths

ARCHIVE = "gpu-runs/archive_pre_instrumented_2026-08-25"
SEEDS = [7, 17, 42, 73, 101]
MLP_RUNS = ["mlp_seed7", "mlp_seed17", "mlp_seed42", "mlp_seed73", "mlp_seed101"]

PRIMARY_GCN = {
    "pca_knn_k24_unweighted": "PCA-kNN k=24",
    "mutual_knn_reference_standard_query_k20_unweighted": "Mutual kNN k=20",
    "bbknn_kperbatch2_donors12": "BBKNN",
}
ABLATION = [
    ("pca_knn_k10_unweighted", "PCA-kNN k=10, unweighted"),
    ("pca_knn_k20_weighted", "PCA-kNN k=20, weighted"),
    ("pca_knn_k50_unweighted", "PCA-kNN k=50, unweighted"),
    ("pca_knn_k20_unweighted_rewired", "PCA-kNN k=20, rewired"),
]


def _load_metrics(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _test(run_dir: Path) -> dict:
    return _load_metrics(run_dir / "metrics_summary.json")["test"]


CLASS_SHORT_NAMES = {
    "naive thymus-derived CD4-positive, alpha-beta T cell": "naive CD4+",
    "central memory CD4-positive, alpha-beta T cell": "central memory CD4+",
    "naive thymus-derived CD8-positive, alpha-beta T cell": "naive CD8+",
    "effector memory CD8-positive, alpha-beta T cell": "effector memory CD8+",
    "effector CD8-positive, alpha-beta T cell": "effector CD8+",
    "gamma-delta T cell": "γδ T",
    "mucosal invariant T cell": "MAIT",
    "CD14-positive monocyte": "monocyte",
    "CD16-positive, CD56-dim natural killer cell, human": "NK dim",
    "CD16-negative, CD56-bright natural killer cell, human": "NK bright",
    "naive B cell": "naive B",
    "platelet": "platelet",
}


def _short_class(name: str) -> str:
    return CLASS_SHORT_NAMES.get(name, name[:24])


def _per_class_summary(summary: dict) -> str:
    pcs = summary["per_class"]
    lo = min(pcs, key=lambda p: p["f1"])
    hi = max(pcs, key=lambda p: p["f1"])
    return f"{lo['f1']:.3f} {_short_class(lo['class_name'])} – {hi['f1']:.3f} {_short_class(hi['class_name'])}"


def _site_f1s(summary: dict) -> tuple[float, float]:
    sites = {s["site"]: s["observed_class_macro_f1"] for s in summary.get("per_site", [])}
    return sites.get("Cambridge", float("nan")), sites.get("Newcastle", float("nan"))


def _label_names(prep_dir: Path) -> list[str]:
    bundle_map = json.loads((prep_dir / "label_mapping.json").read_text(encoding="utf-8"))
    inv = {v: k for k, v in bundle_map.items()}
    return [inv[i] for i in range(len(inv))]


def collect(results_root: Path):
    data: dict[tuple[str, str, int], dict] = {}
    for d in sorted(results_root.iterdir()):
        if not (d / "metrics_summary.json").is_file():
            continue
        name = d.name
        seed = int(name.rsplit("seed", 1)[-1]) if name.rsplit("seed", 1)[-1].isdigit() else None
        if name.startswith("mlp"):
            model = "MLP"
            graph = "none"
            if name == "mlp":
                continue
        elif name.startswith(("gcn_", "graphsage_")):
            model = "GraphSAGE" if name.startswith("graphsage") else "GCN"
            graph = name.split("_", 1)[1].rsplit("_seed", 1)[0]
        else:
            continue
        if seed is None:
            continue
        data[(model, graph, seed)] = _test(d)
    return data


def build_tables(agg_dir: Path, results_root: Path, prep_dir: Path) -> str:
    canon = collect(results_root)
    archive = collect(Path(ARCHIVE))
    labels = _label_names(prep_dir)

    def f1(model, graph, seed, pool=None):
        rec = (pool or canon).get((model, graph, seed))
        return None if rec is None else rec["macro_f1"]

    out: list[str] = []

    mlp_by_seed = {s: f1("MLP", "none", s) for s in SEEDS}
    mlp_f1s = [v for v in mlp_by_seed.values() if v is not None]

    # ---- Table 1 ----
    out.append("### Table 1. Model and graph performance across five seeds\n")
    out.append("|Model|Graph artifact|Seed|Test macro-F1|Overall accuracy|Per-class F1 summary|")
    out.append("|---|---|--:|--:|--:|---|")
    for model, graph_disp, graph_key in [
        ("GCN", "`pca_knn_k24_unweighted`", "pca_knn_k24_unweighted"),
        (
            "GCN",
            "`mutual_knn_reference_standard_query_k20_unweighted`",
            "mutual_knn_reference_standard_query_k20_unweighted",
        ),
        ("GCN", "`bbknn_kperbatch2_donors12`", "bbknn_kperbatch2_donors12"),
        ("GraphSAGE", "`bbknn_kperbatch2_donors12`", "bbknn_kperbatch2_donors12"),
    ]:
        for s in SEEDS:
            rec = canon.get((model, graph_key, s))
            if rec is None:
                continue
            out.append(
                f"|{model}|{graph_disp}|{s}|{rec['macro_f1']:.4f}|{rec['overall_accuracy']:.4f}"
                f"|{_per_class_summary(rec)}|"
            )
    for s in SEEDS:
        rec = canon.get(("MLP", "none", s))
        if rec is None:
            continue
        out.append(
            f"|MLP|None: feature baseline|{s}|{rec['macro_f1']:.4f}"
            f"|{rec['overall_accuracy']:.4f}|{_per_class_summary(rec)}|"
        )

    # ---- Table 2 ----
    out.append("\n### Table 2. Matched graph lift over identically seeded MLP baseline\n")
    out.append("|Model|Graph|Seed|GNN macro-F1|MLP macro-F1|Lift Δ|")
    out.append("|---|---|--:|--:|--:|--:|")
    t2_rows = []
    for model, graph_disp, graph_key in [
        ("GCN", "PCA-kNN k=24", "pca_knn_k24_unweighted"),
        ("GCN", "Mutual kNN k=20", "mutual_knn_reference_standard_query_k20_unweighted"),
        ("GCN", "BBKNN", "bbknn_kperbatch2_donors12"),
        ("GraphSAGE", "BBKNN", "bbknn_kperbatch2_donors12"),
    ]:
        for s in SEEDS:
            g = f1(model, graph_key, s)
            m = mlp_by_seed[s]
            if g is None or m is None:
                continue
            lift = g - m
            t2_rows.append((model, graph_disp, s, g, m, lift))
            out.append(f"|{model}|{graph_disp}|{s}|{g:.4f}|{m:.4f}|{lift:+.4f}|")

    # ---- Table 3 ----
    out.append("\n### Table 3. Summary statistics across five seeds\n")
    out.append(
        "|Model and graph condition|Test macro-F1, mean ± SD|Matched lift, mean ± SD|Minimum lift|Maximum lift|"
    )
    out.append("|---|--:|--:|--:|--:|")
    t3 = []
    for model, graph_disp, graph_key in [
        ("GCN", "PCA-kNN k=24", "pca_knn_k24_unweighted"),
        ("GCN", "Mutual kNN k=20", "mutual_knn_reference_standard_query_k20_unweighted"),
        ("GCN", "BBKNN", "bbknn_kperbatch2_donors12"),
        ("GraphSAGE", "BBKNN", "bbknn_kperbatch2_donors12"),
    ]:
        f1s = [f1(model, graph_key, s) for s in SEEDS]
        lifts = [f1(model, graph_key, s) - mlp_by_seed[s] for s in SEEDS]
        f1s = np.array([x for x in f1s if x is not None])
        lifts = np.array([x for x in lifts if x is not None])
        t3.append(
            (
                f"{model}, {graph_disp}",
                f1s.mean(),
                f1s.std(ddof=1),
                lifts.mean(),
                lifts.std(ddof=1),
                lifts.min(),
                lifts.max(),
            )
        )
        out.append(
            f"|{model}, {graph_disp}|{f1s.mean():.4f} ± {f1s.std(ddof=1):.4f}"
            f"|{lifts.mean():+.4f} ± {lifts.std(ddof=1):.4f}|{lifts.min():+.4f}|{lifts.max():+.4f}|"
        )
    mf = np.array(mlp_f1s)
    out.append(f"|MLP baseline|{mf.mean():.4f} ± {mf.std(ddof=1):.4f}|Reference|—|—|")

    # ---- Table 4 (ablation; GNN side from audited archives, lifts vs canonical MLP) ----
    out.append("\n### Table 4. PCA-kNN graph-variant ablation\n")
    out.append("|Graph variant|Seed 7|Seed 17|Seed 42|Seed 73|Seed 101|Mean lift|")
    out.append("|---|--:|--:|--:|--:|--:|--:|")
    for graph_key, disp in ABLATION:
        pool = (
            canon
            if (
                any((model, graph_key, s) in canon for model in ["GCN"] for s in SEEDS)
                and graph_key == "pca_knn_k50_unweighted"
            )
            else archive
        )
        lifts, cells = [], []
        for s in SEEDS:
            g = f1("GCN", graph_key, s, pool)
            if g is None or mlp_by_seed[s] is None:
                cells.append("—")
                continue
            lift = g - mlp_by_seed[s]
            lifts.append(lift)
            cells.append(f"{lift:+.4f}")
        mean_lift = f"{np.mean(lifts):+.4f}" if lifts else "—"
        out.append(f"|{disp}|" + "|".join(cells) + f"|{mean_lift}|")

    # ---- Table 5 (k50 architectures) ----
    out.append("\n### Table 5. GCN and GraphSAGE on PCA-kNN k=50\n")
    out.append("|Model|Seed 7|Seed 17|Seed 42|Seed 73|Seed 101|Mean lift|")
    out.append("|---|--:|--:|--:|--:|--:|--:|")
    for model in ["GCN", "GraphSAGE"]:
        lifts, cells = [], []
        for s in SEEDS:
            g = f1(model, "pca_knn_k50_unweighted", s)
            lift = g - mlp_by_seed[s] if (g is not None and mlp_by_seed[s] is not None) else None
            cells.append(f"{lift:+.4f}" if lift is not None else "—")
            if lift is not None:
                lifts.append(lift)
        out.append(f"|{model}|" + "|".join(cells) + f"|{np.mean(lifts):+.4f}|")

    # ---- Table 6 (cross-site) ----
    out.append("\n### Table 6. Per-site macro-F1 across five seeds\n")
    out.append(
        "|Model and graph condition|Cambridge test macro-F1|Newcastle test macro-F1|Cross-site drop|"
    )
    out.append("|---|--:|--:|--:|")
    site_conditions = [
        ("GCN", "PCA-kNN k=24", "pca_knn_k24_unweighted"),
        ("GCN", "Mutual kNN k=20", "mutual_knn_reference_standard_query_k20_unweighted"),
        ("GCN", "BBKNN", "bbknn_kperbatch2_donors12"),
        ("GraphSAGE", "BBKNN", "bbknn_kperbatch2_donors12"),
    ]
    for model, disp, key in site_conditions:
        cams, news = [], []
        for s in SEEDS:
            rec = canon.get((model, key, s))
            if rec is None:
                continue
            c, n = _site_f1s(rec)
            cams.append(c)
            news.append(n)
        cams, news = np.array(cams), np.array(news)
        out.append(
            f"|{model}, {disp}|{cams.mean():.4f} ± {cams.std(ddof=1):.4f}"
            f"|{news.mean():.4f} ± {news.std(ddof=1):.4f}|{(cams - news).mean():.4f}|"
        )
    cams, news = [], []
    for s in SEEDS:
        rec = canon.get(("MLP", "none", s))
        if rec is None:
            continue
        c, n = _site_f1s(rec)
        cams.append(c)
        news.append(n)
    cams, news = np.array(cams), np.array(news)
    out.append(
        f"|MLP baseline|{cams.mean():.4f} ± {cams.std(ddof=1):.4f}"
        f"|{news.mean():.4f} ± {news.std(ddof=1):.4f}|{(cams - news).mean():.4f}|"
    )

    # ---- Table 7 (top misclassifications, seed 42) ----
    out.append("\n### Table 7. Top three misclassifications by model at seed 42\n")
    out.append("|Model|Error type|Cells misclassified|Error rate within source class|")
    out.append("|---|---|--:|--:|")
    for model, disp, key in [
        ("GCN", "GCN, BBKNN", "bbknn_kperbatch2_donors12"),
        ("GraphSAGE", "GraphSAGE, BBKNN", "bbknn_kperbatch2_donors12"),
        ("MLP", "MLP", "none"),
    ]:
        rec = canon.get((model, key, 42))
        if rec is None:
            continue
        cm = np.array(rec["confusion_matrix"])
        errors = []
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j and cm[i, j] > 0:
                    errors.append((int(cm[i, j]), i, j))
        errors.sort(reverse=True)
        for cnt, i, j in errors[:3]:
            rate = cnt / cm[i].sum()
            out.append(
                f"|{disp}|{_short_class(labels[i])} → {_short_class(labels[j])}|{cnt}|{rate:.1%}|"
            )

    # ---- New analyses ----
    dj = pd.read_csv(agg_dir / "diagnostics_join" / "diagnostics_join.csv")
    out.append("\n### Table 8. Graph diagnostics vs mean matched lift (one row per graph)\n")
    cols = [
        "graph_name",
        "n_seeds",
        "mean_lift",
        "overall_edge_homophily",
        "train_train_edge_homophily",
        "test_to_train_query_homophily",
        "macro_average_class_purity",
        "mean_train_donor_entropy",
    ]
    cols = [c for c in cols if c in dj.columns]
    out.append("|" + "|".join(cols) + "|")
    out.append("|" + "---|" * len(cols))
    for _, r in dj.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(
                str(int(v))
                if c in ("n_seeds",)
                else (f"{v:+.4f}" if isinstance(v, float) else str(v))
            )
        out.append("|" + "|".join(cells) + "|")

    corr = pd.read_csv(agg_dir / "diagnostics_join" / "correlations.csv")
    out.append("\n### Table 9. Correlation of graph diagnostics with mean matched lift\n")
    out.append("|Feature|Pearson r|Spearman ρ|")
    out.append("|---|--:|--:|")
    for _, r in corr.iterrows():
        out.append(f"|{r['feature']}|{r['pearson_r']:+.3f}|{r['spearman_rho']:+.3f}|")

    emb = pd.read_csv(agg_dir / "embedding_quality" / "embedding_quality.csv")
    emb = emb[emb["partition"] == "test"]
    out.append("\n### Table 10. Embedding quality, test partition (mean across runs per family)\n")
    out.append("|Representation family|Silhouette|kNN accuracy|Centroid separation|")
    out.append("|---|--:|--:|--:|")
    for family in ["raw_pca_input", "mlp", "gcn", "graphsage"]:
        rows = emb[emb["representation_name"].str.split(":").str[0] == family]
        if rows.empty:
            continue
        out.append(
            f"|{family}|{rows['silhouette_euclidean'].mean():.4f}"
            f"|{rows['knn_accuracy'].mean():.4f}"
            f"|{rows['centroid_separation'].mean():.4f}|"
        )

    cal = pd.read_csv(agg_dir / "per_class_batch" / "confidence_summary.csv")
    cal = cal[cal["partition"] == "test"].copy()
    cal["family"] = cal.apply(
        lambda r: (
            r["model_name"]
            if r["graph_name"] == "none"
            else f"{r['model_name']}:{' '.join(r['graph_name'].split('_')[:3])}"
        ),
        axis=1,
    )
    out.append("\n### Table 11. Confidence and calibration, test partition (mean across seeds)\n")
    out.append("|Model · Graph|Accuracy|ECE|Brier|Entropy (nats)|Margin|")
    out.append("|---|--:|--:|--:|--:|--:|")
    for (family,), grp in cal.groupby(["family"]):
        dedup = grp.drop_duplicates(subset=["model_name", "graph_name", "seed"])
        out.append(
            f"|{family}|{dedup['accuracy'].mean():.4f}|{dedup['ece'].mean():.4f}"
            f"|{dedup['brier_score'].mean():.4f}|{dedup['mean_entropy_nats'].mean():.3f}"
            f"|{dedup['mean_margin'].mean():.3f}|"
        )

    dyn = pd.read_csv(agg_dir / "training_dynamics" / "dynamics_summary.csv")
    out.append("\n### Table 12. Training dynamics summary\n")
    out.append("|Model|Graph|Runs|Best val macro-F1|SD|Final train loss|Final val loss|")
    out.append("|---|---|--:|--:|--:|--:|--:|")
    for _, r in dyn.iterrows():
        val_loss = "-" if pd.isna(r["final_val_loss"]) else f"{r['final_val_loss']:.4f}"
        std = 0.0 if pd.isna(r["best_val_f1_std"]) else r["best_val_f1_std"]
        out.append(
            f"|{r['model_name']}|{r['graph_name']}|{int(r['n_runs'])}"
            f"|{r['best_val_f1_mean']:.4f}|{std:.4f}"
            f"|{r['final_train_loss']:.4f}|{val_loss}|"
        )

    return "\n".join(out) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="stephenson_2021_healthy_pbmc")
    parser.add_argument("--split", default="site_stratified_seed42")
    args = parser.parse_args()

    paths = ArtifactPaths.default()
    agg = paths.artifacts_dir / "aggregated" / args.dataset / args.split
    res = paths.artifacts_dir / "results" / args.dataset / args.split
    prep = paths.artifacts_dir / "preprocessed" / args.dataset / args.split

    text = build_tables(agg, res, prep)
    out_path = paths.root_dir / "results" / "reports" / "table_data.md"
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote {out_path}")
