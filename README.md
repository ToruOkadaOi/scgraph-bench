# scgraph-bench: Benchmarking Graph Inductive Benefit in Single-Cell RNA-seq

`scgraph-bench` is a benchmark for evaluating whether cell–cell graphs provide a genuine inductive benefit over strong non-graph baselines for single-cell cell-type annotation(as a test case).

---

## 1. Core Research Question

> **“When do cell–cell graphs provide a genuine inductive benefit over strong non-graph baselines for single-cell cell-type annotation, which graph constructions help, and which measurable graph properties explain success or failure?”**

This is essentially to isolate the effect of `graph construction from model architecture`. A null or negative graph lift over an MLP baseline could be treated as a valid scientific finding.

Standard transductive GNNs (GCN, GraphSAGE) are evaluated under an inductive, donor-held-out protocol to test generalization to unseen donors

---

## 2. Scientific Principles

- **Donor Held Out Evaluation**: Strict inductive evaluation where donors in validation and test partitions are disjoint from training donors.
- **Strict Inductive Connectivity**: Feature matrices ($X$) are computed purely using training-fitted preprocessing. Test nodes connect only to reference training nodes via feature similarity.
- **Identical Fixed Features**: Every model (Logistic Regression, MLP, GNN) is evaluated on the exact same precomputed feature matrix $X$.
- **Graph Diagnostics Without Leakage**: Labels and metadata are used strictly for post hoc diagnostic evaluation (measuring homophily, mixing, degree distributions), never inside graph builders.
- **Reproducible Artifact Registry**: Splits and graphs are frozen to disk with cryptographic SHA-256 validation hashes.

---

## 3. Quick start & Repo Struct.
<details>
<summary>Here!</summary>

### Installation with `uv`

```bash
# Clone the repository
git clone https://github.com/ToruOkadaOi/scgraph-bench.git
cd scgraph-bench

# Create virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Running Checks

```bash
# Run linting and formatting checks
make lint

# Run unit tests
make test

# Run all verification checks
make check
```

---

## Architecture

```text
src/scgraph_bench/
├── config/         # Structured Pydantic/YAML benchmark configs
├── data/           # Dataset registry, schema validator, and loaders
├── splitting/      # Donor-held-out splitters with label stratification
├── preprocessing/  # Training-fitted normalisation, HVG, scaling, and PCA
├── graphs/         # Graph bundles, PCA-kNN, BBKNN, Rewired controls, Diagnostics
├── models/         # Scikit-learn Logistic Regression & PyTorch CPU MLP
├── evaluation/     # Metrics, confusion matrices, and matched graph lift
├── tracking/       # Tidy results schemas and MLflow logging
└── utils/          # Hashing, seed management, logging, path resolution
```
</details>

---

## 4. Protocols and Decision Logs

- [STUDY_PROTOCOL.md](file:///Users/aman/Documents/scgraph-bench/STUDY_PROTOCOL.md): Non-negotiable scientific contract and experimental design.
- [DECISIONS_NEEDED.md](file:///Users/aman/Documents/scgraph-bench/DECISIONS_NEEDED.md): Decision log tracking owner sign-offs and protocol options.
- [AGENTS.md](file:///Users/aman/Documents/scgraph-bench/AGENTS.md): Role boundaries between CPU foundation and GPU training agents.
- [IMPLEMENTATION_STATUS.md](file:///Users/aman/Documents/scgraph-bench/IMPLEMENTATION_STATUS.md): Current completion status across phases.

## 5. Analysis & Diagnostics Tooling

Post hoc analysis over frozen run artifacts (all retroactively applicable to saved `test_probs.npy` / `metrics_summary.json`):

| Script | Purpose |
|---|---|
| `scripts/analyze_per_class_batch.py` | Tidy per-class & per-donor CSVs, matched GNN−MLP per-class deltas, calibration summaries (ECE/Brier/entropy/margin) + plots |
| `scripts/join_diagnostics_results.py` | Joins graph diagnostics (homophily, purity, mixing entropy) with mean lift; correlation analysis |
| `scripts/analyze_embeddings.py` | Embedding geometry (silhouette, kNN-space accuracy vs raw PCA, centroid separation) + UMAP projections |
| `scripts/analyze_training_dynamics.py` | Loss/F1 trajectories, convergence speed, seed stability from persisted `training_history.csv` |
| `scripts/generate_final_report.py` | Consolidated markdown report under `results/reports/` |

### GPU Result Delivery Pipeline

Results produced on remote GPU instances are transferred via a cryptographically verified two-step flow:

```bash
# 1. On the GPU machine (run last, before teardown):
uv run python scripts/package_gpu_results.py            # audits + hashes every file, prints BATCH FINGERPRINT
rsync gpu_results_*.tar.gz localmachine:scgraph-bench/

# 2. On the local machine:
uv run python scripts/receive_gpu_delivery.py gpu_results_*.tar.gz --dry-run   # preview
uv run python scripts/receive_gpu_delivery.py gpu_results_*.tar.gz             # verify + auto-ingest
```

Four verification layers: per-file SHA-256 (transfer corruption), batch aggregate hash (tampering), provenance hash-chain match against local canonical artifacts (stale features/splits/graphs), and independent semantic recomputation of reported metrics from frozen labels. PASS runs are ingested into `artifacts/results/`; FAILs are quarantined under `audits/gpu_runs/<batch>/quarantine/`. Every delivery is recorded in the append-only `audits/gpu_runs/ingestion_log.jsonl`.

## 6. Key findings (Stephenson 2021 PBMC)

Using a site-stratified, multi seed protocol on the Stephenson 2021 healthy PBMC dataset:

GCN consistently underperforms a matched MLP baseline across PCA‑kNN, mutual kNN, and BBKNN graphs (mean matched lift −0.009 to −0.023; *positive lift in only 2 of 15 seed-level comparisons*).

Varying PCA‑kNN construction (k = 10/20/50, weighted edges) does not reverse this pattern; GCN remains below MLP in 4 of 5 seeds for each variant.

Destroying graph topology causes catastrophic failure (macro‑F1 ≈ 0.03–0.05, lift ≈ −0.85), confirming that structured neighborhoods & not just edge presence, are essential for message passing.

GraphSAGE improves over GCN on the same graphs and reaches approximate parity with MLP (small positive mean lift), but the advantage over MLP is modest and seed-dependent (positive in 2–3 of 5 seeds).

GraphSAGE on BBKNN shows the smallest cross-site performance drop, suggesting improved robustness to site-level variation, though this requires confirmation on additional datasets.

Embedding-quality analysis adds a mechanistic explanation: MLP penultimate representations separate cell types substantially better (silhouette ≈ 0.34) than GNN hidden layers (≈ 0.15–0.20), which barely improve on the raw PCA input space (≈ 0.15) — message passing adds little geometric structure over the fixed features.

Full tables, ablations, and error analysis: docs/results-stephenson-2021.md.
