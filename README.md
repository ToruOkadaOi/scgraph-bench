# scgraph-bench: Benchmarking Graph Inductive Benefit in Single-Cell RNA-seq

`scgraph-bench` is a benchmark for evaluating whether cell–cell graphs provide a genuine inductive benefit over strong non-graph baselines for single-cell cell-type annotation(as a test case).

🏳️This is a learning project I am pursuing during my leisure time to question and deepen my understanding of some concepts

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

<details>
<summary>Here!</summary>

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

</details>

## 6. Key findings (Stephenson 2021 PBMC)

Using a site-stratified, multi-seed protocol on the Stephenson 2021 healthy PBMC dataset (canonical audited results; matched MLP reference 0.9026 ± 0.0015):

GCN consistently underperforms a matched MLP baseline across PCA‑kNN, mutual kNN, and BBKNN graphs (mean matched lift −0.017 to −0.030; *positive lift in 0 of 25 seed-level comparisons*).

Varying PCA‑kNN construction (k = 10/20/50, weighted edges) does not reverse this pattern; no ablation variant produced a single positive seed (mean lift −0.022 to −0.017, improving modestly with k).

Destroying graph topology causes catastrophic failure (macro‑F1 ≈ 0.04–0.05, lift ≈ −0.86), confirming that structured neighborhoods — not just edge presence — are essential for message passing.

GraphSAGE improves over GCN on every graph and seed, reaching exact parity with MLP but never surpassing it (mean lift −0.001 to −0.005 across all five constructions; one marginally positive seed of 25).

Homophily anti-correlates with lift: across the five graph conditions, train-train edge homophily and class purity rank-correlate at ρ = −1.0 with mean lift while donor-mixing entropy correlates at +1.0 — homophilous neighborhoods mostly re-encode feature proximity already in the fixed features, and batch-balanced BBKNN edges preserve the most complementary signal.

GraphSAGE on BBKNN shows the smallest cross-site performance drop (0.0170 vs MLP 0.0218), suggesting improved robustness to site-level variation, though the margin is narrow and requires confirmation on additional datasets.

Embedding-quality analysis revealed that representation separability orders exactly like task performance; MLP penultimate embeddings are most class-separable (silhouette ≈ 0.29, kNN accuracy 0.912), GraphSAGE hidden layers approach them (≈ 0.24 / 0.909), GCN trails (≈ 0.28 / 0.894), and raw PCA input sits far below (≈ 0.15 / 0.887). Message passing adds structure over the inputs, but never more than a feature-only network learns from the same features.

Full tables, ablations, and error analysis: [docs/results-stephenson-2021.md](docs/results-stephenson-2021.md)
