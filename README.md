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

## 6. Key findings (Cross-Dataset Synthesis)

Using strict donor-held-out protocols across both the **Stephenson 2021 healthy PBMC** dataset (78,959 cells across 23 donors; matched MLP reference 0.9026 ± 0.0015) and the **GSE164690 HNSCC solid tumor** dataset (136,881 cells across 18 patients; matched MLP reference 0.8361 ± 0.0019):

GCN consistently underperforms the matched MLP baseline across all primary graph constructions in both datasets (mean matched lift −0.017 to −0.030 on PBMC, −0.020 to −0.022 on HNSCC; *positive lift in 0 of 45 non-control seed comparisons*). Symmetrical smoothing ($D^{-1/2} A D^{-1/2}$) blurs boundaries between continuous cell states, resulting in an invariant ~2.0% performance penalty across both liquid and solid tissues.

GraphSAGE improves over GCN on every graph and seed across both datasets, reaching near-exact parity with MLP but never reliably surpassing it (mean lift −0.001 to −0.005 on PBMC, −0.003 to −0.009 on HNSCC). GraphSAGE’s separate root transformation ($W_1 x$) allows the network to downweight noisy neighbor messages and preserve self-node features.

Destroying graph topology reveals a stark mechanistic divergence: under randomized degree-matched controls, GCN suffers a catastrophic collapse (macro-F1 drops to ≈ 0.03–0.05, lift ≈ −0.80 to −0.86) due to forced averaging across random cell types, whereas GraphSAGE retains high predictive accuracy (≈ 0.81–0.89 macro-F1) by effectively pruning corrupted neighbor messages.

Homophily anti-correlates with graph lift: across graph conditions, train-train edge homophily and class purity rank-correlate negatively with lift—highly homophilous neighborhoods mostly re-encode metric proximity already captured by the MLP, while batch-balanced BBKNN and Mutual-kNN retain the most complementary cross-donor signal.

Embedding-quality analysis reveals that representation separability orders identically to downstream task performance: MLP penultimate representations are most class-separable, GraphSAGE hidden layers closely track them, GCN trails, and raw PCA sits far below. Message passing adds topological structure over raw inputs, but never extracts more predictive signal than an identically regularized feature-only network learns from the same features.

- Full Dataset 1 (Stephenson PBMC) tables & ablations: [docs/results-stephenson-2021.md](docs/results-stephenson-2021.md)
- Full Dataset 2 (GSE164690 HNSCC) tables & controls: [docs/results-gse164690-hnscc.md](docs/results-gse164690-hnscc.md)
- Cross-dataset meta-analysis & comparative synthesis: [docs/results-cross-dataset-synthesis.md](docs/results-cross-dataset-synthesis.md)
