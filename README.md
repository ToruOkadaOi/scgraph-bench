# scgraph-bench: Benchmarking Graph Inductive Benefit in Single-Cell RNA-seq

`scgraph-bench` is a benchmark for evaluating whether cell–cell graphs provide a genuine inductive benefit over strong non-graph baselines for single-cell cell-type annotation(as a test case).

---

## 1. Core Research Question

> **“When do cell–cell graphs provide a genuine inductive benefit over strong non-graph baselines for single-cell cell-type annotation, which graph constructions help, and which measurable graph properties explain success or failure?”**

This is essentially to isolate the effect of `graph construction from model architecture`. A null or negative graph lift over an MLP baseline could be treated as a valid scientific finding.

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

## 5. Key findings (Stephenson 2021 PBMC)

Using a site-stratified, multi seed protocol on the Stephenson 2021 healthy PBMC dataset:

GCN consistently underperforms a matched MLP baseline across PCA‑kNN, mutual kNN, and BBKNN graphs (mean matched lift −0.009 to −0.023; *positive lift in only 2 of 15 seed-level comparisons*).

Varying PCA‑kNN construction (k = 10/20/50, weighted edges) does not reverse this pattern; GCN remains below MLP in 4 of 5 seeds for each variant.

Destroying graph topology causes catastrophic failure (macro‑F1 ≈ 0.03–0.05, lift ≈ −0.85), confirming that structured neighborhoods & not just edge presence, are essential for message passing.

GraphSAGE improves over GCN on the same graphs and reaches approximate parity with MLP (small positive mean lift), but the advantage over MLP is modest and seed-dependent (positive in 2–3 of 5 seeds).

GraphSAGE on BBKNN shows the smallest cross-site performance drop, suggesting improved robustness to site-level variation, though this requires confirmation on additional datasets.

Full tables, ablations, and error analysis: docs/results-stephenson-2021.md.
