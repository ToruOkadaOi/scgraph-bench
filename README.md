# scgraph-bench: Benchmarking Graph Inductive Benefit in Single-Cell RNA-seq

`scgraph-bench` is a rigorous, leakage-safe, CPU-first benchmark for evaluating whether cell–cell graphs provide a genuine inductive benefit over strong non-graph baselines for single-cell cell-type annotation.

---

## 1. Core Research Question

> **“When do cell–cell graphs provide a genuine inductive benefit over strong non-graph baselines for single-cell cell-type annotation, which graph constructions help, and which measurable graph properties explain success or failure?”**

This benchmark isolates the effect of **graph construction** from model architecture. A null or negative graph lift over an MLP baseline is treated as a valid scientific finding.

---

## 2. Scientific Principles

- **Donor-Held-Out Evaluation**: Strict inductive evaluation where donors in validation and test partitions are disjoint from training donors.
- **Strict Inductive Connectivity**: Feature matrices ($X$) are computed purely using training-fitted preprocessing. Test nodes connect only to reference training nodes via feature similarity.
- **Identical Fixed Features**: Every model (Logistic Regression, MLP, GNN) is evaluated on the exact same precomputed feature matrix $X$.
- **Graph Diagnostics Without Leakage**: Labels and metadata are used strictly for post hoc diagnostic evaluation (measuring homophily, mixing, degree distributions), never inside graph builders.
- **Reproducible Artifact Registry**: Splits and graphs are frozen to disk with cryptographic SHA-256 validation hashes.

---

## 3. Quick Start (CPU-First)

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

## 4. Repository Architecture

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

---

## 5. Protocols and Decision Logs

- [STUDY_PROTOCOL.md](file:///Users/aman/Documents/scgraph-bench/STUDY_PROTOCOL.md): Non-negotiable scientific contract and experimental design.
- [DECISIONS_NEEDED.md](file:///Users/aman/Documents/scgraph-bench/DECISIONS_NEEDED.md): Decision log tracking owner sign-offs and protocol options.
- [AGENTS.md](file:///Users/aman/Documents/scgraph-bench/AGENTS.md): Role boundaries between CPU foundation and GPU training agents.
- [IMPLEMENTATION_STATUS.md](file:///Users/aman/Documents/scgraph-bench/IMPLEMENTATION_STATUS.md): Current completion status across phases.
