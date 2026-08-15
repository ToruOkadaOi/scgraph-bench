# DECISIONS_NEEDED.md: Owner Decision Log & Scientific Ambiguities

This document tracks scientific ambiguities, protocol choices, design alternatives, and the owner's explicit decisions.

---

## 1. Resolved & Approved Decisions

### Decision 1: Primary Evaluation Protocol (Transductive vs. Strict Inductive)
- **Status**: **APPROVED** by Owner
- **Decision**: The primary benchmark protocol is **strict inductive donor-held-out evaluation**.
- **Specification**: Test nodes may connect to training-reference nodes using features only (bipartite test $\to$ train edges). Test-to-test and val-to-val edges are explicitly disabled in v0.
- **Protocol Impact**: Recorded in [STUDY_PROTOCOL.md](file:///Users/aman/Documents/scgraph-bench/STUDY_PROTOCOL.md).

### Decision 2: Reference Dataset Selection Strategy
- **Status**: **APPROVED** by Owner
- **Decision**: Primary dataset is locked to **Stephenson et al. (2021) Healthy PBMC** (`stephenson_2021_healthy_pbmc`). Kang PBMC is retained for development/smoke testing only. Zheng 68k is rejected (single-donor). HLCA is deferred. Tabula Sapiens blood is retained as a negative feasibility audit.

### Decision 3: Nature of Degree-Preserving Rewired Negative Control
- **Status**: **APPROVED** by Owner
- **Decision**: Degree-preserving rewiring does not guarantee zero homophily. It must be treated as a degree-matched topological control, and its realised homophily must be explicitly measured and reported in diagnostics.

### Decision 4: Graph Builder Input Signatures & Label Leakage Guard
- **Status**: **APPROVED** by Owner
- **Decision**: Graph builders must receive restricted feature matrices ($X$) and allowed metadata tables (`donor_id`, `site`), never `AnnData` objects or cell-type label arrays.

### Decision 5: BBKNN Inductive Inference Design
- **Status**: **APPROVED** by Owner
- **Decision**: Standard BBKNN is not valid for held-out donor inference. BBKNN requires an explicit strict-inductive design connecting test query cells to reference training cells balanced across training batches.

### Decision 6: Explicit HVG Selection Method
- **Status**: **APPROVED** by Owner
- **Decision**: Fixed to `seurat` method calculated on normalized $\log(1+p)$ expression counts (library size normalization with `target_sum=1e4` followed by `log1p`). Distinct from `seurat_v3`.

### Decision 7: Handling Rare Cell Types Across Donor Splits
- **Status**: **APPROVED** by Owner
- **Decision**: Make rare-class inclusion thresholds configurable and record *unsupported*, *low-support*, and *excluded* labels separately.

### Decision 8: RBF Kernel Bandwidth Formulation ($\sigma_k$)
- **Status**: **APPROVED** by Owner
- **Decision**: For each neighborhood size $k \in \{10, 20, 40\}$, compute $\sigma_k$ as the **median Euclidean distance from every training cell to its $k$-th nearest training neighbor** in fixed 50-dim PCA space:
  $$\sigma_k = \text{median}_{i \in \text{train}} \, \|x_i - \text{NN}_k(x_i)\|_2$$
- **Specification**: Separate $\sigma_k$ computed for each $k$, computed strictly on training partition, all RBF quantiles and metadata persisted in graph bundles.

### Decision 9: Frozen Split Git Tracking
- **Status**: **APPROVED** by Owner
- **Decision**: Frozen split JSON files under `splits/` must be committed to Git as immutable experimental contracts.

### Decision 10: Primary Dataset Selection & Provenance
- **Status**: **APPROVED** by Owner
- **Decision**: Pinned to **Stephenson et al. (2021) Healthy PBMC** (`stephenson_2021_healthy_pbmc`) via CZ CELLxGENE Census release `2025-11-08` (Dataset ID: `c7775e88-49bf-4ba2-a03b-93f00447c958` / E-MTAB-10026). Raw count layer `adata.X` verified as pure sparse integer counts.

### Decision 11: Donor Manifest & Cohort Filtering
- **Status**: **APPROVED** by Owner
- **Decision**: Filter out the 6 intravenous LPS challenge donors via versioned Git-tracked manifest [`configs/dataset/stephenson_donor_manifest.csv`](file:///Users/aman/Documents/scgraph-bench/configs/dataset/stephenson_donor_manifest.csv). Retain all **23 unperturbed healthy baseline donors** (12 Cambridge + 11 Newcastle).

### Decision 12: Primary Site-Stratified Donor Split
- **Status**: **APPROVED** by Owner
- **Decision**: Primary frozen split is `site_stratified_seed42`:
  - **Train**: 12 donors (6 Cambridge + 6 Newcastle) $\to$ 38,692 cells
  - **Validation**: 6 donors (3 Cambridge + 3 Newcastle) $\to$ 21,759 cells
  - **Test**: 5 donors (3 Cambridge + 2 Newcastle) $\to$ 18,508 cells
  - Total evaluated cells: 78,959 cells across 23 donors.

### Decision 13: Primary 12-Class Coherent Flat Label Set
- **Status**: **APPROVED** by Owner
- **Decision**: 12 coherent Cell Ontology classes adopted. Broad parent labels (`B cell`, `natural killer cell`, `dendritic cell`) and low-resolution classes excluded from primary macro-F1 task and tracked in `label_policy.yaml`.

---

## 2. Genuinely Unresolved Future Decisions (Open for Phase 6+ / GPU Agent)

### Open Decision 1: GNN Hyperparameter Tuning Space Bounds (GPU Phase)
- **Context**: The GPU training agent will perform hyperparameter sweeps for GCN, GAT, GraphSAGE.
- **Pending Decision**: Exact parameter grid limits (learning rate ranges, weight decay, hidden dimensions [64 vs 128 vs 256], dropout rates [0.1 vs 0.5]) to be formalized before GPU handoff.

### Open Decision 2: Secondary Multi-Tissue / Cross-Dataset Benchmarks (Post-v0)
- **Context**: After completing the v0 primary benchmark on `stephenson_2021_healthy_pbmc`, evaluation on secondary multi-tissue datasets (e.g. cross-tissue Tabula Sapiens with harmonized ontologies) can be explored.
- **Pending Decision**: Selection criteria for post-v0 cross-tissue benchmarks.
