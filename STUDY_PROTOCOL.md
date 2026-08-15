# STUDY_PROTOCOL.md: Immutable Scientific Contract & Experimental Protocol

## 1. Core Research Question

> **“When do cell–cell graphs provide a genuine inductive benefit over strong non-graph baselines for single-cell cell-type annotation, which graph constructions help, and which measurable graph properties explain success or failure?”**

This benchmark is strictly designed to isolate the effect of graph construction from model architecture through controlled, reproducible experiments.

---

## 2. Approved Primary Benchmark Data Contract (v0 Frozen Contract)

The primary v0 benchmark configuration is permanently locked to the following frozen specifications:

1. **Primary Dataset**: `stephenson_2021_healthy_pbmc` (Stephenson et al., *Nature Medicine* 2021, E-MTAB-10026 / DOI: `10.1038/s41591-021-01329-2`).
2. **Data Repository & Pinned Census Version**: Pinned CZ CELLxGENE Census version `2025-11-08`, dataset accession ID `c7775e88-49bf-4ba2-a03b-93f00447c958`.
3. **Raw Expression Source**: `adata.X` containing **100% raw unnormalized, unintegrated, sparse non-negative integer UMI counts** (`is_integer_counts = True`, min: 0, max: 27,701). No pre-integrated embeddings or scaled layers are present.
4. **Donor Population**: Exactly **23 unperturbed healthy baseline control donors** (12 Cambridge donors `C-XXXX` + 11 Newcastle donors `CVXXXX`).
5. **Donor Exclusion Manifest**: Intravenous LPS challenge donors (`IVLPS-1`, `IVLPS-2`, `IVLPS-3`, `IVLPS-4`, `IVLPS-6`, `IVLPS-12`) are excluded via the versioned Git-tracked manifest [`configs/dataset/stephenson_donor_manifest.csv`](file:///Users/aman/Documents/scgraph-bench/configs/dataset/stephenson_donor_manifest.csv).
6. **Primary Frozen Split**: `site_stratified_seed42` (persisted in [`splits/stephenson_2021_healthy_pbmc/site_stratified_seed42.json`](file:///Users/aman/Documents/scgraph-bench/splits/stephenson_2021_healthy_pbmc/site_stratified_seed42.json)):
   - **Train Partition**: 12 donors (6 Cambridge + 6 Newcastle) $\to$ **38,692 cells**.
   - **Validation Partition**: 6 donors (3 Cambridge + 3 Newcastle) $\to$ **21,759 cells**.
   - **Test Partition**: 5 donors (3 Cambridge + 2 Newcastle) $\to$ **18,508 cells**.
   - **Total Evaluated Cells**: **78,959 cells**.
   - **Partition Disjointness**: Train, validation, and test donor sets and cell ID sets are strictly mutually disjoint ($Train \cap Val = \emptyset$, $Train \cap Test = \emptyset$, $Val \cap Test = \emptyset$).
7. **Approved 12-Class Coherent Flat Label Vocabulary**:
   1. `naive thymus-derived CD4-positive, alpha-beta T cell` (12,952 cells)
   2. `CD16-positive, CD56-dim natural killer cell, human` (12,375 cells)
   3. `CD14-positive monocyte` (10,546 cells)
   4. `central memory CD4-positive, alpha-beta T cell` (7,437 cells)
   5. `naive thymus-derived CD8-positive, alpha-beta T cell` (7,432 cells)
   6. `effector CD8-positive, alpha-beta T cell` (6,145 cells)
   7. `effector memory CD8-positive, alpha-beta T cell` (5,410 cells)
   8. `naive B cell` (4,887 cells)
   9. `gamma-delta T cell` (4,733 cells)
   10. `mucosal invariant T cell` (3,361 cells)
   11. `CD16-negative, CD56-bright natural killer cell, human` (1,959 cells)
   12. `platelet` (1,722 cells)
   - Every class is represented with $\ge 868$ train cells, $\ge 216$ validation cells, and $\ge 527$ test cells across partitions.
   - Broad parent categories (`B cell`, `natural killer cell`, `dendritic cell`) and low-resolution/difficult subsets are excluded from the primary macro-F1 task and documented in [`audits/stephenson_2021_healthy_pbmc/label_policy.yaml`](file:///Users/aman/Documents/scgraph-bench/audits/stephenson_2021_healthy_pbmc/label_policy.yaml).
8. **Label Provenance**: Author annotations standardised to **Cell Ontology (CL)** terms by CZ CELLxGENE.

---

## 3. Non-Negotiable Scientific Requirements

1. **Primary Task**: Supervised scRNA-seq cell-type annotation across independent donors.
2. **Primary Evaluation Protocol**: Strict inductive donor-held-out evaluation. Transductive evaluation across all nodes is explicitly prohibited in the primary benchmark.
3. **Inductive Graph Connectivity Semantics (v0)**:
   - Feature representations ($X$) are computed solely using preprocessing models fitted on training donors.
   - For validation and test cells, edges may connect only to training-reference nodes based strictly on feature similarity (bipartite test $\to$ train edges).
   - Test-to-test and validation-to-validation edges are disabled in v0.
   - Validation and test labels must never be accessible during preprocessing, graph construction, hyperparameter tuning, or early stopping.
4. **Input Boundary Separation**: Graph builders receive restricted feature matrices ($X$) and explicitly allowed metadata tables (e.g. `donor_id`, `site`). Graph builders never receive raw `AnnData` objects or cell-type label arrays.
5. **Fixed Feature Matrix**: Every model (Logistic Regression, MLP, GNNs) must receive the exact same precomputed feature matrix $X$ for a given dataset and split.
6. **Zero Preprocessing Leakage**: Library size normalisation parameters, HVG selection, standard scaling (mean and variance), and PCA projection matrices are fitted strictly on training cells. Validation and test cells are transformed using only these training-fitted artifacts.
7. **Explicit HVG Strategy**: The v0 pipeline uses one explicit HVG selection flavor: `seurat` on normalized $\log(1+p)$ counts (target sum $10^4$, followed by `log1p`, computing mean and dispersion across training cells). The `seurat` on $\log(1+p)$ method is distinct from `seurat_v3` (which operates on raw counts via variance-stabilizing transformation); the two methods are not interchangeable.
8. **Degree-Preserving Rewiring as Topology Control**: Rewired graphs do not guarantee zero homophily; they serve as degree-matched topological negative controls where realised edge and node homophily are explicitly measured.
9. **BBKNN Inductive Semantics**: Standard BBKNN creates a full graph across all batches. In strict inductive donor-held-out evaluation, BBKNN must follow an explicit inductive inference design (connecting test cells to nearest training cells per training batch).
10. **Configurable Rare-Class Thresholds**: Rare-class thresholds are explicitly configurable. Labels are transparently tracked and reported under three distinct categories:
    - *Unsupported*: Labels absent from training donors.
    - *Low-support*: Labels with fewer than $N_{\text{min}}$ cells in training or test sets.
    - *Excluded*: Labels filtered out by explicit configuration.
11. **Immutability and Git-Tracking of Frozen Splits**: Frozen split definitions (`splits/{dataset}/{split_id}.json`) are committed to Git as immutable benchmark contracts. Serialized graph bundles (`artifacts/...`) are persisted with cryptographic SHA-256 hashes and reloaded, never silently regenerated.
12. **Honest Reporting of Graph Lift**: The benchmark must report positive, null, or negative graph lift without bias:
    $$\text{Graph Lift} = \text{Macro-F1}_{\text{GNN}} - \text{Macro-F1}_{\text{matched MLP}}$$

---

## 4. Preprocessing Protocol (Phase 5)

```text
Raw Counts (Training Cells Only)
   │
   ├──> Fit Preprocessor:
   │      1. Target sum normalisation (target_sum = 1e4)
   │      2. log1p transformation: log(1 + x)
   │      3. HVG selection (explicit 'seurat' flavor on log1p counts, default: 2,000 genes)
   │      4. Feature scaling (mean & std computed on train HVG matrix; clipping at ±10.0)
   │      5. PCA fitting (50 components fitted strictly on train scaled HVGs)
   │
   └──> Transform:
          - X_train = transform(train_counts)
          - X_val   = transform(val_counts)   [using train-fitted parameters]
          - X_test  = transform(test_counts)  [using train-fitted parameters]
```

---

## 5. Graph Construction Taxonomy & Weighting (Phase 6 & 9)

| Graph Type | Construction Principle | Inductive Test Edge Semantics |
| :--- | :--- | :--- |
| **PCA-kNN** | $k$-nearest neighbors in 50-dim PCA space ($k \in \{10, 20, 40\}$, Euclidean, symmetric). | Test cells query $k$ nearest neighbors among training cells. Test-test edges disabled. |
| **Mutual PCA-kNN** | Reciprocal nearest neighbors. Edge kept only if $u \in \text{kNN}(v)$ and $v \in \text{kNN}(u)$. | Test cell $u$ connects to train cell $v$ if $v \in \text{kNN}(u)$ and $u$ ranks in top-$k$ for $v$. |
| **BBKNN** | Balanced kNN across batches/donors. | Test cell queries $k_{\text{batch}}$ neighbors per training donor batch. |
| **Rewired Control** | Maslov-Sneppen double-edge swap preserving in/out degree distribution. | Reference training graph rewired; test-to-train degree distributions preserved. |

### Primary RBF Kernel Bandwidth Formulation
For graphs configured with `EdgeWeightingMode.RBF_WEIGHTED`, edge weights are computed as:
$$W_{ij} = \exp\left(-\frac{d(x_i, x_j)^2}{2\sigma_k^2}\right)$$
- **Bandwidth Calculation**: For each neighborhood size $k \in \{10, 20, 40\}$, $\sigma_k$ is computed as the **median Euclidean distance from every training cell to its $k$-th nearest training neighbor** in fixed 50-dim PCA space:
  $$\sigma_k = \text{median}_{i \in \text{train}} \, \|x_i - \text{NN}_k(x_i)\|_2$$
- **Separation per $k$**: A distinct $\sigma_k$ is calculated and applied for each neighborhood size $k$.
- **Zero Leakage**: $\sigma_k$ is determined strictly from training pairwise distances; it must never be tuned using validation or test performance.
- **Persistence**: $\sigma_k$ and associated distance distribution quantiles are persisted in graph bundle metadata.

---

## 6. Primary Metrics & Lift Definition

- **Primary Metric**: Macro-F1 across all 12 evaluated classes.
- **Secondary Metrics**: Balanced Accuracy, Accuracy, Macro Precision, Macro Recall, per-class F1, confusion matrices.
- **Matched Baseline Comparison**: The matched MLP must share the identical split ID, random seed, fixed $X$ representation, and label vocabulary.

---

## 7. Diagnostic Measurements (Post Hoc Only)

Labels and donor metadata are used **exclusively** in post hoc diagnostic functions:
- Edge & Node Homophily: Fraction of edges linking identical cell types.
- Neighborhood Class Purity: Mean proportion of same-class neighbors.
- Donor/Batch Mixing Entropy: Dispersion of neighbor connections across distinct donor batches.
- Partition Cross-Edge Counts: Number of train–train vs. test–train edges.
- Graph Topology: Density, connected components, degree distribution (mean, median, min, max, std), isolated node fraction.
