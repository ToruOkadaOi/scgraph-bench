# IMPLEMENTATION_STATUS.md: Project Milestones & Status

## Phase Tracking

| Phase | Description | Status | Test Coverage | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | Package Scaffold, Tooling, CI, Dockerfile, Makefile | **COMPLETED** | Scaffold & CI verified | CPU-only, Ruff/pytest mandatory, Mypy non-blocking. |
| **Phase 2** | Configuration System, Seed Management, Logging, Artifact Hashing, Result Schema | **COMPLETED** | 21 unit tests passing | Pydantic schemas, cryptographic hashing, tidy metric schema. |
| **Phase 3** | Dataset Registry, Donor Manifest & Production Loader | **COMPLETED** | Unit tests passing | Stephenson 2021 healthy PBMC loader with Cell Ontology standardisation & 23 unperturbed donors. |
| **Phase 4** | Site-Stratified Donor-Held-Out Frozen Splitting | **COMPLETED** | Unit tests passing | Site-stratified 12:6:5 donor split saved to `splits/` with full 12-class support (78,959 cells). |
| **Phase 5** | Leakage-Safe Preprocessing Pipeline | **COMPLETED** | 6 unit tests passing | Train-only library sizing, log1p, Seurat HVGs, scaling, PCA basis fitted & serialized. |
| **Phase 6** | PCA-kNN Graph Construction & PyG Serialization | **COMPLETED** | 6 unit tests passing | Strict-inductive PCA-kNN ($k=20$, Euclidean), RBF sigma heuristic, PyG bundle serialization. |
| **Phase 7** | Graph Diagnostics Suite | **COMPLETED** | 5 unit tests passing | Post hoc homophily, macro purity, donor/site mixing entropy, topology diagnostics report. |
| **Phase 8** | Classical Baseline (Logistic Regression) & CPU MLP Smoke Test | **COMPLETED** | 5 unit tests passing | Logistic Regression (test macro-F1: 0.8810) & CPU MLP (test macro-F1: 0.9012) fitted on fixed $X$. |
| **Phase 9** | Mutual kNN, BBKNN, Rewired Negative Control | **COMPLETED** | 3 unit tests passing | Built and diagnosed PCA-kNN (k=20 & k=24 density match), Mutual-kNN, BBKNN, and Rewired Control. |
| **Phase 10**| Evaluation Engine, Results Aggregation & MLflow | **COMPLETED** | 4 unit tests passing | Tidy results aggregation, cryptographic RunManifest, matched graph lift interface, MLflow tracker. |
| **Phase 11**| End-to-End CPU Pipeline & GPU Handoff Document | **COMPLETED** | 58 unit tests passing | End-to-end CPU pipeline (`make pipeline`), `validate_artifacts.py`, and `HANDOFF_TO_GPU.md` finalized. |

---

## Benchmark State & Boundary Declaration

- **Frozen Benchmark Dataset**: `stephenson_2021_healthy_pbmc` (23 unperturbed healthy donors, Cambridge + Newcastle).
- **Frozen Split**: `splits/stephenson_2021_healthy_pbmc/site_stratified_seed42.json` (12 train, 6 val, 5 test donors; 78,959 cells).
- **Preprocessed Feature Bundle**: `artifacts/preprocessed/stephenson_2021_healthy_pbmc/site_stratified_seed42/` (Fixed 50-dim PCA representation strictly fitted on 38,692 training cells; 21,759 val cells and 18,508 test cells projected into training basis with `feature_manifest.json`).
- **Constructed Graph Variants**:
  1. `pca_knn_k20_unweighted`: 2,066,630 edges, Train Homophily: **84.90%**, Val Query Homophily: **82.61%**, Test Query Homophily: **82.11%**, Macro Purity: **81.71%**, Lift over random: **+72.99%**.
  2. `pca_knn_k24_unweighted` (Density Match): 2,474,148 edges, Train Homophily: **84.66%**, Val Query Homophily: **82.39%**, Test Query Homophily: **81.89%**, Macro Purity: **81.45%**, Lift over random: **+72.74%**.
  3. `mutual_knn_reference_standard_query_k20_unweighted`: 1,091,730 edges, Train Homophily: **89.80%**, Val Query Homophily: **82.61%**, Test Query Homophily: **82.11%**, Macro Purity: **83.07%**, Lift over random: **+73.84%**.
  4. `bbknn_kperbatch2_donors12`: 2,703,020 edges, Train Homophily: **76.60%**, Val Query Homophily: **74.79%**, Test Query Homophily: **73.74%**, Macro Purity: **72.98%**, Cross-donor edges: **92.47%**, Lift over random: **+64.71%**.
  5. `rewired_control_pca_knn_seed42`: 2,066,630 edges (exact degree sequence of PCA-kNN), Train Homophily: **11.97%**, Macro Purity: **8.60%**, Lift over random: **+0.65%** (matches random class-composition background baseline: $\sum_c p_c^2 = 10.93\%$).
- **Baseline Models Evaluated**:
  - **Logistic Regression**: Validation Macro-F1 = 0.8752, Test Macro-F1 = **0.8810** (Cambridge: 0.8799, Newcastle: 0.8349).
  - **PyTorch MLP (Seed 42)**: Validation Macro-F1 = 0.8913, Test Macro-F1 = **0.9012** (Cambridge: 0.8943, Newcastle: 0.8717).
- **Label Set**: Approved 12-class flat Cell Ontology vocabulary with full partition representation.
- **Current Execution Boundary**: **CPU IMPLEMENTATION COMPLETE. READY FOR GPU TRAINING AGENT.**
- **GPU Handoff Document**: [`HANDOFF_TO_GPU.md`](file:///Users/aman/Documents/scgraph-bench/HANDOFF_TO_GPU.md).

---

## Testing & Quality Assurance Summary

- **Ruff Lint & Format**: Enforced and passing across all `src/`, `tests/`, `scripts/`.
- **Pytest**: 58 unit tests passing (100% pass rate across config, dataset, splitting, preprocessing, graph construction, diagnostics, baselines, tracking, hashing, seed).
- **Mypy**: Informational type checking configured.
