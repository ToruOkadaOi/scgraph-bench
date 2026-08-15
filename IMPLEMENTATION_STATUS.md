# IMPLEMENTATION_STATUS.md: Project Milestones & Status

## Phase Tracking

| Phase | Description | Status | Test Coverage | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | Package Scaffold, Tooling, CI, Dockerfile, Makefile | **COMPLETED** | Scaffold & CI verified | CPU-only, Ruff/pytest mandatory, Mypy non-blocking. |
| **Phase 2** | Configuration System, Seed Management, Logging, Artifact Hashing, Result Schema | **COMPLETED** | 21 unit tests passing | Pydantic schemas, cryptographic hashing, tidy metric schema. |
| **Phase 3** | Dataset Registry, Donor Manifest & Production Loader | **COMPLETED** | Unit tests passing | Stephenson 2021 healthy PBMC loader with Cell Ontology standardisation & 23 unperturbed donors. |
| **Phase 4** | Site-Stratified Donor-Held-Out Frozen Splitting | **COMPLETED** | Unit tests passing | Site-stratified 12:6:5 donor split saved to `splits/` with full 12-class support (78,959 cells). |
| **Phase 5** | Leakage-Safe Preprocessing Pipeline | **NEXT APPROVED** | - | Train-only library sizing, log1p, Seurat HVGs, scaling, PCA. |
| **Phase 6** | PCA-kNN Graph Construction & PyG Serialization | PENDING | - | Strict inductive test connectivity & RBF median bandwidth. |
| **Phase 7** | Graph Diagnostics Suite | PENDING | - | Post hoc homophily, mixing, topology. |
| **Phase 8** | Classical Baseline (Logistic Regression) & CPU MLP Smoke Test | PENDING | - | Baseline models consuming fixed $X$. |
| **Phase 9** | Mutual kNN, BBKNN, Rewired Negative Control | PENDING | - | Strict inductive BBKNN & degree-matched control. |
| **Phase 10**| Evaluation Engine, Results Aggregation & MLflow | PENDING | - | Tidy results table, matched graph lift. |
| **Phase 11**| End-to-End CPU Pipeline & GPU Handoff Document | PENDING | - | Full test suite, HANDOFF_TO_GPU.md finalized. |

---

## Benchmark State & Boundary Declaration

- **Frozen Benchmark Dataset**: `stephenson_2021_healthy_pbmc` (23 unperturbed healthy donors, Cambridge + Newcastle).
- **Frozen Split**: `splits/stephenson_2021_healthy_pbmc/site_stratified_seed42.json` (12 train, 6 val, 5 test donors; 78,959 cells).
- **Label Set**: Approved 12-class flat Cell Ontology vocabulary with full partition representation.
- **Current Execution Boundary**: **NO preprocessing features ($X$), NO graphs, NO model training, and NO GPU experiments have been executed yet.**
- **Next Milestone**: **Phase 5 (Leakage-Safe Preprocessing Pipeline)**.

---

## Testing & Quality Assurance Summary

- **Ruff Lint & Format**: Enforced and passing across all `src/`, `tests/`, `scripts/`.
- **Pytest**: 29 unit tests passing (100% pass rate).
- **Mypy**: Informational type checking configured.
