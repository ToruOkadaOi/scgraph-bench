# GSE164690 HNSCC Audit Summary

- **Dataset Identifier**: `gse164690_hnscc`
- **Source Study**: Kürten et al. (*Nature Communications* 2021, DOI: `10.1038/s41467-021-27619-4` / GEO: `GSE164690`)
- **H5AD Cache Size**: `1,570,349,592` bytes
- **Generated Timestamp**: `2026-08-26T12:01:45.892728+00:00`

---

## Reconciliation & Invariant Checks

| Check | Expected | Actual / Verified | Status |
| :--- | :--- | :--- | :--- |
| **Sum of Stage 0 donor counts == raw cell count** | 151,680 | 151,680 | **PASS** |
| **Sum of Stage 1 donor counts == donor-filtered cell count** | 151,680 | 151,680 | **PASS** |
| **Sum of Primary donor counts == Primary cell count** | 136,881 | 136,881 | **PASS** |
| **Sum of Split partition counts == Primary cell count** | 136,881 | 136,881 | **PASS** |
| **Raw count matrix is sparse integer counts** | True | True | **PASS** |
| **Retained Donors Count (>= 12 requirement)** | >= 12 | **18 donors** | **PASS** |
| **10:4:4 HPV-Stratified Donor Split Feasibility** | Feasible | **Feasible (10 Train / 4 Val / 4 Test)** | **PASS** |
| **Donor Disjointness across Train/Val/Test** | Mutually Disjoint | Mutually Disjoint | **PASS** |

---

## Filtering Funnel

| stage               | description                                            |   cells |   donors |   cells_dropped |
|:--------------------|:-------------------------------------------------------|--------:|---------:|----------------:|
| 0_raw_geo_tar       | GSE164690 raw 10x MTX count matrices across 51 samples |  151680 |       18 |               0 |
| 1_donor_threshold   | Retain donors with >= 200 cells                        |  151680 |       18 |               0 |
| 2_annotation_filter | Remove unassigned/low-confidence cell annotations      |  136881 |       18 |           14799 |
| 3_primary_labels    | Retain 14 robust primary classes (>= 8 donor presence) |  136881 |       18 |               0 |

---

## Primary Label Support Across Partitions (14 Classes)

| cell_type                       |   train |   val |   test |   Total |
|:--------------------------------|--------:|------:|-------:|--------:|
| B cell                          |    2497 |  2526 |   3048 |    8071 |
| CD4-positive, alpha-beta T cell |   13409 |  7055 |   8394 |   28858 |
| CD8-positive, alpha-beta T cell |    7980 |  3666 |   3507 |   15153 |
| classical monocyte              |   11709 |  2656 |   7273 |   21638 |
| dendritic cell                  |     191 |   160 |    137 |     488 |
| endothelial cell                |    6695 |  1318 |   1698 |    9711 |
| fibroblast                      |    5156 |   963 |    341 |    6460 |
| macrophage                      |    1368 |    81 |    431 |    1880 |
| malignant epithelial cell       |   12571 |  4086 |   2395 |   19052 |
| mast cell                       |     202 |    46 |     73 |     321 |
| natural killer cell             |    9974 |  3566 |   3077 |   16617 |
| non-classical monocyte          |    1601 |   518 |   1439 |    3558 |
| plasma cell                     |     396 |   279 |   1119 |    1794 |
| regulatory T cell               |    1978 |   692 |    610 |    3280 |
| Total                           |   75727 | 27612 |  33542 |  136881 |

---

## Donor Breakdown Across Proposed Split (10 Train / 4 Val / 4 Test)

- **Train Donors (10)**: `HN01, HN02, HN03, HN07, HN08, HN09, HN10, HN11, HN12, HN13` (75,727 cells)
- **Validation Donors (4)**: `HN04, HN14, HN15, HN16` (27,612 cells)
- **Test Donors (4)**: `HN05, HN06, HN17, HN18` (33,542 cells)

---

## Label Policy Summary

- **Primary Robust Classes (14)**: `B cell, CD4-positive, alpha-beta T cell, CD8-positive, alpha-beta T cell, classical monocyte, dendritic cell, endothelial cell, fibroblast, macrophage, malignant epithelial cell, mast cell, natural killer cell, non-classical monocyte, plasma cell, regulatory T cell`
- **Deferred Low-Support Classes (0)**: ``
- **Excluded Sparse Classes (0)**: ``
