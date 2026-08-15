# Tabula Sapiens Blood (Figshare v5) Audit Summary

- **Dataset Identifier**: `tabula_sapiens_blood_figshare_v5`
- **Release Version**: Figshare v5 (Article 14267219)
- **Source File**: `TS_Blood.h5ad.zip` (1,169,692,522 bytes)
- **H5AD SHA-256**: `1da20aac113a92213fdefe32289eea175cb8d6198550c5b1b12d0d411ccaf987`
- **Generated Timestamp**: `2026-08-15T21:51:59.046531+00:00`

---

## Reconciliation & Invariant Checks

| Check | Expected | Actual / Verified | Status |
| :--- | :--- | :--- | :--- |
| **Sum of Stage 0 donor counts == raw cell count** | 50,115 | 50,115 | **PASS** |
| **Sum of Stage 1 donor counts == 10X cell count** | 48,138 | 48,138 | **PASS** |
| **Sum of Stage 3 donor counts == Stage 3 cell count** | 48,138 | 48,138 | **PASS** |
| **Sum of Primary donor counts == Primary cell count** | 16,553 | 16,553 | **PASS** |
| **Sum of Split partition counts == Primary cell count** | 16,553 | 16,553 | **PASS** |
| **Sum of Primary class counts == Primary cell count** | 16,553 | 16,553 | **PASS** |
| **raw_counts layer matches AnnData dimensions** | (50115, 58870) | (50115, 58870) | **PASS** |
| **Retained Donors Count (>= 12 requirement)** | >= 12 | **6 donors (Only 6 donors in TS Blood 10X)** | **FAIL** |
| **9:3:3 Donor Split Feasibility** | 9:3:3 Split | **Not Feasible (Max split: 4:1:1)** | **FAIL** |
| **Donor Disjointness across Train/Val/Test** | Mutually Disjoint | Mutually Disjoint | **PASS** |

---

## Filtering Funnel

| stage               | description                                   |   cells |   donors |   cells_dropped |
|:--------------------|:----------------------------------------------|--------:|---------:|----------------:|
| 0_raw_file          | Complete TS_Blood.h5ad                        |   50115 |        6 |               0 |
| 1_assay_filter      | Filter method == '10X' (remove Smart-seq2)    |   48138 |        6 |            1977 |
| 2_annotation_filter | Remove unassigned/missing cell_ontology_class |   48138 |        6 |               0 |
| 3_donor_threshold   | Filter donors with >= 200 cells               |   48138 |        6 |               0 |
| 4_primary_labels    | Retain 8 robust primary immune classes        |   16553 |        6 |           31585 |

---

## Primary Label Support Across Partitions (8 Classes)

| canonical_label                 |   train |   val |   test |   Total |
|:--------------------------------|--------:|------:|-------:|--------:|
| CD4-positive, alpha-beta T cell |    1132 |  1238 |      0 |    2370 |
| CD8-positive, alpha-beta T cell |     665 |   548 |     16 |    1229 |
| classical monocyte              |    5124 |     0 |   1875 |    6999 |
| memory B cell                   |     703 |    80 |     10 |     793 |
| naive B cell                    |    1736 |   338 |      5 |    2079 |
| natural killer cell             |    2320 |   452 |     99 |    2871 |
| non-classical monocyte          |       8 |     0 |      0 |       8 |
| platelet                        |     122 |     0 |     82 |     204 |
| Total                           |   11810 |  2656 |   2087 |   16553 |

---

## Donor Breakdown Across Proposed Split (4 Train / 1 Val / 1 Test)

- **Train Donors (4)**: `TSP7, TSP10, TSP1, TSP8` (11,810 cells)
- **Validation Donors (1)**: `TSP2` (2,656 cells)
- **Test Donors (1)**: `TSP14` (2,087 cells)

---

## Label Policy Summary

- **Primary Robust Classes (8)**: `CD4-positive, alpha-beta T cell, CD8-positive, alpha-beta T cell, classical monocyte, non-classical monocyte, natural killer cell, naive B cell, memory B cell, platelet`
- **Deferred Low-Support Classes (2)**: `plasmacytoid dendritic cell, myeloid dendritic cell`
- **Excluded Rare / Tissue Classes (17)**: `erythrocyte, cd4-positive, alpha-beta memory t cell, cd8-positive, alpha-beta cytokine secreting effector t cell, neutrophil, type i nk t cell, plasma cell, t cell, naive thymus-derived cd4-positive, alpha-beta t cell, hematopoietic stem cell, basophil, monocyte (unsubtyped), plasmablast, cd24 neutrophil, nampt neutrophil, myeloid progenitor, macrophage, granulocyte`
