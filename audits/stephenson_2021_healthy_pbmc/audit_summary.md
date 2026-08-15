# Stephenson 2021 Healthy PBMC Audit Summary

- **Dataset Identifier**: `stephenson_2021_healthy_pbmc`
- **Source Study**: Stephenson et al. (*Nature Medicine* 2021, E-MTAB-10026)
- **Census Version**: `2025-11-08`
- **H5AD Cache Size**: `1,193,845,026` bytes
- **Generated Timestamp**: `2026-08-15T22:26:31.023094+00:00`

---

## Reconciliation & Invariant Checks

| Check | Expected | Actual / Verified | Status |
| :--- | :--- | :--- | :--- |
| **Sum of Stage 0 donor counts == healthy cell count** | 104,923 | 104,923 | **PASS** |
| **Sum of Stage 1 donor counts == unperturbed cell count** | 97,039 | 97,039 | **PASS** |
| **Sum of Stage 3 donor counts == Stage 3 cell count** | 97,039 | 97,039 | **PASS** |
| **Sum of Primary donor counts == Primary cell count** | 95,749 | 95,749 | **PASS** |
| **Sum of Split partition counts == Primary cell count** | 95,749 | 95,749 | **PASS** |
| **Sum of Primary class counts == Primary cell count** | 95,749 | 95,749 | **PASS** |
| **Raw count matrix is sparse integer counts** | True | True | **PASS** |
| **Retained Donors Count (>= 12 requirement)** | >= 12 | **23 donors** | **PASS** |
| **12:6:6 / 14:5:5 Donor Split Feasibility** | Feasible | **Feasible (12 Train / 6 Val / 5 Test)** | **PASS** |
| **Donor Disjointness across Train/Val/Test** | Mutually Disjoint | Mutually Disjoint | **PASS** |

---

## Filtering Funnel

| stage                | description                                                   |   cells |   donors |   cells_dropped |
|:---------------------|:--------------------------------------------------------------|--------:|---------:|----------------:|
| 0_raw_healthy_census | Stephenson PBMC healthy subset (disease == 'normal')          |  104923 |       29 |               0 |
| 1_cohort_filter      | Exclude IVLPS challenge donors (retain true healthy controls) |   97039 |       23 |            7884 |
| 2_donor_threshold    | Retain donors with >= 200 cells                               |   97039 |       23 |               0 |
| 3_annotation_filter  | Remove unassigned/missing cell_type annotations               |   97039 |       23 |               0 |
| 4_primary_labels     | Retain 26 robust primary classes (>= 10 donor presence)       |   95749 |       23 |            1290 |

---

## Primary Label Support Across Partitions (26 Classes)

| cell_type                                             |   train |   val |   test |   Total |
|:------------------------------------------------------|--------:|------:|-------:|--------:|
| B cell                                                |     135 |    56 |     54 |     245 |
| CD14-low, CD16-positive monocyte                      |     137 |    10 |     51 |     198 |
| CD14-positive monocyte                                |    6902 |   929 |   2715 |   10546 |
| CD16-negative, CD56-bright natural killer cell, human |    1260 |   346 |    353 |    1959 |
| CD16-positive, CD56-dim natural killer cell, human    |    8294 |  2124 |   1957 |   12375 |
| ILC1, human                                           |     125 |    65 |     23 |     213 |
| T follicular helper cell                              |     286 |   379 |    107 |     772 |
| T-helper 22 cell                                      |    2683 |  2318 |   1297 |    6298 |
| central memory CD4-positive, alpha-beta T cell        |    2591 |  4204 |    642 |    7437 |
| class switched memory B cell                          |     612 |   295 |    142 |    1049 |
| dendritic cell                                        |      80 |    24 |     22 |     126 |
| dendritic cell, human                                 |     625 |   137 |    213 |     975 |
| effector CD8-positive, alpha-beta T cell              |    4400 |   681 |   1064 |    6145 |
| effector memory CD8-positive, alpha-beta T cell       |    2581 |  1539 |   1290 |    5410 |
| gamma-delta T cell                                    |    2681 |   875 |   1177 |    4733 |
| immature B cell                                       |     370 |   142 |    138 |     650 |
| mature NK T cell                                      |    2532 |   251 |   1047 |    3830 |
| mucosal invariant T cell                              |    1833 |   722 |    806 |    3361 |
| myeloid dendritic cell                                |     663 |   158 |    231 |    1052 |
| naive B cell                                          |    2728 |  1289 |    870 |    4887 |
| naive thymus-derived CD4-positive, alpha-beta T cell  |    6750 |  2669 |   3533 |   12952 |
| naive thymus-derived CD8-positive, alpha-beta T cell  |    3780 |  2324 |   1328 |    7432 |
| natural killer cell                                   |     128 |    20 |     57 |     205 |
| plasmacytoid dendritic cell                           |     452 |   124 |    102 |     678 |
| platelet                                              |    1098 |   324 |    300 |    1722 |
| unswitched memory B cell                              |     246 |   115 |    138 |     499 |
| Total                                                 |   53972 | 22120 |  19657 |   95749 |

---

## Donor Breakdown Across Proposed Split (12 Train / 6 Val / 5 Test)

- **Train Donors (12)**: `C-8946, C-8940, C-8936, C-8942, C-8930, CV0902, CV0940, CV0917, C-8939, CV0911, CV0929, C-8914` (53,972 cells)
- **Validation Donors (6)**: `C-8943, CV0904, CV0934, CV0926, CV0915, CV0944` (22,120 cells)
- **Test Donors (5)**: `C-8928, C-8937, C-8938, C-8941, CV0939` (19,657 cells)

---

## Label Policy Summary

- **Primary Robust Classes (26)**: `B cell, CD14-low, CD16-positive monocyte, CD14-positive monocyte, CD16-negative, CD56-bright natural killer cell, human, CD16-positive, CD56-dim natural killer cell, human, ILC1, human, T follicular helper cell, T-helper 22 cell, central memory CD4-positive, alpha-beta T cell, class switched memory B cell, dendritic cell, dendritic cell, human, effector CD8-positive, alpha-beta T cell, effector memory CD8-positive, alpha-beta T cell, gamma-delta T cell, immature B cell, mature NK T cell, mucosal invariant T cell, myeloid dendritic cell, naive B cell, naive thymus-derived CD4-positive, alpha-beta T cell, naive thymus-derived CD8-positive, alpha-beta T cell, natural killer cell, plasmacytoid dendritic cell, platelet, unswitched memory B cell`
- **Deferred Low-Support Classes (11)**: `CD34-positive, CD38-negative hematopoietic stem cell, CD8-positive, alpha-beta T cell, IgA plasma cell, IgG plasma cell, IgM plasma cell, T-helper 1 cell, effector memory CD4-positive, alpha-beta T cell, erythrocyte, erythroid progenitor cell, mammalian, plasmablast, regulatory T cell`
- **Excluded Sparse Classes (7)**: `CD4-positive, alpha-beta T cell, T-helper 17 cell, T-helper 2 cell, group 2 innate lymphoid cell, human, hematopoietic precursor cell, monocyte, myeloid lineage restricted progenitor cell`
