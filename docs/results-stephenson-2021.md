# Results

The study evaluated whether graph neural networks (GNNs) provide a reliable improvement over strong non-graph baselines for cell-type classification in single-cell transcriptomics.

Using a controlled, batch-aware protocol on the Stephenson 2021 healthy PBMC dataset (N = 18,508 cells), test cells were held out across the Cambridge and Newcastle sequencing sites.

Graph convolutional networks (GCNs) and GraphSAGE models were compared against matched multilayer perceptron (MLP) baselines across multiple graph-construction strategies, including PCA-kNN, mutual kNN, and BBKNN.

All models were trained and evaluated under identical conditions using five independent random seeds: 7, 17, 42, 73, and 101. The evaluation focused on mean predictive performance and on whether any apparent GNN advantage was consistent across seeds.

## GCN consistently underperforms matched MLP

GCN failed to outperform the matched MLP baseline on any of the three primary graph topologies (Tables 1 and 2). Mean matched lift was negative under every condition:

- **PCA-kNN, k=24:** −0.0118 ± 0.0102, range −0.0194 to +0.0018.
- **Mutual kNN, k=20:** −0.0232 ± 0.0108, range −0.0327 to −0.0084.
- **BBKNN, k-per-batch=2:** −0.0093 ± 0.0107, range −0.0186 to +0.0054.

GCN achieved a positive matched lift in only 2 of 15 seed-level comparisons, both on BBKNN at seed 42 — indicating the occasional improvement was seed-dependent rather than systematic.

### Table 1. Model and graph performance across five seeds

|Model|Graph artifact|Seed|Test macro-F1|Overall accuracy|Per-class F1 summary|
|---|---|--:|--:|--:|---|
|GCN|`pca_knn_k24_unweighted`|7|0.8843|0.8960|0.748 EM CD8+ – 1.000 Monocyte/B cell|
|GCN|`pca_knn_k24_unweighted`|17|0.8857|0.8974|0.752 EM CD8+ – 1.000 Monocyte/B cell|
|GCN|`pca_knn_k24_unweighted`|42|0.8821|0.8942|0.757 EM CD8+ – 1.000 Monocyte/B cell|
|GCN|`pca_knn_k24_unweighted`|73|0.8822|0.8940|0.757 EM CD8+ – 1.000 Monocyte/B cell|
|GCN|`pca_knn_k24_unweighted`|101|0.8838|0.8962|0.753 EM CD8+ – 1.000 Monocyte/B cell|
|GCN|`mutual_knn_reference_standard_query_k20_unweighted`|7|0.8727|0.8856|0.735 EM CD8+ – 1.000 Monocyte/B cell|
|GCN|`mutual_knn_reference_standard_query_k20_unweighted`|17|0.8718|0.8844|0.734 EM CD8+ – 1.000 Monocyte/B cell|
|GCN|`mutual_knn_reference_standard_query_k20_unweighted`|42|0.8719|0.8847|0.735 EM CD8+ – 1.000 Monocyte/B cell|
|GCN|`mutual_knn_reference_standard_query_k20_unweighted`|73|0.8725|0.8852|0.734 EM CD8+ – 1.000 Monocyte/B cell|
|GCN|`mutual_knn_reference_standard_query_k20_unweighted`|101|0.8722|0.8851|0.735 EM CD8+ – 1.000 Monocyte/B cell|
|GCN|`bbknn_kperbatch2_donors12`|7|0.8865|0.8956|0.760 EM CD8+ – 1.000 Monocyte/B cell|
|GCN|`bbknn_kperbatch2_donors12`|17|0.8859|0.8955|0.759 EM CD8+ – 1.000 Monocyte/B cell|
|GCN|`bbknn_kperbatch2_donors12`|42|0.8857|0.8950|0.760 EM CD8+ – 1.000 Monocyte/B cell|
|GCN|`bbknn_kperbatch2_donors12`|73|0.8858|0.8953|0.758 EM CD8+ – 1.000 Monocyte/B cell|
|GCN|`bbknn_kperbatch2_donors12`|101|0.8865|0.8957|0.762 EM CD8+ – 1.000 Monocyte/B cell|
|GraphSAGE|`bbknn_kperbatch2_donors12`|7|0.8992|0.9087|0.780 EM CD8+ – 1.000 Monocyte/B cell|
|GraphSAGE|`bbknn_kperbatch2_donors12`|17|0.9008|0.9105|0.785 EM CD8+ – 1.000 Monocyte/B cell|
|GraphSAGE|`bbknn_kperbatch2_donors12`|42|0.9023|0.9123|0.786 EM CD8+ – 1.000 Monocyte/B cell|
|GraphSAGE|`bbknn_kperbatch2_donors12`|73|0.9033|0.9131|0.790 EM CD8+ – 1.000 Monocyte/B cell|
|GraphSAGE|`bbknn_kperbatch2_donors12`|101|0.9017|0.9123|0.783 EM CD8+ – 1.000 Monocyte/B cell|
|MLP|None: feature baseline|7|0.8878|0.9019|0.746 EM CD8+ – 1.000 Monocyte/B cell|
|MLP|None: feature baseline|17|0.9045|0.9152|0.798 EM CD8+ – 1.000 Monocyte/B cell|
|MLP|None: feature baseline|42|0.8804|0.8965|0.733 EM CD8+ – 1.000 Monocyte/B cell|
|MLP|None: feature baseline|73|0.9016|0.9121|0.791 EM CD8+ – 1.000 Monocyte/B cell|
|MLP|None: feature baseline|101|0.9029|0.9134|0.803 Gamma-delta T – 1.000 Monocyte/B cell|

### Table 2. Matched graph lift over identically seeded MLP baseline

|Model|Graph|Seed|GNN macro-F1|MLP macro-F1|Lift Δ|
|---|---|--:|--:|--:|--:|
|GCN|PCA-kNN k=24|7|0.8843|0.8878|-0.0035|
|GCN|PCA-kNN k=24|17|0.8857|0.9045|-0.0188|
|GCN|PCA-kNN k=24|42|0.8821|0.8804|+0.0018|
|GCN|PCA-kNN k=24|73|0.8822|0.9016|-0.0194|
|GCN|PCA-kNN k=24|101|0.8838|0.9029|-0.0191|
|GCN|Mutual kNN k=20|7|0.8727|0.8878|-0.0151|
|GCN|Mutual kNN k=20|17|0.8718|0.9045|-0.0327|
|GCN|Mutual kNN k=20|42|0.8719|0.8804|-0.0084|
|GCN|Mutual kNN k=20|73|0.8725|0.9016|-0.0290|
|GCN|Mutual kNN k=20|101|0.8722|0.9029|-0.0307|
|GCN|BBKNN|7|0.8865|0.8878|-0.0014|
|GCN|BBKNN|17|0.8859|0.9045|-0.0186|
|GCN|BBKNN|42|0.8857|0.8804|+0.0054|
|GCN|BBKNN|73|0.8858|0.9016|-0.0157|
|GCN|BBKNN|101|0.8865|0.9029|-0.0164|
|GraphSAGE|BBKNN|7|0.8992|0.8878|+0.0113|
|GraphSAGE|BBKNN|17|0.9008|0.9045|-0.0037|
|GraphSAGE|BBKNN|42|0.9023|0.8804|+0.0220|
|GraphSAGE|BBKNN|73|0.9033|0.9016|+0.0017|
|GraphSAGE|BBKNN|101|0.9017|0.9029|-0.0012|

### Table 3. Summary statistics across five seeds

|Model and graph condition|Test macro-F1, mean ± SD|Matched lift, mean ± SD|Minimum lift|Maximum lift|
|---|--:|--:|--:|--:|
|GCN, PCA-kNN k=24|0.8836 ± 0.0015|-0.0118 ± 0.0102|-0.0194|+0.0018|
|GCN, mutual kNN k=20|0.8722 ± 0.0004|-0.0232 ± 0.0108|-0.0327|-0.0084|
|GCN, BBKNN|0.8861 ± 0.0004|-0.0093 ± 0.0107|-0.0186|+0.0054|
|GraphSAGE, BBKNN|0.9014 ± 0.0016|+0.0060 ± 0.0106|-0.0037|+0.0220|
|MLP baseline|0.8954 ± 0.0107|Reference|—|—|

## Graph construction ablation

To test whether the GCN deficit was specific to the chosen PCA-kNN neighborhood size or to unweighted edges, four additional PCA-kNN variants were evaluated: k=10 unweighted, k=20 with Gaussian RBF weights, k=50 unweighted, and k=20 unweighted with deterministic degree-preserving rewiring — all under the same site-stratified split and five-seed protocol.

### Table 4. PCA-kNN graph-variant ablation

|Graph variant|Seed 7|Seed 17|Seed 42|Seed 73|Seed 101|Mean lift|
|---|--:|--:|--:|--:|--:|--:|
|PCA-kNN k=10, unweighted|-0.0091|-0.0249|+0.0002|-0.0225|-0.0224|-0.0157|
|PCA-kNN k=20, weighted|-0.0040|-0.0190|+0.0016|-0.0206|-0.0190|-0.0122|
|PCA-kNN k=50, unweighted|-0.0031|-0.0169|+0.0050|-0.0218|-0.0183|-0.0110|
|PCA-kNN k=20, rewired|-0.8443|-0.8553|-0.8487|-0.8512|-0.8560|Approximately -0.85|

The three non-rewired variants produced the same qualitative pattern as the original PCA-kNN experiment, with GCN below MLP in four of five seeds each; the only non-negative result in each variant occurred at seed 42 (+0.0002, +0.0016, +0.0050 respectively). Mean lift became less negative as neighborhood size increased or edge weights were added (−0.0157 → −0.0122 → −0.0110), but the effect was small, seed-dependent, and did not produce a reliable GCN advantage — there was no clear "sweet spot."

The rewired graph, by contrast, caused a dramatic collapse (~−0.85 mean lift, macro-F1 ≈ 0.03–0.05), serving as a destructive structural control. Although node degrees were preserved, local neighborhood relationships were removed, indicating that GCN message passing depends on the organization of edges, not merely their existence or density. This does not, by itself, establish that the retained PCA-kNN structure is exclusively biological — it may mix biological similarity, technical variation, and batch structure — but the contrast confirms that graph topology materially affects model behavior. In short: real graph construction choices produce modest changes in GCN performance, while destroying neighborhood structure produces catastrophic failure, and the persistent deficit on intact graphs is not explained solely by the choice of k or weighting.

## GraphSAGE on PCA-kNN k=50

To determine whether the GCN deficit was architecture-specific, GraphSAGE (mean aggregation) was evaluated on the PCA-kNN k=50 graph and compared with GCN on the same graph and with the matched MLP baseline.

### Table 5. GCN and GraphSAGE on PCA-kNN k=50

|Model|Seed 7|Seed 17|Seed 42|Seed 73|Seed 101|Mean lift|
|---|--:|--:|--:|--:|--:|--:|
|GCN|-0.0031|-0.0169|+0.0050|-0.0218|-0.0183|-0.0110|
|GraphSAGE|+0.0097|-0.0027|+0.0174|-0.0011|-0.0060|+0.0035|

GraphSAGE improved substantially over GCN on the same graph (macro-F1 roughly 0.012–0.015 higher) and produced a positive mean lift (+0.0035) versus MLP, but the improvement was not fully consistent: positive lift occurred in only two of five seeds (7: +0.0097, 42: +0.0174), with small negative lifts at seeds 17, 73, and 101. This indicates the aggregation rule and architectural design influence how effectively the model uses neighborhood information, though architecture alone was not a silver bullet — GraphSAGE improved over GCN but did not consistently beat the MLP baseline.

## GraphSAGE restores message-passing parity with MLP

On BBKNN, GraphSAGE achieved a macro-F1 of 0.9014 ± 0.0016 versus 0.8954 ± 0.0107 for MLP, a mean matched lift of +0.0060 ± 0.0106 (range −0.0037 to +0.0220), with positive lift in 3 of 5 seeds — outperforming the best GCN condition by roughly 0.0153 macro-F1 points. The direction of the GraphSAGE-over-GCN gain was highly consistent, though the advantage over MLP itself was not: the largest gain was at seed 42 (+0.0220), while seeds 17 and 101 showed slight deficits (−0.0037, −0.0012).

Across both BBKNN and PCA-kNN k=50, GraphSAGE consistently improved over GCN and achieved a small positive mean lift over MLP, but that lift was positive in only 2–3 of 5 seeds each time — a modest, context-dependent advantage rather than a general claim that GraphSAGE consistently beats MLP.

## Graph construction and architecture effects

For GCN, changing the graph produced mean macro-F1 values of 0.8722 (mutual kNN k=20), 0.8836 (PCA-kNN k=24), and 0.8861 (BBKNN) — a range of 0.0139 points. Switching architecture from GCN to GraphSAGE on BBKNN produced a gain of 0.9014 − 0.8861 = 0.0153 macro-F1 points, slightly larger than the graph effect. This comparison should be read cautiously, since GraphSAGE was initially evaluated only on BBKNN and the two factors were not tested in a fully crossed architecture-by-graph design; the PCA-kNN k=50 experiment partially addresses this by comparing both architectures on the same graph, where mean lift moved from about −0.0110 (GCN) to +0.0035 (GraphSAGE).

Overall, the results support a joint role for graph construction and architecture: graph choice can depress GCN macro-F1 by more than 0.01 (e.g., mutual kNN vs. BBKNN), GCN underperforms MLP even on the best intact graphs tested, GraphSAGE improves over GCN on both BBKNN and k=50, GraphSAGE can reach parity with MLP but with modest, seed-dependent gains, and destroying graph topology has a far larger negative effect than ordinary changes in neighborhood size or weighting. Neither factor alone guarantees a benefit over a strong non-graph baseline.

## Cross-site generalization

All primary results used a site-stratified split, with held-out donors from Cambridge and Newcastle, to test whether models generalized across sites rather than relying on site-specific patterns.

### Table 6. Per-site macro-F1 across five seeds

|Model and graph condition|Cambridge test macro-F1|Newcastle test macro-F1|Cross-site drop|
|---|--:|--:|--:|
|GCN, PCA-kNN k=24|0.8696 ± 0.0011|0.8480 ± 0.0022|0.0216|
|GCN, mutual kNN k=20|0.8610 ± 0.0003|0.8292 ± 0.0011|0.0318|
|GCN, BBKNN|0.8720 ± 0.0007|0.8525 ± 0.0009|0.0195|
|GraphSAGE, BBKNN|0.8888 ± 0.0022|0.8719 ± 0.0018|0.0169|
|MLP baseline|0.8879 ± 0.0093|0.8598 ± 0.0178|0.0281|

Newcastle performance was consistently lower than Cambridge for every model, suggesting Newcastle was the harder test site or that residual site-related variation remained after preprocessing. GraphSAGE on BBKNN showed the smallest cross-site drop (0.0169), lower than MLP (0.0281) and all GCN variants (0.0195–0.0318), suggesting GraphSAGE message passing on a batch-balanced graph may improve robustness to site-level variation. This conclusion should remain cautious, since the smaller drop co-occurred with higher overall performance and only one dataset and split were evaluated.

## Error analysis

Confusion matrices for seed 42 (used as a representative seed) showed that the most common errors involved closely related T-cell subtypes, particularly naive and central memory CD4+ T cells.

### Table 7. Top three misclassifications by model at seed 42

|Model|Error type|Cells misclassified|Error rate within source class|
|---|---|--:|--:|
|GCN, BBKNN|Central memory CD4+ → naive CD4+|268|11.4%|
|GCN, BBKNN|Naive CD4+ → central memory CD4+|196|7.1%|
|GCN, BBKNN|Naive CD8+ → naive CD4+|143|7.9%|
|GraphSAGE, BBKNN|Central memory CD4+ → naive CD4+|246|10.4%|
|GraphSAGE, BBKNN|Naive CD4+ → central memory CD4+|155|5.6%|
|GraphSAGE, BBKNN|Effector CD8+ → effector memory CD8+|81|4.1%|
|MLP|Central memory CD4+ → naive CD4+|288|12.2%|
|MLP|Naive CD4+ → central memory CD4+|133|4.8%|
|MLP|Effector memory CD8+ → effector CD8+|117|12.1%|

All models struggled with discrimination between naive and central memory CD4+ T cells, consistent with the broader pattern that the hardest errors occurred between biologically related subtypes with overlapping transcriptional programs. GraphSAGE reduced the central memory CD4+ → naive CD4+ error rate relative to both GCN (11.4%) and MLP (12.2%), reaching 10.4%. For the reverse direction, GraphSAGE (5.6%) improved on GCN (7.1%) but not on MLP, which had the lowest rate (4.8%) — showing GraphSAGE was not uniformly best for every error type. GraphSAGE also showed a lower frequency of effector CD8+/effector memory CD8+ confusion than MLP among the top-error categories. These findings suggest architecture can shift the distribution of errors without changing the underlying biological difficulty of the task; the overall confusion profile remained dominated by T-cell subtype ambiguity.

## Summary

- **1. GCN underperformed the matched MLP baseline.** Mean matched lifts were negative across PCA-kNN \(k=24\) (−0.0118), mutual kNN \(k=20\) (−0.0232), and BBKNN (−0.0093). Positive lift occurred in only 2 of 15 seed-level comparisons, both for BBKNN at seed 42.
- **2. The GCN deficit was not explained by PCA-kNN configuration.** Across \(k=10\), weighted \(k=20\), and unweighted \(k=50\), GCN remained below MLP in four of five seeds. Mean lift improved only modestly, from −0.0157 to −0.0110.
- **3. Graph topology materially affected message passing.** Degree-preserving rewiring caused catastrophic failure, producing macro-F1 values of approximately 0.03–0.05 and matched lifts near −0.85. This indicates that structured neighborhoods, rather than edge presence or density alone, are essential.
- **4. GraphSAGE substantially outperformed GCN on matched graphs.** Mean matched lift was +0.0060 on BBKNN and approximately +0.0035 on PCA-kNN \(k=50\), compared with −0.0093 and −0.0110 for the corresponding GCN conditions.
- **5. GraphSAGE reached approximate parity with MLP but did not establish a robust advantage.** Positive lift occurred in 3 of 5 seeds on BBKNN and 2 of 5 seeds on PCA-kNN \(k=50\); improvements were modest and some seeds showed slight deficits.
- **6. GraphSAGE on BBKNN showed the smallest observed cross-site performance drop.** The drop was 0.0169, compared with 0.0281 for MLP and 0.0195–0.0318 for the GCN variants. This preliminary robustness signal requires confirmation on additional datasets and splits.
- **7. Error profiles remained dominated by biologically related T-cell subtypes.** All models struggled particularly with naive versus central memory CD4+ classification. GraphSAGE shifted some confusion frequencies but did not eliminate the underlying biological difficulty.
- **8. A graph alone does not guarantee improved cell-type classification; any GNN advantage depends jointly on graph construction, architecture, and rigorous multi-seed, cross-site evaluation.**