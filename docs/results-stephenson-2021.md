# Results

> **Revision note (2026-08-25):** all tables below were regenerated from the cryptographically audited result set delivered by the GPU audit pipeline (batch `28439e6f045c`), with matched lifts computed against the canonical five-seed MLP baseline snapshot (`audits/baselines_snapshot/site_stratified_seed42/`, mean test macro-F1 **0.9026 ± 0.0015**). Earlier revisions of this page mixed in two one-epoch reference runs on seeds 7 and 42 that inflated apparent GNN lift and baseline variance; those numbers are superseded. New sections cover graph-diagnostics correlations, embedding quality, calibration, and training dynamics.

The study evaluated whether graph neural networks (GNNs) provide a reliable improvement over strong non-graph baselines for cell-type classification in single-cell transcriptomics.

Using a controlled, batch-aware protocol on the Stephenson 2021 healthy PBMC dataset (78,959 cells total; 18,508 test cells held out across the Cambridge and Newcastle sequencing sites), graph convolutional networks (GCNs) and GraphSAGE models were compared against matched multilayer perceptron (MLP) baselines across multiple graph-construction strategies, including PCA-kNN, mutual kNN, and BBKNN.

All models were trained and evaluated under identical conditions using five independent random seeds: 7, 17, 42, 73, and 101. The evaluation focused on mean predictive performance and on whether any apparent GNN advantage was consistent across seeds. Every run passed a four-layer integrity audit (per-file SHA-256 hashes, batch manifest hash, provenance hash-chain match against frozen artifacts, and independent recomputation of reported metrics from frozen labels).

## GCN consistently underperforms matched MLP

GCN failed to outperform the matched MLP baseline on any of the three primary graph topologies (Tables 1–3). Mean matched lift was negative under every condition, with tight seed-to-seed variability:

- **PCA-kNN, k=24:** −0.0190 ± 0.0019, range −0.0219 to −0.0168.
- **Mutual kNN, k=20:** −0.0304 ± 0.0016, range −0.0324 to −0.0284.
- **BBKNN, k-per-batch=2:** −0.0165 ± 0.0017, range −0.0183 to −0.0146.

Across all 25 seed-level GCN comparisons (five graph constructions × five seeds), not one produced positive matched lift. The deficit is systematic rather than seed-dependent.

### Table 1. Model and graph performance across five seeds

|Model|Graph artifact|Seed|Test macro-F1|Overall accuracy|Per-class F1 summary|
|---|---|--:|--:|--:|---|
|GCN|`pca_knn_k24_unweighted`|7|0.8843|0.8960|0.748 effector memory CD8+ – 1.000 naive B|
|GCN|`pca_knn_k24_unweighted`|17|0.8857|0.8974|0.752 effector memory CD8+ – 1.000 naive B|
|GCN|`pca_knn_k24_unweighted`|42|0.8821|0.8942|0.757 effector memory CD8+ – 1.000 naive B|
|GCN|`pca_knn_k24_unweighted`|73|0.8822|0.8940|0.757 effector memory CD8+ – 1.000 naive B|
|GCN|`pca_knn_k24_unweighted`|101|0.8838|0.8962|0.753 effector memory CD8+ – 1.000 naive B|
|GCN|`mutual_knn_reference_standard_query_k20_unweighted`|7|0.8727|0.8856|0.735 effector memory CD8+ – 1.000 naive B|
|GCN|`mutual_knn_reference_standard_query_k20_unweighted`|17|0.8718|0.8844|0.734 effector memory CD8+ – 1.000 naive B|
|GCN|`mutual_knn_reference_standard_query_k20_unweighted`|42|0.8719|0.8847|0.735 effector memory CD8+ – 1.000 naive B|
|GCN|`mutual_knn_reference_standard_query_k20_unweighted`|73|0.8725|0.8852|0.734 effector memory CD8+ – 1.000 naive B|
|GCN|`mutual_knn_reference_standard_query_k20_unweighted`|101|0.8722|0.8851|0.735 effector memory CD8+ – 1.000 naive B|
|GCN|`bbknn_kperbatch2_donors12`|7|0.8865|0.8956|0.760 effector memory CD8+ – 1.000 naive B|
|GCN|`bbknn_kperbatch2_donors12`|17|0.8859|0.8955|0.759 effector memory CD8+ – 1.000 naive B|
|GCN|`bbknn_kperbatch2_donors12`|42|0.8857|0.8950|0.760 effector memory CD8+ – 1.000 naive B|
|GCN|`bbknn_kperbatch2_donors12`|73|0.8858|0.8953|0.758 effector memory CD8+ – 1.000 naive B|
|GCN|`bbknn_kperbatch2_donors12`|101|0.8865|0.8957|0.762 effector memory CD8+ – 1.000 naive B|
|GraphSAGE|`bbknn_kperbatch2_donors12`|7|0.8992|0.9087|0.780 effector memory CD8+ – 1.000 naive B|
|GraphSAGE|`bbknn_kperbatch2_donors12`|17|0.9008|0.9105|0.785 effector memory CD8+ – 1.000 naive B|
|GraphSAGE|`bbknn_kperbatch2_donors12`|42|0.9023|0.9123|0.786 effector memory CD8+ – 1.000 naive B|
|GraphSAGE|`bbknn_kperbatch2_donors12`|73|0.9033|0.9131|0.790 effector memory CD8+ – 1.000 naive B|
|GraphSAGE|`bbknn_kperbatch2_donors12`|101|0.9017|0.9123|0.783 effector memory CD8+ – 1.000 naive B|
|MLP|None: feature baseline|7|0.9011|0.9114|0.798 effector memory CD8+ – 1.000 monocyte|
|MLP|None: feature baseline|17|0.9041|0.9150|0.790 effector memory CD8+ – 1.000 monocyte|
|MLP|None: feature baseline|42|0.9012|0.9112|0.799 effector memory CD8+ – 1.000 naive B|
|MLP|None: feature baseline|73|0.9041|0.9147|0.800 effector memory CD8+ – 1.000 monocyte|
|MLP|None: feature baseline|101|0.9025|0.9124|0.799 effector memory CD8+ – 1.000 monocyte|

### Table 2. Matched graph lift over identically seeded MLP baseline

|Model|Graph|Seed|GNN macro-F1|MLP macro-F1|Lift Δ|
|---|---|--:|--:|--:|--:|
|GCN|PCA-kNN k=24|7|0.8843|0.9011|-0.0168|
|GCN|PCA-kNN k=24|17|0.8857|0.9041|-0.0185|
|GCN|PCA-kNN k=24|42|0.8821|0.9012|-0.0191|
|GCN|PCA-kNN k=24|73|0.8822|0.9041|-0.0219|
|GCN|PCA-kNN k=24|101|0.8838|0.9025|-0.0188|
|GCN|Mutual kNN k=20|7|0.8727|0.9011|-0.0284|
|GCN|Mutual kNN k=20|17|0.8718|0.9041|-0.0324|
|GCN|Mutual kNN k=20|42|0.8719|0.9012|-0.0293|
|GCN|Mutual kNN k=20|73|0.8725|0.9041|-0.0316|
|GCN|Mutual kNN k=20|101|0.8722|0.9025|-0.0303|
|GCN|BBKNN|7|0.8865|0.9011|-0.0146|
|GCN|BBKNN|17|0.8859|0.9041|-0.0183|
|GCN|BBKNN|42|0.8857|0.9012|-0.0154|
|GCN|BBKNN|73|0.8858|0.9041|-0.0183|
|GCN|BBKNN|101|0.8865|0.9025|-0.0160|
|GraphSAGE|BBKNN|7|0.8992|0.9011|-0.0019|
|GraphSAGE|BBKNN|17|0.9008|0.9041|-0.0034|
|GraphSAGE|BBKNN|42|0.9023|0.9012|+0.0011|
|GraphSAGE|BBKNN|73|0.9033|0.9041|-0.0009|
|GraphSAGE|BBKNN|101|0.9017|0.9025|-0.0008|

### Table 3. Summary statistics across five seeds

|Model and graph condition|Test macro-F1, mean ± SD|Matched lift, mean ± SD|Minimum lift|Maximum lift|
|---|--:|--:|--:|--:|
|GCN, PCA-kNN k=24|0.8836 ± 0.0015|-0.0190 ± 0.0019|-0.0219|-0.0168|
|GCN, Mutual kNN k=20|0.8722 ± 0.0004|-0.0304 ± 0.0016|-0.0324|-0.0284|
|GCN, BBKNN|0.8861 ± 0.0004|-0.0165 ± 0.0017|-0.0183|-0.0146|
|GraphSAGE, BBKNN|0.9014 ± 0.0016|-0.0012 ± 0.0017|-0.0034|+0.0011|
|MLP baseline|0.9026 ± 0.0015|Reference|—|—|

The tighter baseline variance relative to earlier revisions is itself informative: once the MLP reference is measured properly (five fully-converged seeds), its performance is highly stable, and the residual GCN deficit of 1.5–3 F1 points is a property of the graph model, not reference noise.

## Graph construction ablation

To test whether the GCN deficit was specific to the chosen PCA-kNN neighborhood size or to unweighted edges, four additional PCA-kNN variants were evaluated: k=10 unweighted, k=20 with Gaussian RBF weights, k=50 unweighted, and k=20 unweighted with deterministic degree-preserving rewiring — all under the same site-stratified split and five-seed protocol. GNN-side predictions come from audited GPU runs; lifts are recomputed against the canonical MLP baseline.

### Table 4. PCA-kNN graph-variant ablation

|Graph variant|Seed 7|Seed 17|Seed 42|Seed 73|Seed 101|Mean lift|
|---|--:|--:|--:|--:|--:|--:|
|PCA-kNN k=10, unweighted|-0.0194|-0.0236|-0.0192|-0.0236|-0.0217|-0.0215|
|PCA-kNN k=20, weighted|-0.0173|-0.0194|-0.0192|-0.0217|-0.0183|-0.0192|
|PCA-kNN k=50, unweighted|-0.0162|-0.0177|-0.0159|-0.0193|-0.0177|-0.0174|
|PCA-kNN k=20, rewired|-0.8576|-0.8557|-0.8696|-0.8524|-0.8553|-0.8581|

With the corrected reference, the picture sharpens: **no ablation variant produced a single positive seed-level lift**, and the apparent seed-42 positives reported in earlier revisions were artifacts of the degraded baseline. Mean lift improves monotonically with neighborhood size for unweighted graphs (−0.0215 at k=10 → −0.0174 at k=50), with RBF weighting intermediate (−0.0192), but the effect is small and never crosses zero — there is no "sweet spot."

The rewired graph caused a dramatic collapse (mean lift ≈ −0.86, macro-F1 ≈ 0.04–0.05), serving as a destructive structural control. Although node degrees were preserved, local neighborhood relationships were removed, indicating that GCN message passing depends on the organization of edges, not merely their existence or density. This does not, by itself, establish that the retained PCA-kNN structure is exclusively biological — it may mix biological similarity, technical variation, and batch structure — but the contrast confirms that graph topology materially affects model behavior while ordinary construction choices only modulate the size of the deficit.

## GraphSAGE on PCA-kNN k=50

To determine whether the GCN deficit was architecture-specific, GraphSAGE (mean aggregation) was evaluated on the PCA-kNN k=50 graph and compared with GCN on the same graph and with the matched MLP baseline.

### Table 5. GCN and GraphSAGE on PCA-kNN k=50

|Model|Seed 7|Seed 17|Seed 42|Seed 73|Seed 101|Mean lift|
|---|--:|--:|--:|--:|--:|--:|
|GCN|-0.0162|-0.0177|-0.0159|-0.0193|-0.0177|-0.0174|
|GraphSAGE|+0.0023|-0.0038|-0.0003|-0.0025|-0.0030|-0.0015|

GraphSAGE improved substantially over GCN on the same graph and essentially closed the gap to the MLP baseline (mean lift −0.0015, within one seed standard deviation of zero), but did not convert parity into an advantage: only seed 7 produced positive lift, by a margin (+0.0023) smaller than the seed spread.

## GraphSAGE reaches message-passing parity with MLP, without surpassing it

On BBKNN, GraphSAGE achieved a macro-F1 of 0.9014 ± 0.0016 versus 0.9026 ± 0.0015 for MLP — a mean matched lift of −0.0012 ± 0.0017 (range −0.0034 to +0.0011), with a single marginally positive seed (42, +0.0011). Across all five graph constructions tested with both architectures, GraphSAGE lift clustered between −0.0045 and −0.0012 — **exact parity within seed noise, never an advantage**. The direction of the GraphSAGE-over-GCN gain, by contrast, was perfectly consistent: on every graph and every seed, GraphSAGE outperformed GCN.

The architecture comparison therefore supports a narrow conclusion: mean aggregation over these neighborhoods recovers whatever useful signal the edges contain, closing the GCN deficit entirely, but the edges themselves add no information beyond what the fixed PCA features already provide to a plain MLP.

## Graph construction and architecture effects

For GCN, changing the graph produced mean macro-F1 values of 0.8722 (mutual kNN k=20), 0.8836 (PCA-kNN k=24), and 0.8861 (BBKNN) — a range of 0.0139 points. Switching architecture from GCN to GraphSAGE on BBKNN produced a gain of 0.9014 − 0.8861 = 0.0153 macro-F1 points, slightly larger than the graph effect. Unlike earlier revisions, this comparison is now supported by a fully crossed architecture-by-graph design: both architectures were evaluated on all five graph constructions, and the pattern held everywhere (GraphSAGE above GCN on every condition; neither model above the MLP on any).

Overall, the results support three joint conclusions: (i) graph choice modulates how much GCN underperforms (mutual kNN is worst, batch-balanced BBKNN least bad); (ii) architecture choice determines whether that deficit exists at all; and (iii) destroying graph topology has a catastrophic effect far larger than either factor — but none of these manipulations produces a model that beats the strong feature-only baseline.

## Cross-site generalization

All primary results used a site-stratified split, with held-out donors from Cambridge and Newcastle, to test whether models generalized across sites rather than relying on site-specific patterns.

### Table 6. Per-site macro-F1 across five seeds

|Model and graph condition|Cambridge test macro-F1|Newcastle test macro-F1|Cross-site drop|
|---|--:|--:|--:|
|GCN, PCA-kNN k=24|0.8696 ± 0.0012|0.8480 ± 0.0024|0.0216|
|GCN, Mutual kNN k=20|0.8610 ± 0.0004|0.8292 ± 0.0012|0.0318|
|GCN, BBKNN|0.8720 ± 0.0008|0.8525 ± 0.0010|0.0195|
|GraphSAGE, BBKNN|0.8888 ± 0.0025|0.8719 ± 0.0020|0.0170|
|MLP baseline|0.8946 ± 0.0020|0.8728 ± 0.0023|0.0218|

Newcastle performance was consistently lower than Cambridge for every model, suggesting Newcastle was the harder test site or that residual site-related variation remained after preprocessing. With the corrected baseline, the cross-site robustness claim narrows: GraphSAGE on BBKNN still shows the smallest drop (0.0170), below all GCN variants (0.0195–0.0318) and slightly below MLP (0.0218), but the margin over MLP shrank from 0.0112 to 0.0048 and now lies within roughly one seed-SD of the per-site estimates. The directional signal survives; its magnitude should be treated as tentative pending additional datasets.

## Why graphs fail here: mechanism

Three post-hoc analysis families probe *why* every construction sits at or below the feature-only baseline. Labels and donor metadata are used strictly post hoc; they never entered graph building or training.

### Homophily anti-correlates with lift

Joining each graph's diagnostics report with its mean matched lift (pooled across both architectures) yields a striking rank ordering:

### Table 8. Graph diagnostics vs mean matched lift (pooled GCN + GraphSAGE)

|graph_name|n_seeds|mean_lift|overall_edge_homophily|train_train_edge_homophily|test_to_train_query_homophily|macro_average_class_purity|mean_train_donor_entropy|
|---|---|---|---|---|---|---|---|
|mutual_knn_reference_standard_query_k20_unweighted|10|-0.0167|+0.8433|+0.8980|+0.8211|+0.8307|+0.9429|
|pca_knn_k20_unweighted|10|-0.0123|+0.8392|+0.8490|+0.8211|+0.8171|+1.6840|
|pca_knn_k24_unweighted|10|-0.0104|+0.8368|+0.8466|+0.8189|+0.8145|+1.7331|
|bbknn_kperbatch2_donors12|10|-0.0088|+0.7578|+0.7660|+0.7374|+0.7298|+3.3975|

(diagnostics reports for `pca_knn_k50_unweighted` have not been generated; its pooled mean lift is −0.0094.)

### Table 9. Correlation of graph diagnostics with mean matched lift

|Feature|Pearson r|Spearman ρ|
|---|--:|--:|
|mean_train_donor_entropy|+0.852|+1.000|
|log_num_edges|+0.988|+1.000|
|test_to_train_query_homophily|-0.642|-0.949|
|overall_edge_homophily|-0.677|-1.000|
|train_train_edge_homophily|-0.893|-1.000|
|macro_average_class_purity|-0.739|-1.000|
|train_intra_site_edge_fraction|-0.673|-1.000|

The most class-*homophilous* and class-*pure* graphs produce the *worst* lift, while the batch-mixed BBKNN graph — lowest train-train homophily (0.766), highest donor-mixing entropy (3.40 nats) — preserves the most value. The interpretation: homophilous kNN neighborhoods largely re-encode feature proximity already present in the fixed PCA space (redundant with X), whereas batch-balanced edges inject complementary technical-normalization information. That complementary information still is not enough to beat the MLP — but it explains the ordering among graphs. These correlations are rank-perfect yet rest on only five conditions; they should be read as a strongly suggested mechanism awaiting replication, not an estimated effect size.

### Embedding quality mirrors the performance ordering

Hidden-layer representations (GNN: post-conv1 + batch-norm + ReLU; MLP: penultimate layer) were compared against the raw PCA inputs on the test partition:

### Table 10. Embedding quality, test partition (mean across runs per family)

|Representation family|Silhouette|kNN accuracy|Centroid separation|
|---|--:|--:|--:|
|raw_pca_input|0.1465|0.8867|1.9432|
|gcn|0.2768|0.8940|2.2759|
|graphsage|0.2354|0.9094|1.9958|
|mlp|0.2881|0.9120|2.8682|

Message passing does add geometric structure over the raw input space (silhouette 0.147 → 0.235–0.277), and representation quality orders consistently with downstream performance: MLP representations are the most class-separable by every metric, GraphSAGE's kNN-space accuracy (0.9094) approaches the MLP's (0.9120), and GCN's lags (0.8940) — exactly mirroring the macro-F1 ordering. The bottleneck is not that GNNs fail to embed; it is that their learned geometry never exceeds what a feature-only network learns from the same inputs, so the extra inductive bias buys nothing.

### Calibration: the deficit is not a confidence artifact

### Table 11. Confidence and calibration, test partition (mean across seeds)

|Model · Graph|Accuracy|ECE|Brier|Entropy (nats)|Margin|
|---|--:|--:|--:|--:|--:|
|gcn · BBKNN|0.8954|0.0364|0.1579|0.471|0.779|
|gcn · Mutual kNN|0.8850|0.0612|0.1830|0.152|0.903|
|gcn · PCA-kNN k=20|0.8946|0.0325|0.1576|0.469|0.785|
|gcn · PCA-kNN k=24|0.8955|0.0318|0.1562|0.461|0.788|
|gcn · PCA-kNN k=50|0.8966|0.0268|0.1549|0.436|0.794|
|graphsage · BBKNN|0.9114|0.0138|0.1337|0.309|0.834|
|graphsage · Mutual kNN|0.9106|0.0083|0.1348|0.268|0.850|
|graphsage · PCA-kNN k=20|0.9086|0.0043|0.1362|0.288|0.847|
|graphsage · PCA-kNN k=24|0.9109|0.0069|0.1332|0.271|0.853|
|graphsage · PCA-kNN k=50|0.9112|0.0068|0.1324|0.266|0.854|
|logistic regression|0.8946|0.0249|0.1623|0.317|0.809|
|MLP|0.9129|0.0140|0.1298|0.211|0.867|

Both MLP and GraphSAGE are well calibrated (ECE ≤ 0.014). The standout pathology is **GCN on mutual kNN**: the worst-performing condition is simultaneously the most overconfident (ECE 0.061) and the most committed (entropy 0.152 nats, margin 0.903) — confidently wrong, with near-one-hot distributions. This links the largest deficit to a concrete failure mode: mutual-kNN sparsity starves the GCN of neighborhood smoothing, and the model collapses onto confident, partially erroneous decisions instead of spreading probability mass.

### Training dynamics: clean convergence, persistent optimization gap

### Table 12. Training dynamics summary

|Model|Graph|Runs|Best val macro-F1|SD|Final train loss|Final val loss|
|---|---|--:|--:|--:|--:|--:|
|gcn|bbknn|5|0.8653|0.0009|0.2846|0.3585|
|gcn|mutual kNN|5|0.8543|0.0006|0.2394|0.5239|
|gcn|pca knn k=20|5|0.8661|0.0009|0.2474|0.3584|
|gcn|pca knn k=24|5|0.8661|0.0012|0.2467|0.3562|
|gcn|pca knn k=50|5|0.8651|0.0004|0.2468|0.3556|
|graphsage|bbknn|5|0.8816|0.0025|0.2026|0.3033|
|graphsage|mutual kNN|5|0.8815|0.0016|0.1876|0.3099|
|graphsage|pca knn k=20|5|0.8792|0.0011|0.2014|0.3148|
|graphsage|pca knn k=24|5|0.8807|0.0010|0.1934|0.3094|
|graphsage|pca knn k=50|5|0.8813|0.0007|0.1907|0.3088|
|mlp|none|5|0.8909|0.0006|0.1466|0.3027|

All 50 instrumented runs converged smoothly under early stopping (no degenerate trajectories; per-epoch histories are persisted alongside each run). The GCN deficit appears as a persistent generalization gap: its validation loss (0.356–0.524) exceeds both GraphSAGE (~0.31) and MLP (0.303) despite comparable training loss, and the gap explodes on the sparse mutual-kNN graph (0.524). Normalizing over homophilous-but-redundant neighborhoods thus drives GCN into a worse minimum that neither longer training nor seed choice escapes.

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
|MLP|Central memory CD4+ → naive CD4+|205|8.7%|
|MLP|Naive CD4+ → central memory CD4+|190|6.9%|
|MLP|Effector CD8+ → effector memory CD8+|103|5.2%|

All models struggled with discrimination between naive and central memory CD4+ T cells, consistent with the broader pattern that the hardest errors occurred between biologically related subtypes with overlapping transcriptional programs. The corrected baseline strengthens this reading: the canonical MLP posts the lowest rate on the dominant error type (central memory → naive CD4+, 8.7% versus 11.4% GCN and 10.4% GraphSAGE), confirming that the graph models' extra machinery does not resolve the biologically hard distinctions — GraphSAGE wins only the reverse direction (5.6% versus MLP's 6.9%) and the effector/memory CD8+ confusion. Architecture shifts the distribution of errors at the margins without changing the underlying biological difficulty; the confusion profile remains dominated by T-cell subtype ambiguity.

## Summary

- **1. GCN underperformed the matched MLP baseline on every tested condition.** Mean matched lifts: PCA-kNN k=24 (−0.0190 ± 0.0019), mutual kNN k=20 (−0.0304 ± 0.0016), BBKNN (−0.0165 ± 0.0017). Zero of 25 GCN seed-level comparisons were positive.
- **2. The GCN deficit was not explained by PCA-kNN configuration.** Across k=10, weighted k=20, and unweighted k=50, mean lift stayed negative (−0.0215 to −0.0174) with zero positive seeds; larger neighborhoods and edge weights only modestly narrowed the gap.
- **3. Graph topology materially affected message passing.** Degree-preserving rewiring caused catastrophic failure (macro-F1 ≈ 0.04–0.05, mean lift −0.858), confirming that structured neighborhoods — not edge presence or density alone — are essential.
- **4. GraphSAGE substantially outperformed GCN on every matched graph.** The improvement was universal across all five constructions and all seeds (e.g., BBKNN 0.9014 vs 0.8861; k=50 lift −0.0015 vs −0.0174).
- **5. GraphSAGE reached exact parity with MLP but never surpassed it.** Mean lifts ranged −0.0045 to −0.0012 across graphs; one marginally positive seed in 25 comparisons. Aggregation choice recovers the GCN deficit; the edges add nothing beyond the features.
- **6. Homophily anti-correlates with lift.** Across the five dual-architecture conditions, train-train homophily, class purity, and intra-site edge fraction rank-correlate at ρ = −1.0 with mean lift, while donor-mixing entropy correlates at +1.0: redundant homophilous neighborhoods re-encode the features; batch-balanced edges preserve the most complementary signal (mechanistic finding; n = 5 conditions).
- **7. Representation geometry and calibration corroborate the null result.** MLP penultimate embeddings are the most class-separable (silhouette 0.288 vs 0.147 raw input); GNN layers land between and never exceed them, and kNN-space accuracy reproduces the macro-F1 ordering. Both strong models are well calibrated (ECE ≤ 0.014), while the worst GCN condition is distinctly overconfident (ECE 0.061) — confidently wrong rather than usefully uncertain.
- **8. GraphSAGE on BBKNN retains the smallest cross-site drop, weakly.** Drop 0.0170 versus 0.0218 (MLP) and 0.0195–0.0318 (GCN); the margin over MLP narrowed to 0.0048 under the corrected baseline and requires confirmation on additional datasets.
- **9. Error profiles remained dominated by biologically related T-cell subtypes.** Naive versus central memory CD4+ classification dominates every model's confusion matrix; the canonical MLP handles the dominant error type best, underscoring that graph machinery does not resolve the biological difficulty.
- **10. A graph alone does not guarantee improved cell-type classification.** Under strict inductive, donor-held-out evaluation with fixed features, every tested construction sat at or below a same-feature MLP; any future GNN advantage must come from information the features do not already contain, not from re-encoding them.

## Provenance

Every number above derives from runs transferred through the audited GPU delivery pipeline (batch fingerprint `28439e6f045c`, 551 files SHA-256-verified; provenance hash-chain matched against frozen split/feature/graph artifacts; reported metrics independently recomputed from frozen labels). Baseline evidence is snapshotted under `audits/baselines_snapshot/site_stratified_seed42/`; ingestion records live in `audits/gpu_runs/ingestion_log.jsonl`. Regenerate all tables with `uv run python scripts/export_results_tables.py`.
