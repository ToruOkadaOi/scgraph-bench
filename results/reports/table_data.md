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

### Table 4. PCA-kNN graph-variant ablation

|Graph variant|Seed 7|Seed 17|Seed 42|Seed 73|Seed 101|Mean lift|
|---|--:|--:|--:|--:|--:|--:|
|PCA-kNN k=10, unweighted|-0.0194|-0.0236|-0.0192|-0.0236|-0.0217|-0.0215|
|PCA-kNN k=20, weighted|-0.0173|-0.0194|-0.0192|-0.0217|-0.0183|-0.0192|
|PCA-kNN k=50, unweighted|-0.0162|-0.0177|-0.0159|-0.0193|-0.0177|-0.0174|
|PCA-kNN k=20, rewired|-0.8576|-0.8557|-0.8696|-0.8524|-0.8553|-0.8581|

### Table 5. GCN and GraphSAGE on PCA-kNN k=50

|Model|Seed 7|Seed 17|Seed 42|Seed 73|Seed 101|Mean lift|
|---|--:|--:|--:|--:|--:|--:|
|GCN|-0.0162|-0.0177|-0.0159|-0.0193|-0.0177|-0.0174|
|GraphSAGE|+0.0023|-0.0038|-0.0003|-0.0025|-0.0030|-0.0015|

### Table 6. Per-site macro-F1 across five seeds

|Model and graph condition|Cambridge test macro-F1|Newcastle test macro-F1|Cross-site drop|
|---|--:|--:|--:|
|GCN, PCA-kNN k=24|0.8696 ± 0.0012|0.8480 ± 0.0024|0.0216|
|GCN, Mutual kNN k=20|0.8610 ± 0.0004|0.8292 ± 0.0012|0.0318|
|GCN, BBKNN|0.8720 ± 0.0008|0.8525 ± 0.0010|0.0195|
|GraphSAGE, BBKNN|0.8888 ± 0.0025|0.8719 ± 0.0020|0.0170|
|MLP baseline|0.8946 ± 0.0020|0.8728 ± 0.0023|0.0218|

### Table 7. Top three misclassifications by model at seed 42

|Model|Error type|Cells misclassified|Error rate within source class|
|---|---|--:|--:|
|GCN, BBKNN|central memory CD4+ → naive CD4+|268|11.4%|
|GCN, BBKNN|naive CD4+ → central memory CD4+|196|7.1%|
|GCN, BBKNN|naive CD8+ → naive CD4+|143|7.9%|
|GraphSAGE, BBKNN|central memory CD4+ → naive CD4+|246|10.4%|
|GraphSAGE, BBKNN|naive CD4+ → central memory CD4+|155|5.6%|
|GraphSAGE, BBKNN|effector CD8+ → effector memory CD8+|81|4.1%|
|MLP|central memory CD4+ → naive CD4+|205|8.7%|
|MLP|naive CD4+ → central memory CD4+|190|6.9%|
|MLP|effector CD8+ → effector memory CD8+|103|5.2%|

### Table 8. Graph diagnostics vs mean matched lift (one row per graph)

|graph_name|n_seeds|mean_lift|overall_edge_homophily|train_train_edge_homophily|test_to_train_query_homophily|macro_average_class_purity|mean_train_donor_entropy|
|---|---|---|---|---|---|---|---|
|mutual_knn_reference_standard_query_k20_unweighted|10|-0.0167|+0.8433|+0.8980|+0.8211|+0.8307|+0.9429|
|pca_knn_k20_unweighted|10|-0.0123|+0.8392|+0.8490|+0.8211|+0.8171|+1.6840|
|pca_knn_k24_unweighted|10|-0.0104|+0.8368|+0.8466|+0.8189|+0.8145|+1.7331|
|bbknn_kperbatch2_donors12|10|-0.0088|+0.7578|+0.7660|+0.7374|+0.7298|+3.3975|
|pca_knn_k50_unweighted|10|-0.0094|+nan|+nan|+nan|+nan|+nan|

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

### Table 10. Embedding quality, test partition (mean across runs per family)

|Representation family|Silhouette|kNN accuracy|Centroid separation|
|---|--:|--:|--:|
|raw_pca_input|0.1465|0.8867|1.9432|
|mlp|0.2881|0.9120|2.8682|
|gcn|0.2768|0.8940|2.2759|
|graphsage|0.2354|0.9094|1.9958|

### Table 11. Confidence and calibration, test partition (mean across seeds)

|Model · Graph|Accuracy|ECE|Brier|Entropy (nats)|Margin|
|---|--:|--:|--:|--:|--:|
|gcn:bbknn kperbatch2 donors12|0.8954|0.0364|0.1579|0.471|0.779|
|gcn:mutual knn reference|0.8850|0.0612|0.1830|0.152|0.903|
|gcn:pca knn k20|0.8946|0.0325|0.1576|0.469|0.785|
|gcn:pca knn k24|0.8955|0.0318|0.1562|0.461|0.788|
|gcn:pca knn k50|0.8966|0.0268|0.1549|0.436|0.794|
|graphsage:bbknn kperbatch2 donors12|0.9114|0.0138|0.1337|0.309|0.834|
|graphsage:mutual knn reference|0.9106|0.0083|0.1348|0.268|0.850|
|graphsage:pca knn k20|0.9086|0.0043|0.1362|0.288|0.847|
|graphsage:pca knn k24|0.9109|0.0069|0.1332|0.271|0.853|
|graphsage:pca knn k50|0.9112|0.0068|0.1324|0.266|0.854|
|logistic_regression|0.8946|0.0249|0.1623|0.317|0.809|
|mlp|0.9129|0.0140|0.1298|0.211|0.867|

### Table 12. Training dynamics summary

|Model|Graph|Runs|Best val macro-F1|SD|Final train loss|Final val loss|
|---|---|--:|--:|--:|--:|--:|
|gcn|bbknn_kperbatch2_donors12|5|0.8653|0.0009|0.2846|0.3585|
|gcn|mutual_knn_reference_standard_query_k20_unweighted|5|0.8543|0.0006|0.2394|0.5239|
|gcn|pca_knn_k20_unweighted|5|0.8661|0.0009|0.2474|0.3584|
|gcn|pca_knn_k24_unweighted|5|0.8661|0.0012|0.2467|0.3562|
|gcn|pca_knn_k50_unweighted|5|0.8651|0.0004|0.2468|0.3556|
|graphsage|bbknn_kperbatch2_donors12|5|0.8816|0.0025|0.2026|0.3033|
|graphsage|mutual_knn_reference_standard_query_k20_unweighted|5|0.8815|0.0016|0.1876|0.3099|
|graphsage|pca_knn_k20_unweighted|5|0.8792|0.0011|0.2014|0.3148|
|graphsage|pca_knn_k24_unweighted|5|0.8807|0.0010|0.1934|0.3094|
|graphsage|pca_knn_k50_unweighted|5|0.8813|0.0007|0.1907|0.3088|
|mlp|none|5|0.8909|0.0006|0.1466|0.3027|
