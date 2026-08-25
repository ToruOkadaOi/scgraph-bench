# scgraph-bench Analysis Report — 2026-08-25

Dataset: `stephenson_2021_healthy_pbmc` · Split: `site_stratified_seed42`

## 1. Headline: Matched Graph Lift

| model_name | graph_name | n_seeds | gnn_f1 | mlp_f1 | lift | lift_sd |
|---|---|---|---|---|---|---|
| gcn | bbknn_kperbatch2_donors12 | 5 | 0.8861 | 0.9026 | -0.0165 | 0.0017 |
| gcn | mutual_knn_reference_standard_query_k20_unweighted | 5 | 0.8722 | 0.9026 | -0.0304 | 0.0016 |
| gcn | pca_knn_k20_unweighted | 5 | 0.8825 | 0.9026 | -0.0201 | 0.0008 |
| gcn | pca_knn_k24_unweighted | 5 | 0.8836 | 0.9026 | -0.0190 | 0.0019 |
| gcn | pca_knn_k50_unweighted | 5 | 0.8852 | 0.9026 | -0.0174 | 0.0014 |
| graphsage | bbknn_kperbatch2_donors12 | 5 | 0.9014 | 0.9026 | -0.0012 | 0.0017 |
| graphsage | mutual_knn_reference_standard_query_k20_unweighted | 5 | 0.8996 | 0.9026 | -0.0031 | 0.0028 |
| graphsage | pca_knn_k20_unweighted | 5 | 0.8982 | 0.9026 | -0.0045 | 0.0015 |
| graphsage | pca_knn_k24_unweighted | 5 | 0.9008 | 0.9026 | -0.0018 | 0.0024 |
| graphsage | pca_knn_k50_unweighted | 5 | 0.9011 | 0.9026 | -0.0015 | 0.0025 |

## 2. Per-Class Findings (GNN − MLP ΔF1)

**Classes most hurt by graphs:**

| class_name | mean_delta_f1 |
|---|---|
| mucosal invariant T cell | -0.0457 |
| gamma-delta T cell | -0.0323 |
| effector memory CD8-positive, alpha-beta T cell | -0.0303 |
| naive thymus-derived CD8-positive, alpha-beta T cell | -0.0247 |
| naive thymus-derived CD4-positive, alpha-beta T cell | -0.0190 |
| central memory CD4-positive, alpha-beta T cell | -0.0129 |

**Classes most helped by graphs:**

| class_name | mean_delta_f1 |
|---|---|
| CD16-negative, CD56-bright natural killer cell, human | 0.0186 |
| platelet | 0.0097 |
| CD16-positive, CD56-dim natural killer cell, human | 0.0030 |
| naive B cell | 0.0004 |

## 3. Confidence & Calibration (Test)

| model_name | graph_name | seed | accuracy | ece | brier_score | mean_entropy_nats | mean_margin |
|---|---|---|---|---|---|---|---|
| gcn | bbknn_kperbatch2_donors12 | 101 | 0.8957 | 0.0352 | 0.1572 | 0.4652 | 0.7811 |
| gcn | bbknn_kperbatch2_donors12 | 17 | 0.8955 | 0.0371 | 0.1580 | 0.4745 | 0.7781 |
| gcn | bbknn_kperbatch2_donors12 | 42 | 0.8950 | 0.0348 | 0.1574 | 0.4662 | 0.7802 |
| gcn | bbknn_kperbatch2_donors12 | 7 | 0.8956 | 0.0368 | 0.1577 | 0.4722 | 0.7787 |
| gcn | bbknn_kperbatch2_donors12 | 73 | 0.8953 | 0.0384 | 0.1589 | 0.4789 | 0.7767 |
| gcn | mutual_knn_reference_standard_query_k20_unweighted | 101 | 0.8851 | 0.0612 | 0.1834 | 0.1513 | 0.9033 |
| gcn | mutual_knn_reference_standard_query_k20_unweighted | 17 | 0.8844 | 0.0610 | 0.1843 | 0.1550 | 0.9014 |
| gcn | mutual_knn_reference_standard_query_k20_unweighted | 42 | 0.8847 | 0.0618 | 0.1828 | 0.1513 | 0.9034 |
| gcn | mutual_knn_reference_standard_query_k20_unweighted | 7 | 0.8856 | 0.0613 | 0.1823 | 0.1503 | 0.9039 |
| gcn | mutual_knn_reference_standard_query_k20_unweighted | 73 | 0.8852 | 0.0607 | 0.1822 | 0.1526 | 0.9025 |
| gcn | pca_knn_k20_unweighted | 101 | 0.8939 | 0.0379 | 0.1600 | 0.4955 | 0.7780 |
| gcn | pca_knn_k20_unweighted | 17 | 0.8957 | 0.0319 | 0.1568 | 0.4627 | 0.7864 |
| gcn | pca_knn_k20_unweighted | 42 | 0.8943 | 0.0324 | 0.1575 | 0.4680 | 0.7848 |
| gcn | pca_knn_k20_unweighted | 7 | 0.8935 | 0.0344 | 0.1587 | 0.4828 | 0.7814 |
| gcn | pca_knn_k20_unweighted | 73 | 0.8957 | 0.0260 | 0.1548 | 0.4341 | 0.7949 |
| gcn | pca_knn_k24_unweighted | 101 | 0.8962 | 0.0303 | 0.1552 | 0.4462 | 0.7914 |
| gcn | pca_knn_k24_unweighted | 17 | 0.8974 | 0.0303 | 0.1547 | 0.4463 | 0.7914 |
| gcn | pca_knn_k24_unweighted | 42 | 0.8942 | 0.0321 | 0.1567 | 0.4669 | 0.7856 |
| gcn | pca_knn_k24_unweighted | 7 | 0.8960 | 0.0257 | 0.1541 | 0.4330 | 0.7947 |
| gcn | pca_knn_k24_unweighted | 73 | 0.8940 | 0.0408 | 0.1603 | 0.5111 | 0.7752 |
| gcn | pca_knn_k50_unweighted | 101 | 0.8964 | 0.0262 | 0.1546 | 0.4339 | 0.7942 |
| gcn | pca_knn_k50_unweighted | 17 | 0.8970 | 0.0295 | 0.1557 | 0.4469 | 0.7909 |
| gcn | pca_knn_k50_unweighted | 42 | 0.8967 | 0.0253 | 0.1541 | 0.4284 | 0.7955 |
| gcn | pca_knn_k50_unweighted | 7 | 0.8964 | 0.0273 | 0.1555 | 0.4397 | 0.7926 |
| gcn | pca_knn_k50_unweighted | 73 | 0.8964 | 0.0255 | 0.1547 | 0.4324 | 0.7952 |
| graphsage | bbknn_kperbatch2_donors12 | 101 | 0.9123 | 0.0102 | 0.1313 | 0.2843 | 0.8446 |
| graphsage | bbknn_kperbatch2_donors12 | 17 | 0.9105 | 0.0139 | 0.1341 | 0.3122 | 0.8322 |
| graphsage | bbknn_kperbatch2_donors12 | 42 | 0.9123 | 0.0116 | 0.1315 | 0.2975 | 0.8386 |
| graphsage | bbknn_kperbatch2_donors12 | 7 | 0.9087 | 0.0206 | 0.1398 | 0.3562 | 0.8152 |
| graphsage | bbknn_kperbatch2_donors12 | 73 | 0.9131 | 0.0125 | 0.1316 | 0.2927 | 0.8403 |
| graphsage | mutual_knn_reference_standard_query_k20_unweighted | 101 | 0.9094 | 0.0071 | 0.1354 | 0.2689 | 0.8508 |
| graphsage | mutual_knn_reference_standard_query_k20_unweighted | 17 | 0.9125 | 0.0097 | 0.1330 | 0.2574 | 0.8539 |
| graphsage | mutual_knn_reference_standard_query_k20_unweighted | 42 | 0.9076 | 0.0060 | 0.1379 | 0.2833 | 0.8447 |
| graphsage | mutual_knn_reference_standard_query_k20_unweighted | 7 | 0.9124 | 0.0098 | 0.1335 | 0.2621 | 0.8509 |
| graphsage | mutual_knn_reference_standard_query_k20_unweighted | 73 | 0.9112 | 0.0087 | 0.1343 | 0.2670 | 0.8507 |
| graphsage | pca_knn_k20_unweighted | 101 | 0.9079 | 0.0042 | 0.1362 | 0.2850 | 0.8486 |
| graphsage | pca_knn_k20_unweighted | 17 | 0.9117 | 0.0070 | 0.1318 | 0.2582 | 0.8573 |
| graphsage | pca_knn_k20_unweighted | 42 | 0.9076 | 0.0043 | 0.1382 | 0.3005 | 0.8430 |
| graphsage | pca_knn_k20_unweighted | 7 | 0.9069 | 0.0031 | 0.1384 | 0.3062 | 0.8394 |
| graphsage | pca_knn_k20_unweighted | 73 | 0.9087 | 0.0032 | 0.1364 | 0.2912 | 0.8460 |
| graphsage | pca_knn_k24_unweighted | 101 | 0.9123 | 0.0072 | 0.1318 | 0.2593 | 0.8572 |
| graphsage | pca_knn_k24_unweighted | 17 | 0.9113 | 0.0084 | 0.1322 | 0.2631 | 0.8559 |
| graphsage | pca_knn_k24_unweighted | 42 | 0.9128 | 0.0086 | 0.1315 | 0.2551 | 0.8586 |
| graphsage | pca_knn_k24_unweighted | 7 | 0.9076 | 0.0039 | 0.1378 | 0.3058 | 0.8396 |
| graphsage | pca_knn_k24_unweighted | 73 | 0.9106 | 0.0063 | 0.1328 | 0.2701 | 0.8535 |
| graphsage | pca_knn_k50_unweighted | 101 | 0.9102 | 0.0058 | 0.1338 | 0.2749 | 0.8520 |
| graphsage | pca_knn_k50_unweighted | 17 | 0.9103 | 0.0063 | 0.1327 | 0.2665 | 0.8544 |
| graphsage | pca_knn_k50_unweighted | 42 | 0.9108 | 0.0079 | 0.1328 | 0.2636 | 0.8556 |
| graphsage | pca_knn_k50_unweighted | 7 | 0.9129 | 0.0068 | 0.1308 | 0.2575 | 0.8555 |
| graphsage | pca_knn_k50_unweighted | 73 | 0.9118 | 0.0073 | 0.1318 | 0.2672 | 0.8544 |
| logistic_regression | none | 42 | 0.8946 | 0.0249 | 0.1623 | 0.3170 | 0.8088 |
| mlp | none | 42 | 0.9112 | 0.0195 | 0.1324 | 0.1962 | 0.8736 |
| mlp | none | 101 | 0.9124 | 0.0076 | 0.1298 | 0.2391 | 0.8540 |
| mlp | none | 17 | 0.9150 | 0.0147 | 0.1278 | 0.1994 | 0.8718 |
| mlp | none | 42 | 0.9112 | 0.0195 | 0.1324 | 0.1962 | 0.8736 |
| mlp | none | 7 | 0.9114 | 0.0151 | 0.1325 | 0.2109 | 0.8656 |
| mlp | none | 73 | 0.9147 | 0.0132 | 0.1264 | 0.2087 | 0.8689 |

## 4. Per-Donor Stability

| model_name | min | mean | max |
|---|---|---|---|
| gcn | 0.7873 | 0.8746 | 0.9859 |
| graphsage | 0.8234 | 0.8933 | 0.9860 |
| logistic_regression | 0.7979 | 0.8762 | 0.9837 |
| mlp | 0.8357 | 0.8978 | 0.9865 |

## 5. Graph Diagnostics vs Lift

| graph_name | n_seeds | mean_lift | std_lift | overall_edge_homophily | test_to_train_query_homophily | macro_average_class_purity |
|---|---|---|---|---|---|---|
| mutual_knn_reference_standard_query_k20_unweighted | 10 | -0.0167 | 0.0146 | 0.8433 | 0.8211 | 0.8307 |
| pca_knn_k20_unweighted | 10 | -0.0123 | 0.0083 | 0.8392 | 0.8211 | 0.8171 |
| pca_knn_k24_unweighted | 10 | -0.0104 | 0.0093 | 0.8368 | 0.8189 | 0.8145 |
| bbknn_kperbatch2_donors12 | 10 | -0.0088 | 0.0082 | 0.7578 | 0.7374 | 0.7298 |
| pca_knn_k50_unweighted | 10 | -0.0094 | 0.0086 | - | - | - |

**Correlations with mean lift:**

| feature | pearson_r | spearman_rho |
|---|---|---|
| mean_train_donor_entropy | 0.8524 | 1.0000 |
| log_num_edges | 0.9883 | 1.0000 |
| test_to_train_query_homophily | -0.6424 | -0.9487 |
| overall_edge_homophily | -0.6771 | -1.0000 |
| train_train_edge_homophily | -0.8928 | -1.0000 |
| macro_average_class_purity | -0.7394 | -1.0000 |
| train_intra_site_edge_fraction | -0.6726 | -1.0000 |

## 6. Embedding Quality (Test)

| representation_name | silhouette_euclidean | knn_accuracy | centroid_separation | mean_class_radius |
|---|---|---|---|---|
| gcn:bbknn_kperbatch2_donors12 | 0.2651 | 0.8968 | 2.2797 | 3.2236 |
| gcn:bbknn_kperbatch2_donors12 | 0.2688 | 0.8967 | 2.2597 | 3.3384 |
| gcn:bbknn_kperbatch2_donors12 | 0.2550 | 0.8966 | 2.2205 | 3.2523 |
| gcn:bbknn_kperbatch2_donors12 | 0.2697 | 0.8957 | 2.2907 | 3.2157 |
| gcn:bbknn_kperbatch2_donors12 | 0.2629 | 0.8946 | 2.2622 | 3.2965 |
| gcn:mutual_knn_reference_standard_query_k20_unweighted | 0.2976 | 0.8870 | 2.3541 | 5.1197 |
| gcn:mutual_knn_reference_standard_query_k20_unweighted | 0.2992 | 0.8865 | 2.3175 | 5.2769 |
| gcn:mutual_knn_reference_standard_query_k20_unweighted | 0.2879 | 0.8866 | 2.3067 | 5.1664 |
| gcn:mutual_knn_reference_standard_query_k20_unweighted | 0.3034 | 0.8853 | 2.3218 | 5.1085 |
| gcn:mutual_knn_reference_standard_query_k20_unweighted | 0.2923 | 0.8857 | 2.3242 | 5.2345 |
| gcn:pca_knn_k20_unweighted | 0.2748 | 0.8948 | 2.3075 | 3.7009 |
| gcn:pca_knn_k20_unweighted | 0.2743 | 0.8971 | 2.2203 | 3.9642 |
| gcn:pca_knn_k20_unweighted | 0.2605 | 0.8963 | 2.2059 | 3.8384 |
| gcn:pca_knn_k20_unweighted | 0.2721 | 0.8948 | 2.2543 | 3.7714 |
| gcn:pca_knn_k20_unweighted | 0.2649 | 0.8964 | 2.2142 | 3.9607 |
| gcn:pca_knn_k24_unweighted | 0.2745 | 0.8984 | 2.2861 | 3.7513 |
| gcn:pca_knn_k24_unweighted | 0.2767 | 0.8971 | 2.2250 | 3.9650 |
| gcn:pca_knn_k24_unweighted | 0.2644 | 0.8960 | 2.2234 | 3.8131 |
| gcn:pca_knn_k24_unweighted | 0.2708 | 0.8957 | 2.2418 | 3.8325 |
| gcn:pca_knn_k24_unweighted | 0.2713 | 0.8958 | 2.2811 | 3.8184 |
| gcn:pca_knn_k50_unweighted | 0.2856 | 0.8951 | 2.3356 | 3.6985 |
| gcn:pca_knn_k50_unweighted | 0.2900 | 0.8958 | 2.2933 | 3.8853 |
| gcn:pca_knn_k50_unweighted | 0.2715 | 0.8943 | 2.2567 | 3.8229 |
| gcn:pca_knn_k50_unweighted | 0.2844 | 0.8946 | 2.3161 | 3.7576 |
| gcn:pca_knn_k50_unweighted | 0.2823 | 0.8965 | 2.2978 | 3.8707 |
| graphsage:bbknn_kperbatch2_donors12 | 0.2154 | 0.9099 | 1.8520 | 5.1808 |
| graphsage:bbknn_kperbatch2_donors12 | 0.2448 | 0.9093 | 2.0735 | 4.8556 |
| graphsage:bbknn_kperbatch2_donors12 | 0.2309 | 0.9093 | 2.0018 | 4.9590 |
| graphsage:bbknn_kperbatch2_donors12 | 0.2636 | 0.9091 | 2.1217 | 4.5837 |
| graphsage:bbknn_kperbatch2_donors12 | 0.2229 | 0.9108 | 1.8980 | 5.0508 |
| graphsage:mutual_knn_reference_standard_query_k20_unweighted | 0.2182 | 0.9101 | 1.9622 | 5.0530 |
| graphsage:mutual_knn_reference_standard_query_k20_unweighted | 0.2138 | 0.9102 | 1.9670 | 5.0982 |
| graphsage:mutual_knn_reference_standard_query_k20_unweighted | 0.2365 | 0.9096 | 2.0858 | 4.8716 |
| graphsage:mutual_knn_reference_standard_query_k20_unweighted | 0.2189 | 0.9074 | 1.9093 | 5.0559 |
| graphsage:mutual_knn_reference_standard_query_k20_unweighted | 0.2104 | 0.9093 | 1.8969 | 5.1241 |
| graphsage:pca_knn_k20_unweighted | 0.2471 | 0.9086 | 2.0606 | 4.7190 |
| graphsage:pca_knn_k20_unweighted | 0.2262 | 0.9098 | 1.9680 | 5.0152 |
| graphsage:pca_knn_k20_unweighted | 0.2578 | 0.9081 | 2.1518 | 4.6571 |
| graphsage:pca_knn_k20_unweighted | 0.2689 | 0.9092 | 2.1325 | 4.5417 |
| graphsage:pca_knn_k20_unweighted | 0.2470 | 0.9096 | 2.0239 | 4.7392 |
| graphsage:pca_knn_k24_unweighted | 0.2227 | 0.9099 | 1.9150 | 4.9925 |
| graphsage:pca_knn_k24_unweighted | 0.2344 | 0.9097 | 2.0136 | 4.9257 |
| graphsage:pca_knn_k24_unweighted | 0.2256 | 0.9090 | 1.9402 | 5.0120 |
| graphsage:pca_knn_k24_unweighted | 0.2703 | 0.9088 | 2.1366 | 4.5310 |
| graphsage:pca_knn_k24_unweighted | 0.2307 | 0.9101 | 1.9402 | 4.9057 |
| graphsage:pca_knn_k50_unweighted | 0.2414 | 0.9085 | 2.0106 | 4.7987 |
| graphsage:pca_knn_k50_unweighted | 0.2395 | 0.9095 | 2.0316 | 4.8804 |
| graphsage:pca_knn_k50_unweighted | 0.2373 | 0.9101 | 2.0000 | 4.8949 |
| graphsage:pca_knn_k50_unweighted | 0.2307 | 0.9097 | 1.8898 | 4.9428 |
| graphsage:pca_knn_k50_unweighted | 0.2291 | 0.9103 | 1.9136 | 4.9428 |
| mlp | 0.2747 | 0.9114 | 2.8086 | 8.7216 |
| mlp | 0.3116 | 0.9123 | 2.9689 | 8.0263 |
| mlp | 0.2842 | 0.9134 | 2.9212 | 8.4817 |
| mlp | 0.2747 | 0.9114 | 2.8086 | 8.7216 |
| mlp | 0.2791 | 0.9110 | 2.7980 | 8.6134 |
| mlp | 0.3041 | 0.9124 | 2.9041 | 8.2383 |
| raw_pca_input | 0.1465 | 0.8867 | 1.9432 | 9.6050 |

## 7. Training Dynamics

| model_name | graph_name | n_runs | best_val_f1_mean | best_val_f1_std | epochs_trained_mean | final_train_loss | final_val_loss |
|---|---|---|---|---|---|---|---|
| gcn | bbknn_kperbatch2_donors12 | 5.0000 | 0.8653 | 0.0009 | 497.6000 | 0.2846 | 0.3585 |
| gcn | mutual_knn_reference_standard_query_k20_unweighted | 5.0000 | 0.8543 | 0.0006 | 500.0000 | 0.2394 | 0.5239 |
| gcn | pca_knn_k20_unweighted | 5.0000 | 0.8661 | 0.0009 | 419.2000 | 0.2474 | 0.3584 |
| gcn | pca_knn_k24_unweighted | 5.0000 | 0.8661 | 0.0012 | 441.4000 | 0.2467 | 0.3562 |
| gcn | pca_knn_k50_unweighted | 5.0000 | 0.8651 | 0.0004 | 500.0000 | 0.2468 | 0.3556 |
| graphsage | bbknn_kperbatch2_donors12 | 5.0000 | 0.8816 | 0.0025 | 428.8000 | 0.2026 | 0.3033 |
| graphsage | mutual_knn_reference_standard_query_k20_unweighted | 5.0000 | 0.8815 | 0.0016 | 473.8000 | 0.1876 | 0.3099 |
| graphsage | pca_knn_k20_unweighted | 5.0000 | 0.8792 | 0.0011 | 384.6000 | 0.2014 | 0.3148 |
| graphsage | pca_knn_k24_unweighted | 5.0000 | 0.8807 | 0.0010 | 459.8000 | 0.1934 | 0.3094 |
| graphsage | pca_knn_k50_unweighted | 5.0000 | 0.8813 | 0.0007 | 486.8000 | 0.1907 | 0.3088 |
| mlp | none | 5.0000 | 0.8909 | 0.0006 | 78.4000 | 0.1466 | 0.3027 |

## Reproduction Commands

```bash
uv run python scripts/compute_graph_lift_offline.py --dataset stephenson_2021_healthy_pbmc --split site_stratified_seed42
uv run python scripts/analyze_per_class_batch.py --dataset stephenson_2021_healthy_pbmc --split site_stratified_seed42
uv run python scripts/join_diagnostics_results.py --dataset stephenson_2021_healthy_pbmc --split site_stratified_seed42
uv run python scripts/analyze_embeddings.py --dataset stephenson_2021_healthy_pbmc --split site_stratified_seed42
uv run python scripts/analyze_training_dynamics.py --dataset stephenson_2021_healthy_pbmc --split site_stratified_seed42
uv run python scripts/generate_final_report.py --dataset stephenson_2021_healthy_pbmc --split site_stratified_seed42
```
