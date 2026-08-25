# scgraph-bench Analysis Report — 2026-08-25

Dataset: `stephenson_2021_healthy_pbmc` · Split: `site_stratified_seed42`

## 1. Headline: Matched Graph Lift

| model_name | graph_name | n_seeds | gnn_f1 | mlp_f1 | lift | lift_sd |
|---|---|---|---|---|---|---|
| gcn | pca_knn_k10_unweighted | 5 | 0.8811 | 0.8915 | -0.0104 | 0.0106 |
| gcn | pca_knn_k20_unweighted_rewired | 5 | 0.0445 | 0.8915 | -0.8470 | 0.0071 |
| gcn | pca_knn_k20_weighted | 5 | 0.8834 | 0.8915 | -0.0081 | 0.0101 |
| gcn | pca_knn_k24_unweighted | 1 | 0.0716 | 0.8804 | -0.8088 | - |
| gcn | pca_knn_k50_unweighted | 5 | 0.8846 | 0.8915 | -0.0069 | 0.0116 |
| graphsage | bbknn_kperbatch2_donors12 | 5 | 0.7329 | 0.8915 | -0.1586 | 0.3702 |
| graphsage | pca_knn_k50_unweighted | 5 | 0.8991 | 0.8915 | 0.0075 | 0.0100 |

## 2. Per-Class Findings (GNN − MLP ΔF1)

**Classes most hurt by graphs:**

| class_name | mean_delta_f1 |
|---|---|
| naive B cell | -0.4000 |
| CD16-positive, CD56-dim natural killer cell, human | -0.3931 |
| effector CD8-positive, alpha-beta T cell | -0.3684 |
| naive thymus-derived CD8-positive, alpha-beta T cell | -0.3681 |
| CD16-negative, CD56-bright natural killer cell, human | -0.3607 |
| platelet | -0.3571 |

**Classes most helped by graphs:**

| class_name | mean_delta_f1 |
|---|---|
| CD14-positive monocyte | -0.1990 |
| gamma-delta T cell | -0.2572 |
| effector memory CD8-positive, alpha-beta T cell | -0.2961 |
| central memory CD4-positive, alpha-beta T cell | -0.3000 |

## 3. Confidence & Calibration (Test)

| model_name | graph_name | seed | accuracy | ece | brier_score | mean_entropy_nats | mean_margin |
|---|---|---|---|---|---|---|---|
| gcn | pca_knn_k10_unweighted | 17 | 0.8929 | 0.0304 | 0.1605 | 0.4687 | 0.7851 |
| gcn | pca_knn_k10_unweighted | 42 | 0.8944 | 0.0219 | 0.1577 | 0.4249 | 0.7966 |
| gcn | pca_knn_k10_unweighted | 7 | 0.8943 | 0.0244 | 0.1585 | 0.4371 | 0.7936 |
| gcn | pca_knn_k24_unweighted | 42 | 0.1327 | 0.1190 | 0.9352 | 2.1599 | 0.0752 |
| gcn | pca_knn_k24_unweighted | 999 | 0.1552 | 0.1021 | 0.9265 | 2.1578 | 0.0880 |
| graphsage | bbknn_kperbatch2_donors12 | 42 | 0.0642 | 0.1355 | 0.9388 | 2.2873 | 0.0508 |
| logistic_regression | none | 42 | 0.8946 | 0.0249 | 0.1623 | 0.3170 | 0.8088 |
| mlp | none | 42 | 0.8965 | 0.0231 | 0.1564 | 0.3789 | 0.7978 |
| mlp | none | 101 | 0.9134 | 0.0107 | 0.1300 | 0.2310 | 0.8585 |
| mlp | none | 17 | 0.9002 | 0.0228 | 0.1510 | 0.3640 | 0.8047 |
| mlp | none | 42 | 0.8965 | 0.0231 | 0.1564 | 0.3789 | 0.7978 |
| mlp | none | 7 | 0.9019 | 0.0242 | 0.1504 | 0.3693 | 0.8001 |
| mlp | none | 73 | 0.9121 | 0.0182 | 0.1321 | 0.1985 | 0.8707 |

## 4. Per-Donor Stability

| model_name | min | mean | max |
|---|---|---|---|
| gcn | 0.0446 | 0.5767 | 0.9818 |
| graphsage | 0.0291 | 0.0504 | 0.0763 |
| logistic_regression | 0.7979 | 0.8762 | 0.9837 |
| mlp | 0.7964 | 0.8824 | 0.9837 |

## 5. Graph Diagnostics vs Lift

| graph_name | n_seeds | mean_lift | std_lift | overall_edge_homophily | test_to_train_query_homophily | macro_average_class_purity |
|---|---|---|---|---|---|---|
| pca_knn_k24_unweighted | 1 | -0.8088 | 0.0000 | 0.8368 | 0.8189 | 0.8145 |
| bbknn_kperbatch2_donors12 | 1 | -0.8207 | 0.0000 | 0.7578 | 0.7374 | 0.7298 |
| pca_knn_k10_unweighted | 3 | -0.0030 | 0.0041 | - | - | - |

## 6. Embedding Quality (Test)

| representation_name | silhouette_euclidean | knn_accuracy | centroid_separation | mean_class_radius |
|---|---|---|---|---|
| gcn:pca_knn_k24_unweighted | 0.1794 | 0.8776 | 2.3435 | 4.0362 |
| gcn:pca_knn_k24_unweighted | 0.1975 | 0.8805 | 2.3454 | 4.1061 |
| graphsage:bbknn_kperbatch2_donors12 | 0.1460 | 0.8772 | 2.0205 | 6.1648 |
| mlp | 0.3449 | 0.9054 | 3.0466 | 5.9299 |
| mlp | 0.3394 | 0.9072 | 2.9622 | 6.1260 |
| mlp | 0.3449 | 0.9054 | 3.0466 | 5.9299 |
| mlp | 0.3381 | 0.9058 | 3.0517 | 6.0160 |
| raw_pca_input | 0.1465 | 0.8867 | 1.9432 | 9.6050 |

## 7. Training Dynamics

| model_name | graph_name | n_runs | best_val_f1_mean | best_val_f1_std | epochs_trained_mean | final_train_loss | final_val_loss |
|---|---|---|---|---|---|---|---|
| gcn | pca_knn_k24_unweighted | 2.0000 | 0.0781 | 0.0303 | 1.0000 | 2.7504 | 2.5984 |
| graphsage | bbknn_kperbatch2_donors12 | 1.0000 | 0.0165 | - | 1.0000 | 2.6510 | 2.6711 |
| mlp | none | 3.0000 | 0.8714 | 0.0035 | 1.0000 | 0.6995 | 0.3267 |

## Reproduction Commands

```bash
uv run python scripts/compute_graph_lift_offline.py --dataset stephenson_2021_healthy_pbmc --split site_stratified_seed42
uv run python scripts/analyze_per_class_batch.py --dataset stephenson_2021_healthy_pbmc --split site_stratified_seed42
uv run python scripts/join_diagnostics_results.py --dataset stephenson_2021_healthy_pbmc --split site_stratified_seed42
uv run python scripts/analyze_embeddings.py --dataset stephenson_2021_healthy_pbmc --split site_stratified_seed42
uv run python scripts/analyze_training_dynamics.py --dataset stephenson_2021_healthy_pbmc --split site_stratified_seed42
uv run python scripts/generate_final_report.py --dataset stephenson_2021_healthy_pbmc --split site_stratified_seed42
```
