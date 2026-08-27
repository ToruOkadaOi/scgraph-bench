# GPU Delivery Audit Report

- Batch hash: ``
- Runs audited: 57
- Ingested: 0
- Quarantined: 7

## logistic_regression_hpv_stratified_seed42 — FAIL

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ❌ | run=f34c6551fceb… local=102f86568dba… |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.826812 recomputed=0.826812 Δ=0.00e+00 |
| optional_training_history.csv | ⚠️ | absent |
| optional_embeddings_test.npy | ⚠️ | absent |

## mlp_hpv_stratified_seed42_seed101 — FAIL

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ❌ | run=f34c6551fceb… local=102f86568dba… |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.835715 recomputed=0.835715 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## mlp_hpv_stratified_seed42_seed17 — FAIL

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ❌ | run=f34c6551fceb… local=102f86568dba… |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.837082 recomputed=0.837082 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## mlp_hpv_stratified_seed42_seed42 — FAIL

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ❌ | run=f34c6551fceb… local=102f86568dba… |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.839151 recomputed=0.839151 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## mlp_hpv_stratified_seed42_seed42 — FAIL

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ❌ | run=f34c6551fceb… local=102f86568dba… |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.839151 recomputed=0.839151 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## mlp_hpv_stratified_seed42_seed7 — FAIL

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ❌ | run=f34c6551fceb… local=102f86568dba… |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.833839 recomputed=0.833839 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## mlp_hpv_stratified_seed42_seed73 — FAIL

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ❌ | run=f34c6551fceb… local=102f86568dba… |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.834487 recomputed=0.834487 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_bbknn_kperbatch2_donors9_seed101 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | bbknn_kperbatch2_donors9 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.815949 recomputed=0.815949 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_bbknn_kperbatch2_donors9_seed17 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | bbknn_kperbatch2_donors9 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.816251 recomputed=0.816251 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_bbknn_kperbatch2_donors9_seed42 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | bbknn_kperbatch2_donors9 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.817194 recomputed=0.817194 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_bbknn_kperbatch2_donors9_seed7 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | bbknn_kperbatch2_donors9 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.817823 recomputed=0.817823 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_bbknn_kperbatch2_donors9_seed73 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | bbknn_kperbatch2_donors9 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.815129 recomputed=0.815129 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_mutual_knn_reference_standard_query_k20_unweighted_seed101 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | mutual_knn_reference_standard_query_k20_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.813912 recomputed=0.813912 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_mutual_knn_reference_standard_query_k20_unweighted_seed17 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | mutual_knn_reference_standard_query_k20_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.815098 recomputed=0.815098 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_mutual_knn_reference_standard_query_k20_unweighted_seed42 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | mutual_knn_reference_standard_query_k20_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.814066 recomputed=0.814066 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_mutual_knn_reference_standard_query_k20_unweighted_seed7 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | mutual_knn_reference_standard_query_k20_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.814794 recomputed=0.814794 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_mutual_knn_reference_standard_query_k20_unweighted_seed73 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | mutual_knn_reference_standard_query_k20_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.815818 recomputed=0.815818 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_pca_knn_k20_unweighted_seed101 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k20_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.816065 recomputed=0.816065 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_pca_knn_k20_unweighted_seed17 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k20_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.814308 recomputed=0.814308 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_pca_knn_k20_unweighted_seed42 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k20_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.816318 recomputed=0.816318 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_pca_knn_k20_unweighted_seed7 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k20_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.817392 recomputed=0.817392 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_pca_knn_k20_unweighted_seed73 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k20_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.815306 recomputed=0.815306 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_pca_knn_k24_unweighted_seed101 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k24_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.812353 recomputed=0.812353 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_pca_knn_k24_unweighted_seed17 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k24_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.814117 recomputed=0.814117 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_pca_knn_k24_unweighted_seed42 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k24_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.816039 recomputed=0.816039 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_pca_knn_k24_unweighted_seed7 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k24_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.815473 recomputed=0.815473 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_pca_knn_k24_unweighted_seed73 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k24_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.814925 recomputed=0.814925 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_rewired_control_pca_knn_seed42_seed101 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | rewired_control_pca_knn_seed42 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.031415 recomputed=0.031415 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_rewired_control_pca_knn_seed42_seed17 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | rewired_control_pca_knn_seed42 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.030653 recomputed=0.030653 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_rewired_control_pca_knn_seed42_seed42 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | rewired_control_pca_knn_seed42 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.036696 recomputed=0.036696 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_rewired_control_pca_knn_seed42_seed7 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | rewired_control_pca_knn_seed42 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.047134 recomputed=0.047134 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_rewired_control_pca_knn_seed42_seed73 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | rewired_control_pca_knn_seed42 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.026136 recomputed=0.026136 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_bbknn_kperbatch2_donors9_seed101 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | bbknn_kperbatch2_donors9 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.826044 recomputed=0.826044 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_bbknn_kperbatch2_donors9_seed17 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | bbknn_kperbatch2_donors9 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.827257 recomputed=0.827257 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_bbknn_kperbatch2_donors9_seed42 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | bbknn_kperbatch2_donors9 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.834850 recomputed=0.834850 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_bbknn_kperbatch2_donors9_seed7 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | bbknn_kperbatch2_donors9 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.829832 recomputed=0.829832 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_bbknn_kperbatch2_donors9_seed73 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | bbknn_kperbatch2_donors9 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.831567 recomputed=0.831567 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_mutual_knn_reference_standard_query_k20_unweighted_seed101 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | mutual_knn_reference_standard_query_k20_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.833847 recomputed=0.833847 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_mutual_knn_reference_standard_query_k20_unweighted_seed17 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | mutual_knn_reference_standard_query_k20_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.831355 recomputed=0.831355 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_mutual_knn_reference_standard_query_k20_unweighted_seed42 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | mutual_knn_reference_standard_query_k20_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.832285 recomputed=0.832285 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_mutual_knn_reference_standard_query_k20_unweighted_seed7 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | mutual_knn_reference_standard_query_k20_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.833366 recomputed=0.833366 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_mutual_knn_reference_standard_query_k20_unweighted_seed73 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | mutual_knn_reference_standard_query_k20_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.833489 recomputed=0.833489 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_pca_knn_k20_unweighted_seed101 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k20_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.819334 recomputed=0.819334 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_pca_knn_k20_unweighted_seed17 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k20_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.828531 recomputed=0.828531 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_pca_knn_k20_unweighted_seed42 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k20_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.827832 recomputed=0.827832 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_pca_knn_k20_unweighted_seed7 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k20_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.829170 recomputed=0.829170 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_pca_knn_k20_unweighted_seed73 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k20_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.829238 recomputed=0.829238 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_pca_knn_k24_unweighted_seed101 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k24_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.826165 recomputed=0.826165 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_pca_knn_k24_unweighted_seed17 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k24_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.831320 recomputed=0.831320 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_pca_knn_k24_unweighted_seed42 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k24_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.826687 recomputed=0.826687 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_pca_knn_k24_unweighted_seed7 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k24_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.828845 recomputed=0.828845 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_pca_knn_k24_unweighted_seed73 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k24_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.828909 recomputed=0.828909 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_rewired_control_pca_knn_seed42_seed101 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | rewired_control_pca_knn_seed42 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.813875 recomputed=0.813875 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_rewired_control_pca_knn_seed42_seed17 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | rewired_control_pca_knn_seed42 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.812129 recomputed=0.812129 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_rewired_control_pca_knn_seed42_seed42 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | rewired_control_pca_knn_seed42 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.821228 recomputed=0.821228 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_rewired_control_pca_knn_seed42_seed7 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | rewired_control_pca_knn_seed42 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.823572 recomputed=0.823572 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_rewired_control_pca_knn_seed42_seed73 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | rewired_control_pca_knn_seed42 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.802362 recomputed=0.802362 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |
