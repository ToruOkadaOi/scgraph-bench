# GPU Delivery Audit Report

- Batch hash: `28439e6f045c6d32ffdb5ba15a426a934d7cb40a34ac6b64a5f32c7ebafe392a`
- Runs audited: 54
- Ingested: 2
- Quarantined: 0

## gcn_bbknn_kperbatch2_donors12_seed101 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | bbknn_kperbatch2_donors12 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.886524 recomputed=0.886524 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_bbknn_kperbatch2_donors12_seed17 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | bbknn_kperbatch2_donors12 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.885878 recomputed=0.885878 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_bbknn_kperbatch2_donors12_seed42 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | bbknn_kperbatch2_donors12 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.885739 recomputed=0.885739 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_bbknn_kperbatch2_donors12_seed7 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | bbknn_kperbatch2_donors12 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.886461 recomputed=0.886461 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_bbknn_kperbatch2_donors12_seed73 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | bbknn_kperbatch2_donors12 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.885847 recomputed=0.885847 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.872188 recomputed=0.872188 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.871769 recomputed=0.871769 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.871905 recomputed=0.871905 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.872686 recomputed=0.872686 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.872535 recomputed=0.872535 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.881476 recomputed=0.881476 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.884201 recomputed=0.884201 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.882282 recomputed=0.882282 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.880921 recomputed=0.880921 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.883850 recomputed=0.883850 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.883782 recomputed=0.883782 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.885684 recomputed=0.885684 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.882117 recomputed=0.882117 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.884282 recomputed=0.884282 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.882190 recomputed=0.882190 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_pca_knn_k50_unweighted_seed101 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k50_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.884814 recomputed=0.884814 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_pca_knn_k50_unweighted_seed17 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k50_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.886461 recomputed=0.886461 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_pca_knn_k50_unweighted_seed42 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k50_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.885242 recomputed=0.885242 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_pca_knn_k50_unweighted_seed7 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k50_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.884922 recomputed=0.884922 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## gcn_pca_knn_k50_unweighted_seed73 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k50_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.884806 recomputed=0.884806 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_bbknn_kperbatch2_donors12_seed101 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | bbknn_kperbatch2_donors12 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.901738 recomputed=0.901738 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_bbknn_kperbatch2_donors12_seed17 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | bbknn_kperbatch2_donors12 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.900764 recomputed=0.900764 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_bbknn_kperbatch2_donors12_seed42 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | bbknn_kperbatch2_donors12 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.902328 recomputed=0.902328 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_bbknn_kperbatch2_donors12_seed7 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | bbknn_kperbatch2_donors12 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.899155 recomputed=0.899155 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_bbknn_kperbatch2_donors12_seed73 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | bbknn_kperbatch2_donors12 |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.903256 recomputed=0.903256 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.897748 recomputed=0.897748 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.902724 recomputed=0.902724 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.895453 recomputed=0.895453 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.902154 recomputed=0.902154 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.899695 recomputed=0.899695 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.896596 recomputed=0.896596 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.901786 recomputed=0.901786 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.897450 recomputed=0.897450 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.896478 recomputed=0.896478 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.898463 recomputed=0.898463 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.901685 recomputed=0.901685 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.901598 recomputed=0.901598 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.903047 recomputed=0.903047 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.897344 recomputed=0.897344 Δ=0.00e+00 |
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
| macro_f1_recomputation | ✅ | reported=0.900418 recomputed=0.900418 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_pca_knn_k50_unweighted_seed101 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k50_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.899508 recomputed=0.899508 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_pca_knn_k50_unweighted_seed17 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k50_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.900371 recomputed=0.900371 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_pca_knn_k50_unweighted_seed42 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k50_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.900865 recomputed=0.900865 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_pca_knn_k50_unweighted_seed7 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k50_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.903374 recomputed=0.903374 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## graphsage_pca_knn_k50_unweighted_seed73 — PASS

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| graph_artifact_present | ✅ | pca_knn_k50_unweighted |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.901611 recomputed=0.901611 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ✅ | present |

## logistic_regression_site_stratified_seed42 — WARN

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.881007 recomputed=0.881007 Δ=0.00e+00 |
| optional_training_history.csv | ⚠️ | absent |
| optional_embeddings_test.npy | ⚠️ | absent |

## mlp_site_stratified_seed42_seed42 — WARN

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.880354 recomputed=0.880354 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ⚠️ | absent |

## mlp_site_stratified_seed42_seed42 — WARN

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.880354 recomputed=0.880354 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ⚠️ | absent |

## mlp_site_stratified_seed42_seed7 — WARN

| Check | Result | Detail |
|---|---|---|
| manifest_schema | ✅ |  |
| metrics_schema | ✅ |  |
| feature_manifest_hash_match | ✅ | ok |
| split_hash_match | ✅ | ok |
| label_mapping_hash_match | ✅ | ok |
| preds_length | ✅ |  |
| probs_sanity | ✅ | shape_ok=True rows_sum_to_1=True argmax_matches_preds=True |
| confusion_matrix_consistency | ✅ | ok |
| macro_f1_recomputation | ✅ | reported=0.887816 recomputed=0.887816 Δ=0.00e+00 |
| optional_training_history.csv | ✅ | present |
| optional_embeddings_test.npy | ⚠️ | absent |
