# GPU Run Quarantine Notes

Non-canonical run directories excluded from `artifacts/results/` aggregation. Payloads
(npy tensors, ~119 MB) are kept on disk under `manual_quarantine_noncuda/` (gitignored)
and remain recoverable from the original delivery tarball
`gpu_results_site_stratified_seed42_28439e6f045c.tar.gz` (batch fingerprint
`28439e6f045c6d32ffdb5ba15a426a934d7cb40a34ac6b64a5f32c7ebafe392a`).

| Directory | Reason | Disposition |
|---|---|---|
| `gcn_pca_knn_k24_unweighted_seed42` | CPU-device duplicate; blocked ingestion of the canonical CUDA run from delivery 28439e6f045c | Replaced by canonical CUDA run |
| `graphsage_bbknn_kperbatch2_donors12_seed42` | CPU-device duplicate; same as above | Replaced by canonical CUDA run |
| `gcn_pca_knn_k24_unweighted_seed999` | Authentic but non-converged legacy run (best val macro-F1 ≈ 0.0995 ≈ random guessing); would corrupt per-graph seed aggregates | Excluded from aggregation |

All three passed the 4-layer integrity audit (they are internally consistent, honest
records) — they were quarantined for provenance/quality reasons, not corruption.
