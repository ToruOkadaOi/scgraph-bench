#!/bin/zsh
set -e
cd /Users/aman/Documents/scgraph-bench
PY=.venv/bin/python
LOG=gpu-runs/rerun_sweep_2026-08-25.log
echo "=== Instrumented re-run started $(date) ===" >> $LOG

GRAPHS=(pca_knn_k20_unweighted pca_knn_k24_unweighted mutual_knn_reference_standard_query_k20_unweighted bbknn_kperbatch2_donors12 pca_knn_k50_unweighted)

echo "--- MLP 5-seed refresh ---" >> $LOG
$PY scripts/run_mlp_seed_sweep.py >> $LOG 2>&1

for g in $GRAPHS; do
  echo "--- GCN $g ---" >> $LOG
  $PY scripts/run_gcn_sweep.py --graph "$g" --device mps >> $LOG 2>&1
done

for g in $GRAPHS; do
  echo "--- GraphSAGE $g ---" >> $LOG
  $PY scripts/run_graphsage_sweep.py --graph "$g" --device mps >> $LOG 2>&1
done

echo "=== Sweep complete $(date) ===" >> $LOG
