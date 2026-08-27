# Cross-Dataset Meta-Analysis & Benchmark Synthesis

> **Benchmark Scope:** Comprehensive synthesis of single-cell graph neural network performance across two distinct human biological domains:
> 1. **Dataset 1: Stephenson 2021 PBMC** (78,959 cells, 23 donors, 12 immune lineages, circulating blood).
> 2. **Dataset 2: GSE164690 HNSCC** (136,881 cells, 18 donors, 14 lineages, solid tumor microenvironment).
> Total: **215,840 cells** evaluated across **100 audited GNN training runs** and **12 matched MLP baseline runs** (5 independent seeds per condition).

---

## 1. Primary Benchmark Thesis: External Validity Replicated

The core question of this benchmark is: **Do Graph Neural Networks provide a reliable predictive advantage over a non-graph Multilayer Perceptron (MLP) baseline when evaluated under strict, zero-leakage inductive donor splits on fixed feature spaces?**

Across **both** biological domains, the answer is consistently **No**:

```
                              Cross-Dataset Matched Graph Lift Summary
   0.000 ──────────┬──────────────────────────────────────────┬───────────────────────────
                   │ MLP Baseline (Anchor: 0.0000)            │ MLP Baseline (Anchor: 0.0000)
  -0.005 ──────────┼── GraphSAGE (BBKNN: -0.0012) ────────────┼── GraphSAGE (Mutual-kNN: -0.0032)
                   │                                          │   GraphSAGE (BBKNN: -0.0061)
  -0.010 ──────────┼──────────────────────────────────────────┼── GraphSAGE (PCA-kNN: -0.0092)
                   │                                          │
  -0.015 ──────────┼── GCN (BBKNN: -0.0165) ──────────────────┼───────────────────────────
  -0.020 ──────────┼── GCN (PCA-kNN: -0.0190) ────────────────┼── GCN (BBKNN: -0.0196)
                   │                                          │   GCN (PCA-kNN: -0.0202)
  -0.025 ──────────┼──────────────────────────────────────────┼───────────────────────────
  -0.030 ──────────┼── GCN (Mutual-kNN: -0.0304) ─────────────┼── GCN (Mutual-kNN: -0.0213)
                   │                                          │
                   └──────────────────────────────────────────┴───────────────────────────
                        Stephenson 2021 Healthy PBMC                GSE164690 HNSCC Tumor
```

---

## 2. Comparative Performance Matrix

| Metric / Evaluation Dimension | Stephenson 2021 PBMC (Dataset 1) | GSE164690 HNSCC (Dataset 2) | Joint Cross-Dataset Conclusion |
| :--- | :---: | :---: | :--- |
| **Tissue Microenvironment** | Healthy circulating blood | Solid Head & Neck Tumor + PBL | Diverse physiological conditions |
| **Total Cells Analyzed** | 78,959 | 136,881 | 215,840 cells total |
| **Test Cells Evaluated** | 18,508 (2 held-out sites) | 39,698 (5 held-out patients) | 58,206 test cells total |
| **Classes Evaluated** | 12 Cell Ontology classes | 14 Cell Ontology classes | Immune & Non-immune Compartments |
| | | | |
| **MLP Reference Macro-F1** | **$0.9026 \pm 0.0015$** | **$0.8361 \pm 0.0019$** | Highly stable baseline anchor |
| **Logistic Regression Macro-F1**| $0.8875$ | $0.8268$ | Linear model trails MLP by ~1–1.5% |
| | | | |
| **GraphSAGE: Best Graph Variant** | `bbknn_kperbatch2` | `mutual_knn_k20` | Graph-dependent near-parity |
| **GraphSAGE: Best Lift ($\Delta$)** | **$-0.0012 \pm 0.0017$** | **$-0.0032 \pm 0.0026$** | Reaches within $0.1–0.3\%$ of MLP |
| **GraphSAGE: Standard PCA-kNN Lift** | $-0.0045 \pm 0.0019$ | $-0.0092 \pm 0.0043$ | Minor negative lift (~$0.5–0.9\%$) |
| | | | |
| **GCN: Best Graph Variant** | `bbknn_kperbatch2` | `bbknn_kperbatch2` | Batch-balanced graph minimizes loss |
| **GCN: Best Lift ($\Delta$)** | $-0.0165 \pm 0.0017$ | $-0.0196 \pm 0.0020$ | Consistent ~2.0% penalty across datasets |
| **GCN: Standard PCA-kNN Lift** | $-0.0190 \pm 0.0019$ | $-0.0202 \pm 0.0024$ | Exact match between datasets (~$-0.020$) |
| | | | |
| **GCN: Rewired Negative Control** | Severe performance loss | Catastrophic collapse ($0.0344$) | Symmetrical smoothing failure |
| **GraphSAGE: Rewired Control** | Preserved baseline | Preserved baseline ($0.8146$) | Self-loop $W_1 x$ preserves node features |

---

## 3. Key Mechanistic Insights

### 1. The GCN Smoothing Penalty ($D^{-1/2} A D^{-1/2}$)
In both datasets, GCN exhibits a systematic **$\sim 2.0\%$ deficit** compared to the identically parameterized MLP baseline. This deficit stems from the symmetrical normalization operator, which computes:
$$h_i^{(l+1)} = \sigma\left( \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{\tilde{d}_i \tilde{d}_j}} W h_j^{(l)} \right)$$
Because single-cell transcriptomic graphs inevitably contain heterophilous boundary edges (between transitional cell states, shared cell compartments, or batch artifacts), GCN averages representations across class boundaries, causing irreversible feature degradation.

### 2. GraphSAGE Root Isolation ($W_1 x + W_2 \text{mean}(x_{\mathcal{N}})$)
GraphSAGE consistently matches or approaches MLP performance because of its separate transformation matrices for the node's own state and its neighbors:
$$h_i^{(l+1)} = \sigma\left( W_1 h_i^{(l)} + W_2 \cdot \text{AGG}\left(\{h_j^{(l)}, \forall j \in \mathcal{N}(i)\}\right) \right)$$
When neighbor messages are noisy or uninformative, the network learns to weight $W_1$ over $W_2$, effectively falling back to the non-graph MLP representation. This explains why GraphSAGE is robust to randomized topologies ($0.8146$ on rewired control) while GCN collapses completely ($0.0344$).

---

## 4. Benchmark Guardrail Invariants

All 112 evaluated runs adhered to strict methodological guardrails:
1. **Zero Data Leakage**: Graph construction and preprocessing were fitted exclusively on training donors. Test queries connect strictly inductively ($E_{\text{test} \to \text{test}} = \emptyset$).
2. **Immutable Fixed Features**: Every model consumed identical, precomputed 50-dimensional PCA projections.
3. **Cryptographic Provenance**: Every artifact, split, and run was independently verified by the 4-layer CPU audit engine.
