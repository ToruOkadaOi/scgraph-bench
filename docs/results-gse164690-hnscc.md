# Results: GSE164690 Head & Neck Squamous Cell Carcinoma (HNSCC)

> **Dataset Provenance & Cohort:** Kürten et al. (*Nature Communications* 2021, DOI: `10.1038/s41467-021-27619-4` / GEO: `GSE164690`). Evaluated across **136,881 cells** from 18 treatment-naive human patients (`HN01`–`HN18`) across solid tumor CD45$^+$, CD45$^-$, and matched peripheral blood lymphocytes (PBL), annotated across 14 Cell Ontology classes.

This evaluation assesses the external validity of our graph benchmark findings on a complex solid tumor microenvironment (TME). Using an immutable, HPV-stratified donor-held-out split (`hpv_stratified_seed42`: 9 Train, 4 Val, 5 Test donors), Graph Convolutional Networks (GCN) and GraphSAGE were evaluated against identically regularized Multilayer Perceptron (MLP) baselines across 5 graph construction strategies.

All models were evaluated across 5 independent random seeds (`7, 17, 42, 73, 101`) under strict inductive connectivity (0 test-to-test edges, bipartite test-to-train edges only).

---

## 1. Executive Summary of Findings

1. **GCN Underperforms MLP Baseline**: Across all 20 non-control seed comparisons on GSE164690, GCN consistently underperforms the matched MLP baseline (mean lift **$-0.0196$ to $-0.0215$**).
2. **GraphSAGE Reaches Near-Parity**: GraphSAGE closely matches MLP performance across all graph topologies, with Mutual-kNN achieving the closest parity (mean lift **$-0.0032 \pm 0.0026$**).
3. **Topological Negative Control Contrast**:
   - On the degree-matched `rewired_control` (randomized edges, 14.1% homophily), **GCN collapses completely** ($\text{Macro-F1} = 0.0344$, lift $-0.8016$).
   - **GraphSAGE retains resilience** ($\text{Macro-F1} = 0.8146$, lift $-0.0214$) due to its separate root transformation isolating node self-features from corrupted neighbor messages.

---

## 2. Benchmark Performance Summary Across 5 Seeds

| Model Architecture | Graph Construction Variant | Directed Edges | Test Macro-F1 (Mean $\pm$ SD) | Matched Lift Over MLP ($\Delta$) | Minimum Lift | Maximum Lift | Balanced Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLP Reference Baseline** | *No Graph (Node Features Only)* | — | **$0.8361 \pm 0.0019$** | **$0.0000$ (Anchor)** | $0.0000$ | $0.0000$ | **$0.8389$** |
| **Logistic Regression** | *No Graph (Linear Baseline)* | — | $0.8268$ | $-0.0093$ | — | — | $0.8185$ |
| | | | | | | | |
| **GraphSAGE** | `mutual_knn_k20_unweighted` | 1,888,562 | **$0.8329 \pm 0.0009$** | **$-0.0032 \pm 0.0026$** | $-0.0069$ | **$-0.0005$** | **$0.8333$** |
| **GraphSAGE** | `bbknn_kperbatch2_donors9` | 3,507,550 | $0.8299 \pm 0.0031$ | $-0.0061 \pm 0.0030$ | $-0.0098$ | $-0.0029$ | $0.8287$ |
| **GraphSAGE** | `pca_knn_k24_unweighted` | 4,294,976 | $0.8284 \pm 0.0018$ | $-0.0077 \pm 0.0029$ | $-0.0125$ | $-0.0050$ | $0.8264$ |
| **GraphSAGE** | `pca_knn_k20_unweighted` | 3,586,678 | $0.8268 \pm 0.0038$ | $-0.0092 \pm 0.0043$ | $-0.0164$ | $-0.0047$ | $0.8254$ |
| **GraphSAGE** | `rewired_control_pca_knn` | 3,586,678 | $0.8146 \pm 0.0075$ | $-0.0214 \pm 0.0073$ | $-0.0321$ | $-0.0103$ | $0.8023$ |
| | | | | | | | |
| **GCN** | `bbknn_kperbatch2_donors9` | 3,507,550 | $0.8165 \pm 0.0009$ | $-0.0196 \pm 0.0020$ | $-0.0220$ | $-0.0160$ | $0.8167$ |
| **GCN** | `pca_knn_k20_unweighted` | 3,586,678 | $0.8159 \pm 0.0010$ | $-0.0202 \pm 0.0024$ | $-0.0228$ | $-0.0164$ | $0.8148$ |
| **GCN** | `mutual_knn_k20_unweighted` | 1,888,562 | $0.8147 \pm 0.0007$ | $-0.0213 \pm 0.0023$ | $-0.0251$ | $-0.0187$ | $0.8122$ |
| **GCN** | `pca_knn_k24_unweighted` | 4,294,976 | $0.8146 \pm 0.0013$ | $-0.0215 \pm 0.0021$ | $-0.0234$ | $-0.0184$ | $0.8132$ |
| **GCN** | `rewired_control_pca_knn` | 3,586,678 | $0.0344 \pm 0.0072$ | $-0.8016 \pm 0.0077$ | $-0.8084$ | $-0.7867$ | $0.0711$ |

---

## 3. Seed-by-Seed Matched Comparison Table

| Model | Graph Variant | Seed | GNN Macro-F1 | MLP Macro-F1 | Matched Lift ($\Delta$) | Balanced Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **GraphSAGE** | `mutual_knn_k20_unweighted` | 7 | $0.8333$ | $0.8338$ | **$-0.0005$** | $0.8340$ |
| **GraphSAGE** | `mutual_knn_k20_unweighted` | 17 | $0.8329$ | $0.8371$ | $-0.0042$ | $0.8332$ |
| **GraphSAGE** | `mutual_knn_k20_unweighted` | 42 | $0.8322$ | $0.8392$ | $-0.0069$ | $0.8327$ |
| **GraphSAGE** | `mutual_knn_k20_unweighted` | 73 | $0.8337$ | $0.8345$ | **$-0.0008$** | $0.8340$ |
| **GraphSAGE** | `mutual_knn_k20_unweighted` | 101 | $0.8325$ | $0.8357$ | $-0.0032$ | $0.8327$ |
| | | | | | | |
| **GraphSAGE** | `bbknn_kperbatch2_donors9` | 7 | $0.8309$ | $0.8338$ | $-0.0029$ | $0.8300$ |
| **GraphSAGE** | `bbknn_kperbatch2_donors9` | 17 | $0.8315$ | $0.8371$ | $-0.0056$ | $0.8302$ |
| **GraphSAGE** | `bbknn_kperbatch2_donors9` | 42 | $0.8294$ | $0.8392$ | $-0.0098$ | $0.8282$ |
| **GraphSAGE** | `bbknn_kperbatch2_donors9` | 73 | $0.8315$ | $0.8345$ | $-0.0030$ | $0.8301$ |
| **GraphSAGE** | `bbknn_kperbatch2_donors9` | 101 | $0.8262$ | $0.8357$ | $-0.0095$ | $0.8251$ |
| | | | | | | |
| **GCN** | `bbknn_kperbatch2_donors9` | 7 | $0.8174$ | $0.8338$ | $-0.0164$ | $0.8173$ |
| **GCN** | `bbknn_kperbatch2_donors9` | 17 | $0.8151$ | $0.8371$ | $-0.0220$ | $0.8154$ |
| **GCN** | `bbknn_kperbatch2_donors9` | 42 | $0.8172$ | $0.8392$ | $-0.0220$ | $0.8173$ |
| **GCN** | `bbknn_kperbatch2_donors9` | 73 | $0.8154$ | $0.8345$ | $-0.0191$ | $0.8157$ |
| **GCN** | `bbknn_kperbatch2_donors9` | 101 | $0.8173$ | $0.8357$ | $-0.0184$ | $0.8176$ |

---

## 4. Cross-Dataset Comparison (Stephenson PBMC vs GSE164690 HNSCC)

| Property / Finding | Stephenson 2021 PBMC (Dataset 1) | GSE164690 HNSCC (Dataset 2) | External Validity Status |
| :--- | :--- | :--- | :--- |
| **Tissue Microenvironment** | Circulating healthy blood | Solid head & neck tumor + blood | **Diverse biological domains** |
| **Cohort Size & Donors** | 78,959 cells (23 donors) | 136,881 cells (18 donors) | **Large scale verified** |
| **Classes Evaluated** | 12 Cell Ontology classes | 14 Cell Ontology classes | **Immune & Stromal/Epithelial** |
| **MLP Baseline F1** | $0.9026 \pm 0.0015$ | $0.8361 \pm 0.0019$ | **Tight baseline variance** |
| **GCN Matched Lift** | $-0.0165$ to $-0.0304$ | $-0.0196$ to $-0.0215$ | **Replicated (~-2.0% deficit)** |
| **GraphSAGE Matched Lift** | $-0.0012$ to $-0.0045$ | $-0.0032$ to $-0.0092$ | **Replicated (exact near-parity)** |
| **GCN on Rewired Control** | Severe degradation | Complete collapse ($0.0344$) | **Replicated (smoothing flaw)** |
| **GraphSAGE on Rewired Control** | Preserved baseline | Preserved baseline ($0.8146$) | **Replicated (root isolation)** |
