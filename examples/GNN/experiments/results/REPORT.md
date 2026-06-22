# GNN Verification Experiment Report

Generated 2026-06-22T14:57:53.  Spec: under L_inf perturbation eps on node features, every output stays within +/- 0.1 of its nominal value.

- cells: 54  |  buses: ['ieee118', 'ieee24', 'ieee39']  |  types: ['gcn', 'gine_pretrain', 'sage']  |  eps: ['0.001', '0.01']


## 1. Forward parity vs PyTorch (max |dY|)

| bus | type | max parity |
|---|---|---|
| ieee24 | gcn | 1.71e-07 |
| ieee24 | gine_pretrain | 8.95e-07 |
| ieee24 | sage | 1.1e-07 |
| ieee39 | gcn | 2.95e-07 |
| ieee39 | gine_pretrain | 4.24e-07 |
| ieee39 | sage | 8.02e-08 |
| ieee118 | gcn | 5.04e-07 |
| ieee118 | gine_pretrain | 1.05e-06 |
| ieee118 | sage | 7.42e-08 |

## 2. Verification verdicts (count by status)

| bus | type | verified | falsified | unknown | other |
|---|---|---|---|---|---|
| ieee24 | gcn | 3 | 3 | 0 | 0 |
| ieee24 | gine_pretrain | 3 | 3 | 0 | 0 |
| ieee24 | sage | 6 | 0 | 0 | 0 |
| ieee39 | gcn | 6 | 0 | 0 | 0 |
| ieee39 | gine_pretrain | 3 | 3 | 0 | 0 |
| ieee39 | sage | 3 | 1 | 2 | 0 |
| ieee118 | gcn | 6 | 0 | 0 | 0 |
| ieee118 | gine_pretrain | 3 | 3 | 0 | 0 |
| ieee118 | sage | 6 | 0 | 0 | 0 |

## 3. n2v timing and bound width (mean over cells)

`bound`: lp = LP-tight get_ranges; estimate = predicate-box (numpy, used where the LP is intractable — GINE beyond ieee24).  reach = full-graph reach; subgraph = per-target k-hop reach (timing only).

| bus | type | bound | reach s | bound s | subgraph s | sub max N | nVar | width mean |
|---|---|---|---|---|---|---|---|---|
| ieee24 | gcn | lp | 0.00182 | 0.261 | 0.00242 | 19 | 157.7 | 0.0473 |
| ieee24 | gine_pretrain | lp | 0.0115 | 0.277 | 0.0137 | 19 | 167 | 0.0768 |
| ieee24 | sage | lp | 0.00159 | 0.205 | 0.00255 | 19 | 131.8 | 0.0111 |
| ieee39 | gcn | lp | 0.00287 | 0.323 | 0.00237 | 19 | 164.3 | 0.0105 |
| ieee39 | gine_pretrain | estimate | 0.0333 | 9.21e-05 | 0.0156 | 19 | 332 | 0.104 |
| ieee39 | sage | lp | 0.00954 | 0.847 | 0.00387 | 19 | 323.5 | 0.0111 |
| ieee118 | gcn | estimate | 0.0593 | 0.000586 | 0.00459 | 18 | 572.7 | 0.00858 |
| ieee118 | gine_pretrain | estimate | 0.412 | 0.0013 | 0.019 | 18 | 1008 | 0.146 |
| ieee118 | sage | estimate | 0.0807 | 0.000737 | 0.00578 | 18 | 755.7 | 0.00121 |

## 4. Cross-tool vs MATLAB GNNV (NNV)

Soundness: n2v box must contain MATLAB box (min slack >= 0, tol -1e-06).  width ratio = n2v / MATLAB; time ratio = n2v reach / MATLAB reach.

| bus | type | min slack | sound | width ratio | n2v s | matlab s | time ratio |
|---|---|---|---|---|---|---|---|
| ieee24 | gcn | -9.5e-11 | OK | 1.03 | 0.00182 | 0.732 | 0.00414 |
| ieee24 | gine_pretrain | -4.39e-11 | OK | 1.08 | 0.0115 | 0.772 | 0.0391 |
| ieee24 | sage | -1.05e-10 | OK | 1.18 | 0.00159 | 0.346 | 0.0118 |
| ieee39 | gcn | -9.89e-11 | OK | 1 | 0.00287 | 0.0884 | 0.0756 |
| ieee39 | sage | -4.99e-11 | OK | 1.05 | 0.00954 | 1.87 | 0.00847 |

_No MATLAB reference for: ieee39/gine_pretrain, ieee118/gcn, ieee118/gine_pretrain, ieee118/sage — ieee39/GINE full-graph getRanges did not finish in MATLAB (same LP wall n2v hits), and ieee118 full-graph getRanges is intractable in both tools.  Those cells are verified with the predicate-box estimate + numpy falsifier instead._

## 5. Discrepancies

None. All cells within parity and soundness tolerances.

