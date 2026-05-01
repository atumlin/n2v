# GNN Verification Examples

Examples that exercise the GraphStar-based GNN reachability path translated
from the gnnv2/gnnv-saiv26 MATLAB implementation.

## Available examples

### `verify_ieee24_pf.py`

End-to-end verification of a 3-layer GCN power-flow predictor on the IEEE-24
bus system.  Loads a checkpoint exported by `gnn_training/mat_exporter.py`,
checks forward parity against the saved PyTorch reference, runs GraphStar
reachability under an L_inf perturbation on node features, and performs a
sampling-based soundness check on the result.

```
python examples/GNN/verify_ieee24_pf.py --eps 0.01 --samples 32
```

Default checkpoint path:
`/home/verivital/Anne/graph_verification/gnnv2/gnn_training/outputs/ieee24_pf/gcn_pf_ieee24.mat`

Pass `--checkpoint <path>` to point at a different model.

## Supported architectures (so far)

| Architecture | Forward eval | Reachability |
| --- | --- | --- |
| GCN (Kipf & Welling) | yes | yes |
| Sum / mean graph pooling | yes | yes |
| ReLU on GraphStar | yes | yes (approx-star) |
| GINE / HuGINE | not yet | not yet |
| SAGEConv | not yet | not yet |

## Component layout

* `n2v.sets.GraphStar` — node-feature matrix Star (`n2v/sets/graph_star.py`)
* `n2v.nn.layer_ops.gcn_reach` — GCN forward + reach + ReLU bridge
* `n2v.nn.layer_ops.graph_pool_reach` — sum/mean pooling
* `n2v.utils.load_gnn_mat` — checkpoint loader

See `tests/unit/sets/test_graph_star.py`, `tests/unit/layer_ops/test_gcn.py`,
`tests/unit/layer_ops/test_graph_pool.py`, and
`tests/soundness/test_soundness_gcn.py` for the test coverage.
