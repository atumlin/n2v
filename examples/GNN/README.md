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

### `compare_with_matlab.py` and `matlab_reference/`

Cross-tool diff harness comparing n2v's GraphStar reach against the MATLAB
GNNV reference.  The flow is:

1.  **Generate reference (MATLAB, one-time)**

    ```matlab
    cd /home/verivital/Anne/graph_verification/gnnv2/gnnv-saiv26/nnv/code/nnv
    addpath(genpath(pwd()));
    addpath('/home/verivital/Anne/graph_verification/n2v/examples/GNN/matlab_reference');
    generate_ieee24_reference(...
        'instances', [1 2 3], ...
        'eps_values', [1e-3, 1e-2, 5e-2]);
    ```

    Saves `ieee24_pf_reference.mat` next to the script.

2.  **Run diff (Python)**

    ```
    python examples/GNN/compare_with_matlab.py
    ```

#### What the diff reports

* **Soundness slack** (`matlab_lb - n2v_lb` and `n2v_ub - matlab_ub`):
  positive means n2v's box contains MATLAB's box, which is the correct
  direction since both are sound over-approximations of the same true
  reachable set.  Allowed negative slack is LP solver noise (~1e-10).
* **Width inflation** (`width_n2v / width_matlab`): how much wider n2v's
  bounds are.  Computed only on outputs where the MATLAB width is above
  1e-6 to avoid divide-by-zero on always-inactive ReLU neurons.

#### Known approximation gap

At small perturbations (eps ≤ 1e-2) n2v's mean bound width agrees with
MATLAB to within a few percent.  At larger eps the inflation grows
(~1.5–2× mean, up to ~20× on individual outputs at eps=5e-2).

This is a tightness gap, not a soundness regression: n2v's `approx-star`
ReLU classifies neurons as stable/unstable using the cheap predicate-box
estimate, while MATLAB uses an LP solve per neuron.  The estimate
over-counts crossing neurons, so triangle relaxation is applied to more
of them and the gap compounds across layers.  See the comment in
`n2v/nn/layer_ops/relu_reach.py::_relu_single_star_approx` for the
relevant TODO.

The pytest at `tests/soundness/test_cross_tool_gnn.py` enforces
soundness containment as a hard assertion at every (instance, eps)
cell, plus a tightness assertion only at eps ≤ 1e-2 (where the gap is
small enough to be a useful regression signal).

## Component layout

* `n2v.sets.GraphStar` — node-feature matrix Star (`n2v/sets/graph_star.py`)
* `n2v.nn.layer_ops.gcn_reach` — GCN forward + reach + ReLU bridge
* `n2v.nn.layer_ops.graph_pool_reach` — sum/mean pooling
* `n2v.utils.load_gnn_mat` — checkpoint loader

See `tests/unit/sets/test_graph_star.py`, `tests/unit/layer_ops/test_gcn.py`,
`tests/unit/layer_ops/test_graph_pool.py`, and
`tests/soundness/test_soundness_gcn.py` for the test coverage.
