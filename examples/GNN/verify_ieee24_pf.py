"""
Verify a trained GCN power-flow predictor on IEEE-24 with GraphStar reachability.

Loads the .mat checkpoint produced by gnn_training/mat_exporter.py, runs forward
parity against the saved PyTorch predictions, then performs reachability under
an L_inf perturbation on the node-feature matrix.  Reports the bound widths and
checks soundness against random samples.

Usage:
    python verify_ieee24_pf.py [--eps 0.01] [--samples 32] [--instance 0]
"""

import argparse
from pathlib import Path

import numpy as np

from n2v.sets import GraphStar
from n2v.nn.layer_ops.gcn_reach import gcn_stack_evaluate, gcn_stack_reach
from n2v.utils import load_gnn_mat


DEFAULT_CHECKPOINT = Path(
    "/home/verivital/Anne/graph_verification/gnnv2/gnn_training/outputs"
    "/ieee24_pf/gcn_pf_ieee24.mat"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="Path to .mat checkpoint")
    parser.add_argument("--instance", type=int, default=0,
                        help="Test instance index (0-based)")
    parser.add_argument("--eps", type=float, default=0.01,
                        help="L_inf radius on input node features")
    parser.add_argument("--samples", type=int, default=32,
                        help="Random samples used for soundness check")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.exists():
        print(f"checkpoint not found: {args.checkpoint}")
        return 1

    print(f"loading: {args.checkpoint}")
    model = load_gnn_mat(args.checkpoint)
    print(f"  {model.num_layers} GCN layers, has_relu={model.has_relu}")
    print(f"  graph: {model.A_norm.shape[0]} nodes, "
          f"{int(model.A_norm.astype(bool).sum())} entries in A_norm")
    print(f"  test instances: {model.num_test_instances}")

    X = model.X_test[args.instance]
    Y_pred_pt = model.python_predictions[args.instance]
    print(f"\ninstance {args.instance}: X shape {X.shape}")

    # ---- forward parity vs PyTorch reference --------------------------------
    Y_pred_n2v = gcn_stack_evaluate(X, model.A_norm, model.gcn_layers, model.has_relu)
    parity = float(np.max(np.abs(Y_pred_n2v - Y_pred_pt)))
    print(f"forward parity vs PyTorch:  max |dY| = {parity:.3e}")

    # ---- reachability under L_inf perturbation ------------------------------
    print(f"\nrunning reach with eps = {args.eps}")
    gs_in = GraphStar.from_bounds(X - args.eps, X + args.eps, adjacency=model.A_norm)
    out_sets = gcn_stack_reach(gs_in, model.gcn_layers, has_relu=model.has_relu)
    assert len(out_sets) == 1, "approx-star should produce a single output set"
    out = out_sets[0]
    lb, ub = out.get_ranges()
    width = ub - lb
    print(f"reach output: shape {lb.shape}")
    print(f"  bound width:  mean = {width.mean():.4e},  max = {width.max():.4e}")
    print(f"  centre vs PyTorch prediction: max |dY| = "
          f"{float(np.max(np.abs(out.center() - Y_pred_pt))):.3e}")

    # ---- soundness check (sample-and-contain) ------------------------------
    print(f"\nsampling {args.samples} perturbations for soundness check ...")
    rng = np.random.default_rng(args.seed)
    misses = 0
    for _ in range(args.samples):
        delta = rng.uniform(-args.eps, args.eps, size=X.shape)
        Y = gcn_stack_evaluate(X + delta, model.A_norm, model.gcn_layers, model.has_relu)
        if np.any(Y < lb - 1e-5) or np.any(Y > ub + 1e-5):
            misses += 1
    if misses == 0:
        print("  all samples contained: reach is sound on this instance")
    else:
        print(f"  WARNING: {misses}/{args.samples} samples fell outside the reach box")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
