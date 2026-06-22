"""
n2v GNN verification runner: sweeps (bus size x GNN type x eps x instance),
records verdicts + timing, and compares against the MATLAB GNNV reference.

Robustness spec verified at each cell:  under an L_inf perturbation of radius
eps on the input node features, every output stays within +/- DELTA of its
nominal value (the unperturbed prediction).  Verdict is verified / falsified
(with a counterexample) / unknown.

For small graphs (N <= FULL_GETRANGES_MAX_NODES) this is checked on the full
graph; for large graphs it is checked per target node via k-hop subgraph reach
(exact per target, scalable).  Where a MATLAB reference exists, the n2v output
box is checked to contain the MATLAB box (the cross-tool soundness property)
and bound-width / timing ratios are recorded.

Results stream to results/n2v_results.csv (one row per cell) and
results/runs.jsonl.  A soundness or parity discrepancy writes
results/DISCREPANCY.md and, with --stop-on-discrepancy, halts the sweep.

Usage:
    python run_n2v.py [--stop-on-discrepancy] [--buses ieee24 ieee39] \
                      [--types gcn sage] [--task pf]
"""

import argparse
import csv
import json
import time
import traceback
from datetime import datetime

import numpy as np
from scipy.io import loadmat

import config as C
from n2v.sets import GraphStar
from n2v.nn import GraphNeuralNetwork
from n2v.utils import load_gnn_mat, verify_node_bounds, falsify_node_bounds


CSV_FIELDS = [
    "bus", "n_nodes", "type", "task", "instance", "eps",
    "parity", "mode", "bound_method",
    "reach_time_s", "getranges_time_s", "n_predicates",
    "width_mean", "width_max",
    "verify_status", "verify_time_s", "falsify_margin",
    "subgraph_time_s", "subgraph_max_nodes",
    # cross-tool (filled when a MATLAB reference is present)
    "matlab_time_s", "matlab_width_mean", "soundness_min_slack",
    "width_ratio_mean", "time_ratio_n2v_over_matlab",
    "discrepancy",
]


def _timer():
    return time.perf_counter()


def _lp_tractable(model, n_nodes):
    """Whether LP-tight bounds (get_ranges) + LP verify are tractable here.

    These spend their time in a per-output LP over the whole predicate
    polytope.  GCN/SAGE stay cheap through ieee39, but GINE's two ReLUs per
    layer pack the polytope with constraints that make the LP intractable
    beyond ieee24 (measured: ieee39/GINE/eps=1e-2 does not finish, while
    ieee39/SAGE at a similar predicate count finishes in <1s) — and the same
    holds on its 3-hop subgraphs, which cover most of a dense grid.  Where LP
    is intractable we fall back to the predicate-box estimate (pure numpy,
    sound but looser) plus the numpy falsifier; both never stall.
    """
    if model.gine_variant is not None:
        return n_nodes <= C.GINE_FULL_MAX_NODES
    return n_nodes <= C.FULL_GETRANGES_MAX_NODES


def _measure_subgraph(net, gs, targets):
    """Time per-target k-hop subgraph reach (reach only, no LP)."""
    t = _timer(); results = net.reach_subgraph(gs, targets)
    return _timer() - t, max(r.n_sub_nodes for r in results)


def _load_reference(bus, typ, task):
    p = C.ref_path(bus, typ, task)
    if not p.exists():
        return None
    raw = loadmat(str(p))
    return {
        "instances": np.asarray(raw["instances"]).flatten().astype(int),
        "eps_values": np.asarray(raw["eps_values"]).flatten().astype(float),
        "lb": raw["lb"], "ub": raw["ub"],
        "times": np.asarray(raw["times_sec"], dtype=float),
    }


def _ref_cell(ref, instance, eps):
    """Return (lb, ub, time) for a (0-based instance, eps) or None."""
    ii = np.where(ref["instances"] == instance + 1)[0]    # MATLAB is 1-based
    jj = np.where(np.isclose(ref["eps_values"], eps))[0]
    if not len(ii) or not len(jj):
        return None
    i, j = int(ii[0]), int(jj[0])
    return (np.asarray(ref["lb"][i, j], float),
            np.asarray(ref["ub"][i, j], float),
            float(ref["times"][i, j]))


def _cross_tool(row, ref, instance, eps, lb, ub):
    """Fill cross-tool columns: n2v box must contain the MATLAB box."""
    if ref is None:
        return
    cell = _ref_cell(ref, instance, eps)
    if cell is None:
        return
    lb_m, ub_m, t_m = cell
    slack = min((lb_m - lb).min(), (ub - ub_m).min())     # >=0 => n2v contains matlab
    mask = (ub_m - lb_m) > 1e-6
    ratio = float(((ub - lb)[mask] / (ub_m - lb_m)[mask]).mean()) if mask.any() else 1.0
    row.update(matlab_time_s=t_m, matlab_width_mean=float((ub_m - lb_m).mean()),
               soundness_min_slack=float(slack), width_ratio_mean=ratio,
               time_ratio_n2v_over_matlab=row["reach_time_s"] / t_m if t_m else None)


def _full_lp_cell(net, model, X, eps, delta, ref, instance):
    """Full reach + LP-tight bounds + LP verify + falsify + cross-tool compare."""
    row = {"mode": "full", "bound_method": "lp"}
    adj = None if model.gine_variant else model.adjacency
    gs = GraphStar.from_bounds(X - eps, X + eps, adjacency=adj)

    t = _timer(); out = net.reach(gs)[0]; row["reach_time_s"] = _timer() - t
    row["n_predicates"] = out.nVar
    t = _timer(); lb, ub = out.get_ranges(); row["getranges_time_s"] = _timer() - t
    row["width_mean"] = float((ub - lb).mean()); row["width_max"] = float((ub - lb).max())

    Yc = net.evaluate(X)
    spec_lb, spec_ub = Yc - delta, Yc + delta
    t = _timer()
    status = verify_node_bounds([out], spec_lb, spec_ub)
    if status != "verified":
        res = falsify_node_bounds(net.evaluate, X - eps, X + eps, spec_lb, spec_ub,
                                  n_random=1500, n_restarts=6, seed=instance)
        if res.found:
            status, row["falsify_margin"] = "falsified", res.margin
    row["verify_time_s"] = _timer() - t
    row["verify_status"] = status

    row["subgraph_time_s"], row["subgraph_max_nodes"] = _measure_subgraph(
        net, gs, C.SUBGRAPH_TARGETS)
    _cross_tool(row, ref, instance, eps, lb, ub)
    return row


def _estimate_cell(net, model, X, eps, delta, ref, instance):
    """Full reach + predicate-box estimate (no LP) + numpy falsify.

    For checkpoints where the LP path is intractable (GINE beyond ieee24).  The
    estimate is a sound over-approximation, so an estimate box inside the spec
    proves the property; otherwise the numpy falsifier (real forward evals)
    decides falsified vs unknown.  Both steps are pure numpy and never stall.
    """
    row = {"mode": "full", "bound_method": "estimate"}
    adj = None if model.gine_variant else model.adjacency
    gs = GraphStar.from_bounds(X - eps, X + eps, adjacency=adj)

    t = _timer(); out = net.reach(gs)[0]; row["reach_time_s"] = _timer() - t
    row["n_predicates"] = out.nVar
    t = _timer(); lb, ub = out.estimate_ranges(); row["getranges_time_s"] = _timer() - t
    row["width_mean"] = float((ub - lb).mean()); row["width_max"] = float((ub - lb).max())

    Yc = net.evaluate(X)
    spec_lb, spec_ub = Yc - delta, Yc + delta
    t = _timer()
    if np.all(lb >= spec_lb - 1e-9) and np.all(ub <= spec_ub + 1e-9):
        status = "verified"                               # sound: true set ⊆ estimate ⊆ spec
    else:
        res = falsify_node_bounds(net.evaluate, X - eps, X + eps, spec_lb, spec_ub,
                                  n_random=1500, n_restarts=6, seed=instance)
        status = "falsified" if res.found else "unknown"
        if res.found:
            row["falsify_margin"] = res.margin
    row["verify_time_s"] = _timer() - t
    row["verify_status"] = status

    row["subgraph_time_s"], row["subgraph_max_nodes"] = _measure_subgraph(
        net, gs, C.SUBGRAPH_TARGETS)
    _cross_tool(row, ref, instance, eps, lb, ub)          # estimate box still contains MATLAB
    return row


def run(args):
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    csv_path = C.RESULTS / "n2v_results.csv"
    jsonl_path = C.RESULTS / "runs.jsonl"
    disc_path = C.RESULTS / "DISCREPANCY.md"

    buses = [(b, n) for (b, n) in C.BUSES if b in args.buses]
    discrepancies = []

    with open(csv_path, "w", newline="") as fcsv, open(jsonl_path, "w") as fj:
        writer = csv.DictWriter(fcsv, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for bus, n_nodes in buses:
            for typ in args.types:
                ckpt = C.checkpoint(bus, typ, args.task)
                if not ckpt.exists():
                    print(f"SKIP missing checkpoint: {ckpt}")
                    continue
                model = load_gnn_mat(ckpt)
                net = GraphNeuralNetwork.from_model(model)
                ref = _load_reference(bus, typ, args.task)
                lp = _lp_tractable(model, n_nodes)
                print(f"\n=== {bus} ({n_nodes} nodes) / {typ} "
                      f"[{'LP' if lp else 'estimate'}{' +ref' if ref else ''}] ===")

                for inst in C.INSTANCES:
                    X = model.X_test[inst]
                    parity = float(np.max(np.abs(net.evaluate(X) - model.python_predictions[inst])))
                    for eps in C.EPS_VALUES:
                        base = {k: None for k in CSV_FIELDS}
                        base.update(bus=bus, n_nodes=n_nodes, type=typ, task=args.task,
                                    instance=inst, eps=eps, parity=parity)
                        try:
                            if lp:
                                cell = _full_lp_cell(net, model, X, eps, C.DELTA, ref, inst)
                            else:
                                cell = _estimate_cell(net, model, X, eps, C.DELTA, ref, inst)
                            base.update(cell)
                        except Exception as e:
                            base["verify_status"] = f"ERROR: {e}"
                            traceback.print_exc()

                        # discrepancy detection
                        flags = []
                        if parity > C.PARITY_TOL:
                            flags.append(f"parity {parity:.2e} > {C.PARITY_TOL:.0e}")
                        s = base.get("soundness_min_slack")
                        if s is not None and s < -C.SOUNDNESS_TOL:
                            flags.append(f"n2v box does NOT contain MATLAB box (slack {s:.2e})")
                        if flags:
                            base["discrepancy"] = "; ".join(flags)
                            discrepancies.append({**base})

                        writer.writerow(base); fcsv.flush()
                        fj.write(json.dumps({k: base[k] for k in CSV_FIELDS}) + "\n"); fj.flush()
                        t_cell = base.get("reach_time_s") or base.get("subgraph_time_s") or 0.0
                        print(f"  inst {inst} eps {eps:.0e}: {str(base['mode']):22s} "
                              f"verdict={base['verify_status']} time={t_cell:.3f}s"
                              + (f" slack={s:+.1e}" if s is not None else "")
                              + (f"  !!DISCREPANCY: {base['discrepancy']}" if flags else ""), flush=True)

                        if flags and args.stop_on_discrepancy:
                            _write_discrepancy(disc_path, discrepancies)
                            print(f"\nHALTING on discrepancy. See {disc_path}")
                            return 2

    if discrepancies:
        _write_discrepancy(disc_path, discrepancies)
        print(f"\n{len(discrepancies)} discrepancy cell(s); see {disc_path}")
        return 2
    print(f"\nDONE. Results: {csv_path}")
    return 0


def _write_discrepancy(path, rows):
    with open(path, "w") as f:
        f.write(f"# Discrepancies\n\nDetected {len(rows)} cell(s) at "
                f"{datetime.now().isoformat(timespec='seconds')}.\n\n")
        for r in rows:
            f.write(f"- **{r['bus']}/{r['type']} inst{r['instance']} "
                    f"eps{r['eps']:.0e}**: {r['discrepancy']}\n")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--buses", nargs="*", default=[b for b, _ in C.BUSES])
    p.add_argument("--types", nargs="*", default=C.TYPES)
    p.add_argument("--task", default=C.TASK)
    p.add_argument("--stop-on-discrepancy", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
