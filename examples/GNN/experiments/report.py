"""
Generate a markdown report from results/n2v_results.csv (standard library only).

Summarises, across bus size x GNN type: forward parity, verification verdicts,
n2v timing + bound widths, cross-tool comparison vs MATLAB GNNV (soundness
containment, width ratio, time ratio), and any flagged discrepancies.

Usage:  python report.py
"""

import csv
from collections import defaultdict, OrderedDict
from datetime import datetime

import config as C


def _f(x, nd=3):
    if x in (None, "", "None"):
        return "-"
    try:
        return f"{float(x):.{nd}g}"
    except (TypeError, ValueError):
        return str(x)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _key(r):
    return (int(r["n_nodes"]), r["bus"], r["type"])


def main():
    csv_path = C.RESULTS / "n2v_results.csv"
    if not csv_path.exists():
        print(f"no results at {csv_path}; run run_n2v.py first")
        return 1
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    keys = sorted({_key(r) for r in rows})

    def col(r, c):
        return _num(r.get(c))

    def mean(key, c):
        vals = [col(r, c) for r in rows if _key(r) == key and col(r, c) is not None]
        return sum(vals) / len(vals) if vals else None

    def worst(key, c):
        vals = [col(r, c) for r in rows if _key(r) == key and col(r, c) is not None]
        return min(vals) if vals else None

    L = []
    A = L.append
    A("# GNN Verification Experiment Report\n")
    A(f"Generated {datetime.now().isoformat(timespec='seconds')}.  Spec: under L_inf "
      f"perturbation eps on node features, every output stays within +/- {C.DELTA} of "
      f"its nominal value.\n")
    buses = sorted({r['bus'] for r in rows}); types = sorted({r['type'] for r in rows})
    epss = sorted({r['eps'] for r in rows})
    A(f"- cells: {len(rows)}  |  buses: {buses}  |  types: {types}  |  eps: {epss}\n")

    # 1. parity
    A("\n## 1. Forward parity vs PyTorch (max |dY|)\n")
    A("| bus | type | max parity |"); A("|---|---|---|")
    for k in keys:
        pv = [col(r, "parity") for r in rows if _key(r) == k and col(r, "parity") is not None]
        A(f"| {k[1]} | {k[2]} | {_f(max(pv) if pv else None)} |")

    # 2. verdicts
    A("\n## 2. Verification verdicts (count by status)\n")
    statuses = ["verified", "falsified", "unknown"]
    A("| bus | type | " + " | ".join(statuses) + " | other |"); A("|---|---|---|---|---|---|")
    for k in keys:
        cnt = defaultdict(int)
        for r in rows:
            if _key(r) == k:
                cnt[r["verify_status"]] += 1
        other = sum(v for s, v in cnt.items() if s not in statuses)
        A(f"| {k[1]} | {k[2]} | " + " | ".join(str(cnt.get(s, 0)) for s in statuses)
          + f" | {other} |")

    # 3. timing + width
    A("\n## 3. n2v timing and bound width (mean over cells)\n")
    A("`bound`: lp = LP-tight get_ranges; estimate = predicate-box (numpy, used where "
      "the LP is intractable — GINE beyond ieee24).  reach = full-graph reach; subgraph "
      "= per-target k-hop reach (timing only).\n")
    A("| bus | type | bound | reach s | bound s | subgraph s | sub max N | nVar | width mean |")
    A("|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        bms = {r["bound_method"] for r in rows if _key(r) == k}
        bm = "/".join(sorted(b for b in bms if b))
        A(f"| {k[1]} | {k[2]} | {bm} | {_f(mean(k,'reach_time_s'))} | "
          f"{_f(mean(k,'getranges_time_s'))} | {_f(mean(k,'subgraph_time_s'))} | "
          f"{_f(mean(k,'subgraph_max_nodes'),3)} | {_f(mean(k,'n_predicates'),4)} | "
          f"{_f(mean(k,'width_mean'))} |")

    # 4. cross-tool
    A("\n## 4. Cross-tool vs MATLAB GNNV (NNV)\n")
    cross_keys = [k for k in keys
                  if any(_key(r) == k and col(r, "soundness_min_slack") is not None for r in rows)]
    if not cross_keys:
        A("_No MATLAB references present — run generate_references.m._\n")
    else:
        A("Soundness: n2v box must contain MATLAB box (min slack >= 0, tol "
          f"-{C.SOUNDNESS_TOL:.0e}).  width ratio = n2v / MATLAB; time ratio = n2v reach / "
          "MATLAB reach.\n")
        A("| bus | type | min slack | sound | width ratio | n2v s | matlab s | time ratio |")
        A("|---|---|---|---|---|---|---|---|")
        for k in cross_keys:
            ms = worst(k, "soundness_min_slack")
            ok = "OK" if (ms is None or ms >= -C.SOUNDNESS_TOL) else "**FAIL**"
            A(f"| {k[1]} | {k[2]} | {_f(ms)} | {ok} | {_f(mean(k,'width_ratio_mean'))} | "
              f"{_f(mean(k,'reach_time_s'))} | {_f(mean(k,'matlab_time_s'))} | "
              f"{_f(mean(k,'time_ratio_n2v_over_matlab'))} |")
        missing = [f"{k[1]}/{k[2]}" for k in keys if k not in cross_keys]
        if missing:
            A(f"\n_No MATLAB reference for: {', '.join(missing)} — ieee39/GINE full-graph "
              "getRanges did not finish in MATLAB (same LP wall n2v hits), and ieee118 full-"
              "graph getRanges is intractable in both tools.  Those cells are verified with "
              "the predicate-box estimate + numpy falsifier instead._")

    # 5. discrepancies
    A("\n## 5. Discrepancies\n")
    disc = [r for r in rows if r.get("discrepancy")]
    if not disc:
        A("None. All cells within parity and soundness tolerances.\n")
    else:
        A(f"**{len(disc)} cell(s) flagged:**\n")
        for r in disc:
            A(f"- {r['bus']}/{r['type']} inst{r['instance']} eps{r['eps']}: {r['discrepancy']}")

    (C.RESULTS / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"wrote {C.RESULTS / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
