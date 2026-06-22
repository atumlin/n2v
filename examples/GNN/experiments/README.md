# GNN verification experiment suite

Compares **n2v** GraphStar verification against the **MATLAB GNNV (NNV)** reference
across IEEE bus sizes (24 / 39 / 118) and GNN types (GCN / SAGE / GINE), logging
verification verdicts and timing.

**Spec.** Under an L∞ perturbation of radius `eps` on input node features, does every
output stay within ±`DELTA` of its nominal (unperturbed) value? → verified / falsified
(with counterexample) / unknown.

## Files
- `config.py` — experiment matrix + discrepancy thresholds.
- `run_n2v.py` — n2v sweep; writes `results/n2v_results.csv` + `results/runs.jsonl`.
- `generate_references.m` — MATLAB NNV references for ieee24/ieee39 (full-graph).
- `report.py` — renders `results/REPORT.md` from the CSV.

## Run
```bash
# 1. (optional) MATLAB reference for cross-tool comparison (ieee24, ieee39)
NNV=/home/verivital/Anne/graph_verification/gnnv2/gnnv-saiv26/nnv/code/nnv
MR=/home/verivital/Anne/graph_verification/n2v/examples/GNN/matlab_reference
matlab -batch "addpath(genpath('$NNV')); addpath('$MR'); cd('$(pwd)'); generate_references()"

# 2. n2v sweep (halts and writes results/DISCREPANCY.md on a soundness/parity miss)
python run_n2v.py --stop-on-discrepancy

# 3. report
python report.py        # -> results/REPORT.md
```

`run_n2v.py` runs full-graph reach where tractable (N ≤ 45) and per-target k-hop
subgraph reach for larger graphs (ieee118). A discrepancy = forward parity above
tolerance, or the n2v box failing to contain the MATLAB box.
