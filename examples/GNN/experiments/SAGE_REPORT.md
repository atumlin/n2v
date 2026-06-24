# SAGEConv in n2v — How Well Does It Work?

Focused assessment of the GraphSAGE (`SAGEConv`) verification path in n2v, drawn
from the test suite and the n2v-vs-MATLAB-GNNV experiment campaign.

**Verdict: SAGE is the strongest of the three ported layers.**  It is exactly
affine, so its reachability adds *zero* over-approximation; bounds match the
MATLAB NNV reference to LP tolerance while running 14–240× faster, and forward
evaluation is bit-faithful to PyTorch across every checkpoint.

---

## 1. Correctness — forward parity (all checkpoints)

`SAGEConv: Y = X·W_node + (A·X)·W_edge + b` reproduced to ~1e-7 (float32 machine
precision) on every trained checkpoint, PF and OPF, all bus sizes:

| checkpoint | N | layers | max parity |
|---|---:|---:|---|
| ieee24 PF  |  24 | 3 | 1.10e-07 |
| ieee24 OPF |  24 | 3 | 2.05e-07 |
| ieee39 PF  |  39 | 3 | 1.10e-07 |
| ieee39 OPF |  39 | 3 | 1.19e-07 |
| ieee118 PF | 118 | 3 | 8.63e-08 |
| ieee118 OPF| 118 | 3 | 1.31e-07 |

## 2. Reachability is EXACT for the SAGE layer

SAGE is linear in X, so a SAGE layer is a pure affine map on a GraphStar — it
introduces **no** predicate variables and **no** relaxation.  On the IEEE-24
checkpoint (single layer, ε=0.05):

- single-layer reach box vs the **analytic affine box**: max diff **4.4e-16**,
- predicates added by the layer: **0**,
- reach center == forward(center): **exact**.

Consequence: in a multi-layer SAGE stack the *only* source of over-approximation
is the ReLU **between** layers — SAGE itself contributes none.  This is why SAGE
bounds track the reference so tightly (§4) and why its LP stays tractable further
up the bus-size scale than GINE's (§5).

## 3. Tests

19 SAGE-specific tests pass (25 reference "sage"): unit (forward/affine/loader
round-trip), sample-and-contain soundness (single layer + ReLU stacks, ε up to
5e-2), and cross-tool containment vs MATLAB on the IEEE-24/39 references.  SAGE
is also covered by the 850-model soundness stress fuzz (0 violations).

## 4. Cross-tool vs MATLAB GNNV (NNV) — IEEE-24 / IEEE-39

n2v's box must contain NNV's box (soundness); width ratio = n2v / NNV; time ratio
= n2v reach / NNV reach.

| bus | ε | soundness slack | width ratio | n2v reach | NNV reach | speedup |
|---|---|---|---|---|---|---|
| ieee24 | 1e-3 | −5e-12 .. −1e-10 (OK) | **1.00** | ~1.3 ms | 0.35–0.73 s | ~50–240× |
| ieee24 | 1e-2 | −2e-11 .. −4e-11 (OK) | 1.01–2.03* | ~1.9 ms | 0.35 s | ~180–310× |
| ieee39 | 1e-3 | −3e-11 (OK) | **1.00–1.02** | ~4 ms | 0.09 s | ~14–70× |
| ieee39 | 1e-2 | −1e-11 .. −5e-11 (OK) | 1.07–1.13 | ~12–17 ms | 1.9 s | ~120–240× |

*the single 2.03 cell is one output dimension, not the mean.

- **Soundness holds in every cell** — worst slack ≈ −1e-10 (LP solver noise),
  i.e. n2v always over-approximates NNV.  No discrepancies flagged.
- **Tightness is essentially identical** to NNV: mean width ratio ≈ 1.0 at ε=1e-3
  and 1.07–1.13 at ε=1e-2.  (SAGE's exact affine reach is the reason — the small
  gap is the inter-layer ReLU relaxation, shared by both tools.)
- **n2v is 14–310× faster** than MATLAB NNV on the reach.

## 5. Robustness verdicts and timing across bus sizes

Spec: under L∞ perturbation ε, every output stays within ±0.10 of its nominal
value → verified / falsified (counterexample) / unknown.

| bus | ε | verdicts (3 instances) | bound | reach time | nVar |
|---|---|---|---|---|---|
| ieee24  | 1e-3 / 1e-2 | 3 verified / 3 verified | lp | ~1.5 ms | 100–172 |
| ieee39  | 1e-3        | 3 verified              | lp | ~4 ms | 184–194 |
| ieee39  | 1e-2        | 1 falsified, 2 unknown  | lp | ~15 ms | 421–480 |
| ieee118 | 1e-3 / 1e-2 | 6 verified              | estimate | 0.05–0.11 s | 497–1013 |

- SAGE is robust on these models at small ε (all verified); at ε=1e-2 on ieee39
  some outputs leave the ±0.1 band (1 real counterexample) or are unprovable with
  the relaxation (2 unknown).
- **Scales well.**  Full-graph LP bound extraction stays tractable through
  ieee39 (unlike GINE, which needs the predicate-box estimate beyond ieee24).
  At ieee118 SAGE uses the estimate path (sound, numpy) and still verifies all
  cells in ≤0.11 s with very tight widths (mean ≈ 1e-3).

## 6. Bottom line

| dimension | SAGE result |
|---|---|
| forward parity | ~1e-7, all 6 checkpoints (PF+OPF, 24/39/118) |
| reach exactness | exact affine layer (0 predicates, 4e-16 vs analytic) |
| soundness vs NNV | holds everywhere (slack ~−1e-10) |
| tightness vs NNV | ~1.0× width at small ε |
| speed vs NNV | 14–310× faster |
| scalability | LP tractable to ieee39; estimate path clean to ieee118 |
| tests | 19 SAGE tests + stress fuzz, all green |

SAGE works **very well** — it is the most accurate and best-scaling of the three
layers, precisely because it carries no internal nonlinearity for the reach
engine to relax.  The only caveat is the shared inter-layer-ReLU looseness at
larger ε, which is identical to MATLAB NNV and is what produces the handful of
`unknown` verdicts at ieee39/ε=1e-2.
