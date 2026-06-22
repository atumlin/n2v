# GNN Integration — Findings & Status Report

Status of porting the gnnv2 (gnnv-saiv26) MATLAB graph-verification work into
n2v, plus an analysis of the first cross-tool soundness diff against the
MATLAB reference.

Last updated: 2026-05-01.  Branch: `gnn-integration`.

---

## 1. What's done

Four commits on `gnn-integration`:

| Commit | What landed |
| --- | --- |
| `adf68b6` | `GraphStar` set in [n2v/sets/graph_star.py](../../n2v/sets/graph_star.py) — Star with an N×F basis tensor, shared predicates, optional adjacency. flatten/unflatten bridge to existing `Star`. |
| `6b843d4` | GCN forward eval + GraphStar reach in [n2v/nn/layer_ops/gcn_reach.py](../../n2v/nn/layer_ops/gcn_reach.py). `.mat` checkpoint loader in [n2v/utils/gnn_loader.py](../../n2v/utils/gnn_loader.py). |
| `f052ce5` | Sum/mean graph pooling in [n2v/nn/layer_ops/graph_pool_reach.py](../../n2v/nn/layer_ops/graph_pool_reach.py). End-to-end IEEE-24 example. Soundness suite. |
| `0e7379a` | Cross-tool diff harness vs MATLAB GNNV reference, plus 9 parametrized cross-tool soundness tests. |

### Coverage matrix

| Component | Forward eval | Reach (approx-star) | Soundness vs samples | Cross-tool vs MATLAB |
| --- | :---: | :---: | :---: | :---: |
| GCN layer | ✓ | ✓ | ✓ | ✓ |
| ReLU on GraphStar | ✓ | ✓ | ✓ | ✓ |
| Sum / mean pool | ✓ | ✓ (exact) | ✓ | indirect (via stack) |
| Linear readout | reuses existing `linear_reach` | reuses | reuses | indirect |
| SAGEConv | ✓ | ✓ (exact) | ✓ | ✓ |
| GINE / HuGINE (node-only) | ✓ | ✓ | ✓ | ✓ |
| `reachSubgraph` (k-hop) | ✓ | ✓ (exact per target) | ✓ | indirect (= full reach) |
| GINE edge-perturbation | ✓ | ✓ | ✓ | — |

### Added in the SAGE+GINE milestone (2026-06-22)

* **SAGEConv** — exact reach (`Y = X·W_node + A·X·W_edge + b`) in
  [sage_reach.py](../../n2v/nn/layer_ops/sage_reach.py); forward parity 1.1e-7.
* **GINE** — node-only reach in [gine_reach.py](../../n2v/nn/layer_ops/gine_reach.py)
  for both variants: `hugine` (`gine_pretrain`, HuGINEConv: no message ReLU,
  self-loops in `edge_index`, inter-layer ReLU) and `pyg` (`gine_conv`,
  GINEConv: message ReLU + `(1+eps)·x`); forward parity 8.9e-7.
* **Loader** — [gnn_loader.py](../../n2v/utils/gnn_loader.py) now dispatches on
  `model_type` (gcn / sage / gine_conv / gine_pretrain) into a unified
  `GNNModel`; `GCNModel` kept as an alias for back-compat.
* **Wrapper** — [GraphNeuralNetwork](../../n2v/nn/graph_neural_network.py) owns
  layers + graph structure and exposes `.evaluate` / `.reach` / `.from_mat`.
* **Verify-spec** — [gnn_verify.py](../../n2v/utils/gnn_verify.py):
  `verify_node_bounds` / `verify_graph_output` over the flattened output space,
  returning verified / unknown / falsified.
* **Tests** — +57: unit (sage/gine/wrapper/verify), sample-and-contain
  soundness, and **cross-tool containment vs MATLAB NNV** on the IEEE-24 SAGE
  and HuGINE checkpoints (n2v box ⊇ MATLAB box at every (instance, ε) cell).

### Added: k-hop subgraph reach (2026-06-22)

* **`reachSubgraph`** — per-target k-hop locality for scalable verification:
  [GraphStar.extract_subgraph](../../n2v/sets/graph_star.py) (node slice +
  predicate pruning), [khop helpers](../../n2v/utils/subgraph.py) (matrix for
  GCN/SAGE, edge-list for GINE), and
  [GraphNeuralNetwork.reach_subgraph](../../n2v/nn/graph_neural_network.py).
  k = number of message-passing layers.
* **Exact per target.**  On IEEE-24 and IEEE-118, the target node's reach box
  is **bit-identical** (0.0e+00) to full-graph reach for all three layer types
  — only boundary nodes compute unused intermediate values.
* **Scales.**  IEEE-118 (118 nodes) yields 13-38 node subgraphs.  At ε=5e-2 on
  GINE-IEEE-118, full-graph reach is 1.88 s (nVar≈5941) vs 0.145 s per target
  subgraph (nVar≈543, 13 nodes) — LP size tracks subgraph, not full graph.
* +14 tests.

### Added: GINE edge-perturbation + end-to-end example (2026-06-22)

* **Edge perturbation** — `gine_graph_star` / `gine_stack_reach` now accept `E`
  as a `GraphStar` (uncertain edge features), combining node + edge predicate
  spaces via `blkdiag` (`_build_edge_message`).  Sound for both variants
  (0 sample violations); multi-layer is sound-but-loose (edge uncertainty
  treated as independent per layer).  +6 tests.
* **`examples/GNN/verify_gnn.py`** — one script for gcn / sage / gine: forward
  parity, full-graph reach, k-hop subgraph reach (exact per target), per-target
  robust-output spec, soundness.  Runs unchanged on IEEE-24 and IEEE-118.

### Scaling note (important)

Full-graph **reach** is cheap (IEEE-118 GINE ≈ 0.6 s), but full-graph
**`get_ranges`** — one LP per output entry over the whole predicate polytope —
does **not** scale: at IEEE-118 (nVar≈1440) it exceeds 130 s.  The scalable
path is `reach_subgraph` + per-target `get_ranges` (≈0.03 s for 3 targets,
13-25 node subgraphs).  `verify_gnn.py` gates full-graph bound extraction to
graphs ≤ 30 nodes and uses per-target subgraphs above that.

Full suite **1407 passed / 6 skipped / 0 failed**.

### Red-team audit + soundness fix (2026-06-22)

An adversarial review (independent agent + empirical fuzzing) audited every
soundness-critical path.  Result: **one genuine soundness bug found and fixed**,
all other paths confirmed sound.

* **BUG (fixed): `GraphStar.extract_subgraph` constraint pruning.**  When a
  pruned (dead-over-subset) predicate was coupled to a kept predicate by a
  constraint row, the old code deleted the pruned *column* but kept the *row*,
  which silently pinned that predicate to 0 and could shrink the target's reach
  box **below** the true reachable set (unsound).  Reproduced end-to-end: a node
  whose true output reached 6.0 was bounded at 5.5.  Fix: drop any constraint
  row that references a pruned predicate (a sound relaxation); for box inputs —
  the only case in the current pipeline — behaviour is unchanged, so real-
  checkpoint subgraph reach stays bit-exact.  The same flaw exists in the MATLAB
  `GraphStar.extractSubgraph.m` and should be fixed there too.  Regression tests
  added (`test_subgraph.py`).
* **Confirmed sound** (code reading + fuzzing, 0 violations): GINE flatten/
  unflatten order; the predicate-space lifting in the pyg self-loop (relies on
  `relu_star_approx` being append-only, which was verified); the edge-pert
  `blkdiag` alignment; `add_set`'s shared-predicate assumption (only caller
  guarantees it); k-hop completeness (symmetric BFS over-includes, never misses);
  verify-spec sign/direction.  Degenerate cases clean: ε=0 point sets, self-loop-
  only graphs, isolated targets, directed chains, parallel/duplicate edges.

Full suite **1409 passed / 6 skipped / 0 failed**.

### Expanded soundness battery (2026-06-22)

Added `tests/soundness/test_soundness_gnn_fuzz.py` (+18 tests):

* **Property-based fuzz** — random topology / weights / depth (1-3) / ε for all
  four families (gcn, sage, gine-hugine, gine-pyg); reach must contain every
  sampled forward output.
* **Subgraph-reach soundness** — per-target box contains the target's forward
  output (distinct from the earlier *exactness* check).
* **Higher-ε regimes** — ε ∈ {0.05, 0.1} on real checkpoints (more crossing
  neurons / wider relaxation).
* **Degenerate graphs** — ε=0 point sets, self-loops-only, parallel/duplicate
  edges, isolated targets.
* **Verifier soundness (no false "verified")** — a reachable point that
  violates a spec must yield "unknown", never "verified" (a false safety
  certificate is the worst verifier bug).  Checked on synthetic + real models.

Out-of-suite stress (not committed, run on demand): **850 random models**
(600 node-perturbation across all families + 250 edge-perturbation, both
variants, node+edge perturbed), ~34k forward samples — **0 soundness
violations**, worst overshoot 3.4e-13 (LP/float tolerance).

Full suite **1427 passed / 6 skipped / 0 failed**.

### Added: falsification / counterexample search (2026-06-22)

The pipeline was sound-only (verified / unknown).  `n2v/utils/gnn_falsify.py`
adds counterexample search so the "unknown" gap is attacked from the other
side: `falsify_node_bounds` searches the input box for a node-feature matrix
whose forward output escapes the spec (random multi-start + coordinate ascent
to box vertices — the forward map is piecewise-linear, so the violation margin
peaks at a vertex; no autograd needed).

`GraphNeuralNetwork.verify(input_set, spec_lb, spec_ub, ...)` ties it together:
reach + `verify_node_bounds` first (sound); on a non-"verified" result it runs
the falsifier and returns **'verified' / 'falsified' (with a witness) /
'unknown'**.  Falsification is intrinsically trustworthy — every returned
counterexample is a real `evaluate(X)` violation (0 spurious witnesses in
testing).  +8 tests.  Full suite **1435 passed / 6 skipped / 0 failed**.

The pipeline is now import -> spec -> reach -> verify -> **falsify** -> scale.
It remains sound-but-incomplete (no exact-star GNN reach), but "unknown" cases
now get a counterexample search instead of a dead end.

### Test totals
- 31 GraphStar/GCN/pool unit tests in `tests/unit/`.
- 13 sample-and-contain soundness tests in `tests/soundness/test_soundness_gcn.py`.
- 10 cross-tool MATLAB diff tests in `tests/soundness/test_cross_tool_gnn.py`.
- Full pre-existing suite still green.  Combined: **1330 passed / 6 skipped / 0 failed**.

### Forward parity vs PyTorch
On 5 IEEE-24 PF instances, max `|Y_n2v − python_predictions|` ≈ **1.7 × 10⁻⁷**
(float32 ≈ machine precision).  The Python forward pass is bit-faithful to the
trained model.

---

## 2. Cross-tool diff results (IEEE-24 PF, GCN, approx-star)

Reference generated by
[matlab_reference/generate_ieee24_reference.m](matlab_reference/generate_ieee24_reference.m)
(MATLAB R2024a, NNV reach with `linprog`).  Diff reproduced via
`python examples/GNN/compare_with_matlab.py`.

### Aggregate per ε (3 instances)

| ε    | sound slack | n2v width / MATLAB width |       |
|------|-------------|--------------------------|-------|
|      |             | mean                     | max   |
| 1e-3 | −1 × 10⁻¹⁰  | 1.000 – 1.001            | 1.001 – 1.004 |
| 1e-2 | −2 × 10⁻¹¹  | 1.011 – 1.147            | 1.076 – 4.189 |
| 5e-2 |  0          | 1.501 – 1.781            | 4.542 – 21.462 |

### Interpretation

* **Soundness holds at every cell.**  The worst negative slack is ≈ −1 × 10⁻¹⁰,
  which is LP solver tolerance noise.  n2v's box always contains MATLAB's box
  on every output dimension, every instance, every ε.  This is the hard
  correctness property that matters.
* **At small ε, the bounds are essentially identical.**  At ε = 10⁻³ the mean
  width inflation rounds to 1.000.  Even individual outputs agree to <0.5 %.
* **At large ε, n2v widens.**  At ε = 5 × 10⁻² n2v's mean width is 1.5–1.8×
  MATLAB's; individual outputs go up to 21× wider.  Both tools remain sound
  over-approximations of the true reachable set; n2v just loses tightness.

The width-inflation gap is **not a bug or porting error**.  Forward eval
matches PyTorch to 10⁻⁷, the affine and pooling reach paths are exact, and
the cross-tool diff confirms n2v always produces a sound superset of MATLAB.
The gap is a downstream consequence of an algorithmic choice in n2v's ReLU
relaxation that is unrelated to the GNN port.

---

## 3. Why the gap exists (layer-by-layer trace)

Profiled on IEEE-24 PF instance 1 at ε = 5 × 10⁻², comparing the cheap
predicate-box estimate (`estimate_ranges`) to the LP-tight bound
(`get_ranges`) inside n2v alone:

| Layer | nVar | nConstr | LP width mean | est ÷ LP mean | est ÷ LP max |
| --- | ---: | ---: | ---: | ---: | ---: |
| input              |  96 |  192 | —        | 1.000 | 1.000 |
| L0 GCN  (affine)   |  96 |  192 | 9.2e-2   | 1.000 | 1.000 |
| L0 ReLU (+105 cross.) | 201 |  507 | 7.1e-2   | 1.000 | **1.032** |
| L1 GCN  (affine)   | 201 |  507 | 2.2e-1   | **1.122** | **2.007** |
| L1 ReLU (+255 cross.) | 456 | 1272 | 9.7e-2   | 1.088 | 1.792 |
| L2 GCN  (affine)   | 456 | 1272 | 9.2e-1   | **1.339** | 1.755 |
| L2 ReLU (+81  cross.) | 537 | 1515 | 5.8e-1   | 1.167 | 1.537 |

### What's happening, step by step

1. **Input is a box-derived Star.**  Its predicate region is
   `α ∈ [−1, 1]^96` exactly.  The estimate-bound (`max_α V α` over the box)
   equals the LP-bound (`max_α V α` over the polytope `Cα ≤ d`) by
   construction — the polytope *is* the box.

2. **L0 GCN is a pure affine map.**  An affine map of a box-bounded predicate
   region is still box-bounded; predicates haven't been mixed.  No gap.

3. **L0 ReLU is the first place the gap can appear.**  n2v's
   `_relu_single_star_approx` triangle relaxation introduces, for each
   crossing neuron `i`,
   * one fresh predicate variable `α_new`, and
   * three constraints:  `α_new ≥ 0`,  `α_new ≥ x_i`,
     `α_new ≤ slope · (x_i − lb_i)`,
   where `x_i` is the (linear) function of the existing predicates and
   `lb_i / ub_i / slope` come from `estimate_ranges` of `x_i`.

   These three constraints couple `α_new` to the *existing* predicates —
   the constraint matrix `C` ceases to be diagonal.  The predicate region
   becomes a polytope that is strictly inside the box `[pred_lb, pred_ub]^n`.

   At this point estimate-bound starts over-counting.  The gap is still tiny
   here (max 1.032×) because there's only one ReLU layer of coupling so far.

4. **L1 GCN is where the gap explodes.**  The 32 → 32 GCN affine map mixes
   the now-coupled predicates across output dimensions.  Estimate-bound treats
   the 201 predicates as independent in their box; LP exploits the actual 507
   constraints.  Mean inflation jumps to 1.12× and one output is 2× wider in
   the estimate than under LP.

5. **L1 ReLU and L2 GCN compound.**  Each subsequent ReLU adds more crossing
   neurons (the looser estimate means *more* neurons get classified as
   "crossing"), each new triangle adds 3 more constraints, and the next
   GCN affine map mixes them all again.  By L2 the predicate polytope has
   1515 constraints in 537 dimensions — the box-vs-polytope volume gap is
   substantial, so the estimate-bound is substantially loose.

### Mathematical statement

For a Star `x = c + V α` with predicate region `P = {α : Cα ≤ d}`:

* **LP-tight bound**:
  `ub_LP[k] = max { V[k, :] · α : Cα ≤ d, pred_lb ≤ α ≤ pred_ub }`
* **Predicate-box estimate**:
  `ub_box[k] = sum_j max( V[k, j] · pred_lb[j], V[k, j] · pred_ub[j] )`
  (i.e., LP over only the box `pred_lb ≤ α ≤ pred_ub`, ignoring `Cα ≤ d`).

`ub_box ≥ ub_LP` always; equality holds **iff** the inequality `Cα ≤ d`
is redundant given the box, which is true exactly when the predicate
region equals the box.  This holds at construction and through purely
affine maps.  Triangle ReLU adds non-redundant constraints, so equality
breaks the moment the first crossing neuron appears.

### Why this hurts ReLU specifically

The triangle slope used for each crossing neuron is `ub / (ub − lb)`.
A looser `ub / lb` from the estimate yields a flatter slope, which means
the triangle over-approximation is *itself* wider — and that wider output
becomes the input to the next layer's neuron classification, propagating
looseness forward.  This is the mechanism by which the gap compounds.

### Why MATLAB is tighter

MATLAB's `ReluLayer.reach_star_single_input` calls
`Star.getRange(i, lp_solver)` per neuron, which solves an LP over the
full polytope.  Two consequences:

1. Fewer neurons get classified as crossing (LP can certify
   `lb ≥ 0` or `ub ≤ 0` more often than the estimate can).
2. For the ones that do cross, the slope `ub / (ub − lb)` is computed from
   the LP-tight bounds, producing a tighter triangle.

Both effects dampen the gap in subsequent layers.  The cost is paid in
LP solves — one per neuron per ReLU layer, parallelizable but
non-trivial on deep / wide networks.

### What this is *not*

* Not a soundness regression.  n2v's box always contains MATLAB's.
* Not specific to GNNs.  It would manifest the same way on any feedforward
  ReLU network with a non-trivial perturbation.  The GNN structure just
  amplifies it because A_norm aggregation increases the number of
  predicate variables that get mixed in each affine step.
* Not specific to `approx-star` semantics.  It is specific to *which range
  oracle* the approx-star ReLU consults: `estimate_ranges` (predicate-box)
  vs `get_ranges` (LP).

The relevant code path is in
[n2v/nn/layer_ops/relu_reach.py](../../n2v/nn/layer_ops/relu_reach.py)
in `_relu_single_star_approx`, lines ~310 (`estimate_ranges`) and
~352 (`_apply_triangle_approx_multi`).  There is already a TODO comment
at the call site flagging "investigate whether LP-based classification
produces tighter bounds that justify the additional cost".  Out of scope
for the current GNN port; tracked separately.

---

## 4. What still needs to be done

In rough priority order; each item is independent.

### 4.1 Layer types not yet ported

* **SAGEConv** — ✅ done (2026-06-22), exact reach.
* **GINE / HuGINE (node-only)** — ✅ done (2026-06-22), both variants.
* **GINE edge-perturbation** — ✅ done (2026-06-22).  Pass `E` as a `GraphStar`
  to `gine_graph_star` / `gine_stack_reach`; node and edge predicate spaces are
  combined via `blkdiag(C_node, C_edge)` (`_build_edge_message` in
  [gine_reach.py](../../n2v/nn/layer_ops/gine_reach.py)).  Sound for both
  variants (0 sample violations).  Caveat: in a multi-layer stack each layer
  re-combines a fresh edge-predicate copy → sound but loose (edge uncertainty
  treated as independent across layers).  Tightening would thread one shared
  edge-predicate identity through the stack.
* **gine_linear** — simplest GINE variant (`mult{i}` node + `edge{i}` edge
  projection, no MLP); loader/reach not yet wired.
* **`reachSubgraph` (k-hop locality)** — ✅ done (2026-06-22), exact per target.
  Original note retained for context: needed for IEEE-118-scale graphs;
  defers cost by verifying one target node at a time using only its k-hop
  neighborhood.  Reference: `GNN.reachSubgraph` in
  [GNN.m](../../../gnnv2/gnnv-saiv26/nnv/code/nnv/engine/nn/GNN.m).

### 4.2 API ergonomics

* **`GraphNeuralNetwork` wrapper class** — currently the entry point is
  the function `gcn_stack_reach`.  A wrapper analogous to
  [`NeuralNetwork`](../../n2v/nn/neural_network.py) would expose
  `.reach(input_set, method=...)` and own model + adjacency together.
* **PyTorch `state_dict` loader** — alternative to `.mat`.  Useful once
  models start being authored fresh in n2v rather than imported from the
  existing gnn_training pipeline.

### 4.3 Verification specifications

* **Robust output certificate API** — currently the example just prints
  bounds.  Add an analogue of n2v's
  [`verify_specification`](../../n2v/utils/verify_specification.py) that
  takes a per-node or graph-level output predicate and reports
  verified / falsified / unknown.

### 4.4 More benchmarks

* **Graph classification: ENZYMES, PROTEINS** — the SAIV26 paper's other
  evaluation track.  Needs trained checkpoints exported from
  `gnn_training/` for graph-classification heads (none currently in
  `gnn_training/outputs/`).
* **OPF and IEEE-39 / IEEE-118** — already-available checkpoints.  The
  IEEE-118 case will surface whether full-graph reach is tractable or
  whether `reachSubgraph` is required.

### 4.5 Tightness gap (out of scope, but worth noting)

The LP-vs-estimate gap analyzed in §3 is the obvious lever for tighter
bounds at large ε.  Two ways to close it without changing the algorithm:

1. **Use LP for ReLU neuron classification** — drop in `get_ranges`
   instead of `estimate_ranges` in `_relu_single_star_approx`, accept
   the per-neuron LP cost.  Mirrors MATLAB exactly.
2. **Use a Zono pre-pass** — compute approximate ranges via a faster
   zonotope abstraction first (already supported via `precomputed_bounds`
   in `_relu_single_star_approx`) and feed those into the triangle
   slopes.  Cheaper than per-neuron LP, tighter than predicate-box.

Either is a project-wide improvement, not a GNN-specific one.

---

## 5. How to reproduce

```bash
# Run the unit + soundness suite (≈ 35 s)
cd n2v
python -m pytest tests/unit tests/soundness -q

# End-to-end IEEE-24 example
python examples/GNN/verify_ieee24_pf.py --eps 0.01 --samples 32

# Cross-tool diff vs MATLAB reference
python examples/GNN/compare_with_matlab.py

# Regenerate MATLAB reference (only needed if checkpoint changes)
matlab -batch "addpath(genpath('/home/verivital/Anne/graph_verification/gnnv2/gnnv-saiv26/nnv/code/nnv')); addpath('examples/GNN/matlab_reference'); generate_ieee24_reference()"
```
