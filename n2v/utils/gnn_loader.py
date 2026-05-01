"""
Loader for Python-trained GNN checkpoints exported as MATLAB .mat files.

Mirrors gnnv-saiv26/nnv/code/nnv/engine/utils/gnn2nnv.m so trained models
from the gnn_training/ pipeline can be consumed without leaving Python.

Currently supports model_type == 'gcn'; SAGE / GINE / HuGINE follow once
their layer reach ops land.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.io import loadmat


@dataclass
class GCNLayerSpec:
    """One graph-convolution layer (weights + bias). Bias may be None."""
    W: np.ndarray   # (F_in, F_out)
    b: Optional[np.ndarray]  # (F_out,) or None


@dataclass
class NormStats:
    """Optional feature-/target-wise normalization saved alongside the model."""
    X_max: Optional[np.ndarray] = None
    Y_max: Optional[np.ndarray] = None


@dataclass
class GCNModel:
    """Parsed GCN checkpoint with all data needed for evaluation and reach.

    Attributes:
        gcn_layers: List of GCNLayerSpec, one per graph-conv layer.
        has_relu: Whether ReLU is interleaved between layers (matches MATLAB
            gnn2nnv.m default of True).
        A_norm: Pre-normalized adjacency matrix shared across all instances.
        X_test, Y_test: Per-instance test inputs / targets, length = n_instances.
            Each entry has shape (N, F_in) / (N, F_out).
        python_predictions: Reference outputs from the original PyTorch model
            (used as ground truth in parity tests).
        norm_stats: Optional feature/target maxes used to denormalize outputs.
        source_path: The .mat file the model was loaded from.
    """
    gcn_layers: List[GCNLayerSpec]
    has_relu: bool
    A_norm: np.ndarray
    X_test: List[np.ndarray] = field(default_factory=list)
    Y_test: List[np.ndarray] = field(default_factory=list)
    python_predictions: List[np.ndarray] = field(default_factory=list)
    norm_stats: NormStats = field(default_factory=NormStats)
    source_path: Optional[Path] = None

    @property
    def num_layers(self) -> int:
        return len(self.gcn_layers)

    @property
    def num_test_instances(self) -> int:
        return len(self.X_test)


def _decode_model_type(raw) -> str:
    """Pull the scalar model-type string out of scipy's nested ndarray wrappers."""
    arr = np.asarray(raw)
    if arr.dtype.kind in ("U", "S"):
        return str(arr.flatten()[0])
    if arr.dtype == object:
        return _decode_model_type(arr.flatten()[0])
    raise ValueError(f"Cannot decode model_type from {raw!r}")


def _unwrap_per_instance(arr) -> List[np.ndarray]:
    """Convert a (k, 1) object array of per-instance matrices to a list."""
    arr = np.asarray(arr)
    out = []
    for i in range(arr.shape[0]):
        cell = arr[i, 0] if arr.ndim == 2 else arr[i]
        out.append(np.asarray(cell, dtype=np.float64))
    return out


def _gcn_params(best_params) -> List[GCNLayerSpec]:
    """Walk best_params.mult1, mult2, ... extracting (W, b) per layer."""
    record = best_params[0, 0] if best_params.ndim == 2 else best_params[0]
    field_names = list(record.dtype.names or [])
    mult_names = sorted(
        [n for n in field_names if n.startswith("mult")],
        key=lambda n: int(n[4:]),
    )

    layers: List[GCNLayerSpec] = []
    for name in mult_names:
        block = record[name]
        block = block[0, 0] if block.ndim == 2 else block[0]
        W = np.asarray(block["Weights"], dtype=np.float64)
        bias = None
        if "Bias" in block.dtype.names:
            raw_b = np.asarray(block["Bias"], dtype=np.float64).reshape(-1)
            if raw_b.size == W.shape[1]:
                bias = raw_b
        layers.append(GCNLayerSpec(W=W, b=bias))
    return layers


def load_gnn_mat(path) -> GCNModel:
    """Load a GNN .mat checkpoint produced by gnn_training/mat_exporter.py.

    Currently only model_type == 'gcn' is supported.  Other types raise
    NotImplementedError pointing at the layer-op work item that must land
    before they can be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GNN checkpoint not found: {path}")

    raw = loadmat(str(path))
    model_type = _decode_model_type(raw["model_type"]).lower()

    if model_type != "gcn":
        raise NotImplementedError(
            f"GNN model_type '{model_type}' is not yet supported. "
            "GCN is the only loader implemented in this milestone."
        )

    A_norm = np.asarray(raw["ANorm_g"], dtype=np.float64)
    layers = _gcn_params(raw["best_params"])

    has_relu = True
    if "activations" in raw:
        # MATLAB stores 'none' or 'relu'; anything not equal to 'none' keeps ReLU.
        try:
            has_relu = _decode_model_type(raw["activations"]).lower() != "none"
        except Exception:
            has_relu = True

    X_test = _unwrap_per_instance(raw["X_test_g"]) if "X_test_g" in raw else []
    Y_test = _unwrap_per_instance(raw["Y_test_g"]) if "Y_test_g" in raw else []
    py_pred = (
        _unwrap_per_instance(raw["python_predictions"])
        if "python_predictions" in raw
        else []
    )

    norm = NormStats()
    if "X_max" in raw:
        norm.X_max = np.asarray(raw["X_max"], dtype=np.float64).reshape(-1)
    if "Y_max" in raw:
        norm.Y_max = np.asarray(raw["Y_max"], dtype=np.float64).reshape(-1)

    return GCNModel(
        gcn_layers=layers,
        has_relu=has_relu,
        A_norm=A_norm,
        X_test=X_test,
        Y_test=Y_test,
        python_predictions=py_pred,
        norm_stats=norm,
        source_path=path,
    )
