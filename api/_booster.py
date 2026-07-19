"""A dependency-free evaluator for LightGBM's native model text format.

Why: LightGBM hard-imports `scipy.sparse`, and scipy installs to roughly half a gigabyte, which
blows Vercel's 500MB serverless function limit on its own. numpy plus lightgbm without scipy is
only ~31MB, but lightgbm cannot be imported without it. So the serving path parses the model file
and runs the forward pass in numpy, leaving the deployed function tiny.

This is only a *reader*. Training still uses real LightGBM. `tests/test_serve_parity.py` asserts
this evaluator reproduces `lightgbm.Booster.predict` to within 1e-9 on the real feature matrices,
including rows with missing values, so it cannot silently diverge.

Format notes that matter for correctness:
  - `left_child[i] < 0` means the left branch is leaf `-left_child[i] - 1`; same for right.
  - `decision_type` is a bitfield: bit 0 categorical, bit 1 default-left, bits 2-3 missing type
    (0 none, 1 zero, 2 NaN).
  - Multiclass trees are interleaved, so tree t scores class `t % num_class`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Tree:
    split_feature: np.ndarray
    threshold: np.ndarray
    decision_type: np.ndarray
    left_child: np.ndarray
    right_child: np.ndarray
    leaf_value: np.ndarray


@dataclass
class Model:
    trees: list[Tree]
    num_class: int
    objective: str
    n_features: int
    average_output: bool = False


def _parse_block(block: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in block.strip().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def _arr(d: dict[str, str], key: str, dtype) -> np.ndarray:
    raw = d.get(key, "").strip()
    if not raw:
        return np.array([], dtype=dtype)
    return np.array([dtype(x) for x in raw.split()], dtype=dtype)


def load_model(path: str) -> Model:
    with open(path) as fh:
        text = fh.read()
    blocks = text.split("\n\n")

    header = _parse_block(blocks[0])
    num_class = int(header.get("num_class", "1"))
    objective = header.get("objective", "regression").split()[0]
    n_features = int(header.get("max_feature_idx", "-1")) + 1
    average_output = "average_output" in text.split("\n\n")[0]

    trees: list[Tree] = []
    for b in blocks:
        if not b.lstrip().startswith("Tree="):
            continue
        d = _parse_block(b)
        if "leaf_value" not in d:
            continue
        trees.append(
            Tree(
                split_feature=_arr(d, "split_feature", int),
                threshold=_arr(d, "threshold", float),
                decision_type=_arr(d, "decision_type", int),
                left_child=_arr(d, "left_child", int),
                right_child=_arr(d, "right_child", int),
                leaf_value=_arr(d, "leaf_value", float),
            )
        )

    return Model(trees, num_class, objective, n_features, average_output)


def _score_tree(tree: Tree, X: np.ndarray) -> np.ndarray:
    """Vectorless per-row descent. Row counts here are tiny (a single cycle), so clarity wins."""
    n = X.shape[0]
    out = np.empty(n, dtype=float)

    # A stump has no internal nodes: LightGBM writes a single leaf value.
    if tree.split_feature.size == 0:
        out[:] = tree.leaf_value[0] if tree.leaf_value.size else 0.0
        return out

    for i in range(n):
        node = 0
        while True:
            f = tree.split_feature[node]
            thr = tree.threshold[node]
            dt = int(tree.decision_type[node])
            default_left = bool(dt & 2)
            missing_type = (dt >> 2) & 3

            v = X[i, f]
            if not np.isfinite(v):
                go_left = default_left if missing_type == 2 else (thr >= 0.0)
            elif missing_type == 1 and v == 0.0:
                go_left = default_left
            else:
                go_left = v <= thr

            nxt = tree.left_child[node] if go_left else tree.right_child[node]
            if nxt < 0:
                out[i] = tree.leaf_value[-nxt - 1]
                break
            node = nxt
    return out


def predict_raw(model: Model, X: np.ndarray) -> np.ndarray:
    """Raw (pre-link) scores, shape (n_rows, num_class)."""
    n = X.shape[0]
    raw = np.zeros((n, model.num_class), dtype=float)
    for t, tree in enumerate(model.trees):
        raw[:, t % model.num_class] += _score_tree(tree, X)
    if model.average_output and model.trees:
        raw /= max(1, len(model.trees) // model.num_class)
    return raw


def predict(model: Model, X: np.ndarray) -> np.ndarray:
    """Apply the objective's link function, matching Booster.predict output shape."""
    raw = predict_raw(model, X)

    if model.objective.startswith("multiclass"):
        m = raw.max(axis=1, keepdims=True)
        e = np.exp(raw - m)
        return e / e.sum(axis=1, keepdims=True)

    if model.objective.startswith("binary"):
        return (1.0 / (1.0 + np.exp(-raw[:, 0]))).reshape(-1)

    if model.objective.startswith("poisson") or model.objective.startswith("gamma"):
        return np.exp(raw[:, 0]).reshape(-1)

    return raw[:, 0].reshape(-1)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
