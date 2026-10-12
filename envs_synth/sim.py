"""Simulator API requested in DESIGN.md section 2 (works for both tracks; `scenario` = one data row).

observed: list of (label, value) pairs or a {label: value} dict, value = outcome index
          (binary track: 1 = finding, 0 = no finding; ms track: index into checks[i].outcomes).
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core import Model
from eval_api import get_model


def _obs(scn, observed):
    M = get_model(scn)
    if isinstance(observed, dict):
        observed = list(observed.items())
    return M, list(M.observed0) + [(M.lab2i[str(l).upper()], int(v)) for l, v in (observed or [])]


def sample_y(scn, rng):
    """Sample a hidden state from the starting belief. Binary track: returns y in {0,1} (1 = adverse)."""
    M = get_model(scn)
    b = M.belief(M.observed0)
    return int(rng.choice(len(b), p=b))


def reveal(scn, label, rng, y):
    """Sample the outcome of item `label` given hidden state y (binary: y; ms: joint state index)."""
    M = get_model(scn)
    L = M.L[M.lab2i[str(label).upper()]][y]
    return int(rng.choice(len(L), p=L / L.sum()))


def posterior(scn, observed=()):
    """Binary track: P(Y=1 | known + observed). MS track: belief vector over joint states."""
    M, obs = _obs(scn, observed)
    b = M.belief(obs)
    return float(b[1]) if scn["track"] == "binary" else b


def eu(scn, observed=()):
    M, obs = _obs(scn, observed)
    return dict(zip(scn["actions"], (M.U @ M.belief(obs)).tolist()))


def voi(scn, label, observed=()):
    M, obs = _obs(scn, observed)
    return M.item_stats(obs, M.lab2i[str(label).upper()])[0]


def eig(scn, label, observed=()):
    M, obs = _obs(scn, observed)
    return M.item_stats(obs, M.lab2i[str(label).upper()])[1]


def eig_decision(scn, label, observed=()):
    M, obs = _obs(scn, observed)
    return M.item_stats(obs, M.lab2i[str(label).upper()])[2]


def dp_policy(scn, observed=()):
    """Optimal next move given observations: 'S' (stop) or an item label; plus the decision if stopping."""
    M, obs = _obs(scn, observed)
    f = M.dp_first(obs)
    if f["move"] == "S":
        return {"move": "S", "action": scn["actions"][f["action"]], "value": f["v"]}
    return {"move": M.labels[f["item"]], "value": f["v"], "ties": [M.labels[i] for i in f["ties"]]}


def best_action(scn, observed=()):
    e = eu(scn, observed)
    return max(e, key=e.get)
