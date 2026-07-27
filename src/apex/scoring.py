"""Proper scoring rules for a distribution over finishing orders.

Shared by the walk-forward harness and the after-the-fact race scorer, which previously
carried their own copies. Two implementations of a scoring rule is one too many: the whole
point of these numbers is that the backtest and the live test are measured the same way,
and that guarantee is worth more than the forty lines it saves.

- **RPS** (ranked probability score) is the right metric for ordered outcomes. It rewards
  putting mass *near* the truth, unlike log-loss, which only looks at the exact cell.
- **log-loss on win / podium / points** checks the headline probabilities specifically.
- **Spearman** asks the weakest question — does the predicted order match at all.

Lower is better on everything except Spearman.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

EPS = 1e-12


def rps(probs: np.ndarray, actual_ix: np.ndarray) -> float:
    """Mean ranked probability score over the drivers in one race.

    `probs[i, p]` is P(driver i finishes in position p); `actual_ix[i]` is the true
    position index. Squared error between the predicted and the observed step CDF,
    normalised by the number of thresholds.
    """
    n = probs.shape[1]
    out = []
    for i, a in enumerate(actual_ix):
        cdf = np.cumsum(probs[i])
        step = (np.arange(n) >= a).astype(float)
        out.append(np.sum((cdf - step) ** 2) / (n - 1))
    return float(np.mean(out))


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def score(probs: np.ndarray, actual_pos: np.ndarray, digits: int | None = None) -> dict:
    """All five metrics for one race.

    Positions are clipped into the matrix. That matters when scoring the *whole field*:
    a car classified below the width of the probability matrix would otherwise index off
    the end. Where every entrant is inside the matrix — the walk-forward case — the clip
    is a no-op, so both callers get identical numbers from identical code.
    """
    n = probs.shape[1]
    ix = np.clip(actual_pos.astype(int) - 1, 0, n - 1)
    exp_pos = (probs * np.arange(1, n + 1)).sum(1)
    out = {
        "rps": rps(probs, ix),
        "ll_win": logloss(probs[:, 0], (ix == 0).astype(float)),
        "ll_podium": logloss(probs[:, :3].sum(1), (ix < 3).astype(float)),
        "ll_points": logloss(probs[:, :10].sum(1), (ix < 10).astype(float)),
        "spearman": float(spearmanr(exp_pos, actual_pos).statistic),
    }
    return {k: round(v, digits) for k, v in out.items()} if digits else out
