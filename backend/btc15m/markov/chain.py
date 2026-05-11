"""9-state Markov chain over 1-minute BTC % price changes."""
from __future__ import annotations

import bisect
import math
from typing import List, Tuple

from scipy.stats import norm as _norm

STATES: List[str] = [
    "large_down", "mid_down", "small_down", "small_down2",
    "flat",
    "small_up", "small_up2", "mid_up", "large_up",
]
BINS: List[float] = [-0.3, -0.15, -0.05, -0.01, 0.01, 0.05, 0.15, 0.3]
N: int = 9

# Representative drift for each state, in % per minute.
_STATE_DRIFTS_PCT: List[float] = [
    -0.45, -0.225, -0.10, -0.03, 0.0, 0.03, 0.10, 0.225, 0.45,
]


def bin_state(pct_change: float) -> int:
    """Map a % change (e.g. 0.15 for +0.15%) to a state index 0–8."""
    return bisect.bisect(BINS, pct_change)


class MarkovChain:
    def __init__(self) -> None:
        self._counts: List[List[int]] = [[0] * N for _ in range(N)]
        self._total: int = 0
        self._current_state: int = N // 2  # start at flat
        self._dist: List[float] = []

    def add_transition(self, from_state: int, to_state: int) -> None:
        self._counts[from_state][to_state] += 1
        self._total += 1
        self._current_state = to_state

    def get_transition_matrix(self) -> List[List[float]]:
        eps = 1e-8
        matrix: List[List[float]] = []
        for row in self._counts:
            total = sum(row) + eps
            matrix.append([c / total for c in row])
        return matrix

    def get_dominant_state(self) -> Tuple[int, float]:
        """Return (state_index, persistence) for the most self-persistent state."""
        matrix = self.get_transition_matrix()
        persistences = [matrix[i][i] for i in range(N)]
        dominant = max(range(N), key=lambda i: persistences[i])
        return dominant, persistences[dominant]

    def propagate(self, steps: int) -> List[float]:
        """Chapman-Kolmogorov: advance current state distribution by `steps` steps."""
        matrix = self.get_transition_matrix()
        dist: List[float] = [0.0] * N
        dist[self._current_state] = 1.0
        for _ in range(steps):
            new_dist = [0.0] * N
            for j in range(N):
                for i in range(N):
                    new_dist[j] += dist[i] * matrix[i][j]
            dist = new_dist
        self._dist = dist
        return dist

    def p_yes(self, strike: float, spot: float, sigma_approx: float) -> float:
        """Convert the last propagated distribution to P(BTC > strike) via Gaussian approx."""
        if not self._dist:
            return 0.5
        expected_drift_pct = sum(
            self._dist[i] * _STATE_DRIFTS_PCT[i] for i in range(N)
        )
        expected_spot = spot * (1.0 + expected_drift_pct / 100.0)
        if sigma_approx <= 0:
            sigma_approx = max(1.0, spot * 0.001)
        return float(_norm.cdf((expected_spot - strike) / sigma_approx))

    def is_valid(self) -> bool:
        return self._total >= 20
