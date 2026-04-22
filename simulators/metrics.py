"""
Fire Modeling Metrics
=====================
Computes shape and spread metrics between predicted and ground-truth
burned-area grids P_t, G_t ∈ [0,1]^{n}.

Example:

P = [np.random.uniform(0,1,(10,10)) for _ in range(10)]
G = [np.random.uniform(0,1,(10,10)) for _ in range(10)]
fm = FireMetrics(P, G)
fm.summary(0,0)
"""

import numpy as np
from scipy.spatial.distance import cdist
import ot


class FireMetrics:
    """
    Compute shape and spread metrics for fire model evaluation.

    Parameters
    ----------
    P : list[np.ndarray]
        Sequence of predicted grids P_t ∈ [0,1]^{mxn}, indexed by time t.
    G : list[np.ndarray]
        Sequence of ground-truth grids G_t ∈ [0,1]^{mxn}, indexed by time t.
    tau : float
        Burn-level threshold τ ∈ [0,1] used for binarisation and arrival times.
        Default: 0.5.
    """

    def __init__(
        self,
        P: list[np.ndarray],
        G: list[np.ndarray],
        tau: float = 0.5,
    ) -> None:
        if not (0.0 <= tau <= 1.0):
            raise ValueError("tau must be in [0, 1].")

        self.P = [np.asarray(p, dtype=float) for p in P]
        self.G = [np.asarray(g, dtype=float) for g in G]
        self.tau = tau
        self.T = len(P)
        self.m, self.n = self.P[0].shape

        # Pre-compute arrival-time maps (used by spread metrics).
        self._T_P: np.ndarray | None = None
        self._T_G: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _arrival_times(self, grids: list[np.ndarray]) -> np.ndarray:
        """
        T^τ(i,j) = inf{ t | grid_t(i,j) >= τ }.
        Cells never reaching τ are assigned np.inf.
        """
        T_map = np.full((self.m, self.n), np.inf)
        for t, grid in enumerate(grids):
            reached = (grid >= self.tau) & np.isinf(T_map)
            T_map[reached] = t
        return T_map

    @property
    def arrival_time_P(self) -> np.ndarray:
        if self._T_P is None:
            self._T_P = self._arrival_times(self.P)
        return self._T_P

    @property
    def arrival_time_G(self) -> np.ndarray:
        if self._T_G is None:
            self._T_G = self._arrival_times(self.G)
        return self._T_G

    @staticmethod
    def _binarise(grid: np.ndarray, tau: float) -> np.ndarray:
        """Return boolean mask: grid(i,j) >= tau."""
        return grid >= tau

    # ------------------------------------------------------------------
    # Shape metrics  (operate on a single pair of grids at times t, t_)
    # ------------------------------------------------------------------

    def iou(self, t: int, t_tilde: int) -> float:
        """
        Intersection over Union between P_t and G_{t_}.

        IOU = average min(P,G) / average max(P,G)
        """
        P, G = self.P[t], self.G[t_tilde]
        intersection = np.minimum(P, G).sum()
        union = np.maximum(P, G).sum()
        print(union)
        if union == 0.0:
            return 1.0  # both grids are identically zero
        return float(intersection / union)

    def lp_norm(self, t: int, t_tilde: int, p: int = 2) -> float:
        """
        l_p norm between P_t and G_{t_}.

        l_p^p = (1/mn) sum |P - G|^p
        Returns the p-th root (i.e. l_p, not l_p^p).
        """
        P, G = self.P[t], self.G[t_tilde]
        value = np.mean(np.abs(P - G) ** p)
        return float(value ** (1.0 / p))

    def wasserstein2(self, t: int, t_tilde: int) -> float:
        """
        Squared Wasserstein-2 distance W_2^2(mu_t, nu_{t_}).

        mu and nu are discrete probability measures on the grid obtained by
        normalising P_t and G_{t_}.  The optimal transport plan is solved
        exactly via the linear assignment / EMD formulation.

        Returns W_2 (not W_2^2).

        Note
        ----
        Exact OT scales as O((mn)^3) and may be slow for large grids.
        For grids larger than ~30x30 consider using the POT library
        (ot.emd2) with an efficient solver.
        """
        P, G = self.P[t], self.G[t_tilde]

        sum_P = P.sum()
        sum_G = G.sum()

        if sum_P == 0.0 or sum_G == 0.0:
            # One distribution is empty; distance is ill-defined -> return inf.
            return float("inf")

        mu = (P / sum_P).ravel()
        nu = (G / sum_G).ravel()

        # Grid coordinates of every cell.
        ii, jj = np.indices((self.m, self.n))
        coords = np.stack([ii.ravel(), jj.ravel()], axis=1).astype(float)

        # Squared Euclidean cost matrix  C[a,b] = ||coord_a - coord_b||^2
        C = cdist(coords, coords, metric="sqeuclidean")

        W2_sq = ot.emd2(mu, nu, C)

        return float(np.sqrt(W2_sq))

    def hausdorff(self, t: int, t_tilde: int) -> float:
        """
        Hausdorff distance between the τ-thresholded active sets of P_t and G_{t_}.

        Returns np.inf if one of the sets is empty.
        """
        P_bin = self._binarise(self.P[t], self.tau)
        G_bin = self._binarise(self.G[t_tilde], self.tau)

        P_pts = np.argwhere(P_bin).astype(float)
        G_pts = np.argwhere(G_bin).astype(float)

        if len(P_pts) == 0 or len(G_pts) == 0:
            return float("inf")

        # Pairwise Euclidean distances.
        D = cdist(P_pts, G_pts, metric="euclidean")

        # sup_a inf_b d(a,b)  and  sup_b inf_a d(a,b)
        h_PG = D.min(axis=1).max()
        h_GP = D.min(axis=0).max()

        return float(max(h_PG, h_GP))

    # Spread metrics  (use the full time series via arrival-time maps)

    def aatd(self) -> float:
        """
        Average Arrival Time Difference.

        AATD^tau = (1/mn) sum_{i,j} |T_P^tau(i,j) - T_G^tau(i,j)|

        Cells where either arrival time is infinite are excluded.
        """
        T_P = self.arrival_time_P
        T_G = self.arrival_time_G

        valid = np.isfinite(T_P) & np.isfinite(T_G)
        if not valid.any():
            return float("nan")

        return float(np.mean(np.abs(T_P[valid] - T_G[valid])))

    @staticmethod
    def _gradient(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Central-difference gradient on interior cells; forward/backward
        differences on the boundary.  Cells with infinite values propagate
        inf into adjacent gradient estimates.
        """
        # np.gradient handles boundaries with one-sided differences.
        grad_i, grad_j = np.gradient(field)
        return grad_i, grad_j

    def fde(self) -> float:
        """
        Fire Spread Direction Error (FDE^τ).

        Uses the negative gradient of the arrival-time field as the local
        spread direction.  Angular error (in radians) is averaged over all
        grid cells where both direction vectors are non-zero.
        """
        T_P = self.arrival_time_P.copy()
        T_G = self.arrival_time_G.copy()

        # Replace inf with nan so np.gradient ignores them gracefully.
        T_P[~np.isfinite(T_P)] = np.nan
        T_G[~np.isfinite(T_G)] = np.nan

        grad_Pi, grad_Pj = self._gradient(T_P)
        grad_Gi, grad_Gj = self._gradient(T_G)

        # Direction = negative gradient.
        dPi, dPj = -grad_Pi, -grad_Pj
        dGi, dGj = -grad_Gi, -grad_Gj

        # Magnitudes.
        norm_P = np.sqrt(dPi**2 + dPj**2)
        norm_G = np.sqrt(dGi**2 + dGj**2)

        # Valid cells Omega^tau: both gradients finite and non-zero.
        valid = (norm_P > 0) & (norm_G > 0) & np.isfinite(norm_P) & np.isfinite(norm_G)

        if not valid.any():
            return float("nan")

        # Normalise directions.
        dPi_n = dPi[valid] / norm_P[valid]
        dPj_n = dPj[valid] / norm_P[valid]
        dGi_n = dGi[valid] / norm_G[valid]
        dGj_n = dGj[valid] / norm_G[valid]

        # Dot product clipped to [-1, 1] to avoid arccos domain errors.
        dot = np.clip(dPi_n * dGi_n + dPj_n * dGj_n, -1.0, 1.0)
        angles = np.arccos(dot)

        return float(np.mean(angles))

    # Convenience: compute all metrics for a given pair (t, t_)

    def all_shape_metrics(
        self, t: int, t_tilde: int, p: int = 2
    ) -> dict[str, float]:
        """Return a dict with all shape metrics for the pair (t, t_)."""
        return {
            "iou": self.iou(t, t_tilde),
            f"l{p}_norm": self.lp_norm(t, t_tilde, p=p),
            "wasserstein2": self.wasserstein2(t, t_tilde),
            "hausdorff": self.hausdorff(t, t_tilde),
        }

    def all_spread_metrics(self) -> dict[str, float]:
        """Return a dict with all spread metrics (use the full time series)."""
        return {
            "aatd": self.aatd(),
            "fde_radians": self.fde(),
            "fde_degrees": np.degrees(self.fde()) if not np.isnan(self.fde()) else float("nan"),
        }

    def summary(self, t: int, t_tilde: int, p: int = 2) -> dict[str, float]:
        """Return all metrics combined."""
        return {**self.all_shape_metrics(t, t_tilde, p=p), **self.all_spread_metrics()}
    

    @staticmethod
    def aatd_from_maps(
        T_P: np.ndarray,
        T_G: np.ndarray,
        unreachable_value: float = -1,
    ) -> float:
        """
        Average Arrival Time Difference computed directly from two arrival-time maps.

        Parameters
        ----------
        T_P : np.ndarray
            Predicted arrival-time map.
        T_G : np.ndarray
            Ground-truth arrival-time map.
        unreachable_value : float
            Value used for cells never reached by fire in saved maps.
            Default: -1.

        Returns
        -------
        float
            Mean absolute difference on cells reached in both maps.
            Returns nan if there is no common reached cell.
        """
        T_P = np.asarray(T_P, dtype=float)
        T_G = np.asarray(T_G, dtype=float)

        valid = (T_P != unreachable_value) & (T_G != unreachable_value)

        if not np.any(valid):
            return float("nan")

        return float(np.mean(np.abs(T_P[valid] - T_G[valid])))