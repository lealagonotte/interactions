import numpy as np
from typing import Callable, List, Tuple, Optional


class CellularAutomaton:
    """
    Cellular Automaton simulator for forest fire propagation.

    Attributes:
        state_grid (np.ndarray): Current fire state (0: healthy, 1: burning/burnt).
        wind_grid (np.ndarray): Local wind coefficients for each cell.
        height_grid (np.ndarray): Elevation map of the terrain.
        phi (Callable): Function where:
             - phi(Δh > 0) > 1
             - phi(Δh = 0) = 1
             - phi(Δh < 0) < 1
        neighborhood (list): List of (di, dj) offsets for the 9-cell neighborhood.
    """

    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        wind_grid: np.ndarray,
        height_grid: np.ndarray,
        phi: Callable[[float], float],
        burnable_mask: Optional[np.ndarray] = None,
    ) -> None:
        self.state_grid = np.zeros((grid_height, grid_width))
        self.neighborhood = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 0),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        self.wind_grid = wind_grid
        self.height_grid = height_grid
        self.phi = phi
        if burnable_mask is None:
            self.burnable_mask = np.ones((grid_height, grid_width))
        else:
            self.burnable_mask = burnable_mask

    def __repr__(self) -> str:
        h, w = self.state_grid.shape
        active_fire_sum = np.sum(self.state_grid)
        return f"CellularAutomaton(Size: {h}x{w}, Total Fire Intensity: {active_fire_sum:.2f})"

    def get_state(self) -> np.ndarray:
        return self.state_grid

    def initialize_ignition(
        self, start_points: List[Tuple[int, int]], init_states: List[float]
    ) -> None:
        for idx, (i, j) in enumerate(start_points):
            if 0 <= i < self.state_grid.shape[0] and 0 <= j < self.state_grid.shape[1]:
                self.state_grid[i, j] = init_states[idx]

    def evolve(self) -> None:
        rows, cols = self.state_grid.shape
        next_grid = np.zeros_like(self.state_grid)

        for i in range(rows):
            for j in range(cols):
                if self.burnable_mask[i, j] == 0:
                    next_grid[i, j] = 0
                    continue

                total_influence = 0
                for di, dj in self.neighborhood:
                    ni, nj = i + di, j + dj

                    if 0 <= ni < rows and 0 <= nj < cols:
                        s_neighbor_effective = (
                            self.state_grid[ni, nj] * self.burnable_mask[ni, nj]
                        )

                        if s_neighbor_effective > 0:
                            dist_coeff = 0.83 if abs(di) + abs(dj) == 2 else 1.0
                            delta_h = self.height_grid[i, j] - self.height_grid[ni, nj]
                            h_influence = self.phi(delta_h)

                            total_influence += (
                                dist_coeff
                                * self.wind_grid[ni, nj]
                                * h_influence
                                * s_neighbor_effective
                            )

                next_grid[i, j] = self.state_grid[i, j] + total_influence

        self.state_grid = np.clip(next_grid, 0, 1)
