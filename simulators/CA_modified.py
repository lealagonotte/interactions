import numpy as np
from typing import Callable, List, Tuple

########################################## Add age #############################################
class CellularAutomaton_modfied:
    """
    Cellular Automaton simulator for forest fire propagation by adding age-dependent inflammability.

    state_grid:
        0 = healthy
        1 = burning / burnt

    age_grid:
        Fixed initial map chosen by the user.
        It represents stand age / time since last disturbance / fuel maturity.
        It does not evolve during the simulation.
    """

    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        wind_grid: np.ndarray,
        height_grid: np.ndarray,
        age_grid: np.ndarray,
        phi: Callable[[float], float],
        t_max: float = 30.0,
        p_max: float = 1.0,
        alpha_age: float = 2.0,
    ) -> None:

        self.state_grid = np.zeros((grid_height, grid_width), dtype=float)
        self.age_grid = age_grid.astype(float)

        self.neighborhood = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        self.wind_grid = wind_grid
        self.height_grid = height_grid
        self.phi = phi

        # Parameters for age-dependent inflammability -> Peterson-style function
        self.t_max = t_max
        self.p_max = p_max
        self.alpha_age = alpha_age

    def __repr__(self) -> str:
        h, w = self.state_grid.shape
        active_fire_sum = np.sum(self.state_grid)
        return f"CellularAutomaton(Size: {h}x{w}, Total Fire Intensity: {active_fire_sum:.2f})"

    def get_state(self) -> np.ndarray:
        return self.state_grid

    def get_age(self) -> np.ndarray:
        return self.age_grid

    def initialize_ignition(
        self, start_points: List[Tuple[int, int]], init_states: List[float]
    ) -> None:
        for idx, (i, j) in enumerate(start_points):
            if 0 <= i < self.state_grid.shape[0] and 0 <= j < self.state_grid.shape[1]:
                self.state_grid[i, j] = np.clip(init_states[idx], 0.0, 1.0)

    def age_inflammability(self, age: float) -> float:
        """
        Peterson-style age-dependent inflammability:
            p(age) = (1 + p_max)^((age / t_max)^alpha_age) - 1   if age < t_max
                    p_max                                      otherwise

        t_max represents the age at which inflammability saturates.
        Before t_max, inflammability increases slowly at first and then rapidly.
        After t_max, it remains constant at p_max.
        """
        if age < self.t_max:
            return (1 + self.p_max) ** ((age / self.t_max) ** self.alpha_age) - 1
        return self.p_max

    def evolve(self) -> None:
        rows, cols = self.state_grid.shape
        next_grid = np.copy(self.state_grid)

        for i in range(rows):
            for j in range(cols):
                total_influence = 0.0

                for di, dj in self.neighborhood:
                    ni, nj = i + di, j + dj

                    if 0 <= ni < rows and 0 <= nj < cols:
                        dist_coeff = 0.83 if abs(di) + abs(dj) == 2 else 1.0
                        s_neighbor = self.state_grid[ni, nj]

                        delta_h = self.height_grid[i, j] - self.height_grid[ni, nj]
                        h_influence = self.phi(delta_h)

                        total_influence += (
                            dist_coeff
                            * self.wind_grid[ni, nj]
                            * h_influence
                            * s_neighbor
                        )

                inflammability = self.age_inflammability(self.age_grid[i, j])

                next_grid[i, j] = self.state_grid[i, j] + inflammability * total_influence

        self.state_grid = np.clip(next_grid, 0.0, 1.0)





        #################################################### Add Humidity and Age #############################################
class CellularAutomaton_humidity_age:
    """
    Cellular Automaton simulator for forest fire propagation modified with humidity and age effects.

    state_grid:
        0 = healthy
        1 = burning / burnt

    age_grid:
        Fixed initial map chosen by the user.

    moisture_grid:
        Fixed initial humidity/moisture map chosen by the user.

    phi:
        Topographic effect function depending on delta_h.

    psi:
        Moisture effect function depending on local moisture. It should return a value between 0 and 1 representing the dampening effect of moisture on fire spread.
        psi should be a decreasing function of moisture, where psi(0) = 1 (no dampening when completely dry) and psi(max_moisture) = 0 (complete dampening when at maximum moisture).
    """

    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        wind_grid: np.ndarray,
        height_grid: np.ndarray,
        age_grid: np.ndarray,
        moisture_grid: np.ndarray,
        phi: Callable[[float], float],
        psi: Callable[[float], float],
        t_max: float = 30.0,
        p_max: float = 1.0,
        alpha_age: float = 2.0,
    ) -> None:

        self.state_grid = np.zeros((grid_height, grid_width), dtype=float)
        self.age_grid = age_grid.astype(float)
        self.moisture_grid = moisture_grid.astype(float)

        self.neighborhood = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        self.wind_grid = wind_grid
        self.height_grid = height_grid
        self.phi = phi
        self.psi = psi

        self.t_max = t_max
        self.p_max = p_max
        self.alpha_age = alpha_age

    def __repr__(self) -> str:
        h, w = self.state_grid.shape
        active_fire_sum = np.sum(self.state_grid)
        return f"CellularAutomaton(Size: {h}x{w}, Total Fire Intensity: {active_fire_sum:.2f})"

    def get_state(self) -> np.ndarray:
        return self.state_grid

    def get_age(self) -> np.ndarray:
        return self.age_grid

    def get_moisture(self) -> np.ndarray:
        return self.moisture_grid

    def initialize_ignition(
        self, start_points: List[Tuple[int, int]], init_states: List[float]
    ) -> None:
        for idx, (i, j) in enumerate(start_points):
            if 0 <= i < self.state_grid.shape[0] and 0 <= j < self.state_grid.shape[1]:
                self.state_grid[i, j] = np.clip(init_states[idx], 0.0, 1.0)

    def age_inflammability(self, age: float) -> float:
        """
        Peterson-style age-dependent inflammability:
            p(age) = (1 + p_max)^((age / t_max)^alpha_age) - 1   if age < t_max
                    p_max                                      otherwise

        t_max represents the age at which inflammability saturates.
        Before t_max, inflammability increases slowly at first and then rapidly.
        After t_max, it remains constant at p_max.
        """
        if age < self.t_max:
            return (1 + self.p_max) ** ((age / self.t_max) ** self.alpha_age) - 1
        return self.p_max

    def evolve(self, use_age: bool = True, use_moisture: bool = True) -> None:
        """Evolve the cellular automaton by one time step, incorporating age and moisture effects.
        If use_age is True, the age-dependent inflammability will be applied. If use_moisture is True, the moisture effect will be applied."""
        rows, cols = self.state_grid.shape
        next_grid = np.copy(self.state_grid)

        for i in range(rows):
            for j in range(cols):
                total_influence = 0.0

                for di, dj in self.neighborhood:
                    ni, nj = i + di, j + dj

                    if 0 <= ni < rows and 0 <= nj < cols:
                        dist_coeff = 0.83 if abs(di) + abs(dj) == 2 else 1.0
                        s_neighbor = self.state_grid[ni, nj]

                        delta_h = self.height_grid[i, j] - self.height_grid[ni, nj]
                        h_influence = self.phi(delta_h)

                        total_influence += (
                            dist_coeff
                            * self.wind_grid[ni, nj]
                            * h_influence
                            * s_neighbor
                        )

                if use_age:
                    age_factor = self.age_inflammability(self.age_grid[i, j])
                else:
                    age_factor = 1.0

                if use_moisture:
                    moisture_factor = self.psi(self.moisture_grid[i, j])
                else:
                    moisture_factor = 1.0

                next_grid[i, j] = self.state_grid[i, j] + age_factor * moisture_factor * total_influence

        self.state_grid = np.clip(next_grid, 0.0, 1.0)