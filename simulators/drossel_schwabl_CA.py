import random
import numpy as np
from collections import deque

class DrosselSchwablForestFire:
    """CA of Drossel-Schwabl Forest Fire Model with von Neumann neighborhood."""
    EMPTY = 0
    TREE = 1
    FIRE = 2

    def __init__(self, width, height, p=0.01, f=0.0001, initial_tree_density=0.6):
        """
        width, height : grid dimensions
        p : probability that an empty cell grows a tree
        f : probability that a tree spontaneously ignites
        initial_tree_density : initial proportion of cells that are trees at the start of the simulation
        """
        self.width = width
        self.height = height
        self.p = p
        self.f = f

        self.grid = [
            [
                self.TREE if random.random() < initial_tree_density else self.EMPTY
                for _ in range(width)
            ]
            for _ in range(height)
        ]

    def _neighbors_on_fire(self, x, y):
        """ von Neumann neighbourhood: up, down, left, right."""
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.grid[ny][nx] == self.FIRE:
                    return True
        return False

    def step(self):
        """Update the grid according to the rules of the model."""
        new_grid = [[self.EMPTY for _ in range(self.width)] for _ in range(self.height)]

        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]

                if cell == self.FIRE:
                    new_grid[y][x] = self.EMPTY

                elif cell == self.TREE:

                    if self._neighbors_on_fire(x, y) or random.random() < self.f:
                        new_grid[y][x] = self.FIRE
                    else:
                        new_grid[y][x] = self.TREE

                else:  # EMPTY

                    if random.random() < self.p:
                        new_grid[y][x] = self.TREE
                    else:
                        new_grid[y][x] = self.EMPTY

        self.grid = new_grid

    def ignite_random_tree(self):
        """Ignite a random tree in the grid (useful to start the fire if there are no spontaneous ignitions)."""
        trees = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self.grid[y][x] == self.TREE
        ]
        if trees:
            x, y = random.choice(trees)
            self.grid[y][x] = self.FIRE

    def count_burning(self):
        """Number of burning trees in the grid."""
        return sum(cell == self.FIRE for row in self.grid for cell in row)

  ################################################ Second implementation #############################""




EMPTY = 0
TREE = 1
BURNING = 2


class DrosselSchwablFFM:
    def __init__(self, L, theta, fixed="p", p=None, f=None, seed=None):
        """
        Drossel-Schwabl Forest Fire Model with von Neumann neighborhood.    

        Paramters
        ----------
        L : int
            Grid size (LxL).
        theta : float
            p/f ratio, mean number of growth attempts per ignition.
        fixed : str
            "p" if we fix p and compute f,
            "f" if we fix f and compute p.
        p : float
            probaility of tree growth if fixed="p".
        f : float
            probability of spontaneous ignition if fixed="f".
        seed : int, optionnel
            
        """

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.L = L
        self.N = L * L
        self.theta = theta

        if fixed == "p":
            if p is None:
                raise ValueError("You must give p.")
            self.p = p
            self.f = p / theta

        elif fixed == "f":
            if f is None:
                raise ValueError("You must give f.")
            self.f = f
            self.p = theta * f

        else:
            raise ValueError("fixed must be f or p.")

        if not (0 <= self.p <= 1):
            raise ValueError(f"p should be in [0,1]. Ici p={self.p}")

        if not (0 <= self.f <= 1):
            raise ValueError(f"f should be in[0,1]. Ici f={self.f}")

        self.grid = np.zeros((L, L), dtype=np.int8)

        self.time = 0
        self.fire_sizes = []
        self.tree_density = []

    def random_cell(self):
        i = random.randrange(self.L)
        j = random.randrange(self.L)
        return i, j

    def neighbors(self, i, j):
        """
       von Neumann's neighbourhood .

        """
        L = self.L
        return [
            ((i - 1) % L, j),
            ((i + 1) % L, j),
            (i, (j - 1) % L),
            (i, (j + 1) % L),
        ]

    def fire_spread(self, start_i, start_j):
        """
        BFS to spread the fire from the initial cell (start_i, start_j) and count how many trees are burned.
        """
        if self.grid[start_i, start_j] != TREE:
            return 0

        q = deque()
        q.append((start_i, start_j))

        self.grid[start_i, start_j] = BURNING
        burned_cells = []

        while q:
            i, j = q.popleft()
            burned_cells.append((i, j))

            for ni, nj in self.neighbors(i, j):
                if self.grid[ni, nj] == TREE:
                    self.grid[ni, nj] = BURNING
                    q.append((ni, nj))

        for i, j in burned_cells:
            self.grid[i, j] = EMPTY

        return len(burned_cells)

    def step_grassberger(self):
        """
        One step of the simulation following Grassberger's algorithm:
        1. Randomly select a cell.  If it's a tree, ignite it and spread the fire.
        2. Randomly select cells to grow new trees according to the growth probability p.

        """

        i, j = self.random_cell()

        if self.grid[i, j] == TREE:
            fire_size = self.fire_spread(i, j)
        else:
            fire_size = 0

        n_growth_attempts = int(self.theta)
        fractional_part = self.theta - n_growth_attempts

        if random.random() < fractional_part:
            n_growth_attempts += 1

        for _ in range(n_growth_attempts):
            i, j = self.random_cell()
            if self.grid[i, j] == EMPTY:
                self.grid[i, j] = TREE

        self.time += 1
        self.fire_sizes.append(fire_size)
        self.tree_density.append(np.mean(self.grid == TREE))

        return fire_size

    def run(self, steps, burn_in=0):
        """
        Run the simulation for a given number of steps and return the fire sizes and tree densities after burn-in.

        Parameters
        ----------
        steps : int
            Number of steps to run the simulation.
        burn_in : int
            Number of initial steps to discard as burn-in.

        Retourne
        --------
        dict
            Fire sizes, tree densities, final grid state, and model parameters after running the simulation.
        """

        for _ in range(steps):
            self.step_grassberger()

        return {
            "fire_sizes": np.array(self.fire_sizes[burn_in:]),
            "tree_density": np.array(self.tree_density[burn_in:]),
            "grid": self.grid.copy(),
            "p": self.p,
            "f": self.f,
            "theta": self.theta,
        }