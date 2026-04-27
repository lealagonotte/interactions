import random
import numpy as np
from collections import deque

class DrosselSchwablForestFire:
    """Automate cellulaire pour le modèle de Drossel-Schwabl de propagation d'incendie de forêt."""
    EMPTY = 0
    TREE = 1
    FIRE = 2

    def __init__(self, width, height, p=0.01, f=0.0001, initial_tree_density=0.6):
        """
        width, height : dimensions de la grille
        p : probabilité de pousse d'un arbre sur une case vide
        f : probabilité qu'un arbre prenne feu spontanément
        initial_tree_density : densité initiale d'arbres
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
        """Voisinage de von Neumann : haut, bas, gauche, droite."""
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.grid[ny][nx] == self.FIRE:
                    return True
        return False

    def step(self):
        """Fait avancer le système d'un pas de temps."""
        new_grid = [[self.EMPTY for _ in range(self.width)] for _ in range(self.height)]

        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]

                if cell == self.FIRE:
                    # Le feu devient vide
                    new_grid[y][x] = self.EMPTY

                elif cell == self.TREE:
                    # brûle si un voisin brûle
                    # ou feu spontané avec probabilité f
                    if self._neighbors_on_fire(x, y) or random.random() < self.f:
                        new_grid[y][x] = self.FIRE
                    else:
                        new_grid[y][x] = self.TREE

                else:  # EMPTY
                    # pousse d'un arbre avec probabilité p
                    if random.random() < self.p:
                        new_grid[y][x] = self.TREE
                    else:
                        new_grid[y][x] = self.EMPTY

        self.grid = new_grid

    def ignite_random_tree(self):
        """Allume un arbre aléatoirement au début de la simulation pour démarrer un feu."""
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
        """Nombre de cellules actuellement en feu."""
        return sum(cell == self.FIRE for row in self.grid for cell in row)

  ################################################ Second implementation #############################""




EMPTY = 0
TREE = 1
BURNING = 2


class DrosselSchwablFFM:
    def __init__(self, L, theta, fixed="p", p=None, f=None, seed=None):
        """
        Modèle de Drossel-Schwabl Forest Fire Model.

        Paramètres
        ----------
        L : int
            Taille du paysage carré L x L.
        theta : float
            Rapport theta = p / f.
        fixed : str
            "p" si on fixe p et on calcule f.
            "f" si on fixe f et on calcule p.
        p : float
            Probabilité de croissance si fixed="p".
        f : float
            Probabilité d'ignition si fixed="f".
        seed : int, optionnel
            Graine aléatoire.
        """

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.L = L
        self.N = L * L
        self.theta = theta

        if fixed == "p":
            if p is None:
                raise ValueError("Il faut donner p quand fixed='p'.")
            self.p = p
            self.f = p / theta

        elif fixed == "f":
            if f is None:
                raise ValueError("Il faut donner f quand fixed='f'.")
            self.f = f
            self.p = theta * f

        else:
            raise ValueError("fixed doit être 'p' ou 'f'.")

        if not (0 <= self.p <= 1):
            raise ValueError(f"p doit être dans [0,1]. Ici p={self.p}")

        if not (0 <= self.f <= 1):
            raise ValueError(f"f doit être dans [0,1]. Ici f={self.f}")

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
        Voisinage de von Neumann : haut, bas, gauche, droite.
        Conditions périodiques.
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
        Brûle tout le cluster connecté d'arbres contenant la cellule initiale.
        Retourne la taille du feu.
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
        Une étape de simulation selon l'algorithme :

        1. choisir une cellule au hasard ;
        2. si c'est un arbre, déclencher FireSpread ;
        3. faire theta tentatives de croissance.
        """

        # Étape feu
        i, j = self.random_cell()

        if self.grid[i, j] == TREE:
            fire_size = self.fire_spread(i, j)
        else:
            fire_size = 0

        # Étape croissance
        # Si theta n'est pas entier, on utilise une version stochastique.
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
        Lance la simulation.

        Paramètres
        ----------
        steps : int
            Nombre total d'étapes.
        burn_in : int
            Nombre d'étapes initiales à ignorer dans les résultats.

        Retourne
        --------
        dict
            Tailles de feux et densités après burn-in.
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