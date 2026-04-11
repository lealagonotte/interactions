import random


class DrosselSchwablForestFire:
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

  