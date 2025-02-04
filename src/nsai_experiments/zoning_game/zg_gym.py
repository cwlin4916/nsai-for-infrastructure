import random
from enum import Enum

import numpy as np

import gymnasium as gym
from gymnasium import spaces

class Tile(Enum):
    RESIDENTIAL = 1
    COMMERCIAL = 2
    INDUSTRIAL = 3
    DOWNTOWN = 4
    PARK = 5

DEFAULT_OCCURRENCES = {
    Tile.RESIDENTIAL: 14/36,
    Tile.COMMERCIAL: 6/36,
    Tile.INDUSTRIAL: 7/36,
    Tile.DOWNTOWN: 5/36,
    Tile.PARK: 4/36
}

class ZoningGameEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size = 6, initially_filled_frac = 0.4, occurrences = DEFAULT_OCCURRENCES):
        """
        Arguments (all have sensible defaults):
          `grid_size = 6`: side length of the square grid
          `initially_filled_frac = 0.4`: fraction of cells that start the game filled
          `occurrences = DEFAULT_OCCURRENCES`: dict from Tile to how commonly that tile occurs; will be normalized
        """
        self.grid_size = grid_size
        self.initially_filled_frac = initially_filled_frac
        self.occurrences = {k: v/sum(occurrences.values()) for k, v in occurrences.items()}

        self.tile_grid, self.tile_queue = self._generate_problem()
        self.observation_space = spaces.MultiDiscrete([[len(Tile)]*self.grid_size]*self.grid_size)
        self.action_space = spaces.Discrete(self.grid_size*self.grid_size)

    def _generate_problem(self):
        return 1, 1
