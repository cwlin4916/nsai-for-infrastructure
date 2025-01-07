import random

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from . import line_extending_game_tools as lgt

class WindowlessLineExtendingGameEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, grid_rows = 10, grid_cols = 10, n_features = (2, 3, 4), max_reward = 100, max_moves = None):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.n_features = n_features
        self.pyrandom_state = random.getstate()
        self.problem = self.solution = self.state = None
        self.n_moves = None
        self.max_reward = max_reward
        self.max_moves = grid_rows*grid_cols if max_moves is None else max_moves

        self.observation_space = spaces.MultiBinary((grid_rows, grid_cols))
        self.action_space = spaces.Discrete(grid_rows*grid_cols)
    
    def _get_obs(self):
        return self.state
    
    def _get_info(self):
        return {}
    
    def reset(self, seed = None, options = None):
        assert options is None
        random.setstate(self.pyrandom_state)
        super().reset(seed=seed)
        if seed is not None: random.seed(seed)
        self.problem, self.solution = lgt.generate_problem(10, 10, *[random.randrange(n) for n in self.n_features])
        self.state = self.problem
        self.n_moves = 0
        # lgt.display_grid(self.problem)
        self.pyrandom_state = random.getstate()
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        coords = (action // self.grid_cols, action % self.grid_cols)
        self.state[coords] = not self.state[coords]
        self.n_moves += 1
        terminated = np.array_equal(self.state, self.solution) 
        truncated = self.n_moves >= self.max_moves and not terminated
        reward = self.max_reward-np.sum(self.solution ^ self.state) if terminated or truncated else 0
        return self._get_obs(), reward, terminated, truncated, self._get_info()

gym.register(
    id="leg/WindowlessLineExtendingGameEnv-v0",
    entry_point=WindowlessLineExtendingGameEnv
)
