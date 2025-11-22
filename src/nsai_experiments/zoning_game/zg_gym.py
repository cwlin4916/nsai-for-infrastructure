from enum import Enum
import logging
import io
import numbers
import math

import numpy as np

import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Tile(Enum):
    EMPTY = 0
    RESIDENTIAL = 1
    COMMERCIAL = 2
    INDUSTRIAL = 3
    DOWNTOWN = 4
    PARK = 5

class Location(Enum):
    BOARD_CENTER = 0
    BOARD_EDGE = 1
    BOARD_CORNER = 2
    BOARD_VERTICAL_MEDIAN = 3
    BOARD_HORIZONTAL_MEDIAN = 4

DEFAULT_OCCURRENCES = {
    Tile.RESIDENTIAL: 14/36,
    Tile.COMMERCIAL: 6/36,
    Tile.INDUSTRIAL: 7/36,
    Tile.DOWNTOWN: 5/36,
    Tile.PARK: 4/36
}

def blank_of_size(grid_size, fill_tile = Tile.EMPTY):  # 0.333
    return np.ones((grid_size, grid_size), dtype=np.int32) * fill_tile.value

_ORTHOGONAL_MASK_ROW = np.array([0, 2, 1, 1])
_ORTHOGONAL_MASK_COL = np.array([1, 1, 2, 0])

def orthogonal_neighbors(padded_grid, my_row, my_col):
    "Given a zero-padded tile grid and the row and column in unpadded coordinates of a particular tile, return a list of orthogonal (non-diagonal) neighbors"
    return padded_grid[_ORTHOGONAL_MASK_ROW + my_row, _ORTHOGONAL_MASK_COL + my_col]  # north, south, east, west

def neighbor_score(padded_grid, my_row, my_col, neighbor_spec):
    "Calculate a score based on a weighted count of neighbors with weights specified in neighbor_spec"

    # PERF note that I prototyped a NumPy-only version of this function (taking a properly reformatted neighbor_spec) for performance:
    #   neighbors = orthogonal_neighbors(padded_grid, my_row, my_col)[:, np.newaxis]
    #   keys = neighbor_spec[:, 0][np.newaxis, :]
    #   weights = neighbor_spec[:, 1]
    #   matching_count = (neighbors == keys).sum(axis=0)
    #   return np.dot(matching_count, weights)
    # ; tragically it was slightly slower
    
    score = 0
    neighbors = orthogonal_neighbors(padded_grid, my_row, my_col)
    for key, weight in neighbor_spec:
        score += weight * (neighbors == key.value).sum()
        # TODO the following ended up being slower so it's not urgent to figure this out,
        # but as far as I can tell these should be completely equivalent and yet they're
        # very subtly not -- why?
        # score += weight * np.count_nonzero(neighbors == key.value)
    return score

def calc_distance_to_coords(from_row, from_col, to_row, to_col):
    """
    Calculate a distance from some tile coordinates to some others. If either of the
    destination coords is None, calculate the distance to the line specified by the extant
    coord.
    """
    if to_row is None and to_col is None: raise ValueError()
    if to_row is None: to_row = from_row
    if to_col is None: to_col = from_col
    return math.hypot(to_col-from_col, to_row-from_row)

def calc_distance_to_location(tile_grid, from_row, from_col, location):
    [grid_size] = set(tile_grid.shape)
    grid_max = grid_size - 1  # maximum index
    grid_mid = grid_max / 2  # 'coordinate' of midpoint
    calc_distance_to = lambda to_row, to_col: calc_distance_to_coords(from_row, from_col, to_row, to_col)
    match location:
        case Location.BOARD_CENTER:
            result = calc_distance_to(grid_mid, grid_mid)
        case Location.BOARD_EDGE:
            result = min(
                calc_distance_to(0, None),
                calc_distance_to(grid_max, None),
                calc_distance_to(None, 0),
                calc_distance_to(None, grid_max),
            )
        case Location.BOARD_CORNER:
            result = min(
                calc_distance_to(0, 0),
                calc_distance_to(grid_max, 0),
                calc_distance_to(0, grid_max),
                calc_distance_to(grid_max, grid_max),
            )
        case Location.BOARD_VERTICAL_MEDIAN:
            result = calc_distance_to(None, grid_mid)
        case Location.BOARD_HORIZONTAL_MEDIAN:
            result = calc_distance_to(grid_mid, None)
        case _:
            raise ValueError()
    assert isinstance(result, numbers.Number), f"After calculating distance to {location}, result {result} is a {type(result)}"
    return result

def calc_distance_to_tile(tile_grid, from_row, from_col, to_object):
    assert to_object.value != -1
    this_object = tile_grid[from_row, from_col]
    tile_grid[from_row, from_col] = -1
    instances = np.argwhere(tile_grid == to_object.value)
    tile_grid[from_row, from_col] = this_object

    diffs = instances - (from_row, from_col)
    distances = np.hypot(diffs[:, 0], diffs[:, 1])
    result = np.min(distances) if len(distances) > 0 else max(tile_grid.shape)
    assert isinstance(result, numbers.Number)
    return result

def eval_tile_indiv_score(padded_grid, my_row, my_col):
    "Given a padded tile grid and the row and column in unpadded coordinates the of a particular tile, evaluate how well the grid satisfies that tile's objectives."
    unpadded_grid = padded_grid[1:-1, 1:-1]
    [grid_size] = set(unpadded_grid.shape)
    my_tile = Tile(unpadded_grid[my_row, my_col])
    match my_tile:
        case Tile.EMPTY:
            # Rules for EMPTY: no objectives, score is always zero
            return 0
        case Tile.RESIDENTIAL:
            # Rules for RESIDENTIAL: +1 for adjacent RESIDENTIAL, +2 for adjacent PARK, -3 for adjacent INDUSTRIAL
            return neighbor_score(padded_grid, my_row, my_col, [(Tile.RESIDENTIAL, +1), (Tile.PARK, +2), (Tile.INDUSTRIAL, -3)])
        case Tile.COMMERCIAL:
            # Rules for COMMERCIAL: +1 for adjacent RESIDENTIAL, +4 for adjacent DOWNTOWN
            return neighbor_score(padded_grid, my_row, my_col, [(Tile.RESIDENTIAL, +1), (Tile.DOWNTOWN, +4)])
        case Tile.INDUSTRIAL:
            # Rules for INDUSTRIAL: +1 for being within grid_size/6 of either the x-center line or the y-center line of the board (suppose there are railroads there)
            dist_from_horiz = calc_distance_to_location(unpadded_grid, my_row, my_col, Location.BOARD_HORIZONTAL_MEDIAN)
            dist_from_vert = calc_distance_to_location(unpadded_grid, my_row, my_col, Location.BOARD_VERTICAL_MEDIAN)
            return 1 * ((dist_from_horiz <= grid_size/6) or (dist_from_vert <= grid_size/6))
        case Tile.DOWNTOWN:
            # Rules for DOWNTOWN: +2 for being within (grid_size/6) Euclidean distance of the center of the grid, +4 for adjacent DOWNTOWN, -2 for adjacent INDUSTRIAL
            dist_from_center = calc_distance_to_location(unpadded_grid, my_row, my_col, Location.BOARD_CENTER)
            distance_criterion = dist_from_center <= grid_size/6
            return 2*distance_criterion + neighbor_score(padded_grid, my_row, my_col, [(Tile.DOWNTOWN, +4), (Tile.INDUSTRIAL, -2)])
        case Tile.PARK:
            # Rules for PARK: +3 for adjacent DOWNTOWN, -2 for adjacent INDUSTRIAL
            return neighbor_score(padded_grid, my_row, my_col, [(Tile.DOWNTOWN, +3), (Tile.INDUSTRIAL, -2)])
        case other:
            raise ValueError(f"Invalid tile: {other}")

def pad_grid(unpadded_grid):
    "Pad the grid with a one-width border of Tile.EMPTY"
    return np.pad(unpadded_grid, 1, mode="constant", constant_values=Tile.EMPTY.value)

def pad_grid_inplace(place, unpadded_grid):
    "Same as `pad_grid` but write the padded grid to `place`, which may be a pre-initialized array"
    # if place.dtype != unpadded_grid.dtype:
    #     raise ValueError(f"`place.dtype` must be same as `unpadded_grid.dtype`, got {place.dtype} and {unpadded_grid.dtype}")
    desired_shape = (unpadded_grid.shape[0]+2, unpadded_grid.shape[1]+2)
    if place.shape != desired_shape:
        place.resize(desired_shape, refcheck=False)  # is this unsafe?
        place[:, :] = Tile.EMPTY.value
    place[1:-1, 1:-1] = unpadded_grid

class ZoningGameEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self,
                 grid_size = 6, initially_filled_frac = 0.4, occurrences = DEFAULT_OCCURRENCES,
                 render_mode = "ansi", max_moves = 100, populate_info = True):
        """
        Arguments (all have sensible defaults):
          `grid_size = 6`: side length of the square grid
          `initially_filled_frac = 0.4`: fraction of cells that start the game filled
          `occurrences = DEFAULT_OCCURRENCES`: dict from Tile to how commonly that tile occurs; will be normalized
          `render_mode = "ansi"`: gymnasium render mode: `"ansi"` returns a string-like output, `None` does no rendering
        """
        self.grid_size = grid_size
        total_grid_cells = self.grid_size*self.grid_size
        self.initially_filled_frac = initially_filled_frac
        self.occurrences = {k: v/sum(occurrences.values()) for k, v in occurrences.items()}
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if max_moves < total_grid_cells:
            logger.warning(f"max_moves={max_moves} is less than total number of cells, {total_grid_cells}")
        self.max_moves = max_moves
        self.populate_info = populate_info

        self.tile_grid, self.tile_queue, self.n_moves = None, None, None
        grid_space = spaces.MultiDiscrete([[len(Tile)]*self.grid_size]*self.grid_size)  # Grid is 2d space where each element has len(Tile) options
        queue_space = spaces.MultiDiscrete([len(Tile)]*total_grid_cells)  # Queue is 1d space where each element has len(Tile) options
        self.observation_space = spaces.Tuple((grid_space, queue_space))  # Note this could also be represented as a single matrix of shape 2 x (grid_size*grid_size)
        self.action_space = spaces.Discrete(total_grid_cells)

    def _get_obs(self):
        return (self.tile_grid, self.tile_queue)
    
    def _get_terminated_truncated(self):
        terminated = (all(self.tile_queue == 0))
        truncated = self.n_moves >= self.max_moves
        return terminated, truncated
    
    def _get_average_scores(self):
        padded_grid = pad_grid(self.tile_grid)
        scores = {tile: [] for tile in Tile if tile is not Tile.EMPTY}
        for row, col in np.ndindex(self.tile_grid.shape):
            current_tile = Tile(self.tile_grid[row, col])
            if current_tile is not Tile.EMPTY:
                score = eval_tile_indiv_score(padded_grid, row, col)
                scores[current_tile].append(score)
        average_scores = {tile: (sum(scores[tile]) / len(scores[tile]) if scores[tile] else 0) for tile in scores}
        return {"average_scores": average_scores}
    
    def _get_info(self, average_scores = False):
        info = {}
        if self.populate_info and average_scores:
            info.update(self._get_average_scores())
        return info

    def reset(self, seed = None, options = None):
        assert options is None
        super().reset(seed=seed)
        self.n_moves = 0
        self.tile_grid, self.tile_queue = self._generate_problem()
        logger.debug(f"tile_grid:\n{self.tile_grid}")
        logger.debug(f"tile_queue:\n{self.tile_queue}")
        return self._get_obs(), self._get_info()

    def _generate_problem(self):
        "Create and return random `tile_grid` and `tile_queue` given instance config"
        tile_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        tile_options = list(self.occurrences.keys())  # Tile enum instances
        tile_values = [tile.value for tile in tile_options]  # ints
        tile_p = list(self.occurrences.values())

        n_filled = 0
        for row, col in np.ndindex(tile_grid.shape):
            if self.np_random.random() < self.initially_filled_frac:
                selected_tile = self.np_random.choice(tile_values, p=tile_p)
                logger.debug(f"Filling ({row}, {col}) with {Tile(selected_tile)}")
                tile_grid[row, col] = selected_tile
                n_filled += 1
        
        n_unfilled = self.grid_size*self.grid_size - n_filled
        filled_queue = self.np_random.choice(tile_values, p=tile_p, size=n_unfilled)
        tile_queue = np.concatenate((filled_queue, np.zeros(n_filled, dtype=filled_queue.dtype)))
        return tile_grid, tile_queue
    
    def render(self):
        "Render given `self.render_mode`. For `render_mode=ansi`, can print the results like `print(my_env.render().read())`."
        if self.render_mode is None: return
        assert self.render_mode == "ansi"
        buf = io.StringIO()
        terminated, truncated = self._get_terminated_truncated()
        print(f"Tile grid:\n{self.tile_grid}", file=buf)
        print(f"Tile queue (leftmost next): {self.tile_queue}", file=buf)
        print(f"where {', '.join([f'{x.value} = {x.name}' for x in Tile])}.", file=buf)
        print(f"After {self.n_moves} moves, current grid score is {self._eval_tile_grid_score()};", end=" ", file=buf)
        print(f"terminated = {terminated}, truncated = {truncated}.", file=buf)
        buf.seek(0)
        return buf
    
    def step(self, action, on_invalid=None):
        """
        Options for `on_invalid`: `None` does nothing, `"warn"` logs a warning, `"error"`
        raises an error
        """
        if isinstance(action, tuple):
            if len(action) != 1:
                raise ValueError(f"If action is a tuple, it must have length 1, got length {len(action)}")
            action, = action
        coords = (action // self.grid_size, action % self.grid_size)
        if Tile(self.tile_grid[*coords]) is not Tile.EMPTY:
            if on_invalid is not None:
                invalid_msg = f"Action {action} (coords {tuple(map(int, coords))}) is invalid, skipping"
                if on_invalid == "warn": logger.warning(invalid_msg)
                elif on_invalid == "error": raise ValueError(invalid_msg)
        else:
            self.tile_grid[*coords] = self.tile_queue[0]
            self.tile_queue[:-1] = self.tile_queue[1:]
            self.tile_queue[-1] = 0
        self.n_moves += 1

        terminated, truncated = self._get_terminated_truncated()
        reward = self._eval_tile_grid_score() if terminated or truncated else 0  # only reward at the end
        if terminated:
            logger.info(f"Finished with reward {reward}")
        return self._get_obs(), reward, terminated, truncated, self._get_info(average_scores = terminated or truncated)
    
    def _eval_tile_grid_score(self):
        "Use `eval_tile_indiv_score` to compute the sum score across the whole tile grid"
        padded_grid = pad_grid(self.tile_grid)
        total_score = 0
        for row, col in np.ndindex(self.tile_grid.shape):
            score_incr = eval_tile_indiv_score(padded_grid, row, col)
            total_score += score_incr
            current_tile = Tile(self.tile_grid[row, col])
            if current_tile is not Tile.EMPTY:
                logger.debug(f"Adding {score_incr} to score for tile {current_tile.name} at {(row, col)}")
        return total_score

def flatten_zg_obs(sub_obs):
    return np.concat((sub_obs[0].flatten(), sub_obs[1]))

class ZoningGameObservationWrapper(gym.ObservationWrapper):
    def __init__(self, sub_env):
        super().__init__(sub_env)
        total_grid_cells = sub_env.unwrapped.grid_size*sub_env.unwrapped.grid_size
        self.observation_space = spaces.MultiDiscrete([len(Tile)]*total_grid_cells*2)
    
    def observation(self, sub_obs):
        return flatten_zg_obs(sub_obs)

gym.register(
    id="zg/ZoningGameEnv-v0",
    entry_point=ZoningGameEnv
)
