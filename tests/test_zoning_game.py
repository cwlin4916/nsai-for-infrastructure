import numpy as np

from nsai_experiments.zoning_game.zg_gym import ZoningGameEnv, Tile, eval_tile_indiv_score, pad_grid

def _blank_of_size(grid_size, fill_tile = Tile.EMPTY):
    return np.ones((grid_size, grid_size), dtype=np.int32) * fill_tile.value

def _compute_location_based_score(grid_size, fill_tile):
    myenv = ZoningGameEnv(grid_size=grid_size)
    result = np.zeros((grid_size, grid_size), np.int32)
    for row, col in np.ndindex(result.shape):
        myenv.reset(seed=0)
        myenv.tile_grid = _blank_of_size(grid_size)
        myenv.tile_grid[row, col] = fill_tile.value
        padded_grid = pad_grid(myenv.tile_grid)
        result[row, col] = eval_tile_indiv_score(padded_grid, row, col)
    return result

def test_indiv_score():
    "Aims to thoroughly test `eval_tile_indiv_score`."

    myenv = ZoningGameEnv(grid_size=6)

    ## Tile.EMPTY
    # Generate a bunch of random grids and verify that all the empty tiles score 0
    i = 0
    for seed in range(10):
        myenv.reset(seed=seed)
        padded_grid = pad_grid(myenv.tile_grid)
        for row, col in np.ndindex(myenv.tile_grid.shape):
            if Tile(myenv.tile_grid[row, col]) is Tile.EMPTY:
                assert eval_tile_indiv_score(padded_grid, row, col) == 0
                i += 1
    assert i >= 100  # Make sure that was actually a substantial number of test cases

    ## Tile.RESIDENTIAL
    # Test with various neighbor configurations
    base_grid = _blank_of_size(myenv.grid_size)
    base_grid[1, 1] = Tile.RESIDENTIAL.value

    # No neighbors -> score is 0
    myenv.reset(seed=0)
    myenv.tile_grid = base_grid.copy()
    assert eval_tile_indiv_score(pad_grid(myenv.tile_grid), 1, 1) == 0

    # RESIDENTIAL neighbor -> +1
    myenv.reset(seed=0)
    myenv.tile_grid = base_grid.copy()
    myenv.tile_grid[1, 0] = Tile.RESIDENTIAL.value
    assert eval_tile_indiv_score(pad_grid(myenv.tile_grid), 1, 1) == 1

    # PARK neighbor -> +2
    myenv.reset(seed=0)
    myenv.tile_grid = base_grid.copy()
    myenv.tile_grid[1, 2] = Tile.PARK.value
    assert eval_tile_indiv_score(pad_grid(myenv.tile_grid), 1, 1) == 2

    # INDUSTRIAL neighbor -> -3
    myenv.reset(seed=0)
    myenv.tile_grid = base_grid.copy()
    myenv.tile_grid[0, 1] = Tile.INDUSTRIAL.value
    assert eval_tile_indiv_score(pad_grid(myenv.tile_grid), 1, 1) == -3

    # Diagonal doesn't count as neighboring
    myenv.reset(seed=0)
    myenv.tile_grid = base_grid.copy()
    myenv.tile_grid[0, 0] = Tile.RESIDENTIAL.value
    assert eval_tile_indiv_score(pad_grid(myenv.tile_grid), 1, 1) == 0

    # Effects are additive
    myenv.reset(seed=0)
    myenv.tile_grid = base_grid.copy()
    myenv.tile_grid[1, 0] = Tile.RESIDENTIAL.value
    myenv.tile_grid[1, 2] = Tile.PARK.value
    myenv.tile_grid[0, 1] = Tile.INDUSTRIAL.value
    myenv.tile_grid[2, 1] = Tile.RESIDENTIAL.value
    assert eval_tile_indiv_score(pad_grid(myenv.tile_grid), 1, 1) == 1

    ## Tile.COMMERCIAL
    # Test with various neighbor configurations
    base_grid = _blank_of_size(myenv.grid_size)
    base_grid[1, 1] = Tile.COMMERCIAL.value

    # No neighbors -> score is 0
    myenv.reset(seed=0)
    myenv.tile_grid = base_grid.copy()
    assert eval_tile_indiv_score(pad_grid(myenv.tile_grid), 1, 1) == 0

    # RESIDENTIAL neighbor -> +1
    myenv.reset(seed=0)
    myenv.tile_grid = base_grid.copy()
    myenv.tile_grid[1, 0] = Tile.RESIDENTIAL.value
    assert eval_tile_indiv_score(pad_grid(myenv.tile_grid), 1, 1) == 1

    # DOWNTOWN neighbor -> +4
    myenv.reset(seed=0)
    myenv.tile_grid = base_grid.copy()
    myenv.tile_grid[1, 0] = Tile.DOWNTOWN.value
    assert eval_tile_indiv_score(pad_grid(myenv.tile_grid), 1, 1) == 4

    ## Tile.INDUSTRIAL
    # Map of the good locations for grid_size=6:
    industrial_reward_map_6 = np.array([
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
    ])
    # Map of the good locations for grid_size=5:
    industrial_reward_map_5 = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ])
    # Map of the good locations for grid_size=7:
    industrial_reward_map_7 = np.array([
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
    ])
    for grid_size, answer in [(6, industrial_reward_map_6), (5, industrial_reward_map_5), (7, industrial_reward_map_7)]:
        result = _compute_location_based_score(grid_size, Tile.INDUSTRIAL)
        assert (result == answer).all()
    
    ## Tile.DOWNTOWN
    # First, test the distance scoring like Tile.INDUSTRIAL:
    downtown_reward_map_6 = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 0, 0],
        [0, 0, 2, 2, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ])
    # Map of the good locations for grid_size=5:
    downtown_reward_map_5 = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    # Map of the good locations for grid_size=7:
    downtown_reward_map_7 = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0],
        [0, 0, 2, 2, 2, 0, 0],
        [0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])
    for grid_size, answer in [(6, downtown_reward_map_6), (5, downtown_reward_map_5), (7, downtown_reward_map_7)]:
        result = _compute_location_based_score(grid_size, Tile.DOWNTOWN)
        assert (result == answer).all()
    
    # Next, test neighbors
    base_grid = _blank_of_size(myenv.grid_size)
    base_grid[1, 1] = Tile.DOWNTOWN.value

    # No neighbors -> score is 0
    myenv.reset(seed=0)
    myenv.tile_grid = base_grid.copy()
    assert eval_tile_indiv_score(pad_grid(myenv.tile_grid), 1, 1) == 0

    # DOWNTOWN neighbor -> +4
    myenv.reset(seed=0)
    myenv.tile_grid = base_grid.copy()
    myenv.tile_grid[1, 0] = Tile.DOWNTOWN.value
    assert eval_tile_indiv_score(pad_grid(myenv.tile_grid), 1, 1) == 4

    # INDUSTRIAL neighbor -> -2
    myenv.reset(seed=0)
    myenv.tile_grid = base_grid.copy()
    myenv.tile_grid[1, 0] = Tile.INDUSTRIAL.value
    assert eval_tile_indiv_score(pad_grid(myenv.tile_grid), 1, 1) == -2
    
    ## Tile.PARK
    # Test with various neighbor configurations
    base_grid = _blank_of_size(myenv.grid_size)
    base_grid[1, 1] = Tile.PARK.value

    # No neighbors -> score is 0
    myenv.reset(seed=0)
    myenv.tile_grid = base_grid.copy()
    assert eval_tile_indiv_score(pad_grid(myenv.tile_grid), 1, 1) == 0

    # DOWNTOWN neighbor -> +3
    myenv.reset(seed=0)
    myenv.tile_grid = base_grid.copy()
    myenv.tile_grid[1, 0] = Tile.DOWNTOWN.value
    assert eval_tile_indiv_score(pad_grid(myenv.tile_grid), 1, 1) == 3

    # INDUSTRIAL neighbor -> -2
    myenv.reset(seed=0)
    myenv.tile_grid = base_grid.copy()
    myenv.tile_grid[1, 0] = Tile.INDUSTRIAL.value
    assert eval_tile_indiv_score(pad_grid(myenv.tile_grid), 1, 1) == -2

def _test_head_to_head(initial_grid, initial_queue, correct_move, test_reversed = True):
    """
    Given `initial_grid`, a tile grid with two spots left to fill, and `initial_queue`, the
    two last tiles in the queue, verifies that playing `correct_move` and then the only
    possible other move results in a greater final score than playing the moves the other
    way around. If `test_reversed = True`, complete the entire test again with the tile
    queue and correct move reversed.
    """

    assert np.count_nonzero(initial_grid == Tile.EMPTY.value) == 2
    assert len(initial_queue) == 2

    def __initialize_env():
        myenv.reset(seed=0)
        myenv.n_moves = np.count_nonzero(myenv.grid_size*myenv.grid_size - 2 - myenv.tile_grid)
        myenv.tile_grid = initial_grid.copy()
        myenv.tile_queue = np.zeros_like(myenv.tile_queue)
        myenv.tile_queue[:2] = initial_queue

    myenv = ZoningGameEnv(grid_size=6)
    __initialize_env()
    myenv.step(correct_move, on_invalid = "error")
    [other_move] = np.flatnonzero(myenv.tile_grid == Tile.EMPTY.value)
    _, score1, terminated, truncated, _ = myenv.step(other_move, on_invalid = "error")
    assert terminated
    assert not truncated

    __initialize_env()
    myenv.step(other_move, on_invalid = "error")
    myenv.step(correct_move, on_invalid = "error")
    _, score2, terminated, truncated, _ = myenv.step(other_move)
    assert terminated
    assert not truncated

    assert score1 > score2

    if test_reversed:
        _test_head_to_head(initial_grid, initial_queue[::-1], other_move, test_reversed = False)

def test_final_scoring():
    """
    Performs a series of manually constructed head-to-head comparisons between two ways of
    making the last two moves in a zoning game instance and verifies that the correct one
    wins.
    """

    t0 = Tile.EMPTY.value
    tr = Tile.RESIDENTIAL.value
    ti = Tile.INDUSTRIAL.value

    # The world is half RESIDENTIAL, half INDUSTRIAL; put the remaining RESIDENTIAL and INDUSTRIAL tiles with the others
    _test_head_to_head(np.array([
        [tr, tr, tr, ti, ti, ti],
        [tr, tr, tr, ti, ti, ti],
        [tr, tr, tr, ti, t0, ti],
        [tr, tr, tr, ti, ti, ti],
        [tr, t0, tr, ti, ti, ti],
        [tr, tr, tr, ti, ti, ti],
    ]), [tr, ti], 25)
