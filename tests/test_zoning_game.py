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
