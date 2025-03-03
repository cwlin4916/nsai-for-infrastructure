import numpy as np
from nltk.tree import Tree

from nsai_experiments.zoning_game.zg_gym import ZoningGameEnv, Tile, eval_tile_indiv_score, pad_grid, blank_of_size
from nsai_experiments.zoning_game.zg_policy import play_one_game, create_policy_random, create_policy_indiv_greedy, create_policy_total_greedy
from nsai_experiments.zoning_game.zg_cfg import ZONING_GAME_GRAMMAR, RuleNT, generate_one_probabilistic, format_ruleset, parse_to_ast, parse_to_nt, interpret_grid

FAST_TEST = False  # Whether to skip some extra test cases to speed up the tests

# Some aliases to make it easier to build tile grids from scratch
t0 = Tile.EMPTY.value
tr = Tile.RESIDENTIAL.value
tc = Tile.COMMERCIAL.value
ti = Tile.INDUSTRIAL.value
td = Tile.DOWNTOWN.value
tp = Tile.PARK.value

T_ = True
F_ = False

def _compute_location_based_score(grid_size, fill_tile):
    myenv = ZoningGameEnv(grid_size=grid_size)
    result = np.zeros((grid_size, grid_size), np.int32)
    for row, col in np.ndindex(result.shape):
        myenv.reset(seed=0)
        myenv.tile_grid = blank_of_size(grid_size)
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
    base_grid = blank_of_size(myenv.grid_size)
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
    base_grid = blank_of_size(myenv.grid_size)
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
    base_grid = blank_of_size(myenv.grid_size)
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
    base_grid = blank_of_size(myenv.grid_size)
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

    # The world is half RESIDENTIAL, half INDUSTRIAL; put the remaining RESIDENTIAL and INDUSTRIAL tiles with the others
    _test_head_to_head(np.array([
        [tr, tr, tr, ti, ti, ti],
        [tr, tr, tr, ti, ti, ti],
        [tr, tr, tr, ti, t0, ti],
        [tr, tr, tr, ti, ti, ti],
        [tr, t0, tr, ti, ti, ti],
        [tr, tr, tr, ti, ti, ti],
    ]), [tr, ti], 25)

    # The world is all PARK; put the DOWNTOWN in the middle and the RESIDENTIAL elsewhere
    _test_head_to_head(np.array([
        [tp, tp, tp, tp, tp, tp],
        [tp, tp, tp, tp, tp, tp],
        [tp, tp, t0, tp, tp, tp],
        [tp, tp, tp, tp, tp, tp],
        [tp, tp, tp, tp, t0, tp],
        [tp, tp, tp, tp, tp, tp],
    ]), [td, tr], 14)

    # The world is all RESIDENTIAL; put the INDUSTRIAL near the center line and the COMMERCIAL elsewhere
    _test_head_to_head(np.array([
        [tr, tr, tr, tr, tr, tr],
        [tr, t0, tr, t0, tr, tr],
        [tr, tr, tr, tr, tr, tr],
        [tr, tr, tr, tr, tr, tr],
        [tr, tr, tr, tr, tr, tr],
        [tr, tr, tr, tr, tr, tr],
    ]), [ti, tc], 9)

def test_policy_basics():
    policy_creators_to_test = [create_policy_random, create_policy_indiv_greedy, create_policy_total_greedy]
    env = ZoningGameEnv()
    for create_policy in policy_creators_to_test:
        for policy_seed in range(0, 5 if FAST_TEST else 10):
            my_policy = create_policy(seed=policy_seed)
            for env_seed in range(10, 15 if FAST_TEST else 20):
                final_env, obs, reward, terminated, truncated, info = \
                    play_one_game(my_policy, env=env, seed=env_seed, on_invalid="error")
                assert final_env == env
                assert terminated
                assert not truncated
                tile_grid, tile_queue = obs
                assert np.count_nonzero(tile_grid == Tile.EMPTY.value) == 0
                assert np.count_nonzero(tile_queue) == 0

def test_greedy_policies():
    env = ZoningGameEnv()
    env.reset(seed=0)
    # In this grid, the individually greedy best choice for the park is in between the two
    # downtowns (+6 individual) but the "total greedy" best choice is between the four
    # residentials (+8 group)
    test_grid = np.array([
        [t0, tr, t0, t0, t0, t0],
        [tr, t0, tr, t0, t0, t0],
        [t0, tr, td, t0, td, t0],
        [t0, t0, t0, t0, t0, t0],
        [t0, t0, t0, t0, t0, t0],
        [t0, t0, t0, t0, t0, t0],
    ])
    test_queue = np.zeros(env.grid_size*env.grid_size, dtype=env.tile_queue.dtype)
    test_queue[:np.sum(test_grid == Tile.EMPTY.value)] = tp

    for policy_seed in range(0, 10):
        indiv_greedy_policy = create_policy_indiv_greedy(seed=policy_seed)
        env.reset(seed=0)
        env.tile_grid = test_grid.copy()
        env.tile_queue = test_queue.copy()
        obs = env.step(35)[0]
        assert indiv_greedy_policy(obs) == 15
    
    for policy_seed in range(0, 10):
        total_greedy_policy = create_policy_total_greedy(seed=policy_seed)
        env.reset(seed=0)
        env.tile_grid = test_grid.copy()
        env.tile_queue = test_queue.copy()
        obs = env.step(35)[0]
        assert total_greedy_policy(obs) == 7

def test_policy_intercomparison():
    "Test that policies we expect to perform better than random actually do, etc."
    policy_creators_to_test = [create_policy_random, create_policy_indiv_greedy, create_policy_total_greedy]
    results = {}  # dict from policy creator name to sum of rewards when playing with that policy
    env = ZoningGameEnv()
    for create_policy in policy_creators_to_test:
        results[create_policy.__name__] = 0
        for policy_seed in range(0, 5 if FAST_TEST else 10):
            my_policy = create_policy(seed=policy_seed)
            for env_seed in range(10, 15 if FAST_TEST else 20):
                _, _, reward, _, _, _ = play_one_game(my_policy, env=env, seed=env_seed, on_invalid="error")
                results[create_policy.__name__] += reward
    
    assert results["create_policy_indiv_greedy"] > results["create_policy_random"]
    assert results["create_policy_total_greedy"] > results["create_policy_random"]
    assert results["create_policy_total_greedy"] > results["create_policy_indiv_greedy"]

def test_cfg_generation_and_parsing():
    for generation_seed in range(30 if FAST_TEST else 100):
        generated_tokens = generate_one_probabilistic(ZONING_GAME_GRAMMAR, seed=generation_seed)
        generated_ruleset = format_ruleset(generated_tokens)

        ast = parse_to_ast(generated_ruleset)
        assert isinstance(ast, Tree)

        nt1 = parse_to_nt(generated_ruleset)
        nt2 = parse_to_nt(ast)
        assert nt1 == nt2
        assert all([isinstance(r, RuleNT) for r in nt2])

def test_cfg_interpreter():
    test_grid_1 = np.array([
        [tr, t0, t0, tr, tr, tr],
        [t0, t0, t0, t0, t0, t0],
        [t0, t0, t0, t0, t0, t0],
        [t0, t0, t0, t0, t0, t0],
        [t0, t0, t0, tr, t0, t0],
        [t0, t0, t0, t0, t0, t0],
    ])
    assert (interpret_grid(
        "RESIDENTIAL must be_within 3 tiles_of RESIDENTIAL ;",
        test_grid_1
    ) == np.array([
        [T_, T_, T_, T_, T_, T_],
        [T_, T_, T_, T_, T_, T_],
        [T_, T_, T_, T_, T_, T_],
        [T_, T_, T_, T_, T_, T_],
        [T_, T_, T_, F_, T_, T_],
        [T_, T_, T_, T_, T_, T_],
    ])).all()
    assert (interpret_grid(
        "RESIDENTIAL must form_fewer_than 2 separate_clusters ;",
        test_grid_1
    ) == np.array([
        [F_, T_, T_, F_, F_, F_],
        [T_, T_, T_, T_, T_, T_],
        [T_, T_, T_, T_, T_, T_],
        [T_, T_, T_, T_, T_, T_],
        [T_, T_, T_, F_, T_, T_],
        [T_, T_, T_, T_, T_, T_],
    ])).all()
    assert (interpret_grid(
        "RESIDENTIAL must form_cluster_with_fewer_than 2 tiles ;",
        test_grid_1
    ) == np.array([
        [T_, T_, T_, F_, F_, F_],
        [T_, T_, T_, T_, T_, T_],
        [T_, T_, T_, T_, T_, T_],
        [T_, T_, T_, T_, T_, T_],
        [T_, T_, T_, T_, T_, T_],
        [T_, T_, T_, T_, T_, T_],
    ])).all()
    assert (interpret_grid(
        "RESIDENTIAL must ( not be_within 3 tiles_of RESIDENTIAL ) ;",
        test_grid_1
    ) == np.array([
        [F_, T_, T_, F_, F_, F_],
        [T_, T_, T_, T_, T_, T_],
        [T_, T_, T_, T_, T_, T_],
        [T_, T_, T_, T_, T_, T_],
        [T_, T_, T_, T_, T_, T_],
        [T_, T_, T_, T_, T_, T_],
    ])).all()

    test_grid_2 = np.array([
        [ti, ti, ti, td, td, td],
        [ti, ti, ti, td, td, td],
        [ti, ti, ti, td, td, td],
        [td, td, td, ti, ti, ti],
        [td, td, td, ti, ti, ti],
        [td, td, td, ti, ti, ti],
    ])
    assert (interpret_grid(
        "DOWNTOWN must be_within 1 tiles_of BOARD_CENTER ;",
        test_grid_2
    ) == np.array([
        [T_, T_, T_, F_, F_, F_],
        [T_, T_, T_, F_, F_, F_],
        [T_, T_, T_, T_, F_, F_],
        [F_, F_, T_, T_, T_, T_],
        [F_, F_, F_, T_, T_, T_],
        [F_, F_, F_, T_, T_, T_],
    ])).all()
    assert (interpret_grid(
        "DOWNTOWN must ( be_within 1 tiles_of BOARD_VERTICAL_MEDIAN and be_within 1 tiles_of BOARD_HORIZONTAL_MEDIAN ) ;",
        test_grid_2
    ) == np.array([
        [T_, T_, T_, F_, F_, F_],
        [T_, T_, T_, F_, F_, F_],
        [T_, T_, T_, T_, F_, F_],
        [F_, F_, T_, T_, T_, T_],
        [F_, F_, F_, T_, T_, T_],
        [F_, F_, F_, T_, T_, T_],
    ])).all()
    assert (interpret_grid(
        """
        DOWNTOWN must be_within 1 tiles_of BOARD_VERTICAL_MEDIAN ;
        DOWNTOWN must be_within 1 tiles_of BOARD_HORIZONTAL_MEDIAN ;
        """,
        test_grid_2
    ) == np.array([
        [T_, T_, T_, F_, F_, F_],
        [T_, T_, T_, F_, F_, F_],
        [T_, T_, T_, T_, F_, F_],
        [F_, F_, T_, T_, T_, T_],
        [F_, F_, F_, T_, T_, T_],
        [F_, F_, F_, T_, T_, T_],
    ])).all()
    assert (interpret_grid(
        "INDUSTRIAL must ( be_within 1 tiles_of BOARD_VERTICAL_MEDIAN or be_within 1 tiles_of BOARD_HORIZONTAL_MEDIAN ) ;",
        test_grid_2
    ) == np.array([
        [F_, F_, T_, T_, T_, T_],
        [F_, F_, T_, T_, T_, T_],
        [T_, T_, T_, T_, T_, T_],
        [T_, T_, T_, T_, T_, T_],
        [T_, T_, T_, T_, F_, F_],
        [T_, T_, T_, T_, F_, F_],
    ])).all()
