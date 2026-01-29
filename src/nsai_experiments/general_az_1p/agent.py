from typing import Any, Callable
import warnings
import copy
import time
from multiprocessing import Pool
import logging
import itertools
import os
from copy import deepcopy
from pathlib import Path
import pickle

import numpy as np

from .game import Game
from .policy_value_net import PolicyValueNet
from .mcts import MCTS, entab
from .setup_utils import THREAD_VARS

# Debugging flags:
DETAILED_DEBUG = False
PIT_NO_MCTS = True
PRINT_ALL_EPOCHS = True

class Agent():
    # Constants
    RNG_NAMES = ["mcts", "train", "eval", "external_policy"]  # Names for the random number generators to construct
    POLICY_RESERVED_NAMES = set(["old_net", "new_net", "old_net_no_mcts", "new_net_no_mcts"])
    save_file_name = "agent_checkpoint.pkl"

    # State
    game: Game
    net: PolicyValueNet
    all_training_examples: list[list[tuple[Any, tuple[Any, float]]]]  # Accumulated training examples (state, (policy, reward))
    rngs: dict[str, np.random.Generator]

    # Config
    n_games_per_train: int  # Number of games to play for training examples per training call
    n_games_per_eval: int  # Number of games to play in the pitting step per training call
    n_past_iterations_to_train: int | None  # Number of past iterations to use for training, where an iteration contains `n_games_per_train` games; if None, use all past iterations
    threshold_to_keep: float  # Threshold for win rate to keep the new network
    reward_discount: float  # For the case where only reward is at the end, set to 1.0 to use end reward at all steps
    mcts_params: dict  # Passed through to MCTS constructor
    n_procs: int | None  # Number of processes to use in the `multiprocessing.Pool()`; if None, use all available cores, if < 0 do not use multiprocessing
    random_seeds: dict[str, int]  # Seeds for the RNGs
    external_policy: Callable | None  # If not None, a policy function to use rather than NN+MCTS for move selection
    external_policy_creators_to_pit: dict[str, Callable]  # A dictionary name -> creator of policy creators for pitting. Currently policies are recreated in each pitting session using a new random seed each time

    def __init__(self, game: Game, net: PolicyValueNet,
                 n_games_per_train: int = 100,
                 n_games_per_eval: int = 20,
                 n_past_iterations_to_train: int = 20,
                 threshold_to_keep: float = 0.55,
                 use_gating: bool = True,
                 reward_discount: float = 1.0,
                 mcts_params: dict | None = None,
                 n_procs: int | None = None,
                 random_seeds: dict[str, int] | None = None,
                 external_policy = None,

                 external_policy_creators_to_pit = {}):
        self.game = game
        self.n_games_per_train = n_games_per_train
        self.n_games_per_eval = n_games_per_eval
        self.n_past_iterations_to_train = n_past_iterations_to_train
        self.threshold_to_keep = threshold_to_keep
        self.use_gating = use_gating
        self.net = net
        self.reward_discount = reward_discount
        self.mcts_params = mcts_params if mcts_params is not None else {}
        self.n_procs = n_procs
        self.external_policy = external_policy
        self.run_start_time = int(time.time())
        print(f"Agent config: {n_games_per_train=}, {n_games_per_eval=}, {n_past_iterations_to_train=}, {threshold_to_keep=}, {use_gating=}, {reward_discount=}, {mcts_params=}, {n_procs=}, {external_policy=}, {external_policy_creators_to_pit=}")

        if self.n_procs is None or self.n_procs >= 0:
            if not all([os.environ.get(thread_var, None) == "1" for thread_var in THREAD_VARS]):
                warnings.warn(f"You have elected to use multiprocessing, but NumPy multithreading is not disabled. This may lead to thread oversubscription. You can disable NumPy multithreading by setting the environment variables {','.join(THREAD_VARS)} to 1 before importing NumPy, or disable multiprocessing by passing n_procs=-1 to the Agent constructor.")
        self._construct_rngs(random_seeds if random_seeds is not None else {})
        self.all_training_examples = []

        if self.external_policy is not None:
            print("Will use external policy rather than NN+MCTS for move selection")
        self._setup_external_policy_creators_to_pit(external_policy_creators_to_pit)

        # Metrics history
        self.history = {
            'iteration': [],
            'reward_mean': [],
            'reward_std': [],
            'loss_policy': [],
            'loss_value': [],
            'game_length': []
        }

    def _construct_rngs(self, random_seeds: dict[str, int]):
        self.rngs = {}
        for rng_name in self.RNG_NAMES:
            seed = random_seeds.get(rng_name, None)
            self.rngs[rng_name] = np.random.default_rng(seed)
        if all(rng_name in random_seeds or (rng_name == "external_policy" and self.external_policy is None) for rng_name in self.RNG_NAMES):
            print(f"RNG seeds are fully specified")
        else:
            print(f"RNG seeds are not fully specified, using nondeterministic seeds for: {', '.join(rng_name for rng_name in self.RNG_NAMES if rng_name not in random_seeds)}")

    def _setup_external_policy_creators_to_pit(self, external_policy_creators_to_pit):
        unique_policy_names = set(external_policy_creators_to_pit.keys())
        if len(external_policy_creators_to_pit) != len(unique_policy_names):
            raise ValueError("External policies must have unique names")
        conflicting_names = unique_policy_names & self.POLICY_RESERVED_NAMES
        if conflicting_names:
            raise ValueError(f"External policy names conflict with reserved agent keys: {conflicting_names}")
        self.external_policy_creators_to_pit = external_policy_creators_to_pit

        # It would be nice to call the creators here, but then we get unpickleable
        # functions, problematic for multiprocessing. Probably the best long-term solution
        # is to have object-oriented policies rather than relying on closures. For now we'll
        # just recreate them each pitting cycle in each pitting process.

        # self.external_policies_to_pit = {
        #     name: copy.deepcopy(policy_creator(seed = self._randseed("external_policy")))
        #     for name, policy_creator in external_policy_creators_to_pit.items()
        # }

    def play_single_game(self, max_moves: int = 10_000, random_seed: int | None = None, msg = "", add_noise=False):
        train_examples = []
        rewards = []
        self.cumulative_reward = 0.0
        mcts = MCTS(self.game, self.net, **self.mcts_params)
        rng = np.random.default_rng(random_seed)
        for i in range(max_moves):
            if msg: print(msg, f"starting move {i}")
            if self.external_policy is None:
                move_probs = mcts.perform_simulations(entab(msg, f", m{i+1}"), add_noise=add_noise)
                self.game = mcts.game  # TODO HACK because MCTS modifies the game state in place
                train_examples.append((deepcopy(self.game.obs), (move_probs, None)))  # PERF deepcopy often unnecessary
                
                # Sample from probabilities (flatten for choice, then convert to tuple)
                flat_probs = move_probs.flatten()
                flat_idx = rng.choice(len(flat_probs), p=flat_probs)
                selected_move = np.unravel_index(flat_idx, move_probs.shape)
                # Distinguish scalar actions from 1-tuple actions
                if len(self.game.action_space.shape) == 0: selected_move, = selected_move
                assert selected_move in self.game.action_space
                
                if msg: print(msg, "obs", self.game.obs, "hobs", self.game.hashable_obs, "move_probs", move_probs, "selmove", selected_move)
                # print(f"Taking move {selected_move} with probability {move_probs[selected_move]:.2f}")  # TODO logging
            else:
                selected_move = self.external_policy(self.game.obs)
                if msg: print(msg, "external policy selmove", selected_move)
            self.game.step_wrapper(selected_move)
            # Temporary, for ZLAZ debugging:
            # print(self.game.info["last_prod"])
            # print(self.game.env.unwrapped.stringify_program())
            rewards.append(self.game.reward)
            self.cumulative_reward += self.game.reward
            if self.game.terminated or self.game.truncated:
                break
        else:
            # In this case, we might not have any reward to work with
            warnings.warn(f"`play_single_game` timed out after {max_moves} moves without termination/truncation, returning no training examples")
            return []
        
        # Propagate rewards backwards through steps
        for i in range(len(rewards) - 1, 0, -1):
            rewards[i-1] += self.reward_discount * rewards[i]
        
        # Attach rewards to training examples
        for i in range(len(train_examples)):
            state, (policy, _) = train_examples[i]
            train_examples[i] = (state, (policy, rewards[i]))
        
        return train_examples
    
    def _play_for_examples(self, i, reset_seed, mcts_seed):
        # print(i)
        logging.getLogger().setLevel(logging.WARN)
        self.game.reset_wrapper(seed=reset_seed)
        logging.getLogger().setLevel(logging.WARN)
        self.game.reset_wrapper(seed=reset_seed)
        return self.play_single_game(random_seed=mcts_seed, msg = f"game {i}" if DETAILED_DEBUG else "", add_noise=True)

    def _play_for_eval(self, i, reset_seed, mcts_seed, external_policy_seed, self_before_training, try_without_mcts = False, pit_external_policy_creators_to_pit = False):
        if DETAILED_DEBUG: print(i)
        logging.getLogger().setLevel(logging.WARN)
        self.game.reset_wrapper(seed=reset_seed)
        self_before_training.game = copy.deepcopy(self.game)
        if DETAILED_DEBUG: print("obs", self.game.obs, self_before_training.game.obs)

        all_agents = {
            "old_net": self_before_training,
            "new_net": self
        }
        if try_without_mcts:
            self_before_training_no_mcts = copy.deepcopy(self_before_training)
            self_before_training_no_mcts.mcts_params = {"n_simulations": -1}
            self_no_mcts = copy.deepcopy(self)
            self_no_mcts.mcts_params = {"n_simulations": -1}
            all_agents |= {
                "old_net_no_mcts": self_before_training_no_mcts,
                "new_net_no_mcts": self_no_mcts,
            }

        if pit_external_policy_creators_to_pit:
            for (policy_name, policy_creator) in self.external_policy_creators_to_pit.items():
                policy_self = copy.deepcopy(self)
                policy = policy_creator(seed = external_policy_seed)
                policy_self.external_policy = policy
                all_agents[policy_name] = policy_self

        # NOTE here we are using the final reward to compare performance -- we may in
        # fact want to use a (weighted?) sum of stepwise rewards

        results = {}
        for label, agent in all_agents.items():
            agent.play_single_game(random_seed=mcts_seed, msg = f"{label} game {i}" if DETAILED_DEBUG else "", add_noise=False)
            results[label] = agent.cumulative_reward

        if DETAILED_DEBUG: print(f"Reward from old: {results['old_net']:.2f}, Reward from new: {results['new_net']:.2f}")
        if DETAILED_DEBUG and try_without_mcts:
            print(f"Reward from old no MCTS: {results['old_net_no_mcts']:.2f}, Reward from new no MCTS: {results['new_net_no_mcts']:.2f}")
        return results

    def _starmap(self, fn, arg_tuples):
        """
        Call `fn(*args)` for each `args in arg_tuples`. If `self.n_procs` is None or >= 0,
        construct a `multiprocessing.Pool` and use `Pool.starmap`; otherwise use
        `itertools.starmap`.
        """
        if self.n_procs is None or self.n_procs >= 0:
            with Pool(processes=self.n_procs) as pool:
                results = pool.starmap(fn, arg_tuples)
        else:
            results = list(itertools.starmap(fn, arg_tuples))
        return results
    
    def _randseed(self, rng_name: str):
        "Extract a random integer from RNG `rng_name` suitable for seeding another RNG"
        return int(self.rngs[rng_name].integers(2**31-1))

    def play_and_train(self):
        # Play a bunch of games and keep track of the training examples
        new_train_examples = []  # PERF consider using a deque for efficiency

        start_time = time.time()
        multiprocessing_stash = self.push_multiprocessing()
        arg_tuples = [(i, self._randseed("train"), self._randseed("mcts")) for i in range(self.n_games_per_train)]
        train_example_sets = self._starmap(self._play_for_examples, arg_tuples)
        self.pop_multiprocessing(multiprocessing_stash)
        for train_examples in train_example_sets:
            new_train_examples.extend(train_examples)
        elapsed = time.time() - start_time
        print(f"..games done in {elapsed:.2f} seconds")
        
        self.all_training_examples.append(new_train_examples)
        if self.n_past_iterations_to_train is not None and len(self.all_training_examples) > self.n_past_iterations_to_train:
            self.all_training_examples.pop(0)
        print(f"Training examples lengths: {[len(x) for x in self.all_training_examples]}")
        flat_examples = list(itertools.chain.from_iterable(self.all_training_examples))
        print(f"Total value: {sum(x[1][1] for x in flat_examples):.2f}")

        # Save an old Agent to pit ourselves against, then train the network
        self.game.reset_wrapper()
        self_before_training = copy.deepcopy(self)

        # Sanity check: game states and network predictions on current state should be the
        # same before we train, assuming the network is deterministic. PyTorch inherent
        # nondeterminism seems to be large enough that we sometimes need the isclose
        # assert all(self.game.obs == self_before_training.game.obs)
        assert self.game.hashable_obs == self_before_training.game.hashable_obs
        p1, v1 = self.net.predict(self.game.obs)
        p2, v2 = self_before_training.net.predict(self_before_training.game.obs)
#        print (p1, p2)
        assert all(np.isclose(p1, p2))
        assert np.isclose(v1, v2)

        print(f"Training on {len(flat_examples)} examples")
        # for i, (state, (policy, reward)) in enumerate(flat_examples):
        #     print(f"Example {i+1}/{len(flat_examples)}: state={state}, policy={policy}, reward={reward}")
        start_time = time.time()
        # Capture training losses (could be a dict or list depending on implementation)
        _, _, train_losses = self.net.train(flat_examples, **({"print_all_epochs": True} if PRINT_ALL_EPOCHS else {}))
        
        # Breakdown losses if available (BitStringPolicyValueNet will return a dict)
        if isinstance(train_losses, list) and len(train_losses) > 0:
            # Handle legacy case (list of total losses)
            if isinstance(train_losses[0], dict):
                 loss_policy = train_losses[-1].get('policy', 0.0)
                 loss_value = train_losses[-1].get('value', 0.0)
            else:
                 loss_policy = 0.0
                 loss_value = train_losses[-1] # Fallback
        else:
             loss_policy = 0.0
             loss_value = 0.0

        elapsed = time.time() - start_time
        print(f"..training done in {elapsed:.2f} seconds")

        # Evaluate & Gating
        if self.use_gating:
            eval_stats = self.pit(self_before_training)
            score = eval_stats['score']
            
            if score >= self.threshold_to_keep:
                print("Keeping the new network")
            else:
                print("Reverting to the old network")
                self.net = self_before_training.net
        else:
            print("Accepted by default (gating disabled)")
            # Create dummy stats for history to avoid breaking structure
            eval_stats = {
                'score': 1.0,
                'new_reward_mean': np.nan,
                'new_reward_std': np.nan,
                'old_reward_mean': np.nan,
                'old_reward_std': np.nan
            }
        
        # Calculate average game length
        avg_game_len = np.mean([len(game_trace) for game_trace in train_example_sets])

        # Record History
        iter_idx = len(self.history['iteration']) + 1
        self.history['iteration'].append(iter_idx)
        self.history['reward_mean'].append(eval_stats['new_reward_mean'])
        self.history['reward_std'].append(eval_stats['new_reward_std'])
        self.history['loss_policy'].append(loss_policy)
        self.history['loss_value'].append(loss_value)
        self.history['game_length'].append(avg_game_len)

        return train_losses

    def pit(self, self_before_training):
        "Play a bunch of games to evaluate new vs. old networks"
        
        start_time = time.time()
        my_multiprocessing_stash = self.push_multiprocessing()
        before_multiprocessing_stash = self_before_training.push_multiprocessing()
        # print("pred on old", self_before_training.game.obs, self_before_training.net.predict(self_before_training.game.obs))
        # print("pred on new", self.game.obs, self.net.predict(self.game.obs))
        arg_tuples = [(i, self._randseed("eval"), self._randseed("mcts"), self._randseed("external_policy"), self_before_training, PIT_NO_MCTS, True) for i in range(self.n_games_per_eval)]
        eval_results = self._starmap(self._play_for_eval, arg_tuples)
        self_before_training.pop_multiprocessing(before_multiprocessing_stash)
        self.pop_multiprocessing(my_multiprocessing_stash)
        elapsed = time.time() - start_time
        print(f"..evaluation done in {elapsed:.2f} seconds")

        eval_results_keys = eval_results[0].keys()
        for res in eval_results:
            assert res.keys() == eval_results_keys
        eval_results_arrays = {key: np.array([res[key] for res in eval_results]) for key in eval_results_keys}

        # Print stats, keep new network iff it wins >= threshold_to_keep fraction of games
        old_rewards = eval_results_arrays["old_net"]
        new_rewards = eval_results_arrays["new_net"]
        print(f"Old network+MCTS average reward: {old_rewards.mean():.4f}, min: {old_rewards.min():.4f}, max: {old_rewards.max():.4f}, stdev: {old_rewards.std():.4f}")
        print(f"New network+MCTS average reward: {new_rewards.mean():.4f}, min: {new_rewards.min():.4f}, max: {new_rewards.max():.4f}, stdev: {new_rewards.std():.4f}")

        if PIT_NO_MCTS:
            old_rewards_no_mcts = eval_results_arrays["old_net_no_mcts"]
            new_rewards_no_mcts = eval_results_arrays["new_net_no_mcts"]
            print(f"Old bare network average reward: {old_rewards_no_mcts.mean():.4f}, min: {old_rewards_no_mcts.min():.4f}, max: {old_rewards_no_mcts.max():.4f}, stdev: {old_rewards_no_mcts.std():.4f}")
            print(f"New bare network average reward: {new_rewards_no_mcts.mean():.4f}, min: {new_rewards_no_mcts.min():.4f}, max: {new_rewards_no_mcts.max():.4f}, stdev: {new_rewards_no_mcts.std():.4f}")

        for policy_name in self.external_policy_creators_to_pit.keys():
            rewards = eval_results_arrays[policy_name]
            print(f"External policy \"{policy_name}\" average reward: {rewards.mean():.2f}, min: {rewards.min():.2f}, max: {rewards.max():.2f}, stdev: {rewards.std():.2f}")

        wins = np.sum((new_rewards > old_rewards) & ~(np.isclose(new_rewards, old_rewards)))
        ties = np.sum(np.isclose(new_rewards, old_rewards))
        losses = np.sum((new_rewards < old_rewards) & ~(np.isclose(new_rewards, old_rewards)))
        assert wins + ties + losses == self.n_games_per_eval
        score = (wins + ties / 2) / self.n_games_per_eval  # a tie is half a win
        print(f"New network won {wins} and tied {ties} out of {self.n_games_per_eval} games ({score:.2%} wins where ties are half wins)")
        
        return {
            'score': score,
            'new_reward_mean': new_rewards.mean(),
            'new_reward_std': new_rewards.std(),
            'old_reward_mean': old_rewards.mean(),
            'old_reward_std': old_rewards.std()
        }
    
    def play_train_multiple(self, n_trains: int, start_at = 0, checkpoint_every = None, checkpoint_dir = "general_az_1p_checkpoint"):
        for i in range(start_at, n_trains):
            print(f"\nTraining iteration {i+1} of {n_trains}: will play {self.n_games_per_train} games, train, and evaluate on {self.n_games_per_eval} games")
            self.play_and_train()
            if checkpoint_every is not None and (i + 1) % checkpoint_every == 0:
                checkpoint_subdir = Path(checkpoint_dir) / f"{self.run_start_time}_iter_{i+1}"
                print(f"Saving intermediate checkpoint to {checkpoint_subdir}")
                self.save_checkpoint(checkpoint_subdir)
                print(f"Saving intermediate checkpoint to {checkpoint_subdir}")
                self.save_checkpoint(checkpoint_subdir)
                self.net.save_checkpoint(checkpoint_subdir)
        return self.history

    def push_multiprocessing(self):
        my_info = self.all_training_examples
        self.all_training_examples = None

        net_info = self.net.push_multiprocessing()
        return (my_info, net_info)
    
    def pop_multiprocessing(self, stash):
        my_info, net_info = stash
        self.all_training_examples = my_info
        self.net.pop_multiprocessing(net_info)

    def save_checkpoint(self, save_dir):
        "Save all state but the network, which you must handle separately. Does not save config."

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        with (save_dir / self.save_file_name).open("wb") as file:
            pickle.dump({
                "game": self.game,
                "all_training_examples": self.all_training_examples,
                "rngs": self.rngs
            }, file)

    def load_checkpoint(self, save_dir, exclude_keys = []):
        "Load all state but the network, which you must handle separately. Does not load config."

        save_dir = Path(save_dir)
        save_file = save_dir / self.save_file_name
        if not save_file.exists():
            raise FileNotFoundError(f"Checkpoint file {save_file} does not exist.")
        with open(save_file, "rb") as file:
            checkpoint = pickle.load(file)
        if "rngs" not in exclude_keys: self.rngs = checkpoint["rngs"]
        if "game" not in exclude_keys: self.game = checkpoint["game"]
        if "all_training_examples" not in exclude_keys:
            # TODO extremely weird behavior: reassigning this destroys multiprocessing
            # performance (no matter whether directly, with deepcopy, with list
            # comprehensions, etc.), but clearing it and extending it works fine. Figure out why...
            self.all_training_examples.clear()
            self.all_training_examples.extend(checkpoint["all_training_examples"])
