from collections.abc import Iterable
import hashlib
import io
import itertools
from multiprocessing.pool import Pool
import re
import logging

import gymnasium as gym
from nltk import PCFG
import numpy as np
from tqdm import tqdm

from nsai_experiments.zoning_game.zg_cfg import ZONING_GAME_GRAMMAR
from nsai_experiments.zoning_game.zg_gym import ZoningGameEnv
from nsai_experiments.zoning_game.zg_policy import play_one_game
from nsai_experiments.zoning_game.zg_cfg_policy import create_policy_cfg_with_fallback
from .zg_policy import create_policy_indiv_greedy

def extract_grammar_symbols(grammar):
    """Extract terminals and nonterminals from a grammar.
    
    Args:
        grammar: NLTK PCFG grammar object
        
    Returns:
        tuple: (terminals, nonterminals, num_tokens)
    """
    terminals = set()
    nonterminals = set()
    for production in grammar.productions():
        nonterminals.add(production.lhs())
        for symbol in production.rhs():
            if hasattr(symbol, 'symbol'):
                nonterminals.add(symbol)
            else:
                terminals.add(symbol)
    return terminals, nonterminals

def zl_evaluate_policies_for_seed(policy_seed, ruleset, fallback_policy_creator, env_seeds, env, on_invalid):
    logging.getLogger("nsai_experiments.zoning_game.zg_cfg").setLevel(logging.ERROR)
    logging.getLogger("nsai_experiments.zoning_game.zg_gym").setLevel(logging.ERROR)

    ruleset_policy = create_policy_cfg_with_fallback(ruleset, fallback_policy_creator, seed=policy_seed)

    local_ruleset_scores = []
    local_ruleset_infos = []

    for env_seed in env_seeds:
        _, _, ruleset_reward, _, _, ruleset_info = play_one_game(ruleset_policy, env=env, seed=env_seed, on_invalid=on_invalid)
        local_ruleset_scores.append(ruleset_reward)
        local_ruleset_infos.append(ruleset_info)
    return local_ruleset_scores, local_ruleset_infos

def zl_evaluate_ruleset(ruleset, policy_seeds, env_seeds, on_invalid=None, env_kwargs={}, n_procs=-1, use_tqdm=False):
    env = ZoningGameEnv(**env_kwargs)

    env_seeds_is_2d = isinstance(env_seeds[0], Iterable)
    if env_seeds_is_2d:
        if not len(env_seeds) == len(policy_seeds):
            raise ValueError("If env_seeds is 2D, its length must match that of policy_seeds")

    arg_tuples = [(policy_seed, ruleset, create_policy_indiv_greedy, env_seeds[i] if env_seeds_is_2d else env_seeds, env, on_invalid)
             for (i, policy_seed) in enumerate(policy_seeds)]
    
    if n_procs is None or n_procs >= 0:
        with Pool(processes=n_procs) as pool:
            results = pool.starmap(zl_evaluate_policies_for_seed, arg_tuples)
    else:
        results = list(itertools.starmap(zl_evaluate_policies_for_seed, 
                        tqdm(arg_tuples) if use_tqdm else arg_tuples))
    
    flat_scores = [score for sublist in results for score in sublist[0]]
    return np.array(flat_scores, dtype=np.float32).mean()

class ZoningLangEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}
    DEFAULT_MAX_LENGTH = 512
    DEFAULT_PAD_TOKEN = "<PAD>"
    DEFAULT_MAX_MOVES = 2048

    def __init__(self, grammar: PCFG = ZONING_GAME_GRAMMAR, max_program_length = DEFAULT_MAX_LENGTH,
                 pad_token = DEFAULT_PAD_TOKEN, max_moves = DEFAULT_MAX_MOVES, render_mode = "ansi", env_kwargs = {},
                 eval_n = 10, eval_policy_creator = create_policy_indiv_greedy, eval_num_procs = -1):
        super().__init__()

        self.env_kwargs = env_kwargs
        self.grammar = grammar
        self.max_length = max_program_length
        self.max_productions = len(grammar.productions())
        self.pad_token = pad_token
        self.max_moves = max_moves
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.eval_n = eval_n  # NOTE: right now we actually run eval_n**2 games
        self.eval_policy_creator = eval_policy_creator
        self.eval_num_procs = eval_num_procs

        # Extract terminals and nonterminals
        self.terminals, self.nonterminals = extract_grammar_symbols(grammar)
        
        # Create bidirectional mappings between symbols and integers
        # Reserve 0 for pad token
        all_symbols = [self.pad_token] + list(self.terminals) + list(self.nonterminals)
        self.symbol_to_int = {symbol: i for i, symbol in enumerate(all_symbols)}
        self.int_to_symbol = {i: symbol for i, symbol in enumerate(all_symbols)}
        self.num_tokens = len(all_symbols)
        
        # Create set of nonterminal indices for quick lookup
        self.nonterminal_indices = {self.symbol_to_int[nt] for nt in self.nonterminals}
        self.pad_token_int = self.symbol_to_int[self.pad_token]

        # Build lookup dictionary from nonterminal symbol to matching productions
        self._nonterminal_to_productions = {}
        for nonterminal in self.nonterminals:
            matching = [(i, prod) for i, prod in enumerate(grammar.productions()) 
                       if prod.lhs() == nonterminal]
            self._nonterminal_to_productions[nonterminal] = matching

        # Cache for evaluation results: (program_str, eval_rand) -> score
        self._eval_cache = {}

        # Action space is (index, production)
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(self.max_length), gym.spaces.Discrete(self.max_productions)))
        # Observation space is the current sequence of token indices
        self.observation_space = gym.spaces.MultiDiscrete([self.num_tokens] * self.max_length)

        self.last_prod = None
    
    def _get_obs(self):
        """Get the current observation as a list of token indices."""
        obs = [self.symbol_to_int[symbol] for symbol in self.current_program]
        # Pad the observation to max_length
        obs += [self.pad_token_int] * (self.max_length - len(obs))
        return obs
    
    def get_indices_of_nonterminals(self, obs):
        """Get the indices of all nonterminals in an observation.
        
        Args:
            obs: List of token indices (observation from environment)
            
        Returns:
            list: Indices of all nonterminals in the observation
        """
        indices = []
        for i, token_int in enumerate(obs):
            if token_int != self.pad_token_int and token_int in self.nonterminal_indices:
                indices.append(i)
        return indices
    
    def get_productions_for_nonterminal(self, nonterminal):
        """Get all productions that can be applied to a nonterminal.
        
        Args:
            nonterminal: Either a nonterminal symbol or a nonterminal integer
            
        Returns:
            list: List of (production_index, production) tuples where production.lhs() matches the nonterminal
        """
        # Convert integer to symbol if needed
        if isinstance(nonterminal, int):
            nonterminal = self.int_to_symbol[nonterminal]
        
        # Direct dictionary lookup (will raise KeyError if nonterminal not found)
        return self._nonterminal_to_productions[nonterminal]
    
    def decode_obs(self, obs):
        """Convert an observation (list of integers) to human-readable symbols.
        
        Args:
            obs: List of token indices
            
        Returns:
            list: List of symbols (strings or NLTK symbols)
        """
        return [str(self.int_to_symbol[token_int]) for token_int in obs]
    
    def _get_terminated_truncated(self):
        terminated = all(symbol not in self.nonterminals for symbol in self.current_program)
        truncated = self.n_moves >= self.max_moves
        return terminated, truncated
    
    def _get_info(self):
        return {
            "last_prod": self.last_prod
            }
    
    def reset(self, seed = None, options = None):
        print(f"RESETTING GYM with {seed=}")
        assert options is None
        super().reset(seed = seed)
        self.current_program = [self.grammar.start()]
        self.n_moves = 0
        self.eval_rand = self.np_random.integers(0, 2**31 - 1)
        return self._get_obs(), self._get_info()
    
    def stringify_program(self, add_newlines=True):
        program_str = " ".join(str(symbol) for symbol in self.current_program)
        if add_newlines:
            program_str = re.sub(r";\s*", ";\n", program_str)
        return program_str
    
    def render(self):
        "Render given `self.render_mode`. For `render_mode=ansi`, can print the results like `print(my_env.render().read())`."
        if self.render_mode is None: return
        assert self.render_mode == "ansi"
        buf = io.StringIO()
        terminated, truncated = self._get_terminated_truncated()
        print(f"Current program:\n'''\n{self.stringify_program()}\n'''", file=buf)
        print(f"terminated = {terminated}, truncated = {truncated}.", file=buf)
        buf.seek(0)
        return buf
    
    def step(self, action, on_invalid=None, use_tqdm=False):
        """
        Options for `on_invalid`: `None` does nothing, `"warn"` logs a warning, `"error"`
        raises an error
        """
        token_i, production_i = action
        
        # Look up the production
        production = self.grammar.productions()[production_i]
        
        # Check if the token index is valid
        if token_i >= len(self.current_program):
            # Invalid action: token index out of bounds
            if on_invalid == "warn":
                print(f"Warning: token_i {token_i} is out of bounds (program length: {len(self.current_program)})")
            elif on_invalid == "error":
                raise ValueError(f"token_i {token_i} is out of bounds (program length: {len(self.current_program)})")
            return self._get_obs(), 0, *self._get_terminated_truncated(), self._get_info()
        
        # Get the token at the specified index
        token = self.current_program[token_i]
        
        # Check if the production can be applied to this token
        # The production's LHS must match the token (token must be a nonterminal that matches production.lhs())
        if token != production.lhs():
            # Invalid action: production cannot be applied to this token
            if on_invalid == "warn":
                print(f"Warning: production {production} cannot be applied to token {token} at index {token_i}")
            elif on_invalid == "error":
                raise ValueError(f"production {production} cannot be applied to token {token} at index {token_i}")
            return self._get_obs(), 0, *self._get_terminated_truncated(), self._get_info()
        
        # Apply the production: replace the token with the RHS of the production
        rhs = list(production.rhs())
        
        # Create new program by replacing token at token_i with the RHS
        new_program = (
            self.current_program[:token_i] +  # Everything before the token
            rhs +                              # The production's RHS
            self.current_program[token_i + 1:] # Everything after the token
        )
        
        # Check if the new program exceeds max length (overflow check)
        if len(new_program) > self.max_length:
            # Invalid action: applying this production would overflow the max length
            if on_invalid == "warn":
                print(f"Warning: applying production would exceed max_length ({len(new_program)} > {self.max_length})")
            elif on_invalid == "error":
                raise ValueError(f"applying production would exceed max_length ({len(new_program)} > {self.max_length})")
            return self._get_obs(), 0, *self._get_terminated_truncated(), self._get_info()
        
        # Valid action: update the current program
        self.current_program = new_program
        self.n_moves += 1

        self.last_prod = production

        terminated, truncated = self._get_terminated_truncated()
        reward = self._eval_reward(on_invalid=on_invalid, use_tqdm=use_tqdm) if terminated or truncated else 0  # only reward at the end
        # if terminated:
        #     print(f"Finished with reward {reward}")
        
        # Return observation, reward, terminated, truncated, info
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _eval_reward(self, on_invalid, use_tqdm=False):
        program_str = self.stringify_program()
        program_hash = int(hashlib.sha256(program_str.encode()).hexdigest(), 16) % 2**20  # fancy hash for determinism across runs
        # TODO think more deeply about this, I think it's actually bad to always use the
        # same random seeds to evaluate a given program but it'd be nice if we could keep
        # this for caching's sake
        rand_base = self.eval_rand + program_hash

        # Check cache
        cache_key = (program_str, rand_base)
        if cache_key in self._eval_cache:
            ruleset_score = self._eval_cache[cache_key]
            return ruleset_score

        policy_seeds = range(rand_base, rand_base + self.eval_n)
        env_seeds = [range(a, a + self.eval_n) for a in [rand_base*i for i in range(self.eval_n)]]
        print(f"Cache miss for {cache_key=}, {program_hash=}, {self.eval_rand=}")
        ruleset_score = zl_evaluate_ruleset(program_str,
                         policy_seeds=policy_seeds,
                         env_seeds=env_seeds,
                         on_invalid=on_invalid,
                         env_kwargs=self.env_kwargs,
                         n_procs=self.eval_num_procs,
                         use_tqdm=use_tqdm)
        print(f"Evaluation complete, ruleset score: {ruleset_score}")
        
        # Store in cache
        self._eval_cache[cache_key] = ruleset_score
        
        return ruleset_score
