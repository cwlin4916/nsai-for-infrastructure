import io
import re

import gymnasium as gym
from nltk import PCFG

from nsai_experiments.zoning_game.zg_cfg import ZONING_GAME_GRAMMAR

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

class ZoningLangEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}
    DEFAULT_MAX_LENGTH = 512
    DEFAULT_PAD_TOKEN = "<PAD>"
    DEFAULT_MAX_MOVES = 2048

    def __init__(self, grammar: PCFG = ZONING_GAME_GRAMMAR, max_program_length = DEFAULT_MAX_LENGTH, pad_token = DEFAULT_PAD_TOKEN, max_moves = DEFAULT_MAX_MOVES, render_mode = "ansi", env_kwargs = {}):
        super().__init__()

        self.env_kwargs = env_kwargs
        self.grammar = grammar
        self.max_length = max_program_length
        self.max_productions = len(grammar.productions())
        self.pad_token = pad_token
        self.max_moves = max_moves
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

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

        # Action space is (index, production)
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(self.max_length), gym.spaces.Discrete(self.max_productions)))
        # Observation space is the current sequence of token indices
        self.observation_space = gym.spaces.MultiDiscrete([self.num_tokens] * self.max_length)
    
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
        return {}
    
    def reset(self, seed = None):
        super().reset(seed = seed)
        self.current_program = [self.grammar.start()]
        self.n_moves = 0
        return self._get_obs(), self._get_info()
    
    def render(self):
        "Render given `self.render_mode`. For `render_mode=ansi`, can print the results like `print(my_env.render().read())`."
        if self.render_mode is None: return
        assert self.render_mode == "ansi"
        buf = io.StringIO()
        terminated, truncated = self._get_terminated_truncated()
        program_str = " ".join(str(symbol) for symbol in self.current_program)
        program_str = re.sub(r";\s*", ";\n", program_str)  # newline after semicolon for legibility
        print(f"Current program:\n'''\n{program_str}\n'''", file=buf)
        print(f"terminated = {terminated}, truncated = {truncated}.", file=buf)
        buf.seek(0)
        return buf
    
    def step(self, action, on_invalid=None):
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

        terminated, truncated = self._get_terminated_truncated()
        reward = self._eval_reward() if terminated or truncated else 0  # only reward at the end
        if terminated:
            print(f"Finished with reward {reward}")
        
        # Return observation, reward, terminated, truncated, info
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _eval_reward(self):
        return 100  # TODO
