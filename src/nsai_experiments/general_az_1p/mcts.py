import warnings
from typing import Any

import numpy as np

from .game import Game
from .policy_value_net import PolicyValueNet

EPS = 1e-8  # Add to UCB numerator to avoid zeroing out the policy when total_N is 0

def entab(s, addl):
    return "  " + s + addl if s else ""

class MCTSTreeNode():
    direct_reward: float  # Any direct reward from the Game for being in this state
    is_terminal_state: bool  # Whether this is an end state (terminated or truncated)
    nn_policy: Any  # The result of the policy network for this state
    nn_value: Any  # The result of the value network for this state
    action_mask: Any  # A bit vector of action validity for this state
    total_N: int  # Total number of visits to this node, should equal sum(action_N.values())
    action_Q: dict[tuple, float]  # Q value for each action from this state
    action_N: dict[tuple, int]  # Number of visits for each action from this state

    def __init__(self, direct_reward: float, is_terminal_state: bool):
        self.direct_reward = direct_reward
        self.is_terminal_state = is_terminal_state

        self.nn_policy = None
        self.nn_value = None
        self.action_mask = None

        self.total_N = 0
        self.action_Q = {}
        self.action_N = {}

class MCTS():
    game: Game
    net: PolicyValueNet
    nodes: dict[Any, MCTSTreeNode]  # Maps game state to MCTSTreeNode
    n_simulations: int  # Number of simulations to perform in each search
    temperature: float  # Temperature for exponentiated visit counts in the final answer
    c_exploration: float  # Exploration constant for UCB

    def __init__(self, game: Game, net: PolicyValueNet,
                 n_simulations: int = 25,
                 temperature: float = 1.0,
                 c_exploration: float = 1.0):
        self.game = game
        self.net = net
        self.nodes = {}

        self.n_simulations = n_simulations
        self.temperature = temperature
        self.c_exploration = c_exploration

    def perform_simulations(self, msg):
        """
        Perform `n_simulations` simulations from the current game state, then return move
        probabilities from exponentiated visit counts. Special case: if `n_simulations` is
        negative, directly query the policy network rather than performing any simulations
        (results are still exponentiated).
        """
        mystate = self.game.hashable_obs
        if msg: print(msg, "at start of perform_simulations, obs is", self.game.obs)

        if self.n_simulations < 0:
            if msg: print(msg, "n_simulations < 0, directly querying policy net")
            counts, _, _ = self.query_net_masked(msg)
        else:
            for i in range(self.n_simulations):
                old_game_state = self.game.stash_state()
                self.search(entab(msg,  f", simulation {i+1}/{self.n_simulations}"))
                self.game = self.game.unstash_state(old_game_state)
                assert mystate == self.game.hashable_obs
            
            mynode = self.nodes[mystate]
            
            # Build counts array directly in the policy shape
            counts = np.zeros_like(mynode.nn_policy)
            for action, count in mynode.action_N.items():
                counts[action] = count
        
        if msg: print(msg, "mynode", mynode, "counts", counts)
        counts = counts ** (1./self.temperature)
        probs = counts / counts.sum()
        # For debugging:
        # print(probs)
        
        return probs
        
    def search(self, msg) -> float:
        # NOTE does not guarantee that self.game will be in the same state upon exit
        mystate = self.game.hashable_obs
        # TODO HACK to get the step count
        if msg: print(msg, "BEGIN: searching state", self.game.obs, "step_count", self.game.step_count, "hash", hash(self.game.hashable_obs), "len nodes", len(self.nodes))

        # Initialize node if we've not been here before
        if mystate not in self.nodes:
            reward: float = self.game.reward  # type: ignore
            is_terminal: bool = self.game.terminated or self.game.truncated  # type: ignore
            if msg: print(msg, "adding node", self.game.obs, "hash", hash(self.game.hashable_obs), "terminated", self.game.terminated, "truncated", self.game.truncated, "reward", reward)
            self.nodes[mystate] = MCTSTreeNode(reward, is_terminal)
        mynode = self.nodes[mystate]
        
        # Base case: in a terminal state, return the direct reward
        if mynode.is_terminal_state:
            if msg: print(msg, "Reached terminal state", self.game.obs, "reward", mynode.direct_reward)
            return mynode.direct_reward
        
        # Base case: at a new node, query the policy-value network
        if mynode.nn_policy is None:
            if msg: print(msg, "Reached unexpanded node")
            assert mynode.nn_value is None
            
            mypolicy, myvalue, myaction_mask = self.query_net_masked(msg)
            mynode.nn_policy = mypolicy
            mynode.nn_value = myvalue
            mynode.action_mask = myaction_mask
            return myvalue
        
        # Recursive case: select the best action using UCB with action masking and descend the tree
        ucbs = self.calc_masked_ucbs(mynode, entab(msg, " ucb"))
        best_action = np.unravel_index(np.argmax(ucbs), ucbs.shape)
        if msg: print(msg, "-> taking action", best_action, "based on UCBs", ucbs)
        self.game.step_wrapper(best_action)
        reward = self.search(entab(msg, " recurse"))
        # reward = self.search("")
        
        self.update_edge(mynode, best_action, reward)
        mynode.total_N += 1
        
        return reward

    def query_net(self, msg):
        "Query the policy-value network and perform some basic validation"
        mypolicy, myvalue = self.net.predict(self.game.obs)
        if msg: print(msg, "Queried NN: policy", mypolicy, "value", myvalue)
        if len(myvalue.shape) != 0:
            raise ValueError(f"Expected value to be scalar, got {myvalue.shape}")
        return mypolicy, myvalue
    
    def query_net_masked(self, msg):
        "Run `query_net` and use the game's action mask to zero out invalid moves in the policy"
        mypolicy, myvalue = self.query_net(msg)

        myaction_mask = self.game.get_action_mask()
        sum_mask = myaction_mask.sum()
        if sum_mask == 0:
            raise ValueError("Action mask says no valid moves")
        mypolicy *= myaction_mask
        sum_policy = mypolicy.sum()
        if sum_policy > 0:
            mypolicy /= sum_policy
        else:
            warnings.warn("All valid moves have zero probability, ignoring probabilities")
            mypolicy += myaction_mask / sum_mask

        return mypolicy, myvalue, myaction_mask
    
    def calc_masked_ucbs(self, mynode: MCTSTreeNode, msg) -> np.ndarray:
        # TODO PERF this was super optimized and then the implementation changed for multidimensional actions, need to reoptimize
        # Get all valid action indices as tuples
        valid_actions = list(zip(*np.nonzero(mynode.action_mask)))

        # Initialize UCB array with -inf for invalid actions
        all_ucbs = np.full(mynode.nn_policy.shape, -np.inf)
        
        # Calculate UCB for each valid action
        for action in valid_actions:
            q = mynode.action_Q.get(action, 0.0)
            n = mynode.action_N.get(action, 0)
            ucb = q + self.c_exploration * mynode.nn_policy[action] * np.sqrt(mynode.total_N + EPS) / (1 + n)
            all_ucbs[action] = ucb
            
            if msg:
                print(msg, "calc_ucb for action", action, "action Q", q, "c_exploration", self.c_exploration, "nn policy", mynode.nn_policy[action], "total N", mynode.total_N, "action N", n)
                print(msg, "ucb result", ucb)
        
        return all_ucbs

    def update_edge(self, mynode: MCTSTreeNode, action: int, reward: float):
        if action not in mynode.action_N:
            assert action not in mynode.action_Q
            mynode.action_N[action] = 0  # could use a collections.Counter for this
            mynode.action_Q[action] = 0.0
        
        mynode.action_Q[action] = (mynode.action_N[action] * mynode.action_Q[action] + reward) / (1 + mynode.action_N[action])
        mynode.action_N[action] += 1
