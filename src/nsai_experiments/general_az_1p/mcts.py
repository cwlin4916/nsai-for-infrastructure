import copy
import warnings
from typing import Any

import numpy as np

from .game import Game
from .policy_value_net import PolicyValueNet

EPS = 1e-8  # Add to UCB numerator to avoid zeroing out the policy when total_N is 0

class MCTSTreeNode():
    direct_reward: float  # Any direct reward from the Game for being in this state
    is_terminal_state: bool  # Whether this is an end state (terminated or truncated)
    nn_policy: Any  # The result of the policy network for this state
    nn_value: Any  # The result of the value network for this state
    action_mask: Any  # A bit vector of action validity for this state
    total_N: int  # Total number of visits to this node, should equal sum(action_N.values())
    action_Q: dict[int, float]  # Q value for each action from this state
    action_N: dict[int, int]  # Number of visits for each action from this state

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
                 n_simulations: int = 1000,
                 temperature: float = 1.0,
                 c_exploration: float = 1.0):
        self.game = game
        self.net = net
        self.nodes = {}

        self.n_simulations = n_simulations
        self.temperature = temperature
        self.c_exploration = c_exploration

    def perform_simulations(self):
        """
        Perform `n_simulations` simulations from the current game state, then return move
        probabilities from exponentiated visit counts
        """
        mystate = self.game.hashable_obs
        for i in range(self.n_simulations):
            old_game = self.game
            self.game = copy.deepcopy(self.game)
            self.search()
            self.game = old_game
            assert mystate == self.game.hashable_obs
        
        mynode = self.nodes[mystate]
        counts = [mynode.action_N.get(a, 0) for a in range(len(mynode.nn_policy))]  # dictionary lookup misses in the case of action masked
        counts = np.array(counts) ** (1./self.temperature)
        probs = counts / sum(counts)
        return probs
        
    def search(self) -> float:
        # NOTE does not guarantee that self.game will be in the same state upon exit
        mystate = self.game.hashable_obs

        # Initialize node if we've not been here before
        if mystate not in self.nodes:
            reward: float = self.game.reward  # type: ignore
            is_terminal: bool = self.game.terminated or self.game.truncated  # type: ignore
            self.nodes[mystate] = MCTSTreeNode(reward, is_terminal)
        mynode = self.nodes[mystate]
        
        # Base case: in a terminal state, return the direct reward
        if mynode.is_terminal_state:
            return mynode.direct_reward
        
        # Base case: at a new node, query the policy-value network
        if mynode.nn_policy is None:
            assert mynode.nn_value is None
            
            mypolicy, myvalue = self.net.predict(self.game.obs)
            myaction_mask = self.game.get_action_mask()
            
            if sum(myaction_mask) == 0:
                raise ValueError("Action mask says no valid moves")
            mypolicy *= myaction_mask
            sum_policy = mypolicy.sum()
            if sum_policy > 0:
                mypolicy /= sum_policy
            else:
                warnings.warn("All valid moves have zero probability, ignoring probabilities")
                mypolicy += myaction_mask / sum(myaction_mask)
            
            mynode.nn_policy = mypolicy
            mynode.nn_value = myvalue
            mynode.action_mask = myaction_mask
            return myvalue
        
        # Recursive case: select the best action using UCB with action masking and descend the tree
        ucbs = [self.masked_ucb(mynode, action) for action in range(len(mynode.nn_policy))]
        best_action: int = np.argmax(ucbs)  # type: ignore
        self.game.step_wrapper(best_action)
        reward = self.search()
        
        self.update_edge(mynode, best_action, reward)
        mynode.total_N += 1
        
        return reward

    # TODO if we decide to keep MCTSTreeNode, maybe some of these should be methods of that class
    def calc_ucb(self, mynode: MCTSTreeNode, action: int) -> float:
        assert mynode.total_N == sum(mynode.action_N.values())
        myaction_Q = mynode.action_Q.get(action, 0.0)
        myaction_N = mynode.action_N.get(action, 0)
        return myaction_Q + self.c_exploration * mynode.nn_policy[action] * \
            np.sqrt(mynode.total_N + EPS) / (1 + myaction_N)
    
    def masked_ucb(self, mynode: MCTSTreeNode, action: int) -> float:
        if not mynode.action_mask[action]:
            return -np.inf
        return self.calc_ucb(mynode, action)
    
    def update_edge(self, mynode: MCTSTreeNode, action: int, reward: float):
        if action not in mynode.action_N:
            assert action not in mynode.action_Q
            mynode.action_N[action] = 0  # could use a collections.Counter for this
            mynode.action_Q[action] = 0.0
        
        mynode.action_Q[action] = (mynode.action_N[action] * mynode.action_Q[action] + reward) / (1 + mynode.action_N[action])
        mynode.action_N[action] += 1
