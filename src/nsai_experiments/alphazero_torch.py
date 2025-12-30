#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-player Alpha Zero
@author: Thomas Moerland, Delft University of Technology
"""

import numpy as np
import torch
import random
import math

myseed = 1
torch.manual_seed(myseed)
np.random.seed(myseed)
random.seed(myseed)


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary


import argparse
import os
import time
import copy
from gymnasium import wrappers
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from .new_games_old_engine.helpers import argmax, check_space, store_safely, stable_normalizer, smooth, symmetric_remove, Database
from .new_games_old_engine.make_game import make_game
from .new_games_old_engine.netgame import make_netgame

from nsai_experiments import line_extending_game_gym
from nsai_experiments.line_extending_game_gym import WindowlessLineExtendingGameEnv

#### Neural Networks ##
class Model(nn.Module):

    def __init__(self,Env,lr,n_hidden_layers,n_hidden_units):
        super(Model, self).__init__()

        # Check the Gym environment
        self.action_dim, self.action_discrete  = check_space(Env.action_space)
        self.state_dim, self.state_discrete  = check_space(Env.observation_space)
        if not self.action_discrete: 
            raise ValueError('Continuous action space not implemented')
        dim = np.array(math.prod(np.array(self.state_dim).reshape(-1)))#*6  # TODO hardcoded

        self.flatten_layer = nn.Flatten()
        self.lin1 = nn.Linear(dim, n_hidden_units)
        self.lin2 = nn.Linear(n_hidden_units, n_hidden_units)
        self.linV = nn.Linear(n_hidden_units, 1)
        self.linpi = nn.Linear(n_hidden_units, self.action_dim)
        self.softmax = nn.Softmax(dim=1) #self.action_dim)
        # self.bn1 = nn.BatchNorm2d(n_hidden_units)
        # self.bn2 = nn.BatchNorm2d(n_hidden_units)

        self.activation = nn.ELU()

        self.n_hidden_layers = n_hidden_layers

        # Loss and optimizer
#        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)  
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        # self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr, weight_decay=1e-4)

        summary(self, (1,dim))

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.int64)
        # x = torch.nn.functional.one_hot(x, num_classes=6)  # TODO hardcoded
        x = torch.as_tensor(x, dtype=torch.float32)
        x = self.flatten_layer(x)
        x = self.activation(self.lin1(x))
        for i in range(self.n_hidden_layers-1):
            x = self.activation(self.lin2(x))
        return x        
    
    
    def train_once(self,sb,Vb,pib):
        # print ("TRAINING <sb vb pib>", sb.shape, Vb.shape, pib.shape)
        # print (sb[0,:], Vb[0,:], pib[0,:])

        #self.train()
        # one epoch of training
        log_pi_hat = self.predict_pi_logits(sb)
        self.predict_V(sb)
        # forward pass
        # Loss
        lossV = nn.MSELoss()
        outputV = lossV(torch.Tensor(Vb), self.V_hat)
        losspi = nn.CrossEntropyLoss()
        outputpi = losspi(log_pi_hat, torch.Tensor(pib))
        loss = outputpi + outputV
        # backward pass
        self.optimizer.zero_grad()
        loss.backward()
        # update weights
        self.optimizer.step()
    
    def predict_V(self,s):
        x = self.forward(s)
        self.V_hat = self.linV(x)  # value head
        return self.V_hat
        
    def predict_pi_logits(self,s):
        x = self.forward(s)
        log_pi_hat = self.linpi(x)
        self.pi_hat = log_pi_hat
        return log_pi_hat

    def predict_pi(self,s):
        log_pi_hat = self.predict_pi_logits(s)
        self.pi_hat = self.softmax(log_pi_hat) # policy head           
        return self.pi_hat

##### MCTS functions #####
      
class Action():
    ''' Action object '''
    def __init__(self,index,parent_state,Q_init=0.0):
        self.index = index
        self.parent_state = parent_state
        self.W = 0.0
        self.n = 0
        self.Q = float(Q_init)
                
    def add_child_state(self,s1,r,terminal,model):
        self.child_state = State(s1,r,terminal,self,self.parent_state.na,model)
        return self.child_state
        
    def update(self,R):
        R = R.detach().numpy()
        self.n += 1
        self.W += R
        self.Q = float(self.W/self.n)

class State():
    ''' State object '''

    def __init__(self,index,r,terminal,parent_action,na,model):
        ''' Initialize a new state '''
        self.index = copy.deepcopy(index) # state
        self.r = r # reward upon arriving in this state
        self.terminal = terminal # whether the domain terminated in this state
        self.parent_action = parent_action
        self.n = 0
        self.model = model
        
        self.evaluate()
        # Child actions
        self.na = na
        self.child_actions = [Action(a,parent_state=self,Q_init=self.V.detach().numpy()) for a in range(na)]
        self.priors = model.predict_pi(index[None,]).flatten()
    
    def select(self,c=1.5):
        ''' Select one of the child actions based on UCT rule '''
        priors = self.priors.detach().numpy()
        UCT = np.array([child_action.Q + prior * c * (np.sqrt(self.n + 1)/(child_action.n + 1)) for child_action,prior in zip(self.child_actions,priors)]) 
        winner = argmax(UCT)
        return self.child_actions[winner]

    def evaluate(self):
        ''' Bootstrap the state value '''
#        print(self.index)
        self.V = np.squeeze(self.model.predict_V(self.index[None,])) if not self.terminal else torch.Tensor(np.array(0.0))          

    def update(self):
        ''' update count on backward pass '''
        self.n += 1
        
class MCTS():
    ''' MCTS object '''

    def __init__(self,root,root_index,model,na,gamma):
        self.root = None
        self.root_index = root_index
        self.model = model
        self.na = na
        self.gamma = gamma
    
    def search(self,n_mcts,c,Env):
        ''' Perform the MCTS search from the root '''
        if self.root is None:
            self.root = State(self.root_index,r=0.0,terminal=False,parent_action=None,na=self.na,model=self.model) # initialize new root
        else:
            self.root.parent_action = None # continue from current root
        if self.root.terminal:
            raise(ValueError("Can't do tree search from a terminal state"))

        for i in range(n_mcts):
            # print(f"n_mcts {i}")
            state = self.root # reset to root for new trace
            mcts_env = copy.deepcopy(Env) # copy original Env to rollout from
#            print ("(Re)Starting MCTS episode <s>", mcts_env.state)

            while not state.terminal: 
                action = state.select(c=c)
                s1,r,terminated,truncated,_ = mcts_env.step(action.index)
#                print ("MCTS step <a, s, r, t>", action.index, s1, r, t)
                if hasattr(action,'child_state'):
    # This goes by the ACTION having children, not the state; involves the question of whether we can deal with _continuous_ states where 
    # we never get the exact same state again;  this is NOT a problem; as long as the env is deterministic, the same action from the same
    # state always reaches the same next state, so indeed we CAN reuse the MCTS tree within an episode 
    # (noting that each episode we build a brand new tree, so a new initial state is not a problem)
                    state = action.child_state # select
#                    print ("existing state" , state.index)
                    continue
                else:
                    state = action.add_child_state(s1,r,terminated or truncated,self.model) # expand
        #This makes a new State, which uses the network to predict pi and V but then ends this iteration of the search
        # This is where we could do a "rollout" of some length using pi, and backing out the "reward + value(from the V-net)" from a deeper level
                    # print ("expand new child state", state.index)
                    break # note this ends episode.
            # if state.terminal:
            #     print("state is terminal", state.index)

            # Back-up 
            R = state.V         
            while state.parent_action is not None: # loop back-up until root is reached
                R = state.r + self.gamma * R 
                action = state.parent_action
                action.update(R)
                state = action.parent_state
                state.update()                
    
    def return_results(self,temp):
        ''' Process the output at the root node '''
        counts = np.array([child_action.n for child_action in self.root.child_actions])
        Q = np.array([child_action.Q for child_action in self.root.child_actions])
        pi_target = stable_normalizer(counts,temp)
        V_target = np.sum((counts/np.sum(counts))*Q)[None]
        return self.root.index,pi_target,V_target
    
    def forward(self,a,s1):
        ''' Move the root forward '''
        if not hasattr(self.root.child_actions[a],'child_state'):
            self.root = None
            self.root_index = s1
        elif np.linalg.norm(self.root.child_actions[a].child_state.index.astype(np.float32, copy=False) - s1.astype(np.float32, copy=False)) > 0.01:
            # point here is that S1 is the result of actually taking the action in the env, whereas
            # self.root.child_actions[a].child_state.index is the state reached DURING MCTS when we tried that action
            # in the root state; they should be the same in a non-stochastic environment.  Since this (the contruction of a MCTS tree 
            # that is shared between steps) all happens DURING
            # a SINGLE episode, it doesn't matter if the env is randomly initialized in reset().
            print('Warning: this domain seems stochastic. Not re-using the subtree for next search. '+
                  'To deal with stochastic environments, implement progressive widening.')
            self.root = None
            self.root_index = s1            
        else:
            self.root = self.root.child_actions[a].child_state

    def dump_tree(self):
        def recursive_part(state, level):
            sp = ' '
            print(sp*level*3 + "state", state.index, state.n, "T" if state.terminal else " ")
            for child in state.child_actions:
                print (sp*level*3 + "child action", child.index, child.n, child.W, child.Q)
                if hasattr(child,'child_state'):
                    recursive_part(child.child_state, level+1)
        
        print ("DUMP tree")
        recursive_part(self.root, 0)



def test_model(model, env, plot=False):
    nsamples = 100
    xs = []
    Vs = []
    Vstar = []
    ncorrect = 0
    for i in range(nsamples):
        state = env.observation_space.sample()
        teststate = state[None,]
        V = model.predict_V(teststate).detach().numpy()
        pi = model.predict_pi(teststate).detach().numpy()
        correct_val = len(state.flatten()) - state.sum()
        best_pi = argmax(pi)
        correct_pi = state.flatten()[best_pi] == 0
        ncorrect +=  correct_pi
        if (plot):
            print (i, V[0], correct_val, V[0]-correct_val, "                ", correct_pi)
        xs.append(i)
        Vs.append(V[0])
        Vstar.append(correct_val)

    if plot:
        fig,ax = plt.subplots(1,figsize=[7,5])

        ax.scatter(xs,Vs, color='red')
        ax.scatter(xs,Vstar, color='green')

        ax.set_ylabel('Value')
        ax.set_xlabel('TestSample',color='darkred')
        plt.savefig(os.getcwd()+'/val_fn_test.png',bbox_inches="tight",dpi=300)

    Vs = np.array(Vs).flatten()
    Vstar = np.array(Vstar).flatten()
    prop_correct = ncorrect/nsamples
    val_err_vect = Vs - Vstar
    val_err = np.linalg.norm(val_err_vect)
    #mine = np.sqrt(sum([(Vs[i] - Vstar[i])**2 for i in range(nsamples)]))
    #me2 = np.sqrt(val_err_vect.dot(val_err_vect))
    #print (prop_correct, val_err, mine, Vs.shape, Vstar.shape, val_err_vect.shape)
    return prop_correct, val_err

#### Agent ##
def agent(game,n_ep,n_mcts,max_ep_len,lr,c,gamma,data_size,batch_size,temp,n_hidden_layers,n_hidden_units, nsites):
    ''' Outer training loop '''
    #tf.reset_default_graph()
    episode_returns = [] # storage
    timepoints = []
    # Environments
    if game.lower()[0:3] != "net":
        Env = make_game(game)
    else:
        Env = make_netgame(nsites=nsites)

    D = Database(max_size=data_size,batch_size=batch_size)        
    model = Model(Env=Env,lr=lr,n_hidden_layers=n_hidden_layers,n_hidden_units=n_hidden_units)  
    t_total = 0 # total steps   
    R_best = -np.inf
 
    for ep in range(n_ep):    
        start = time.time()
        R = 0.0 # Total return counter
        a_store = []
        seed = myseed  #np.random.randint(1e7) # draw some Env seed
        s = Env.reset(seed=seed)[0]

        mcts = MCTS(root_index=s,root=None,model=model,na=model.action_dim,gamma=gamma) # the object responsible for MCTS searches                             
        for t in range(max_ep_len):
            # print(t)
            # MCTS step
            mcts.search(n_mcts=n_mcts,c=c,Env=Env) # perform a forward search
            state,pi,V = mcts.return_results(temp) # extract the root output
            # print ("Results returned", state, pi, V)
            # mcts.dump_tree()  # This is very illuminating!
            D.store((state,V,pi))

            # Make the true step
            a = np.random.choice(len(pi),p=pi)
            a_store.append(a)
            s1,r,terminated,truncated,_ = Env.step(a)
            r = float(r)
            # test: does NN predict same action?
            netpi = model.predict_pi(state[None,])
            netpi = netpi.detach().numpy()
            neta = argmax(netpi)
            mctsa = argmax(pi)

          #  print ("True step <a,s,r,t,pi, neta, mctsa>", a, s1, r, terminated or truncated, pi, netpi, mctsa, neta)
            R += r
            t_total += n_mcts # total number of environment steps (counts the mcts steps)                

            if terminated or truncated:
                break
            else:
                mcts.forward(a,s1)
        
        # Finished episode
        episode_returns.append(R) # store the total episode return
        timepoints.append(t_total) # store the timestep count of the episode return
        store_safely(os.getcwd(),'result',{'R':episode_returns,'t':timepoints})  

        if R > R_best:
            a_best = a_store
            seed_best = seed
            R_best = R
        
        # Train
        D.reshuffle()
        for epoch in range(1):
            for sb,Vb,pib in D:
                model.train_once(sb,Vb,pib)
        if game.lower()[0:3] == "net":
            pi_correct, val_err = test_model(model, Env, plot=False)
            print('Finished episode {}, total return: {}, total time: {} sec,  prop_pi_correct: {},   val_err: {}'.format(
                ep,np.round(R,2),np.round((time.time()-start),1), pi_correct, val_err))
        else:
            print('Finished episode {}, total return: {}, total time: {} sec'.format(
                ep,np.round(R,2),np.round((time.time()-start),1)))

    pi_correct, val_err = test_model(model, Env, plot=True)

    # Return results
    return episode_returns, timepoints, a_best, seed_best, R_best

#### Command line call, parsing and plotting ##
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
#    parser.add_argument('--game', default='CartPole-v0',help='Training environment')
    parser.add_argument('--game', default='net',help='Training environment')
    parser.add_argument('--n_ep', type=int, default=500, help='Number of episodes')
    parser.add_argument('--n_mcts', type=int, default=40, help='Number of MCTS traces per step')
    parser.add_argument('--max_ep_len', type=int, default=300, help='Maximum number of steps per episode')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--c', type=float, default=1.5, help='UCT constant')
    parser.add_argument('--temp', type=float, default=1.0, help='Temperature in normalization of counts to policy target')
    parser.add_argument('--gamma', type=float, default=1.0, help='Discount parameter')
    parser.add_argument('--data_size', type=int, default=1000, help='Dataset size (FIFO)')
    parser.add_argument('--batch_size', type=int, default=32, help='Minibatch size')
    parser.add_argument('--window', type=int, default=10, help='Smoothing window for visualization')
    parser.add_argument('--nsites', type=int, default=20, help='size of grid to play netgame on')

    parser.add_argument('--n_hidden_layers', type=int, default=2, help='Number of hidden layers in NN')  
    parser.add_argument('--n_hidden_units', type=int, default=128, help='Number of units per hidden layers in NN')

    
    args = parser.parse_args()
    episode_returns,timepoints,a_best,seed_best,R_best = agent(game=args.game,n_ep=args.n_ep,n_mcts=args.n_mcts,
                                        max_ep_len=args.max_ep_len,lr=args.lr,c=args.c,gamma=args.gamma,
                                        data_size=args.data_size,batch_size=args.batch_size,temp=args.temp,
                                        n_hidden_layers=args.n_hidden_layers,n_hidden_units=args.n_hidden_units,
                                        nsites = args.nsites)

    # Finished training: Visualize
    fig,ax = plt.subplots(1,figsize=[7,5])
    total_eps = len(episode_returns)
    episode_returns = smooth(episode_returns,args.window,mode='valid') 
    ax.plot(symmetric_remove(np.arange(total_eps),args.window-1),episode_returns,linewidth=4,color='darkred')
    ax.set_ylabel('Return')
    ax.set_xlabel('Episode',color='darkred')
    plt.savefig(os.getcwd()+'/learning_curve.png',bbox_inches="tight",dpi=300)
    


#     print('Showing best episode with return {}'.format(R_best))
# #    Env = make_game(args.game)
#     Env = make_netgame()
#     Env = wrappers.Monitor(Env,os.getcwd() + '/best_episode',force=True)
#     for runs in range(1000):
#         Env.reset()
#         Env.seed(seed_best)
#         done = False
#         i = 0
#         while not done:
#             a = a_best[i]
#             _,_,done,trunc,_ = Env.step(a)
#             done = done or trunc
#             Env.render()
#             i += 1
