# grammar game stuff

import gymnasium as gym
import random
from gymnasium import error, spaces, utils
import numpy as np

from nltk import CFG, Production, nonterminals, Nonterminal
from nltk.parse.generate import generate
from nltk.parse.recursivedescent import RecursiveDescentParser


        # self.grammar = CFG.fromstring(
        #     """
        # S -> NP VP
        # PP -> P NP
        # NP -> Det N | NP PP
        # VP -> V NP | VP PP
        # Det -> 'a' | 'the'
        # N -> 'dog' | 'cat'
        # V -> 'chased' | 'sat'
        # P -> 'on' | 'in'
        # """
        # )

        # self.grammar = CFG.fromstring(
        #     """
        # S ->  I L R
        # I -> AV '=' A ';' VV '=' V ';' 

        # #   eol -> ';'
        # #   eoc -> ':'

        # L -> 'while' V C '0' ':' E ';' 'end' 
        # E ->  AV '=' A | VV '=' V | E ';' E
        # AV -> 'x' | 'y' | 'z'
        # VV -> 'a' | 'b' | 'c'
        # C -> '>'  
        # A ->  A2A1 '(' AV ')' | A2A2 '(' VV ',' AV ')' | '[]' | A '+' A
        # A2A1 -> 'head'
        # A2A2 -> 'prepend'
        # V -> A2V '(' AV ')' |  V '+' V | '0'
        # A2V -> 'len' | 'tail' | 'mod10' | 'div10'
        
        # R -> AV '=' A2A2 '(' VV ',' AV ')' ';' 'return' AV
        # """
        # )

class GrammarEnv(gym.Env):
    def __init__(self, grammarstr, desired_state_len=None):
        self.desired_state_len = desired_state_len
        self.init_grammar(grammarstr, desired_state_len = None)
        self.observation_space = spaces.MultiDiscrete([self.nsym for i in range(self.state_len)])
        self.action_space = spaces.MultiDiscrete([self.state_len, self.nprods])
        self.reset()
        self.parser = RecursiveDescentParser(self.grammar)

    def init_grammar(self, grammarstr, desired_state_len=None):
        self.grammar = CFG.fromstring(grammarstr)
        self.grammarstr = grammarstr

        idx = 0
        self.symdict = {}  # from strings to indices, which we will use as tokens
        self.tokenlist = [] # from index back to the string for the symbol
        self.nonterms = [] # token index of nonterms
        self.productions = [] # tokenized global list of productions
        self.proddict = {} # tokenized, dictionary version of productions on per-nonterm basis; entries are index into global list
        self.maxprods = 0 # maximum number of productions for any one nonterminal; for how big action space needs to be
        prods = self.grammar.productions()
        self.nprods = len(prods) # total number of prods for "n X p" approach to action space (all nonterms x all productions)
        for p in prods:
            self.maxprods = max(self.maxprods, len(p))
            lhs = p._lhs
            if lhs._symbol not in self.symdict.keys():
                self.symdict[lhs._symbol] = idx
                self.tokenlist.append(lhs._symbol)
                idx += 1
            if self.symdict[lhs._symbol] not in self.proddict.keys():
                self.proddict[self.symdict[lhs._symbol]] = []

            ptok = []
            for s in p._rhs:
                if isinstance(s, Nonterminal):
                    s = s._symbol
                if s not in self.symdict.keys():
                    self.symdict[s] = idx
                    self.tokenlist.append(s)
                    idx += 1
                ptok.append(self.symdict[s])
            self.proddict[self.symdict[lhs._symbol]].append(len(self.productions))
            self.productions.append(ptok)

        for p in prods:
            lhs = p._lhs
            if self.symdict[lhs._symbol] not in self.nonterms:
                self.nonterms.append(self.symdict[lhs._symbol])

        self.nsym = idx # number of real symbols
        self.pad_tok = idx  # use this as filler
        self.nnonterms = len(self.nonterms)
        if self.desired_state_len is not None:
            self.state_len = self.desired_state_len
        else:
            self.state_len = self.nsym * 5 # arbitrary; this is where padding and masking, etc comes in

        # build action mask... should be n x p binary mask, where n is max sentence length, p is number of possible prodcutions
        # ...self.action_mask = np.zeros((self.state_len, self.nnonterms))
        # unfortunately, the exact mask is state-dependent!... so we will have a function instead

# It is a difficult question to figure out how close to "full" we let the state get because the productions add variable
# numbers of tokens to the state

    def get_action_mask(self, state=None):
        if state is None:
            state = self.state
            ll = self.real_state_len
        else: 
            statelist =  state.tolist()
            if self.pad_tok not in statelist:
                ll = len(statelist)
                print ("BAD (FULL or OVERFLOWED) STATE")
            else: 
                ll = statelist.index(self.pad_tok) # real length
        mask = np.zeros((self.state_len, self.nprods))   # assume all actions invalid
        for i in range(ll):
            if state[i] in self.nonterms:
                prods = self.proddict[state[i]]
                for j in prods:
                    mask[i, j] = 1  #  valid ones are (nonterm-specific) productions for nonterms in state.
                    # prevent overflow before it happens:
                    if ll + (len(self.productions[j])-1)  >= self.state_len:  # if cur len + additional tokens do to this production > length of state buffer
                        # if jth production will cause the state to overflow
                        mask[i,j] = 0
        return mask
        

    def has_nonterms(self, state):
        for i in range(len(state)):
            if state[i] in self.nonterms:
                return True
            if state[i] == self.pad_tok:
                return False
        return False

    def decode_state(self, state):
        s = ""
        for i in range(len(state)):
            if state[i] == self.pad_tok:
                return s
            s += '  %s' % self.tokenlist[state[i]]
        return s

    def step(self, action):
        r = 0
        done = False
        ntinstate = action[0]  ### position (index) of nonterminal in state vector; we will replace it with one of its productions
        nontermidx = self.state[ntinstate]
#        prodidx = action[1] ### index of the production (in the possible prods for the chosen nonterminal) that we will replace it with
        prodidx = action[1] ### index of the production (in global list of all productions) that we will replace it with
        rhs = self.productions[prodidx]
        # shift by diff in len: (I realized later this could be replace by np.insert(), pretty sure?)
        ll = len(rhs) 
#        self.state = np.insert(self.state, ntinstate, rhs)  ## this doesn't replace lhs!
        if self.real_state_len + ll - 1 > self.state_len:
            # overflow!just declare episode over even though there are still non-terms,
            # and return an empty rule
            done = True
            rule = None
            r = -1 #??
        else:
            if ll > 0:
                self.state[ntinstate+ll:-1] = self.state[ntinstate+1:-ll]
                self.state[ntinstate:ntinstate+ll] =  rhs[:]
            else:
                self.state[ntinstate:-2] = self.state[ntinstate+1:-1]
                self.state[self.real_state_len - 1:-1] = self.pad_tok  # make sure last shifted symbol not left behind!

            done = not self.has_nonterms(self.state)
            self.real_state_len += ll - 1 # keep track of how much is just padding 
            rule = self.decode_state(self.state) if done else None
        # if rule is not None:
        #     print ("STEP", rule)
        # print ("STEP", action, self.state, self.decode_state(self.state), done)   
        return self.state, r, done, False, {'rule':rule}
    
    def step_with_mask(self, action):
        mask = self.get_action_mask()
#        print ("MASK", mask)
        if mask[action[0], action[1]]:
            return self.step(action)
        else:
            print ("INVALID", action, self.state)
            return self.state, 0, False, False, {}
        
    def decode_and_step(self, action):
        # action is index into flattened (n-bits x n-productions) array
        a = [action // self.action_space.nvec[1], action % self.action_space.nvec[1]]
        return self.step_with_mask(a)

    def reset(self, seed = None):
        self.state = [self.pad_tok for _ in range(self.observation_space.shape[0])]
        self.state[0] = self.symdict['S']
        self.state = np.array(self.state)
        self.real_state_len = 1
        return self.state


class GrammarAgent():
    def __init__(self, game_env, grammar_env):
        self.grammar_env = grammar_env
        self.game_env = game_env

    def random_action(self):
        # find non-terminal
        found = False
        i = 0
        state = self.grammar_env.state
        nonterms = []
#        while i < self.grammar_env.nsym and i < len(state):
        while i < len(state):
            if state[i] in self.grammar_env.nonterms:
                nonterms.append(i)
                found = True
            i += 1
        if not found:
            return -1  # no non-terms, rule is complete, we shouldn't be here.
        
        # pick random nonterm and random production
        nonterminstate = random.choice(nonterms) # note, INDEX of nonterm in state
        nonterm = state[nonterminstate]
        nprods = len(self.grammar_env.productions[nonterm])  ####  NEED token, not it's index in state, to get to right productions
        p = random.randrange(nprods)  # note, INDEX of production, not the list of rhs terms themselves
        act = [nonterminstate, p]
        return act

    def truly_random_action(self):
#        act = [random.randrange(self.grammar_env.state_len), random.randrange(self.grammar_env.nprods)]
        act = [random.randrange(self.grammar_env.real_state_len), random.randrange(self.grammar_env.nprods)]
        return act

    def random_valid_action(self):
        # find non-terminal
        found = False
        i = 0
        state = self.grammar_env.state
        nonterms = []
#        while i < self.grammar_env.nsym and i < len(state):
        while i < len(state):
            if state[i] in self.grammar_env.nonterms:
                nonterms.append(i)
                found = True
            i += 1
        if not found:
            return -1  # no non-terms, rule is complete, we shouldn't be here.
        
        # pick random nonterm and random production
        nonterminstate = random.choice(nonterms) # note, INDEX of nonterms in state
        nonterm = state[nonterminstate]  # actual non term in question
        prods = self.grammar_env.proddict[nonterm]  ####  NEED token, not it's index in state, to get to right productions
            # from the proddict we get the list of GLOBAL production indices that are possible for this nonterm.
        p = random.choice(prods)  # p is now the index in the GLOBAL list (of ALL productions) corresponding to one that
            # is allowed for the chosen nonterm.  note, it is the INDEX of production, not the list of rhs terms themselves
        act = [nonterminstate, p]
        return act

    def generate_one(self, grammar, item, depth):
        # generates one random full "sentence" (nothing but terminals) from the given grammar
        if depth > 0:
            if isinstance(item, Nonterminal):
                prods = grammar.productions(lhs=item)
                print ("LHS", item, prods)
                prod = random.choice(prods) # uniform sampling of productions; the only randomness here
                rhs = prod.rhs()
                rhslist = []
                for p in rhs:
                    frag = self.generate_one(grammar, p, depth - 1) # recurse: rhs fragments as lhs's
                    rhslist += frag
                return rhslist
            else:
                return [item]
        else:
            return [item]


    def apply_rules(self, rules):
        # (pure virtual function)
        # run the rules ('program') generated by a grammar, return an action in the sub-environment in which the rules are 
        # meant to describe optimal action
        raise NotImplementedError


