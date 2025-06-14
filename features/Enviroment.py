import sys
sys.path.append('../Tree')

from mutation_tree import MutationTree
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

# Input: Ground Truth LLH, num mutations, num cells, input data matrix, fp rate, fn rate, stop eps
class MutTreeEnv(gym.Env):
    def __init__(self, n_mut, n_cells, alpha, beta, eps = 1):
        super(MutTreeEnv, self).__init__()
        
        self.n_mut = n_mut
        self.n_cells = n_cells
        self.tree = None
        self.eps = eps
        self.gt_llh = None # ground truth likelihood
        self.current_llh = None
        self.data = None
        self.alpha = alpha
        self.beta = beta
        self.all_spr = None

        self.observation_space = spaces.Box(low=0, high=1, shape=(18,), dtype=np.int8)

        self.action_space = spaces.Discrete(1)

    def step(self, action_idx):

        spr = self.all_spr[action_idx]
        
        self.tree.perf_spr(spr[0], spr[1])

        new_llh = self.tree.conditional_llh(self.data, self.alpha, self.beta)

        new_llh = self.tree.conditional_llh(self.data, self.alpha, self.beta)
        if (self.alpha == 0) and (self.beta == 0):
            reward = 30*(new_llh - self.current_llh)
            done = (round(new_llh, 2) == 1)
        else:
            reward = (new_llh - self.current_llh)/abs(self.gt_llh)
            done = abs(new_llh - self.gt_llh) < self.eps
            
        if done: reward = 25
        self.current_llh = new_llh
        self.all_spr = self.get_valid_actions()
        self.action_space = spaces.Discrete(len(self.all_spr))
        return (self.get_observation(),
                self.all_spr,
                reward, 
                done)
    
    def reset(self, gt_llh, data):
        self.data = data
        self.gt_llh = gt_llh
        self.tree = MutationTree(self.n_mut, self.n_cells)
        pvec = np.repeat(self.n_mut, self.n_mut + 1)
        pvec[-1] = -1
        self.tree.use_parent_vec(pvec, 5)
        self.current_llh = self.tree.conditional_llh(self.data, self.alpha, self.beta)
        self.all_spr = self.get_valid_actions()
        self.action_space = spaces.Discrete(len(self.all_spr))
        return self.get_observation(), self.all_spr
        
    def get_observation(self):
        return self.tree

    def get_valid_actions(self):
        return self.tree.all_spr()
    
    def render(self):
        self.tree.to_graphviz("render.png")