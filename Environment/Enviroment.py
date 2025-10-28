import sys
sys.path.append('../')
sys.path.append('../Tree')

from mutation_tree import MutationTree
from helper import timeit
import gymnasium as gym
from gymnasium import spaces
import numpy as np
#import torch

# Input: Ground Truth LLH, num mutations, num cells, input data matrix, fp rate, fn rate, stop eps
class MutTreeEnv(gym.Env):
    def __init__(self, alpha, beta, eps = 2):
        super(MutTreeEnv, self).__init__()
        
        self.n_mut = None
        self.n_cells = None
        self.tree = None
        self.eps = eps
        self.gt_llh = None # ground truth likelihood
        self.current_llh = None
        self.data = None
        self.alpha = alpha
        self.beta = beta
    
    def step(self, action_idx, swap):

        #i = int(action_idx // (self.n_mut+1))
        #j = int(action_idx % (self.n_mut+1))
        if swap: self.tree.swap(action_idx)
        else: self.tree.perf_spr(action_idx[0], action_idx[1])

        new_llh = self.tree.conditional_llh(self.data, self.alpha, self.beta)
        if (self.alpha == 0) and (self.beta == 0):
            reward = 30*(new_llh - self.current_llh)
            done = (round(new_llh, 2) == 1)
        else:
            reward = (new_llh - self.current_llh)/abs(self.gt_llh)
            done = abs(new_llh - self.gt_llh) < self.eps
        if done: reward = 20
        self.current_llh = new_llh

        return (self.get_observation(),
                reward, 
                done)
    
    def reset(self, gt_llh, data):
        self.data = data
        self.n_mut = self.data.shape[0]
        self.n_cells = self.data.shape[1]
        self.gt_llh = gt_llh
        self.tree = MutationTree(self.n_mut, self.n_cells)
        pvec = np.repeat(self.n_mut, self.n_mut + 1)
        pvec[-1] = -1
        self.tree.use_parent_vec(pvec, self.n_mut)
        self.current_llh = self.tree.conditional_llh(self.data, self.alpha, self.beta)
        return self.get_observation()
    
    def get_observation(self):
        return self.tree
    
    def get_valid_actions(self):
        pass
    
    def render(self):
        self.tree.to_graphviz("render.png")