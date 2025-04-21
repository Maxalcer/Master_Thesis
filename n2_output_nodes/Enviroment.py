import sys
sys.path.append('../Tree')

from mutation_tree import MutationTree
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

def scale(x):
    if x > 0:
        x += 10
        return np.log(x)
    else:
        x -= 10
        return -np.log(-x)

# Input: Ground Truth LLH, num mutations, num cells, input data matrix, fp rate, fn rate, stop eps
class MutTreeEnv(gym.Env):
    def __init__(self, n_mut, n_cells, alpha, beta, eps = 2):
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

        self.observation_space = spaces.Box(low=0, high=1, shape=(n_mut * (n_mut+1),), dtype=np.int8)

        self.action_space = spaces.Discrete((n_mut+1)*n_mut)
    
    def step(self, action_idx):

        i = int(action_idx // (self.n_mut+1))
        j = int(action_idx % (self.n_mut+1))
        all_spr = self.get_valid_actions()

        if all_spr[i,j] == 0:
            return self.get_observation(), -50, False, True  # Invalid move penalty
        
        self.tree.perf_spr(i, j)

        new_llh = self.tree.conditional_llh(self.data, self.alpha, self.beta)

        reward = (new_llh - self.current_llh)#/abs(self.gt_llh)
        #reward = np.tanh(reward)
        #reward = 1/(abs(new_llh - self.gt_llh)+ 0.1)
        done = abs(new_llh - self.gt_llh) < self.eps
        if done: reward = 100
        self.current_llh = new_llh

        return self.get_observation(), reward, done, False
    
    def reset(self, gt_llh, data):
        self.data = data
        self.gt_llh = gt_llh
        self.tree = MutationTree(self.n_mut, self.n_cells)
        pvec = np.repeat(self.n_mut, self.n_mut + 1)
        pvec[-1] = -1
        self.tree.use_parent_vec(pvec, 5)
        self.current_llh = self.tree.conditional_llh(self.data, self.alpha, self.beta)
        return self.get_observation()
    
    def get_observation(self):
        A_T = self.tree.ancestor_matrix
        return A_T.flatten()
    
    def get_valid_actions(self):
        return self.tree.all_spr()
    
    def render(self):
        self.tree.to_graphviz("render.png")