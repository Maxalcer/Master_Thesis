import sys
sys.path.append('../Tree')

from mutation_tree import MutationTree
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

def scale(x):
    if x < -0.1: return -1
    elif x > 0.1: return 1
    else: return 0

# Input: Ground Truth LLH, num mutations, num cells, input data matrix, fp rate, fn rate, stop eps
class MutTreeEnv(gym.Env):
    def __init__(self, n_mut, n_cells, alpha, beta, device, eps = 1):
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
        self.device = device

        self.observation_space = spaces.Box(low=0, high=1, shape=(n_mut * (n_mut+1),), dtype=np.int8)

        self.action_space = spaces.Discrete(1)

    def step(self, action_idx):

        spr = self.all_spr[action_idx]
        
        self.tree.perf_spr(spr[0], spr[1])

        new_llh = self.tree.conditional_llh(self.data, self.alpha, self.beta)

        reward = 10*(new_llh - self.current_llh)/abs(self.gt_llh)
        #reward = new_llh - self.gt_llh
        done = abs(new_llh - self.gt_llh) < self.eps
        if done: reward = 50
        #done = False
        self.current_llh = new_llh
        self.all_spr = self.get_valid_actions()
        self.action_space = spaces.Discrete(len(self.all_spr))
        return (self.get_observation(),
                torch.tensor([reward], device=self.device), 
                torch.tensor([bool(done)], device=self.device))
    
    def reset(self, gt_llh, data):
        self.data = data
        self.gt_llh = gt_llh
        self.tree = MutationTree(self.n_mut, self.n_cells)
        pvec = np.repeat(self.n_mut, self.n_mut + 1)
        pvec[-1] = -1
        self.tree.use_parent_vec(pvec, self.n_mut)
        self.current_llh = self.tree.conditional_llh(self.data, self.alpha, self.beta)
        self.all_spr = self.get_valid_actions()
        self.action_space = spaces.Discrete(len(self.all_spr))
        return self.get_observation()
    
    def get_observation(self):
        A_T = self.tree.ancestor_matrix
        return torch.tensor(A_T.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)
    """
    def get_valid_actions(self):
        all_moves = self.tree.all_spr()
        features = [[move[0], move[1], self.tree.distance(move[0], move[1]), self.tree.get_subtree_size(move[0])] for move in all_moves]
        return features
    """   
    def get_valid_actions(self):
        return self.tree.all_spr()
        
    def render(self):
        self.tree.to_graphviz("render.png")