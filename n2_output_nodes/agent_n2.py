import sys
sys.path.append('../Enviroment')
sys.path.append('../Network')
sys.path.append('../Agent')
sys.path.append('../Tree')
sys.path.append('../')
from helper import *
from mutation_tree import MutationTree
from Enviroment import MutTreeEnv
from Network import DQN
from agent import Agent

import numpy as np
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as learningrate_scheduler
import torch.nn as nn
from itertools import count
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Agent_N2_Nodes(Agent):
    def __init__(self, n_mut, n_cells, alpha, beta, device = "cuda"):
        super().__init__(device)
        self.env = MutTreeEnv(alpha=alpha, beta=beta)
        tree_dim = n_mut*(n_mut+1)
        mat_dim = n_mut*n_cells
        out_dim = n_mut*(n_mut+1)
        self.n_mut = n_mut
        self.n_cells = n_cells
        self.policy_net = DQN(tree_dim, mat_dim, out_dim).to(self.device)
        self.target_net = DQN(tree_dim, mat_dim, out_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def get_state_actions(self, tree, data):
        state = torch.tensor(tree.ancestor_matrix.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)
        actions = ~torch.tensor(tree.all_possible_spr.flatten(), dtype=torch.bool, device=self.device).unsqueeze(0)
        return state, actions
    
    def transform_matrix(self, matrix):
        return torch.tensor(matrix.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0) 
    
    def transform_action(self, action, actions, possible_actions):
        i = int(action.item() // (self.n_mut+1))
        j = int(action.item() % (self.n_mut+1))
        return action, [i,j]

    def predict_step(self, state, actions, matrix):
        with torch.no_grad(): 
            q_vals = self.policy_net(state, matrix)
            q_vals = q_vals.masked_fill(actions, float('-inf'))
            return torch.argmax(q_vals, dim=1)
        
    def predict_step_soft(self, state, actions, matrix, temperature):
        with torch.no_grad():
            q_vals = self.policy_net(state, matrix)
            q_vals = q_vals.masked_fill(actions, float('-inf'))
            probs = torch.softmax(q_vals.view(-1) / temperature, dim=0)
            action = torch.multinomial(probs, num_samples=1)
            return action.view(1, 1)

    def predict_step_epsilon(self, state, actions, matrix, eps_threshold):

        sample = random.random()

        if sample > eps_threshold:
            with torch.no_grad():
                q_vals = self.policy_net(state, matrix)
                q_vals = q_vals.masked_fill(actions, float('-inf'))
                return torch.argmax(q_vals, dim=1).unsqueeze(0)
        else:
            poss_spr = torch.nonzero(actions.squeeze(), as_tuple=False).squeeze()
            return torch.tensor([[poss_spr[torch.randint(0, poss_spr.size(0), (1,))]]], device=self.device, dtype=torch.long)

    def predict_step_epsilon_soft(self, state, actions, matrix, temperature, eps_threshold):

        sample = random.random()
        if sample > eps_threshold:
            with torch.no_grad():
                q_vals = self.policy_net(state, matrix)
                q_vals = q_vals.masked_fill(actions, float('-inf'))
                probs = torch.softmax(q_vals.view(-1) / temperature, dim=0)
                action = torch.multinomial(probs, num_samples=1)
                return action.view(1, 1)
        else:
            poss_spr = torch.nonzero(actions.squeeze(), as_tuple=False).squeeze()
            return torch.tensor([[poss_spr[torch.randint(0, poss_spr.size(0), (1,))]]], device=self.device, dtype=torch.long)

    def get_state_action_values(self, state, action, matrix):        
        return self.policy_net(state, matrix).gather(1, action)
    
    def get_max_next_state_action_values(self, next_state, next_actions, matrix):
        next_state = torch.cat(next_state)
        next_actions = torch.cat(next_actions)
        with torch.no_grad():
            q_vals = self.policy_net(next_state, matrix)
            q_vals = q_vals.masked_fill(next_actions, float('-inf'))
            next_predicted_actions = q_vals.argmax(dim=1, keepdim=True)
            max_next_state_action_values = self.target_net(next_state, matrix).gather(1, next_predicted_actions).squeeze()
            max_next_state_action_values = torch.clamp(max_next_state_action_values, min=-10, max=20)
        return max_next_state_action_values