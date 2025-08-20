import sys
sys.path.append('../Agent')
sys.path.append('../Network')
sys.path.append('../Enviroment')
sys.path.append('../Tree')
sys.path.append('../')
from helper import read_data, read_newick, Scheduler
from mutation_tree import MutationTree
from Enviroment import MutTreeEnv
from Network import DQN
from agent import Agent
import params as P

import numpy as np
import math
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as learningrate_scheduler
import torch.nn as nn
from itertools import count
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Agent_Features(Agent):
    def __init__(self, n_mut, n_cells, alpha, beta, device = "cuda"):
        super().__init__(device)
        self.env = MutTreeEnv(alpha=alpha, beta=beta)
        dim = n_mut*n_cells
        self.n_mut = n_mut
        self.n_cells = n_cells
        self.policy_net = DQN(18, dim, 1).to(self.device)
        self.target_net = DQN(18, dim, 1).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
    
    def get_state_actions(self, tree, data):
        all_vectors = tree.feature_vectors_sub(data, self.alpha, self.beta)
        all_vectors = torch.tensor(all_vectors, dtype=torch.float32, device=self.device)
        return all_vectors, all_vectors
    
    def transform_matrix(self, matrix):
        return torch.tensor(matrix.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0) 

    def predict_step(self, state, actions, matrix):
        matrix = matrix.repeat(state.shape[0], 1)
        with torch.no_grad(): q_vals = self.policy_net(state, matrix)  
        return torch.argmax(q_vals)
    
    def predict_step_soft(self, state, actions, matrix, temperature):
        with torch.no_grad():
            matrix = matrix.repeat(state.shape[0], 1)
            q_vals = self.policy_net(state, matrix)  
            probs = torch.softmax(q_vals.view(-1) / temperature, dim=0)
            action = torch.multinomial(probs, num_samples=1)
            return action.view(1, 1)

    def predict_step_epsilon(self, state, actions, matrix, eps_threshold):

        sample = random.random()

        if sample > eps_threshold: 
            return self.predict_step(state, matrix)
        else: return torch.tensor([[random.randint(0, state.shape[0] -1)]], device=self.device, dtype=torch.long)
    
    def transform_action(self, action, actions, possible_actions):
        i = int(action.item() // (self.n_mut+1))
        j = int(action.item() % (self.n_mut+1))
        return actions[action], [i,j]

    def predict_step_epsilon_soft(self, state, actions, matrix, eps_threshold, temperature):
        sample = random.random()
        if sample > eps_threshold:
            with torch.no_grad():
                matrix = matrix.repeat(state.shape[0], 1)
                q_vals = self.policy_net(state, matrix)  
                probs = torch.softmax(q_vals.view(-1) / temperature, dim=0)
                action = torch.multinomial(probs, num_samples=1)
                return action.view(1, 1)
        else: return torch.tensor([[random.randint(0, state.shape[0] -1)]], device=self.device, dtype=torch.long)

    def get_state_action_values(self, state, action, matrix): 
        return self.policy_net(action.squeeze(1), matrix).squeeze()

    def get_max_next_state_action_values(self, next_state, next_actions, matrix):
        num_actions_per_item = [state.shape[0] for state in next_state]
        expanded_mutation_matrix = torch.cat([matrix[i].unsqueeze(0).repeat(n, 1) for i, n in enumerate(num_actions_per_item)], dim=0).to(self.device)
        next_state = torch.cat(next_state)
        max_next_state_action_values = []

        with torch.no_grad(): q_vals = self.target_net(next_state, expanded_mutation_matrix)

        start_idx = 0
        for i in range(P.BATCH_SIZE):
            q_vals_state = q_vals[start_idx:start_idx + num_actions_per_item[i]] 
            max_next_state_action_values.append(q_vals_state.max().item())  
            start_idx += num_actions_per_item[i]
            
        max_next_state_action_values = torch.tensor(max_next_state_action_values, dtype=torch.float32, device=self.device)
        max_next_state_action_values = torch.clamp(max_next_state_action_values, min=-10, max=20)
        return max_next_state_action_values