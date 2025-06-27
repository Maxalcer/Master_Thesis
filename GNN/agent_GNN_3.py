import sys
sys.path.append('../Enviroment')
sys.path.append('../Network')
sys.path.append('../Agent')
sys.path.append('../Tree')
sys.path.append('../')
from helper import *
from mutation_tree import MutationTree
from Enviroment import MutTreeEnv
from Network_GNN_3 import DQN_GNN
from agent import Agent
import params as P

import numpy as np
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as learningrate_scheduler
import torch.nn as nn
from itertools import count
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch

class Agent_GNN(Agent):
    def __init__(self, n_mut, n_cells, alpha, beta, device = "cuda"):
        super().__init__(device)
        self.env = MutTreeEnv(n_mut=n_mut, n_cells=n_cells, alpha=alpha, beta=beta)
        mut_dim = n_mut*n_cells
        self.policy_net = DQN_GNN(mut_dim).to(self.device)
        self.target_net = DQN_GNN(mut_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def get_state_actions(self, tree):
        state = torch.tensor(tree.edge_index, dtype=torch.long, device=self.device).unsqueeze(0)
        actions = tree.all_possible_spr
        indices = np.argwhere(actions == 1)
        num_indices = indices.shape[0]
        features = np.zeros((num_indices, 6, 2), dtype=np.float32)
        features[np.arange(num_indices), indices[:, 0], :] = [0, 1]
        features[np.arange(num_indices), indices[:, 1], :] = [1, 0]
        features = torch.tensor(features, dtype=torch.float32, device=self.device)
        ident = torch.eye(self.n_mut+1).repeat(num_indices, 1, 1).to(self.device)
        features = torch.cat([ident, features], dim = -1)
        return state, features
    
    def transform_matrix(self, matrix):
        return torch.tensor(matrix.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0) 

    def get_graph_data(self, state, actions):
        data_list = [Data(x=actions[i], edge_index=state[i]).to(self.device) for i in range(len(actions))]
        return data_list
    
    def predict_step(self, state, actions, matrix):
        state = state.repeat(actions.shape[0], 1, 1)
        data_list = self.get_graph_data(state, actions)
        data_batch = Batch.from_data_list(data_list).to(self.device)
        matrix = matrix.repeat(actions.shape[0], 1)
        with torch.no_grad(): 
            q_vals = self.policy_net(data_batch.x, data_batch.edge_index, matrix, data_batch.batch)
            return torch.argmax(q_vals, dim=0)
        
    def predict_step_soft(self, state, actions, matrix, temperature):
        state = state.repeat(actions.shape[0], 1, 1)
        data_list = self.get_graph_data(state, actions)
        data_batch = Batch.from_data_list(data_list).to(self.device)
        matrix = matrix.repeat(actions.shape[0], 1)
        with torch.no_grad():
            q_vals = self.policy_net(data_batch.x, data_batch.edge_index, matrix, data_batch.batch)
            probs = torch.softmax(q_vals.view(-1) / temperature, dim=0)
            action = torch.multinomial(probs, num_samples=1)
            return action.view(1, 1)

    def predict_step_epsilon(self, state, actions, matrix, eps_threshold):

        sample = random.random()

        if sample > eps_threshold:
            state = state.repeat(actions.shape[0], 1, 1)
            data_list = self.get_graph_data(state, actions)
            data_batch = Batch.from_data_list(data_list).to(self.device)
            matrix = matrix.repeat(actions.shape[0], 1)
            with torch.no_grad():
                q_vals = self.policy_net(data_batch.x, data_batch.edge_index, matrix, data_batch.batch)
                return torch.argmax(q_vals, dim=0).unsqueeze(0)
        else: return torch.tensor([[random.randint(0, actions.shape[0]-1)]], device=self.device, dtype=torch.long)

    def predict_step_epsilon_soft(self, state, actions, matrix, temperature, eps_threshold):
        sample = random.random()
        if sample > eps_threshold:
            state = state.repeat(actions.shape[0], 1, 1)
            data_list = self.get_graph_data(state, actions)
            data_batch = Batch.from_data_list(data_list).to(self.device)
            matrix = matrix.repeat(actions.shape[0], 1)
            with torch.no_grad():
                q_vals = self.policy_net(data_batch.x, data_batch.edge_index, matrix, data_batch.batch)
                probs = torch.softmax(q_vals.view(-1) / temperature, dim=0)
                action = torch.multinomial(probs, num_samples=1)
                return action.view(1, 1)
        else: return torch.tensor([[random.randint(0, actions.shape[0]-1)]], device=self.device, dtype=torch.long)

    def get_state_action_values(self, state, action, matrix):
        #action = action.squeeze(1)
        data_list = self.get_graph_data(state, action)
        data_batch = Batch.from_data_list(data_list).to(self.device)
        return self.policy_net(data_batch.x, data_batch.edge_index, matrix, data_batch.batch).squeeze()
    
    def get_max_next_state_action_values(self, next_state, next_actions, matrix):
        num_actions_per_item = [action.shape[0] for action in next_actions]
        expanded_matrix = torch.cat([matrix[i].unsqueeze(0).repeat(n, 1) for i, n in enumerate(num_actions_per_item)], dim=0)
        expanded_states = torch.cat([next_state[i].repeat(n, 1, 1) for i, n in enumerate(num_actions_per_item)], dim=0)
        next_actions = torch.cat(next_actions)
        data_list = self.get_graph_data(expanded_states, next_actions)
        data_batch = Batch.from_data_list(data_list).to(self.device)
        
        with torch.no_grad(): 
            q_vals = self.policy_net(data_batch.x, data_batch.edge_index, expanded_matrix, data_batch.batch)
            q_vals_split = torch.split(q_vals, num_actions_per_item)
            argmax_indices = [q.argmax() for q in q_vals_split]
            q_vals_target = self.target_net(data_batch.x, data_batch.edge_index, expanded_matrix, data_batch.batch)
            q_vals_target_split = torch.split(q_vals_target, num_actions_per_item)
            max_next_state_action_values = torch.stack([q[i] for q, i in zip(q_vals_target_split, argmax_indices)]).to(self.device)
            max_next_state_action_values = torch.clamp(max_next_state_action_values, min=-10, max=25)
        return max_next_state_action_values.squeeze()