import sys
sys.path.append('../Enviroment')
sys.path.append('../Network')
sys.path.append('../Agent')
sys.path.append('../Tree')
sys.path.append('../')
from helper import *
from mutation_tree import MutationTree
from Enviroment import MutTreeEnv
from Network_GNN_2 import DQN_GNN
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
from torch_geometric.data import Data, Batch

class Agent_GNN(Agent):
    def __init__(self, n_mut, n_cells, alpha, beta, device = "cuda"):
        super().__init__(device)
        self.env = MutTreeEnv(n_mut=n_mut, n_cells=n_cells, alpha=alpha, beta=beta)
        #tree_dim = n_mut+1
        tree_dim = n_cells
        mut_dim = n_mut*n_cells
        out_dim = n_mut*(n_mut+1)
        self.policy_net = DQN_GNN(tree_dim, out_dim).to(self.device)
        self.target_net = DQN_GNN(tree_dim, out_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def get_state_actions(self, tree):
        state = torch.tensor(tree.edge_index, dtype=torch.long, device=self.device).unsqueeze(0)
        actions = ~torch.tensor(tree.all_possible_spr.flatten(), dtype=torch.bool, device=self.device).unsqueeze(0)
        return state, actions
    
    def transform_matrix(self, matrix):
        matrix = np.vstack((matrix, np.ones((1, matrix.shape[1]), dtype=matrix.dtype)))
        return torch.tensor(matrix, dtype=torch.float32, device=self.device).unsqueeze(0) 

    def transform_action(self, action, actions, possible_actions):
        i = int(action // (self.n_mut+1))
        j = int(action % (self.n_mut+1))
        action_indx = [i, j]
        return action, action_indx
    
    """
    def get_graph_data(self, state):
        return [Data(x=torch.eye(self.n_mut+1).to(self.device), edge_index=state[i]).to(self.device) for i in range(state.size(0))]
    
    """
    def get_graph_data(self, state, matrix):
        return [Data(x=matrix[i], edge_index=state[i]).to(self.device) for i in range(state.size(0))]
    

    def predict_step(self, state, actions, matrix):
        #data_list = self.get_graph_data(state)
        data_list = self.get_graph_data(state, matrix)
        data_batch = Batch.from_data_list(data_list).to(self.device)
        with torch.no_grad(): 
            #q_vals = self.policy_net(data_batch.x, data_batch.edge_index, matrix, data_batch.batch)
            q_vals = self.policy_net(data_batch.x, data_batch.edge_index, data_batch.batch)
            q_vals = q_vals.masked_fill(actions, float('-inf'))
            return torch.argmax(q_vals, dim=1)
        
    def predict_step_soft(self, state, actions, matrix, temperature):
        #data_list = self.get_graph_data(state)
        data_list = self.get_graph_data(state, matrix)
        data_batch = Batch.from_data_list(data_list).to(self.device)
        with torch.no_grad():
            #q_vals = self.policy_net(data_batch.x, data_batch.edge_index, matrix, data_batch.batch)
            q_vals = self.policy_net(data_batch.x, data_batch.edge_index, data_batch.batch)
            q_vals = q_vals.masked_fill(actions, float('-inf'))
            probs = torch.softmax(q_vals.view(-1) / temperature, dim=0)
            action = torch.multinomial(probs, num_samples=1)
            return action.view(1, 1)

    def predict_step_epsilon(self, state, actions, matrix, eps_threshold):

        sample = random.random()

        if sample > eps_threshold:
            #data_list = self.get_graph_data(state)
            data_list = self.get_graph_data(state, matrix)
            data_batch = Batch.from_data_list(data_list).to(self.device)
            with torch.no_grad():
                #q_vals = self.policy_net(data_batch.x, data_batch.edge_index, matrix, data_batch.batch)
                q_vals = self.policy_net(data_batch.x, data_batch.edge_index, data_batch.batch)
                q_vals = q_vals.masked_fill(actions, float('-inf'))
                return torch.argmax(q_vals, dim=1).unsqueeze(0)
        else: return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def predict_step_epsilon_soft(self, state, actions, matrix, temperature, eps_threshold):

        sample = random.random()
        if sample > eps_threshold:
            #data_list = self.get_graph_data(state)
            data_list = self.get_graph_data(state, matrix)
            data_batch = Batch.from_data_list(data_list).to(self.device)
            with torch.no_grad():
                #q_vals = self.policy_net(data_batch.x, data_batch.edge_index, matrix, data_batch.batch)
                q_vals = self.policy_net(data_batch.x, data_batch.edge_index, data_batch.batch)
                q_vals = q_vals.masked_fill(actions, float('-inf'))
                probs = torch.softmax(q_vals.view(-1) / temperature, dim=0)
                action = torch.multinomial(probs, num_samples=1)
                return action.view(1, 1)
        else: return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def get_state_action_values(self, state, action, matrix):
        #data_list = self.get_graph_data(state)
        data_list = self.get_graph_data(state, matrix)
        data_batch = Batch.from_data_list(data_list).to(self.device)
        #return self.policy_net(data_batch.x, data_batch.edge_index, matrix, data_batch.batch).gather(1, action)
        return self.policy_net(data_batch.x, data_batch.edge_index, data_batch.batch).gather(1, action)
    
    def get_max_next_state_action_values(self, next_state, next_actions, matrix):
        #data_list = self.get_graph_data(next_state)
        data_list = self.get_graph_data(next_state, matrix)
        data_batch = Batch.from_data_list(data_list).to(self.device)
        with torch.no_grad():
            #q_vals = self.policy_net(data_batch.x, data_batch.edge_index, matrix, data_batch.batch)
            q_vals = self.policy_net(data_batch.x, data_batch.edge_index, data_batch.batch)
            q_vals = q_vals.masked_fill(next_actions, float('-inf'))
            next_predicted_actions = q_vals.argmax(dim=1, keepdim=True)
            #max_next_state_action_values = self.target_net(data_batch.x, data_batch.edge_index, matrix, data_batch.batch).gather(1, next_predicted_actions).squeeze()
            max_next_state_action_values = self.target_net(data_batch.x, data_batch.edge_index, data_batch.batch).gather(1, next_predicted_actions).squeeze()
            max_next_state_action_values = torch.clamp(max_next_state_action_values, min=-10, max=25)
        return max_next_state_action_values