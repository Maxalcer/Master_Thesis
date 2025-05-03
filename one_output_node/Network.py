import random
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, tree_size, mutation_size, hidden_size=128, layers = 5):
        super(DQN, self).__init__()
        self.tree_features = nn.ModuleList()
        self.mutation_features = nn.ModuleList()
        self.combined = nn.ModuleList()

        self.tree_features.append(nn.Linear(tree_size, hidden_size))
        self.mutation_features.append(nn.Linear(mutation_size, hidden_size))
        self.combined.append(nn.Linear(hidden_size*3, hidden_size))

        for _ in range(layers):
            self.tree_features.append(nn.Linear(hidden_size, hidden_size))
            self.mutation_features.append(nn.Linear(hidden_size, hidden_size))
            self.combined.append(nn.Linear(hidden_size, hidden_size))

        self.out = nn.Linear(hidden_size, 1)

    def forward(self, tree_x, mutation_x):
        for layer in self.tree_features:
            tree_x = F.relu(layer(tree_x))

        for layer in self.mutation_features:
            mutation_x = F.relu(layer(mutation_x))
        
        tree_std = tree_x.std(dim=0)             # std across actions for each feature
        tree_variation = tree_std.mean().item()  # overall average feature variation

        print(f"Tree feature variation across actions: {tree_variation:.6f}")

        x = torch.cat((tree_x, mutation_x, tree_x*mutation_x), dim=1)

        for layer in self.combined:
            x = F.relu(layer(x))
        return self.out(x)*10
    
Transition = namedtuple('Transition', ('state', 'action', 'matrix', 'next_state_actions', 'reward', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)