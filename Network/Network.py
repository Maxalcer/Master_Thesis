import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, tree_size, mutation_size, n_actions, layers = 3, hidden_size = 1024):
        super(DQN, self).__init__()
        
        self.tree_features = nn.ModuleList()
        self.mutation_features = nn.ModuleList()
        self.combined = nn.ModuleList()

        self.tree_features.append(nn.Linear(tree_size, hidden_size))
        self.mutation_features.append(nn.Linear(mutation_size, hidden_size))
        self.combined.append(nn.Linear(64*3, hidden_size))

        self.tree_features.append(nn.LayerNorm(hidden_size))
        self.mutation_features.append(nn.LayerNorm(hidden_size))
        self.combined.append(nn.LayerNorm(hidden_size))

        self.tree_features.append(nn.ReLU())
        self.mutation_features.append(nn.ReLU())
        self.combined.append(nn.ReLU())

        for _ in range(layers):
            self.tree_features.append(nn.Linear(hidden_size, hidden_size))
            self.mutation_features.append(nn.Linear(hidden_size, hidden_size))
            self.combined.append(nn.Linear(hidden_size, hidden_size))

            self.tree_features.append(nn.LayerNorm(hidden_size))
            self.mutation_features.append(nn.LayerNorm(hidden_size))
            self.combined.append(nn.LayerNorm(hidden_size))

            self.tree_features.append(nn.ReLU())
            self.mutation_features.append(nn.ReLU())
            self.combined.append(nn.ReLU())

        self.tree_features.append(nn.Linear(hidden_size, 64))
        self.mutation_features.append(nn.Linear(hidden_size, 64))
        self.combined.append(nn.Linear(hidden_size, hidden_size))

        self.tree_features.append(nn.LayerNorm(64))
        self.mutation_features.append(nn.LayerNorm(64))
        self.combined.append(nn.LayerNorm(hidden_size))
        
        self.tree_features.append(nn.ReLU())
        self.mutation_features.append(nn.ReLU())
        self.combined.append(nn.ReLU())

        self.out = nn.Linear(hidden_size, n_actions)
        
    def forward(self, tree_x, mutation_x):

        for layer in self.tree_features:
            tree_x = layer(tree_x)
        
        for layer in self.mutation_features:
            mutation_x = layer(mutation_x)

        x = torch.cat((tree_x, mutation_x, tree_x*mutation_x), dim=1)

        for layer in self.combined:
            x = layer(x)

        return self.out(x)
    
