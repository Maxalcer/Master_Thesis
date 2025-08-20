import random
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_max_pool
from torch_geometric.nn.aggr import AttentionalAggregation
import torch.nn.functional as F

class DQN_GNN(nn.Module):

    def __init__(self, tree_size, mutation_size, n_actions, layers = 3, hidden_size = 1024):
        super(DQN_GNN, self).__init__()
        
        self.gnn_1 = GATConv(tree_size, hidden_size)
        self.gnn_norm_1 = nn.LayerNorm(hidden_size)
        self.gnn_2 = GATConv(hidden_size, hidden_size)
        self.gnn_norm_2 = nn.LayerNorm(hidden_size)
        self.gnn_3 = GATConv(hidden_size, hidden_size)
        self.gnn_norm_3 = nn.LayerNorm(hidden_size)
        self.gnn_4 = GATConv(hidden_size, hidden_size)
        self.gnn_norm_4 = nn.LayerNorm(hidden_size)
        self.gnn_5 = GATConv(hidden_size, 64)
        self.gnn_norm_5 = nn.LayerNorm(64)

        self.gate_nn = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
            )
        self.pool = AttentionalAggregation(self.gate_nn)

        self.mutation_features = nn.ModuleList()
        self.combined = nn.ModuleList()

        self.mutation_features.append(nn.Linear(mutation_size, hidden_size))
        self.combined.append(nn.Linear(64*3, hidden_size))

        self.mutation_features.append(nn.LayerNorm(hidden_size))
        self.combined.append(nn.LayerNorm(hidden_size))

        self.mutation_features.append(nn.ReLU())
        self.combined.append(nn.ReLU())

        for _ in range(layers):

            self.mutation_features.append(nn.Linear(hidden_size, hidden_size))
            self.combined.append(nn.Linear(hidden_size, hidden_size))

            self.mutation_features.append(nn.LayerNorm(hidden_size))
            self.combined.append(nn.LayerNorm(hidden_size))

            self.mutation_features.append(nn.ReLU())
            self.combined.append(nn.ReLU())

        self.mutation_features.append(nn.Linear(hidden_size, 64))
        self.combined.append(nn.Linear(hidden_size, hidden_size))

        self.mutation_features.append(nn.LayerNorm(64))
        self.combined.append(nn.LayerNorm(hidden_size))
        
        self.mutation_features.append(nn.ReLU())
        self.combined.append(nn.ReLU())

        self.out = nn.Linear(hidden_size, n_actions)
        
    def forward(self, tree_x, edge_index, mutation_x, batch):

        tree_x = self.gnn_1(tree_x, edge_index)
        tree_x = self.gnn_norm_1(tree_x).relu()
        tree_x = self.gnn_2(tree_x, edge_index)
        tree_x = self.gnn_norm_2(tree_x).relu()
        tree_x = self.gnn_3(tree_x, edge_index)
        tree_x = self.gnn_norm_3(tree_x).relu()
        tree_x = self.gnn_4(tree_x, edge_index)
        tree_x = self.gnn_norm_4(tree_x).relu()
        tree_x = self.gnn_5(tree_x, edge_index)
        tree_x = self.gnn_norm_5(tree_x).relu()

        tree_x = self.pool(tree_x, batch) 
        
        for layer in self.mutation_features:
            mutation_x = layer(mutation_x)

        x = torch.cat((tree_x, mutation_x, tree_x*mutation_x), dim=1)

        for layer in self.combined:
            x = layer(x)

        return self.out(x)
    
