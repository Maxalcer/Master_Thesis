import random
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_max_pool
import torch.nn.functional as F

class DQN_GNN(nn.Module):

    def __init__(self, mutation_size, layers = 3, hidden_size = 512):
        super(DQN_GNN, self).__init__()
        
        self.gnn_1 = GCNConv(8, 256)
        self.gnn_norm_1 = nn.LayerNorm(256)
        self.gnn_2 = GCNConv(256, 256)
        self.gnn_norm_2 = nn.LayerNorm(256)

        self.tree_out = nn.Linear(256, 64)
        self.tree_out_norm = nn.LayerNorm(64)

        #self.tree_features = nn.ModuleList()
        self.mutation_features = nn.ModuleList()
        self.combined = nn.ModuleList()
        #self.tree_norm = nn.ModuleList()

        #self.tree_features.append(GCNConv(tree_size, hidden_size//8))
        self.mutation_features.append(nn.Linear(mutation_size, hidden_size))
        self.combined.append(nn.Linear(64*3, hidden_size))

        for _ in range(layers):
            #self.tree_features.append(GCNConv(hidden_size//8, hidden_size//8))
            self.mutation_features.append(nn.Linear(hidden_size, hidden_size))
            self.combined.append(nn.Linear(hidden_size, hidden_size))

            #self.tree_norm.append(nn.LayerNorm(hidden_size//8))
            self.mutation_features.append(nn.LayerNorm(hidden_size))
            self.combined.append(nn.LayerNorm(hidden_size))

            self.combined.append(nn.LeakyReLU())
            self.mutation_features.append(nn.LayerNorm(hidden_size))

        #self.tree_out = nn.Linear(hidden_size, 64)
        self.mutation_features.append(nn.Linear(hidden_size, 64))
        self.combined.append(nn.Linear(hidden_size, hidden_size))

        self.out = nn.Linear(hidden_size, 1)
        
    def forward(self, tree_x, edge_index, mutation_x, batch):

        #for layer, norm in zip(self.tree_features, self.tree_norm):
        #    tree_x = layer(tree_x, edge_index)
        #    tree_x = norm(tree_x)
        #    tree_x = F.leaky_relu(tree_x)
        tree_x = self.gnn_1(tree_x, edge_index)
        tree_x = F.leaky_relu(self.gnn_norm_1(tree_x))
        #tree_x = F.leaky_relu(tree_x)
        tree_x = self.gnn_2(tree_x, edge_index)
        tree_x = F.leaky_relu(self.gnn_norm_2(tree_x))
        #tree_x = F.leaky_relu(tree_x)
        tree_x = global_max_pool(tree_x, batch)
        tree_x = self.tree_out(tree_x)
        tree_x = F.leaky_relu(self.tree_out_norm(tree_x))
        #tree_x = F.leaky_relu(tree_x)

        for layer in self.mutation_features:
            mutation_x = layer(mutation_x)

        x = torch.cat((tree_x, mutation_x, tree_x*mutation_x), dim=1)

        for layer in self.combined:
            x = layer(x)

        return self.out(x)
    
