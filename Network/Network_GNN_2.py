import random
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, global_max_pool
import torch.nn.functional as F

class DQN_GNN(nn.Module):

    def __init__(self, tree_size, n_actions):
        super(DQN_GNN, self).__init__()
        
        self.gnn_1 = GCNConv(tree_size, 512)
        #self.gnn_norm_1 = nn.LayerNorm(512)
        self.gnn_2 = GCNConv(512, 512)
        #self.gnn_norm_2 = nn.LayerNorm(512)
        self.gnn_3 = GCNConv(512, 512)
        #self.gnn_norm_3 = nn.LayerNorm(512)

        self.fcn_1 = nn.Linear(512, 512)
        self.fcn_norm_1 = nn.LayerNorm(512)
        self.fcn_2 = nn.Linear(512, 512)
        self.fcn_norm_2 = nn.LayerNorm(512)
        self.fcn_3 = nn.Linear(512, 512)
        self.fcn_norm_3 = nn.LayerNorm(512)
        self.out = nn.Linear(512, n_actions)
        
    def forward(self, tree_x, edge_index, batch):

        tree_x = self.gnn_1(tree_x, edge_index).relu()
        #tree_x = self.gnn_norm_1(tree_x).relu()
        tree_x = self.gnn_2(tree_x, edge_index).relu()
        #tree_x = self.gnn_norm_2(tree_x).relu()
        tree_x = self.gnn_3(tree_x, edge_index).relu()
        #tree_x = self.gnn_norm_3(tree_x).relu()

        x = global_max_pool(tree_x, batch)
        
        x = self.fcn_1(x).relu()
        #x = self.fcn_norm_1(x).relu()
        x = self.fcn_2(x).relu()
        #x = self.fcn_norm_2(x).relu()
        x = self.fcn_3(x).relu()
        #x = self.fcn_norm_3(x).relu()

        return self.out(x)
    
