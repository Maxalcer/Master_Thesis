import random
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, global_max_pool
from torch_geometric.nn.aggr import AttentionalAggregation
import torch.nn.functional as F

class DQN_GNN(nn.Module):

    def __init__(self, tree_size, n_actions, hidden_size = 1024):
        super(DQN_GNN, self).__init__()
        
        self.gnn_1 = GATConv(tree_size, hidden_size)
        self.gnn_norm_1 = nn.LayerNorm(hidden_size)
        self.gnn_2 = GATConv(hidden_size, hidden_size)
        self.gnn_norm_2 = nn.LayerNorm(hidden_size)
        self.gnn_3 = GATConv(hidden_size, hidden_size)
        self.gnn_norm_3 = nn.LayerNorm(hidden_size)
        self.gnn_4 = GATConv(hidden_size, hidden_size)
        self.gnn_norm_4 = nn.LayerNorm(hidden_size)
        self.gnn_5 = GATConv(hidden_size, hidden_size)
        self.gnn_norm_5 = nn.LayerNorm(hidden_size)

        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
            )
        self.pool = AttentionalAggregation(self.gate_nn)

        self.fcn_1 = nn.Linear(hidden_size, hidden_size)
        self.fcn_norm_1 = nn.LayerNorm(hidden_size)
        self.fcn_2 = nn.Linear(hidden_size, hidden_size)
        self.fcn_norm_2 = nn.LayerNorm(hidden_size)
        self.fcn_3 = nn.Linear(hidden_size, hidden_size)
        self.fcn_norm_3 = nn.LayerNorm(hidden_size)
        self.fcn_4 = nn.Linear(hidden_size, hidden_size)
        self.fcn_norm_4 = nn.LayerNorm(hidden_size)
        self.fcn_5 = nn.Linear(hidden_size, hidden_size)
        self.fcn_norm_5 = nn.LayerNorm(hidden_size)
        self.out = nn.Linear(hidden_size, n_actions)
        
    def forward(self, tree_x, edge_index, batch):

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

        x = self.pool(tree_x, batch)
        
        x = self.fcn_1(x)
        x = self.fcn_norm_1(x).relu()
        x = self.fcn_2(x)
        x = self.fcn_norm_2(x).relu()
        x = self.fcn_3(x)
        x = self.fcn_norm_3(x).relu()
        x = self.fcn_4(x)
        x = self.fcn_norm_4(x).relu()
        x = self.fcn_5(x)
        x = self.fcn_norm_5(x).relu()

        return self.out(x)
    
