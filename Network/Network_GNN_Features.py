import random
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_max_pool
from torch_geometric.nn.aggr import AttentionalAggregation
import torch.nn.functional as F

class DQN_GNN(nn.Module):

    def __init__(self):
        super(DQN_GNN, self).__init__()
        
        self.gnn_1 = GATConv(17, 1024)
        self.gnn_norm_1 = nn.LayerNorm(1024)
        self.gnn_2 = GATConv(1024, 1024)
        self.gnn_norm_2 = nn.LayerNorm(1024)
        self.gnn_3 = GATConv(1024, 1024)
        self.gnn_norm_3 = nn.LayerNorm(1024)
        self.gnn_4 = GATConv(1024, 1024)
        self.gnn_norm_4 = nn.LayerNorm(1024)
        self.gnn_5 = GATConv(1024, 1024)
        self.gnn_norm_5 = nn.LayerNorm(1024)

        self.gate_nn = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
            )
        self.pool = AttentionalAggregation(self.gate_nn)

        self.fcn_1 = nn.Linear(1024, 1024)
        self.fcn_norm_1 = nn.LayerNorm(1024)
        self.fcn_2 = nn.Linear(1024, 1024)
        self.fcn_norm_2 = nn.LayerNorm(1024)
        self.fcn_3 = nn.Linear(1024, 1024)
        self.fcn_norm_3 = nn.LayerNorm(1024)
        self.fcn_4 = nn.Linear(1024, 1024)
        self.fcn_norm_4 = nn.LayerNorm(1024)
        self.fcn_5 = nn.Linear(1024, 1024)
        self.fcn_norm_5 = nn.LayerNorm(1024)
        self.out = nn.Linear(1024, 1)
        
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
    
