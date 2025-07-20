import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()

        self.fcn_1 = nn.Linear(33, 1024)
        self.fcn_norm_1 = nn.LayerNorm(1024)
        self.fcn_2 = nn.Linear(1024, 1024)
        self.fcn_norm_2 = nn.LayerNorm(1024)
        self.fcn_3 = nn.Linear(1024, 1024)
        self.fcn_norm_3 = nn.LayerNorm(1024)
        self.fcn_4 = nn.Linear(1024, 1024)
        self.fcn_norm_4 = nn.LayerNorm(1024)
        self.fcn_5 = nn.Linear(1024, 1024)
        self.fcn_norm_5 = nn.LayerNorm(1024)
        self.fcn_6 = nn.Linear(1024, 1024)
        self.fcn_norm_6 = nn.LayerNorm(1024)
        self.out = nn.Linear(1024, 1)
        
    def forward(self, x):

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
        x = self.fcn_6(x)
        x = self.fcn_norm_6(x).relu()

        return self.out(x)
    
