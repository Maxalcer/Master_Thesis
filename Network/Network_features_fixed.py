import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()

        self.fcn_1 = nn.Linear(27, 512)
        self.fcn_norm_1 = nn.LayerNorm(512)
        self.fcn_2 = nn.Linear(512, 512)
        self.fcn_norm_2 = nn.LayerNorm(512)
        self.fcn_3 = nn.Linear(512, 512)
        self.fcn_norm_3 = nn.LayerNorm(512)
        self.out = nn.Linear(512, 1)
        
    def forward(self, x):

        x = self.fcn_1(x)
        x = self.fcn_norm_1(x).relu()
        x = self.fcn_2(x)
        x = self.fcn_norm_2(x).relu()
        x = self.fcn_3(x)
        x = self.fcn_norm_3(x).relu()

        return self.out(x)
    
