import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, tree_size, mutation_size, n_actions):
        super(DQN, self).__init__()
        self.tree_layer1 = nn.Linear(tree_size, tree_size*2)
        self.mutation_layer1 = nn.Linear(mutation_size, mutation_size*2)
        #self.tree_layer2 = nn.Linear(128, 128)
        #self.mutation_layer2 = nn.Linear(128, 128)

        self.layer1 = nn.Linear((tree_size+mutation_size)*2, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, tree_input, mutation_input):
        tree_x = F.relu(self.tree_layer1(tree_input))
        mutation_x = F.relu(self.mutation_layer1(mutation_input))
        #tree_x = F.relu(self.tree_layer2(tree_x))
        #mutation_x = F.relu(self.mutation_layer2(mutation_x))

        x = torch.cat((tree_x, mutation_x), dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
Transition = namedtuple('Transition', ('state', 'matrix', 'action', 'next_state', 'reward', 'done'))

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