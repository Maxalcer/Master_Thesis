import sys
sys.path.append('../Enviroment')
sys.path.append('../Tree')
sys.path.append('../')
from read import read_data, read_newick
from env import MutTreeEnv
from mutation_tree import MutationTree
from Network import DQN, ReplayMemory

import numpy as np
import torch
from itertools import count

# Data Parameters
n_mut = 5
n_cells = 5
alpha = 0.01
beta = 0.2
data_path = "/home/max/Master_Thesis/Data"

device = "cpu"

policy_net = torch.load("trained_net.py", map_location=torch.device('cpu'), weights_only=False)

all_data = read_data(data_path)
all_trees = read_newick(data_path)

for data in all_data:
    data[data == 97] = 0

gt_tree = MutationTree(n_mut, n_cells)
gt_tree.use_newick_string(all_trees[0])
gt_llh = gt_tree.conditional_llh(all_data[0], alpha, beta)

env = MutTreeEnv(gt_llh, n_mut, n_cells, all_data[0], alpha, beta)

sum = 0

for i in range(len(all_data)):
    gt_tree = MutationTree(n_mut, n_cells)
    gt_tree.use_newick_string(all_trees[i])
    gt_llh = gt_tree.conditional_llh(all_data[i], alpha, beta)
    state = env.reset(gt_llh, all_data[i])
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action_values = []
        for action in env.all_spr:
            action_tensor = torch.tensor([action], dtype=torch.float32, device=device)
            sa_input = torch.cat([state, action_tensor], dim=1)
            action_values.append(policy_net(sa_input).item())

        observation, reward, done, _ = env.step(np.argmax(action_values))
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        state = next_state
        if done or (t > 100):
            break
    sum += t

print(sum/len(all_data))