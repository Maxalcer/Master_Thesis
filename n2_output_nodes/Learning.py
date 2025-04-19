import sys
#sys.path.append('../Enviroment')
#sys.path.append('../Tree')
sys.path.append('../')
from read import read_data, read_newick
from Enviroment import MutTreeEnv
from mutation_tree import MutationTree
from Network import DQN, ReplayMemory

import numpy as np
import math
import random
import torch
import torch.optim as optim
import torch.nn as nn
from itertools import count
from collections import namedtuple

# Data Parameters
n_mut = 5
n_cells = 5
alpha = 0.01
beta = 0.2
data_path = "/home/max/Master_Thesis/Data"

#Learning Parameters
batch_size = 24
episodes = 10
gamma = 0.99
eps_start = 0.9
eps_end = 0.05
eps_decay = 1000
tau = 0.005
lr = 1e-4
device = "cuda"

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            #q_vals = policy_net(state)
            #q_vals[torch.tensor([mask])] = -float('inf')
            return policy_net(state).max(1).indices.view(1, 1)
    else: return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

policy_net = DQN(n_mut*(n_mut+1), n_mut*(n_mut+1)).to(device)

optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
memory = ReplayMemory(10000)

all_data = read_data(data_path)
all_trees = read_newick(data_path)

for data in all_data:
    data[data == 97] = 0

learning_curve = []

env = MutTreeEnv(n_mut, n_cells, alpha, beta)

def optimize_model():
    if len(memory) < batch_size:
        return
    
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions)) 
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)
    done_batch = torch.cat(batch.done)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    with torch.no_grad(): max_next_state_action_values = policy_net(next_state_batch).max(1)[0]
    not_done_mask = ~done_batch

    y = reward_batch + gamma * max_next_state_action_values * not_done_mask.squeeze()
    criterion = nn.MSELoss()
    loss = criterion(state_action_values.squeeze().float(), y.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

for i in range(len(all_data)):
    for _ in range(episodes):

        gt_tree = MutationTree(n_mut, n_cells)
        gt_tree.use_newick_string(all_trees[i])
        gt_llh = gt_tree.conditional_llh(all_data[i], alpha, beta)
        state = env.reset(gt_llh, all_data[i])
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        mse = 0
        for t in count():
            #mask = env.get_valid_actions().flatten()
            #mask = mask == 0
            action = select_action(state)
            observation, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            done = torch.tensor([bool(done)], device=device)
            memory.push(state, action, next_state, reward, done)
            state = next_state
            if done or (t > 20):
                break
            loss = optimize_model()
            if loss is not None:
                mse += loss
        learning_curve.append(mse)
    perc = round(100*i/len(all_data), 2)
    if ((perc % 1) == 0): print(perc, "%, MSE:", mse/(t+1))
np.save("learning_curve.npy", np.array(learning_curve))

torch.save(policy_net, "trained_net.py")