import sys
#sys.path.append('../Enviroment')
#sys.path.append('../Tree')
sys.path.append('../')
from read import read_data, read_newick
from mutation_tree import MutationTree
from Enviroment import MutTreeEnv
from Network import DQN, ReplayMemory, Transition

import numpy as np
import math
import random
import torch
import torch.optim as optim
import torch.nn as nn
from itertools import count
import matplotlib.pyplot as plt

class Scheduler:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        self.steps = 0

    def step(self):
        self.steps += 1

    def get_instance(self):
        return self.end + (self.start - self.end) * math.exp(-self.steps / self.decay)

class Agent():
    def __init__(self, n_mut, n_cells, alpha, beta, device = "cuda"):
        self.device = device
        self.env = MutTreeEnv(n_mut=n_mut, n_cells=n_cells, alpha=alpha, beta=beta)
        dim = n_mut*(n_mut+1)
        self.policy_net = DQN(dim).to(self.device)
        self.target_net = DQN(dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learning_curve = None
        self.rewards = []
    
    def save_net(self, path):
        torch.save(self.policy_net, path)

    def load_net(self, path):
        self.policy_net = torch.load(path, map_location=torch.device(self.device), weights_only=False)

    def save_learning_curve(self, path):
        if self.learning_curve is not None:
            np.save(path, self.learning_curve)
        else: print("learn first")

    def plot_learning_curve(self, path):
        if self.learning_curve is not None:
            plt.plot(self.learning_curve)
            plt.savefig(path)
            plt.show()
        else: print("learn first")

    def plot_rewards(self, path):
        if self.learning_curve is not None:
            plt.plot(self.rewards)
            plt.savefig(path)
            plt.show()
        else: print("learn first")

    # Needs Batch Dim!
    def predict_step(self, state):
        q_vals = torch.zeros(len(self.env.all_spr))
        for i in range(len(q_vals)):
            spr = torch.tensor([self.env.all_spr[i]], dtype=torch.float32, device=self.device)
            with torch.no_grad(): q_vals[i] = self.policy_net(torch.cat([state, spr], dim=1))  
        return torch.argmax(q_vals)
    
    def predict_step_epsilon(self, state, eps_threshold):

        sample = random.random()

        if sample > eps_threshold:
            q_vals = torch.zeros(len(self.env.all_spr))
            for i in range(len(q_vals)):
                spr = torch.tensor([self.env.all_spr[i]], dtype=torch.float32, device=self.device)
                with torch.no_grad(): q_vals[i] = self.policy_net(torch.cat([state, spr], dim=1))  
            return torch.argmax(q_vals)
        else: return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def predict_step_epsilon_soft(self, state, eps_threshold, temperature):

        sample = random.random()
        if sample > eps_threshold:
            with torch.no_grad():
                q_vals = torch.zeros(len(self.env.all_spr))
                for i in range(len(q_vals)):
                    spr = torch.tensor([self.env.all_spr[i]], dtype=torch.float32, device=self.device)
                    with torch.no_grad(): q_vals[i] = self.policy_net(torch.cat([state, spr], dim=1)) 
                probs = torch.softmax(q_vals.view(-1) / temperature, dim=0)
                action = torch.multinomial(probs, num_samples=1)
                return action.view(1, 1)
        else: return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def __optimize_model(self, memory, batch_size, gamma, optimizer):
        if len(memory) < batch_size:
            return

        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions)) 
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        next_state_spr_batch = batch.next_state_spr
        done_batch = torch.cat(batch.done)

        state_action_values = self.policy_net(torch.cat([state_batch, action_batch.float()], dim=1)).squeeze()
        max_next_state_action_values = []

        for i in range(batch_size):
            next_state = next_state_batch[i].unsqueeze(0)
            possible_actions = next_state_spr_batch[i]  # a list of valid actions for this next_state

            q_vals = []
            for action in possible_actions:
                action_tensor = torch.tensor([action], dtype=torch.float32, device=self.device)
                sa_input = torch.cat([next_state, action_tensor], dim=1)
                with torch.no_grad(): q_val = self.target_net(sa_input.unsqueeze(0)).item()
                q_vals.append(q_val)
            max_next_state_action_values.append(max(q_vals))

        max_next_state_action_values = torch.tensor(max_next_state_action_values, dtype=torch.float32, device=self.device)
        not_done_mask = ~done_batch

        y = reward_batch + gamma * max_next_state_action_values * not_done_mask
   
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values.view(-1).float(), y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def train_net(self, data_path, batch_size, episodes, lr = 1e-4, gamma = 0.99, tau = 0.005):
        
        optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        memory = ReplayMemory(10000)

        all_data = read_data(data_path)
        all_trees = read_newick(data_path)

        for data in all_data:
            data[data == 97] = 0

        self.learning_curve = []

        eps_scheduler = Scheduler(start=0.9, end=0.05, decay=1000)
        temp_scheduler = Scheduler(start=2, end=0.5, decay=5000)

        min_reward = 100
        max_reward = 0

        for i in range(len(all_data)):

            gt_tree = MutationTree(self.env.n_mut, self.env.n_cells)
            gt_tree.use_newick_string(all_trees[i])
            gt_llh = gt_tree.conditional_llh(all_data[i], self.env.alpha, self.env.beta)

            for _ in range(episodes):            
                
                state = self.env.reset(gt_llh, all_data[i])
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                mse = 0
                for t in count():
                    action_idx = self.predict_step_epsilon_soft(state, eps_scheduler.get_instance(), temp_scheduler.get_instance())
                    action = self.env.all_spr[action_idx.item()]
                    eps_scheduler.step()
                    observation, reward, done, _ = self.env.step(action_idx.item())
                    if reward > max_reward: max_reward = reward
                    if reward < min_reward: min_reward = reward
                    self.rewards.append(reward)
                    reward = torch.tensor([reward], device=self.device)
                    action = torch.tensor([action], dtype=torch.long, device=self.device)
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                    done = torch.tensor([bool(done)], device=self.device)
                    next_state_spr = self.env.all_spr
                    memory.push(state, action, next_state, next_state_spr, reward, done)
                    state = next_state
                    loss = self.__optimize_model(memory, batch_size, gamma, optimizer)
                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
                        self.target_net.load_state_dict(target_net_state_dict)
                    if loss is not None: mse += loss
                    if done or (t > 20): break                                            
                temp_scheduler.step()   
                self.learning_curve.append(mse)
            perc = round(100*i/len(all_data), 2)
            if ((perc % 1) == 0): 
                print(perc, "%, MSE:", mse/(t+1))
        self.learning_curve = np.array(self.learning_curve)
        self.steps_done = 0
        print(min_reward, max_reward)
        del memory

    def solve_tree(self, data, max_iter, gt_tree = None):
        state = self.env.reset(-float('inf'), data)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        last1 = self.env.tree
        last2 = None
        stop = 0
        while stop < max_iter:
            action_idx = self.predict_step(state)
            action_idx = int(action_idx.item())
            observation, reward, done, invalid = self.env.step(action_idx)
            if invalid: print("invalid move")
            if (last2 is not None) or invalid:
                if self.env.tree == last2: stop += 1
            state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            last2 = last1
            last1 = self.env.tree
        last1_llh = last1.conditional_llh(data, self.env.alpha, self.env.beta)
        last2_llh = last2.conditional_llh(data, self.env.alpha, self.env.beta)
        if last1_llh > last2_llh: return last1
        else: return last2