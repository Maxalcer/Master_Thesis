import sys
#sys.path.append('../Enviroment')
#sys.path.append('../Tree')
sys.path.append('../')
from read import *
from mutation_tree import MutationTree
from Enviroment import MutTreeEnv
from Network import DQN, ReplayMemory, Transition

import numpy as np
import math
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as learningrate_scheduler
import torch.nn as nn
from itertools import count
from sklearn.model_selection import train_test_split
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
        self.env = MutTreeEnv(n_mut=n_mut, n_cells=n_cells, alpha=alpha, beta=beta, device=device)
        tree_dim = n_mut*(n_mut+1)
        mat_dim = n_mut*n_cells
        out_dim = n_mut*(n_mut+1)
        self.policy_net = DQN(tree_dim, mat_dim, out_dim).to(self.device)
        self.target_net = DQN(tree_dim, mat_dim, out_dim).to(self.device)
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
    def predict_step(self, state, matrix):
        with torch.no_grad(): 
            q_vals = self.policy_net(state, matrix)
            return torch.argmax(q_vals, dim=1)
        
    def predict_step_soft(self, state, matrix, temperature):
        with torch.no_grad():
            q_vals = self.policy_net(state, matrix)
            probs = torch.softmax(q_vals.view(-1) / temperature, dim=0)
            action = torch.multinomial(probs, num_samples=1)
            return action.view(1, 1)

    
    def predict_step_epsilon(self, state, matrix, eps_threshold):

        sample = random.random()

        if sample > eps_threshold:
            with torch.no_grad():
                q_vals = self.policy_net(state, matrix)
                return torch.argmax(q_vals, dim=1).unsqueeze(0)
        else: return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def predict_step_epsilon_soft(self, state, matrix, eps_threshold, temperature):

        sample = random.random()
        if sample > eps_threshold:
            with torch.no_grad():
                q_vals = self.policy_net(state, matrix)
                probs = torch.softmax(q_vals.view(-1) / temperature, dim=0)
                #print("Q:", q_vals)
                #print("P:", probs)
                action = torch.multinomial(probs, num_samples=1)
                return action.view(1, 1)
        else: return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def __optimize_model(self, memory, batch_size, gamma, optimizer):
        if len(memory) < batch_size:
            return

        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions)) 
        state_batch = torch.cat(batch.state)
        matrix_batch = torch.cat(batch.matrix)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)

        state_action_values = self.policy_net(state_batch, matrix_batch).gather(1, action_batch)
        #with torch.no_grad(): max_next_state_action_values = self.policy_net(next_state_batch).max(1)[0]
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch, matrix_batch).argmax(dim=1, keepdim=True)
            max_next_state_action_values = self.target_net(next_state_batch, matrix_batch).gather(1, next_actions).squeeze()
        not_done_mask = ~done_batch

        y = reward_batch + gamma * max_next_state_action_values * not_done_mask
   
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values.view(-1).float(), y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def train_net(self, data_path, batch_size, episodes, lr = 1e-3, gamma = 0.99, tau = 0.005):
        
        optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        memory = ReplayMemory(10000)

        all_data = read_data(data_path)
        all_trees = read_newick(data_path)

        data_train, data_test, trees_train, trees_test = train_test_split(all_data, all_trees, test_size=0.30)

        self.learning_curve = []

        eps_scheduler = Scheduler(start=0.9, end=0.05, decay=1000)
        temp_scheduler = Scheduler(start=2, end=0.5, decay=5000)
        #lr_scheduler = learningrate_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=10000)

        min_reward = 100
        max_reward = 0

        for e in range(episodes):
            for i in range(len(data_train)):

                self.policy_net.train()
                gt_tree = MutationTree(self.env.n_mut, self.env.n_cells)
                gt_tree.use_newick_string(trees_train[i])
                gt_llh = gt_tree.conditional_llh(data_train[i], self.env.alpha, self.env.beta)
                matrix = torch.tensor(data_train[i].flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)          
                
                state = self.env.reset(gt_llh, data_train[i])
                mse = 0

                for t in count():
                    action = self.predict_step_epsilon_soft(state, matrix, eps_scheduler.get_instance(), temp_scheduler.get_instance())
                    eps_scheduler.step()
                    next_state, reward, done, invalid = self.env.step(action.item())

                    if reward > max_reward: max_reward = reward
                    if reward < min_reward: min_reward = reward
                    self.rewards.append(reward.item())

                    memory.push(state, matrix, action, next_state, reward, done)

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
                #lr_scheduler.step()
                self.learning_curve.append(round(mse/(t+1), 2))

                perc = round(100*(e*len(data_train)+i)/(len(data_train)*episodes), 2)
                if ((perc % 1) == 0):
                    acc = self.test_net(data_test, trees_test)
                    print(perc, "%, MSE:", round(mse/(t+1), 2), ", Test Acc:", acc)

        self.learning_curve = np.array(self.learning_curve)
        self.steps_done = 0
        print(min_reward, max_reward)
        del memory

    def test_net(self, test_data, test_trees):

        self.policy_net.eval()
        solved = 0
        for i in range(len(test_data)):
            gt_tree = MutationTree(5,5)
            gt_tree.use_newick_string(test_trees[i])
            gt_llh = gt_tree.conditional_llh(test_data[i], self.env.alpha, self.env.beta)
            done = False
            steps = 0
            state = self.env.reset(gt_llh, test_data[i])
            matrix = torch.tensor(test_data[i].flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)
            while not done and steps <= 5:
                action = self.predict_step_soft(state, matrix, 0.5)
                state, reward, done, invalid = self.env.step(action.item())
                if done: solved += 1
                steps += 1
        return round(solved/len(test_data), 2)

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