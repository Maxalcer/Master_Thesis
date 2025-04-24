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
from sklearn.model_selection import train_test_split

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

    def get_input(self, state, actions):
        all_spr = torch.tensor(actions, dtype=torch.float32, device=self.device)
        state_exp = state.expand(len(actions), -1)
        return torch.cat((state_exp, all_spr), dim=1)

    # Needs Batch Dim!
    def predict_step(self, state):
        input = self.get_input(state, self.env.all_spr)
        with torch.no_grad(): q_vals = self.policy_net(input)  
        return torch.argmax(q_vals)
    
    def predict_step_soft(self, state, temperature):
            with torch.no_grad():
                input = self.get_input(state, self.env.all_spr)
                q_vals = self.policy_net(input)  
                probs = torch.softmax(q_vals.view(-1) / temperature, dim=0)
                action = torch.multinomial(probs, num_samples=1)
                return action.view(1, 1)

    def predict_step_epsilon(self, state, eps_threshold):

        sample = random.random()

        if sample > eps_threshold:  return self.predict_step(state)
        else: return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def predict_step_epsilon_soft(self, state, eps_threshold, temperature):

        sample = random.random()
        if sample > eps_threshold:
            with torch.no_grad():
                input = self.get_input(state, self.env.all_spr)
                q_vals = self.policy_net(input)  
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
        next_state_actions_batch = batch.next_state_actions
        done_batch = torch.cat(batch.done)

        state_action_values = self.policy_net(torch.cat([state_batch, action_batch.float()], dim=1)).squeeze()
        max_next_state_action_values = []

        #all_next_state_actions = [self.get_input(next_state_batch[i].unsqueeze(0), next_state_spr_batch[i]) for i in range(batch_size)]
        all_next_state_actions = torch.cat(next_state_actions_batch, dim=0)

        with torch.no_grad(): q_vals = self.target_net(all_next_state_actions)

        start_idx = 0
        for i in range(batch_size):
            num_actions = len(next_state_actions_batch[i])
            q_vals_state = q_vals[start_idx:start_idx + num_actions] 
            max_next_state_action_values.append(q_vals_state.max().item())  
            start_idx += num_actions
            
        """
        for i in range(batch_size):
            next_state = next_state_batch[i].unsqueeze(0)
            possible_actions = next_state_spr_batch[i]  # a list of valid actions for this next_state

            input = self.get_input(next_state, possible_actions)
            with torch.no_grad(): q_vals = self.target_net(input)
            max_next_state_action_values.append(max(q_vals))
        """
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

        data_train, data_test, trees_train, trees_test = train_test_split(all_data, all_trees, test_size=0.30)

        self.learning_curve = []

        eps_scheduler = Scheduler(start=0.9, end=0.05, decay=1000)
        temp_scheduler = Scheduler(start=2, end=0.5, decay=5000)

        min_reward = 100
        max_reward = 0

        for i in range(len(data_train)):

            self.policy_net.train()

            gt_tree = MutationTree(self.env.n_mut, self.env.n_cells)
            gt_tree.use_newick_string(trees_train[i])
            gt_llh = gt_tree.conditional_llh(data_train[i], self.env.alpha, self.env.beta)

            for _ in range(episodes):            
                
                state = self.env.reset(gt_llh, data_train[i])
                mse = 0
                
                for t in count():
                    action_idx = self.predict_step_epsilon_soft(state, eps_scheduler.get_instance(), temp_scheduler.get_instance())
                    action = torch.tensor([self.env.all_spr[action_idx.item()]], dtype=torch.long, device=self.device)
                    eps_scheduler.step()
                    next_state, reward, done = self.env.step(action_idx.item())

                    if reward > max_reward: max_reward = reward
                    if reward < min_reward: min_reward = reward
                    self.rewards.append(reward.item())
                    
                    next_state_actions = self.get_input(next_state, self.env.all_spr)
                    memory.push(state, action, next_state_actions, reward, done)

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
                self.learning_curve.append(mse/(t+1))

            perc = round(100*i/len(data_train), 2)
            if ((perc % 1) == 0):
                acc = self.test_net(data_test, trees_test)
                print(perc, "%, MSE:", mse/(t+1), ", Acc:", acc)

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
            while not done and steps <= 20:
                action = self.predict_step_soft(state, 0.5)
                state, reward, done = self.env.step(action.item())
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