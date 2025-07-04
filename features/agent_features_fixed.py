import sys
sys.path.append('../Enviroment')
sys.path.append('../Tree')
sys.path.append('../')
from helper import read_data, read_newick, Scheduler
from mutation_tree import MutationTree
from Enviroment import MutTreeEnv
from Network_Features_fixed import DQN 
import params as P

import numpy as np
import math
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as learningrate_scheduler
import torch.nn as nn
from itertools import count
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state_action', 'next_state_actions', 'reward', 'done'))

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

class Agent_Features_Fixed():
    def __init__(self, n_mut, n_cells, alpha, beta, device = "cuda"):
        self.device = device
        self.env = MutTreeEnv(n_mut=n_mut, n_cells=n_cells, alpha=alpha, beta=beta)
        dim = n_mut*n_cells
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.learning_curve = None
        self.performance_test = None
        self.performance_train = None
    
    @property
    def n_mut(self):
        return self.env.n_mut
    
    @property
    def n_cells(self):
        return self.env.n_cells

    @property
    def data(self):
        return self.env.data

    @property
    def alpha(self):
        return self.env.alpha

    @property
    def beta(self):
        return self.env.beta

    @property
    def all_spr(self):
        return self.env.all_spr
    
    @property
    def tree(self):
        return self.env.tree

    def save_net(self, path):
        torch.save(self.policy_net, path)

    def load_net(self, path):
        self.policy_net = torch.load(path, map_location=torch.device(self.device), weights_only=False)

    def save_learning_curve(self, path):
        if self.learning_curve is not None:
            np.save(path+"_loss.npy", self.learning_curve)
            np.save(path+"_perf_train.npy", self.performance_train)
            np.save(path+"_perf_test.npy", self.performance_test)
        else: print("learn first")

    def get_state_actions(self, state, data):
        all_vectors = state.feature_vectors(data, self.alpha, self.beta)
        return torch.tensor(all_vectors, dtype=torch.float32, device=self.device)
    
    def transform_action(self, action, actions, possible_actions):
        indices = np.argwhere(possible_actions == 1)
        action_indx = indices[action.item()]
        action = actions[action.item()].unsqueeze(0)
        return action, action_indx
    
    def predict_step(self, state_actions):
        with torch.no_grad(): q_vals = self.policy_net(state_actions)  
        return torch.argmax(q_vals)
    
    def predict_step_soft(self, state_actions, temperature):
            with torch.no_grad():
                q_vals = self.policy_net(state_actions)  
                probs = torch.softmax(q_vals.view(-1) / temperature, dim=0)
                action = torch.multinomial(probs, num_samples=1)
                return action.view(1, 1)

    def predict_step_epsilon(self, state_actions, eps_threshold):

        sample = random.random()

        if sample > eps_threshold: 
            return self.predict_step(state_actions)
        else: return torch.tensor([[random.randint(0, state_actions.shape[0]-1)]], device=self.device, dtype=torch.long)

    def predict_step_epsilon_soft(self, state_actions, eps_threshold, temperature):

        sample = random.random()
        if sample > eps_threshold:
            with torch.no_grad():
                q_vals = self.policy_net(state_actions)  
                probs = torch.softmax(q_vals.view(-1) / temperature, dim=0)
                action = torch.multinomial(probs, num_samples=1)
                return action.view(1, 1)
        else: return torch.tensor([[random.randint(0, state_actions.shape[0]-1)]], device=self.device, dtype=torch.long)

    def get_state_action_values(self, state_action):
        return self.policy_net(state_action).squeeze()
    
    def get_max_next_state_action_values(self, next_state_actions):
        num_actions_per_item = [state_action.shape[0] for state_action in next_state_actions]
        next_state_actions = torch.cat(next_state_actions)
        with torch.no_grad(): 
            q_vals = self.policy_net(num_actions_per_item)
            q_vals_split = torch.split(q_vals, num_actions_per_item)
            argmax_indices = [q.argmax() for q in q_vals_split]
            q_vals_target = self.target_net(num_actions_per_item)
            q_vals_target_split = torch.split(q_vals_target, num_actions_per_item)
            max_next_state_action_values = torch.stack([q[i] for q, i in zip(q_vals_target_split, argmax_indices)]).to(self.device)
            max_next_state_action_values = torch.clamp(max_next_state_action_values, min=-8, max=20)
        return max_next_state_action_values.squeeze()

    def __optimize_model(self, memory, batch_size, optimizer):
        if len(memory) < batch_size:
            return

        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions)) 
        state_action_batch = torch.cat(batch.state_action)
        reward_batch = torch.cat(batch.reward)
        next_state_actions_batch = batch.next_state_actions
        done_batch = torch.cat(batch.done)
        not_done_mask = ~done_batch

        state_action_values = self.get_state_action_values(state_action_batch)
        max_next_state_action_values = self.get_max_next_state_action_values(next_state_actions_batch)
        
        y = reward_batch + P.GAMMA * max_next_state_action_values * not_done_mask
   
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values.view(-1).float(), y.float())
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()

    def train_net(self, data_path):
        
        min_rew = float('inf')
        max_rew = float('-inf')

        optimizer = optim.AdamW(self.policy_net.parameters(), lr=P.LR, amsgrad=True)
        memory = ReplayMemory(10000)
        noisy = (self.alpha != 0) | (self.beta != 0)

        all_data = read_data(data_path, noisy = noisy, validation = False)
        all_trees = read_newick(data_path)

        data_train, data_test, trees_train, trees_test = train_test_split(all_data, all_trees, test_size=0.30)

        self.learning_curve = []
        self.performance_test = []
        self.performance_train = []

        eps_scheduler = Scheduler(start=0.9, end=0.05, decay=P.EPISODES*len(data_train)/20)
        temp_scheduler = Scheduler(start=2, end=0.5, decay=P.EPISODES*len(data_train)/3)
        lr_scheduler = learningrate_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=len(data_train)*P.EPISODES/2)
        
        last_perc = -1
        for e in range(P.EPISODES):
            for i in range(len(data_train)):

                self.policy_net.train()
                gt_tree = MutationTree(self.n_mut, self.n_cells, trees_train[i])
                gt_llh = gt_tree.conditional_llh(data_train[i], self.alpha, self.beta)      
                tree = self.env.reset(gt_llh, data_train[i])
                mse = 0
                
                for t in count():
                    state_actions = self.get_state_actions(tree, data_train[i])
                    action = self.predict_step_epsilon_soft(state_actions, temp_scheduler.get_instance(), eps_scheduler.get_instance())
                    state_action, action_indx = self.transform_action(action, state_actions, tree.all_possible_spr)                    
                    tree, reward, done = self.env.step(action_indx) 
                    if reward > max_rew: max_rew = reward
                    if reward < min_rew: min_rew = reward
                    
                    next_state_actions = self.get_state_actions(tree, data_train[i])
                    reward = torch.tensor([reward], device=self.device)
                    done = torch.tensor([bool(done)], device=self.device)
                    memory.push(state_action, next_state_actions, reward, done)

                    state_actions = next_state_actions
                    loss = self.__optimize_model(memory, optimizer)

                    for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                        target_param.data.mul_(1.0 - P.TAU).add_(policy_param.data, alpha=P.TAU)

                    if loss is not None:     
                        mse += loss.item()
                    if done or (t >= P.HORIZON): break   

                if loss is not None:                                                            
                    eps_scheduler.step()
                    temp_scheduler.step()
                    lr_scheduler.step()   
                self.learning_curve.append(round(mse/(t+1), 4))

                perc = int(100*(e*len(data_train)+i)/(len(data_train)*P.EPISODES))
                if (perc != last_perc):
                    train_acc = self.test_net(data_train, trees_train)
                    test_acc = self.test_net(data_test, trees_test)
                    self.performance_test.append(test_acc)
                    self.performance_train.append(train_acc)
                    print(perc, "%, MSE:", round(mse/(t+1), 4), ", Test Acc:", test_acc, ", Train Acc:", train_acc)
                    with open("log.txt", "a") as f:
                        f.write(f"{perc}%, MSE: {round(mse/(t+1), 4)}, Test Acc: {test_acc}, Train Acc: {train_acc}\n")
                    last_perc = perc
            
        train_acc = self.test_net(data_train, trees_train)
        test_acc = self.test_net(data_test, trees_test)
        print("Test Acc:", test_acc, "  Train Acc:", train_acc)
        self.learning_curve = np.array(self.learning_curve)
        self.performance_test = np.array(self.performance_test)
        self.performance_train = np.array(self.performance_train)
        self.steps_done = 0
        del memory

    def test_net(self, test_data, test_trees):

        self.policy_net.eval()

        perf = 0
        c = 0
        for i in range(len(test_data)):
            gt_tree = MutationTree(self.n_mut, self.n_cells, test_trees[i])
            gt_llh = gt_tree.conditional_llh(test_data[i], self.alpha, self.beta)
            done = False
            steps = 0
            tree = self.env.reset(gt_llh, test_data[i])
            start_llh = self.env.current_llh
            state_actions = self.get_state_actions(tree, test_data[i])
            while steps <= P.HORIZON:
                last_llh = self.env.current_llh
                action = self.predict_step(state_actions)
                tree, reward, done = self.env.step(action.item())
                state_actions = self.get_state_actions(tree, test_data[i])
                steps += 1
            end_llh = max(last_llh, self.env.current_llh)
            if round(start_llh - gt_llh, 5) != 0: 
                perf += (abs(end_llh - start_llh)/abs(gt_llh - start_llh))
                c += 1
        return round(perf/c, 4)