import sys
sys.path.append('../Enviroment')
sys.path.append('../Tree')
sys.path.append('../Network')
sys.path.append('../')
from helper import *
import params as P
from mutation_tree import MutationTree
from Enviroment import MutTreeEnv
from Network import DQN

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
from torch_geometric.data import Data, Batch

class Agent():
    def __init__(self, device):
        self.device = device
        self.env = None
        self.policy_net = None
        self.target_net = None
        self.learning_curve = None
        self.perfomance_train = None
        self.perfomance_test = None
    
    @property
    def n_mut(self):
        return self.env.n_mut
    
    @property
    def n_cells(self):
        return self.env.n_cells

    @property
    def alpha(self):
        return self.env.alpha

    @property
    def beta(self):
        return self.env.beta

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

    def plot_learning_curve(self, path):
        if self.learning_curve is not None:
            N = len(self.learning_curve)
            x_full = np.arange(N)
            eval_x = np.linspace(0, N - 1, 100)  

            fig, ax1 = plt.subplots()

            line1, = ax1.plot(x_full, self.learning_curve, color='blue', label='Train Loss')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Train Loss', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')

            ax2 = ax1.twinx()
            line2, = ax2.plot(eval_x, self.performance_test, color='red', label='Eval Score')
            ax2.set_ylabel('Eval Score', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            lines = [line1, line2]
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels, loc='center right')

            fig.tight_layout()
            plt.savefig(path)
            plt.show()
        else: print("learn first")

    def get_state_actions(self, tree):
        pass
    
    def transform_matrix(self, matrix):
        pass
    
    def transform_action(self, action, actions, possible_actions):
        pass

    def predict_step(self, state, actions, matrix):
        pass
        
    def predict_step_soft(self, state, actions, matrix, temperature):
        pass

    def predict_step_epsilon(self, state, actions, matrix, eps_threshold):
        pass

    def predict_step_epsilon_soft(self, state, actions, matrix, temperature, eps_threshold):
        pass

    def get_state_action_values(self, state, action, matrix):
        pass
    
    def get_max_next_state_action_values(self, next_state, next_actions, matrix):
        pass

    def __optimize_model(self, memory, optimizer):
        if len(memory) < P.BATCH_SIZE:
            return

        transitions = memory.sample(P.BATCH_SIZE)
        batch = Transition(*zip(*transitions)) 
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        matrix_batch = torch.cat(batch.matrix)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = batch.next_state
        next_action_batch = batch.next_actions
        done_batch = torch.cat(batch.done)
        not_done_mask = ~done_batch

        state_action_values = self.get_state_action_values(state_batch, action_batch, matrix_batch)
        max_next_state_action_values = self.get_max_next_state_action_values(next_state_batch, next_action_batch, matrix_batch)
        
        y = reward_batch + P.GAMMA * max_next_state_action_values * not_done_mask

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values.view(-1).float(), y.float())
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        optimizer.step()
        return loss

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
        self.performance_train = []
        self.performance_test = []

        eps_scheduler = Scheduler(start=0.9, end=0.05, decay=P.EPISODES*len(data_train)/20)
        temp_scheduler = Scheduler(start=2, end=0.5, decay=P.EPISODES*len(data_train)/3)
        lr_scheduler = learningrate_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=len(data_train)*P.EPISODES/2)

        last_perc = -1
        for e in range(P.EPISODES):
            for i in range(len(data_train)):

                self.policy_net.train()
                gt_tree = MutationTree(self.n_mut, self.n_cells, trees_train[i])
                gt_llh = gt_tree.conditional_llh(data_train[i], self.alpha, self.beta)
                matrix = self.transform_matrix(data_train[i])         
                tree = self.env.reset(gt_llh, data_train[i])
                mse = 0

                for t in count():
                    state, actions = self.get_state_actions(tree)
                    action = self.predict_step_epsilon_soft(state, actions, matrix, temp_scheduler.get_instance(), eps_scheduler.get_instance())
                    action, action_indx = self.transform_action(action, actions, tree.all_possible_spr)                    
                    tree, reward, done = self.env.step(action_indx) 
                    if reward > max_rew: max_rew = reward
                    if reward < min_rew: min_rew = reward
                    reward = torch.tensor([reward], device=self.device)
                    done = torch.tensor([bool(done)], device=self.device)
                    next_state, next_actions = self.get_state_actions(tree)
                    memory.push(state, matrix, actions[action.item()].unsqueeze(0), next_state, next_actions, reward, done)

                    state = next_state
                    actions = next_actions
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

        print(min_rew, max_rew)
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
            state, actions = self.get_state_actions(tree)
            start_llh = self.env.current_llh
            matrix = self.transform_matrix(test_data[i])
            while steps <= P.HORIZON:
                last_llh = self.env.current_llh
                action = self.predict_step(state, actions, matrix)
                indices = np.argwhere(tree.all_possible_spr == 1)
                tree, reward, done = self.env.step(indices[action.item()])
                state, actions = self.get_state_actions(tree)
                steps += 1
            end_llh = max(last_llh, self.env.current_llh)
            if round(start_llh - gt_llh, 5) != 0: 
                perf += (abs(end_llh - start_llh)/abs(gt_llh - start_llh))
                c += 1
        return round(perf/c, 4)