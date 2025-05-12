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
        self.target_net.eval()
        self.learning_curve = None
        self.perfomance = None
    
    def save_net(self, path):
        torch.save(self.policy_net, path)

    def load_net(self, path):
        self.policy_net = torch.load(path, map_location=torch.device(self.device), weights_only=False)

    def save_learning_curve(self, path):
        if self.learning_curve is not None:
            np.save("loss_"+path, self.learning_curve)
            np.save("perf_"+path, self.performance)
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
            line2, = ax2.plot(eval_x, self.performance, color='red', label='Eval Score')
            ax2.set_ylabel('Eval Score', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            lines = [line1, line2]
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels, loc='center right')

            fig.tight_layout()
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
    def predict_step(self, state, matrix, move_mask):
        with torch.no_grad(): 
            q_vals = self.policy_net(state, matrix)
            q_vals = q_vals.masked_fill(move_mask, float('-inf'))
            return torch.argmax(q_vals, dim=1)
        
    def predict_step_soft(self, state, matrix, move_mask, temperature):
        with torch.no_grad():
            q_vals = self.policy_net(state, matrix)
            q_vals = q_vals.masked_fill(move_mask, float('-inf'))
            probs = torch.softmax(q_vals.view(-1) / temperature, dim=0)
            action = torch.multinomial(probs, num_samples=1)
            return action.view(1, 1)

    
    def predict_step_epsilon(self, state, matrix, move_mask, eps_threshold):

        sample = random.random()

        if sample > eps_threshold:
            with torch.no_grad():
                q_vals = self.policy_net(state, matrix)
                q_vals = q_vals.masked_fill(move_mask, float('-inf'))
                return torch.argmax(q_vals, dim=1).unsqueeze(0)
        else: return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def predict_step_epsilon_soft(self, state, matrix, move_mask, eps_threshold, temperature):

        sample = random.random()
        if sample > eps_threshold:
            with torch.no_grad():
                q_vals = self.policy_net(state, matrix)
                q_vals = q_vals.masked_fill(move_mask, float('-inf'))
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
        matrix_batch = torch.cat(batch.matrix)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        move_mask_batch = torch.cat(batch.move_mask)
        done_batch = torch.cat(batch.done)

        state_action_values = self.policy_net(state_batch, matrix_batch).gather(1, action_batch)
        #with torch.no_grad(): max_next_state_action_values = self.policy_net(next_state_batch).max(1)[0]
        with torch.no_grad():
            q_vals = self.policy_net(next_state_batch, matrix_batch)
            q_vals = q_vals.masked_fill(move_mask_batch, float('-inf'))
            next_actions = q_vals.argmax(dim=1, keepdim=True)
            max_next_state_action_values = self.target_net(next_state_batch, matrix_batch).gather(1, next_actions).squeeze()
            max_next_state_action_values = torch.clamp(max_next_state_action_values, min=-10, max=25)
        not_done_mask = ~done_batch

        y = reward_batch + gamma * max_next_state_action_values * not_done_mask
   
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values.view(-1).float(), y.float())
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        optimizer.step()
        return loss

    def train_net(self, data_path, batch_size, episodes, lr = 1e-4, gamma = 0.99, tau = 0.005):
        
        optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        memory = ReplayMemory(10000)
        noisy = (self.env.alpha != 0) | (self.env.beta != 0)

        all_data = read_data(data_path, noisy = noisy, validation = False)
        all_trees = read_newick(data_path)

        data_train, data_test, trees_train, trees_test = train_test_split(all_data, all_trees, test_size=0.30)

        self.learning_curve = []
        self.performance = []

        eps_scheduler = Scheduler(start=0.9, end=0.05, decay=episodes*len(data_train)/20)
        temp_scheduler = Scheduler(start=2, end=0.5, decay=episodes*len(data_train)/3)
        lr_scheduler = learningrate_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=len(data_train)*episodes/2)
        """
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',             # 'min' for loss, 'max' for accuracy
        factor=0.5,             # multiply LR by this factor
        patience=100,            # how many steps to wait before reducing
        threshold=1e-4
        )
        """
        last_perc = -1
        for e in range(episodes):
            for i in range(len(data_train)):

                self.policy_net.train()
                gt_tree = MutationTree(self.env.n_mut, self.env.n_cells)
                gt_tree.use_newick_string(trees_train[i])
                gt_llh = gt_tree.conditional_llh(data_train[i], self.env.alpha, self.env.beta)
                matrix = torch.tensor(data_train[i].flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)          
                
                state, move_mask = self.env.reset(gt_llh, data_train[i])
                mse = 0

                for t in count():
                    action = self.predict_step_epsilon_soft(state, matrix, move_mask, eps_scheduler.get_instance(), temp_scheduler.get_instance())
                    next_state, move_mask, reward, done = self.env.step(action.item())
                    memory.push(state, matrix, action, next_state, move_mask, reward, done)

                    state = next_state
                    loss = self.__optimize_model(memory, batch_size, gamma, optimizer)

                    for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                        target_param.data.mul_(1.0 - tau).add_(policy_param.data, alpha=tau)

                    if loss is not None:     
                        mse += loss.item()
                    if done or (t > 10): break
                    #if t > 10: break
                if loss is not None:                                                            
                    eps_scheduler.step()
                    temp_scheduler.step()
                    lr_scheduler.step() 
                self.learning_curve.append(round(mse/(t+1), 4))
                
                perc = int(100*(e*len(data_train)+i)/(len(data_train)*episodes))
                if (perc != last_perc):
                    acc = self.test_net(data_test, trees_test)
                    self.performance.append(acc)
                    print(perc, "%, MSE:", round(mse/(t+1), 4), ", Test Acc:", acc)
                    curr_lr = round(lr_scheduler.get_last_lr()[0], 6)
                    curr_temp = round(temp_scheduler.get_instance(), 6)
                    curr_eps = round(eps_scheduler.get_instance(), 6)
                    print("LR:",curr_lr , "temp:", curr_temp, "eps:", curr_eps)
                    with open("log.txt", "a") as f:
                        f.write(f"{perc}%, MSE: {round(mse/(t+1), 4)}, Test Acc: {acc}\n")
                    last_perc = perc

        train_acc = self.test_net(data_train, trees_train)
        test_acc = self.test_net(data_test, trees_test)
        print("Test Acc:", test_acc, "  Train Acc:", train_acc)
        self.learning_curve = np.array(self.learning_curve)
        self.performance = np.array(self.performance)
        self.steps_done = 0
        del memory

    def test_net(self, test_data, test_trees):

        self.policy_net.eval()
        perf = 0
        c = 0
        for i in range(len(test_data)):
            gt_tree = MutationTree(self.env.n_mut, self.env.n_cells)
            gt_tree.use_newick_string(test_trees[i])
            gt_llh = gt_tree.conditional_llh(test_data[i], self.env.alpha, self.env.beta)
            done = False
            steps = 0
            state, move_mask = self.env.reset(gt_llh, test_data[i])
            start_llh = self.env.current_llh
            matrix = torch.tensor(test_data[i].flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)
            while steps <= 10:
                last_llh = self.env.current_llh
                action = self.predict_step(state, matrix, move_mask)
                state, move_mask, reward, done = self.env.step(action.item())
                steps += 1
            end_llh = max(last_llh, self.env.current_llh)
            if round(start_llh - gt_llh, 5) != 0: 
                perf += (abs(end_llh - start_llh)/abs(gt_llh - start_llh))
                c += 1
        return round(perf/c, 4)

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