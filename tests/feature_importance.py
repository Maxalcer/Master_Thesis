import sys
sys.path.append('../')
sys.path.append('../Network')
sys.path.append('../Tree')
sys.path.append('../GNN_Features')
sys.path.append('../features')
sys.path.append('../Enviroment')
from helper import *

from mutation_tree import MutationTree
from agent_features_fixed import Agent_Features_Fixed

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from captum.attr import Occlusion
import torch

alpha = 0.01
beta = 0.2

all_state_actions = []

agent_feat = Agent_Features_Fixed(alpha, beta)
agent_feat.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/feature/trained_net_fixed_long.py")

all_data, _ = read_data("/home/mi/maxa55/Master_Thesis/Data/test/features", noisy=True, validation= True)
data = all_data[0]

agent_feat.policy_net.eval()

n_mut = data.shape[0]
n_cells = data.shape[1]
tree = MutationTree(n_mut, n_cells)
pvec = np.repeat(n_mut, n_mut + 1)
pvec[-1] = -1
tree.use_parent_vec(pvec, n_mut)
for _ in range(n_mut*2):
    state_actions = agent_feat.get_state_actions(tree, data)
    all_state_actions.append(state_actions)
    action = agent_feat.predict_step(state_actions)
    state_action = state_actions[action.item()].unsqueeze(0)             
    swap_idx = state_action[0, 28]
    
    if (swap_idx != -1):
        action_indx = int(swap_idx.item())
        tree.swap(action_indx)
    else:
        indices = np.argwhere(tree.all_possible_spr == 1)
        action_indx = indices[action.item()]
        tree.perf_spr(action_indx[0], action_indx[1])
print("tree solved")
occlusion = Occlusion(agent_feat.policy_net)
inputs = torch.cat(all_state_actions, dim=0)
attributions = occlusion.attribute(
        inputs=inputs,
        sliding_window_shapes=(1,),
        strides=(1,),
        baselines=0
    )
global_importance = attributions.abs().mean(dim=0)

df = pd.DataFrame({
    "feature_index": list(range(len(global_importance))),
    "importance": global_importance.cpu().numpy()
})
df.to_csv("/home/mi/maxa55/Master_Thesis/Results/Plots/CSVs/feature_importance_long.csv", index=False)