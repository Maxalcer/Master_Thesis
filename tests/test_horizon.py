import sys
sys.path.append('Network')
sys.path.append('Tree')
sys.path.append('GNN_Features')
sys.path.append('features')
sys.path.append('Enviroment')
from helper import *
from greedy import solve_greedy
from SCITE_solve import solve_scite
from mutation_tree import MutationTree
from agent_GNN_Features import Agent_GNN_Features
from agent_features_fixed import Agent_Features_Fixed

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#import torch

alpha = 0.01
beta = 0.2

horizons = [6, 12, 24, 36, 48]

results = []

all_data = read_data("/home/mi/maxa55/Master_Thesis/Data/test/beta/20", noisy=True, validation= True)
all_newick = read_newick("/home/mi/maxa55/Master_Thesis/Data/test/beta/20", validation= True)

agent_gnn = Agent_GNN_Features(alpha, beta)
agent_gnn.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/gnn_features/trained_net_swap.py")

agent_feat = Agent_Features_Fixed(alpha, beta)
agent_feat.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/feature/trained_net_fixed_noisy_mix.py")

for horizon in horizons:

    for i in range(len(all_data)):
        n_mut = all_data[i].shape[0]
        n_cells = all_data[i].shape[1]

        gt_tree = MutationTree(n_mut, n_cells, all_newick[i])
        gt_llh = round(gt_tree.conditional_llh(all_data[i], 0.01, 0.2), 4)

        start_llh, end_llh = agent_gnn.solve_tree(all_data[i], horizon)
        perf_gnn = min(1, round(abs(end_llh - start_llh)/abs(gt_llh - start_llh), 4))

        start_llh, end_llh = agent_feat.solve_tree(all_data[i], horizon)
        perf_feat = min(1, round(abs(end_llh - start_llh)/abs(gt_llh - start_llh), 4))

        results.append({'Method': "FCNN-3", 'Horizon': horizon, 'Perf': perf_feat})
        results.append({'Method': "GNN-4", 'Horizon': horizon, 'Perf': perf_gnn})

df = pd.DataFrame(results)
df.to_csv("/home/mi/maxa55/Master_Thesis/Results/Plots/horizon.csv", index=False)


plt.figure(figsize=(10, 6))
sns.boxplot(x='Horizon', y='Perf', hue='Method', data=df)
plt.title('Performance Comparison by Horizon')
plt.ylabel('Performance')
plt.xlabel('Horinzon')
plt.legend(title='Method')
plt.tight_layout()
plt.savefig("/home/mi/maxa55/Master_Thesis/Results/Plots/Horinzon.png")
plt.show()