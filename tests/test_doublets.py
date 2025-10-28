import sys
sys.path.append('../')
sys.path.append('../Network')
sys.path.append('../Tree')
sys.path.append('../GNN_Features')
sys.path.append('../features')
sys.path.append('../Enviroment')
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
dblt_rates = [0, 0.01, 0.05, 0.1, 0.2]

results = []
all_newick = read_newick("/home/mi/maxa55/Master_Thesis/Data/test/doublet", validation= True)

agent_gnn = Agent_GNN_Features(alpha, beta)
agent_gnn.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/gnn_features/trained_net_15.py")

agent_feat = Agent_Features_Fixed(alpha, beta)
agent_feat.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/feature/trained_net_fixed_mix_15.py")

for dr in dblt_rates:

    all_data, matrix_paths = read_data_double("/home/mi/maxa55/Master_Thesis/Data/test/doublet", dr, noisy=True, validation= True)

    for i in range(len(all_data)):
        n_mut = all_data[i].shape[0]
        n_cells = all_data[i].shape[1]

        gt_tree = MutationTree(n_mut, n_cells, all_newick[i])
        gt_llh = round(gt_tree.conditional_llh(all_data[i], 0.01, 0.2), 4)

        start_llh, end_llh = agent_gnn.solve_tree(all_data[i], n_mut*2)
        perf_gnn = min(1, round(abs(end_llh - start_llh)/abs(gt_llh - start_llh), 4))

        start_llh, end_llh = agent_feat.solve_tree(all_data[i], n_mut*2)
        perf_feat = min(1, round(abs(end_llh - start_llh)/abs(gt_llh - start_llh), 4))

        start_llh, end_llh = solve_greedy(all_data[i], alpha, beta)
        perf_greedy = min(1, round(abs(end_llh - start_llh)/abs(gt_llh - start_llh), 4))

        start_llh, end_llh = solve_scite(all_data[i], matrix_paths[i], alpha, beta)
        perf_scite = min(1, round(abs(end_llh - start_llh)/abs(gt_llh - start_llh), 4))

        results.append({'Method': "FCNN-3", 'doub_rt': dr, 'Perf': perf_feat})
        results.append({'Method': "GNN-4", 'doub_rt': dr, 'Perf': perf_gnn})
        results.append({'Method': "Greedy", 'doub_rt': dr, 'Perf': perf_greedy})
        results.append({'Method': "SCITE", 'doub_rt': dr, 'Perf': perf_scite})

df = pd.DataFrame(results)
df.to_csv("/home/mi/maxa55/Master_Thesis/Results/Plots/CSVs/doublets_not_trained.csv", index=False)


plt.figure(figsize=(10, 6))
sns.boxplot(x='doub_rt', y='Perf', hue='Method', data=df)
plt.title('Performance Comparison by Doublet Rate')
plt.ylabel('Performance')
plt.xlabel('Doublet Rate')
plt.legend(title='Method')
plt.tight_layout()
plt.savefig("/home/mi/maxa55/Master_Thesis/Results/Plots/doublets_not_trained.png")
plt.show()