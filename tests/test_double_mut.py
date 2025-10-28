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

results = []

agent_gnn = Agent_GNN_Features(alpha, beta)
agent_gnn.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/gnn_features/trained_net_double.py")

agent_feat = Agent_Features_Fixed(alpha, beta)
agent_feat.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/feature/trained_net_fixed_double.py")

for d in range(2):

    all_data, matrix_paths = read_data(f"/home/mi/maxa55/Master_Thesis/Data/test/double_mut/{d}", noisy=True, validation= True)
    all_newick = read_newick(f"/home/mi/maxa55/Master_Thesis/Data/test/double_mut/{d}", validation= True)

    for i in range(len(all_data)):
        n_mut = all_data[i].shape[0]
        n_cells = all_data[i].shape[1]

        gt_tree = MutationTree(n_mut, n_cells, all_newick[i])
        gt_llh = round(gt_tree.conditional_llh(all_data[i], 0.01, 0.2), 4)

        start_llh, end_llh = agent_gnn.solve_tree(all_data[i], n_mut*2)
        perf_gnn = min(1, round(abs(end_llh - start_llh)/abs(gt_llh - start_llh), 4))

        start_llh, end_llh = agent_feat.solve_tree(all_data[i], n_mut*2)
        perf_feat = min(1, round(abs(end_llh - start_llh)/abs(gt_llh - start_llh), 4))

        #start_llh, end_llh = solve_greedy(all_data[i], alpha, beta)
        perf_greedy = 0#min(1, round(abs(end_llh - start_llh)/abs(gt_llh - start_llh), 4))

        #start_llh, end_llh = solve_scite(all_data[i], matrix_paths[i], alpha, beta)
        perf_scite = 0#min(1, round(abs(end_llh - start_llh)/abs(gt_llh - start_llh), 4))

        results.append({'Method': "FCNN-3", 'double_mut': d, 'Perf': perf_feat})
        results.append({'Method': "GNN-4", 'double_mut': d, 'Perf': perf_gnn})
        results.append({'Method': "Greedy", 'double_mut': d, 'Perf': perf_greedy})
        results.append({'Method': "SCITE", 'double_mut': d, 'Perf': perf_scite})

df = pd.DataFrame(results)
df.to_csv("/home/mi/maxa55/Master_Thesis/Results/Plots/CSVs/double_mut_trained.csv", index=False)

"""
plt.figure(figsize=(10, 6))
sns.boxplot(x='double_mut', y='Perf', hue='Method', data=df)
#plt.title('Performance Comparison by Doublet Rate')
plt.ylabel('Performance')
plt.xlabel('Double Mutations')
plt.legend(title='Method')
plt.tight_layout()
plt.savefig("/home/mi/maxa55/Master_Thesis/Results/Plots/double_mut.png")
plt.show()
"""