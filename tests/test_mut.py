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

all_data = read_data("/home/mi/maxa55/Master_Thesis/Data/test/n_mut", noisy=True, validation= True)
all_newick = read_newick("/home/mi/maxa55/Master_Thesis/Data/test/n_mut", validation= True)

matrix_paths = glob.glob("/home/mi/maxa55/Master_Thesis/Data/test/n_mut/valid*.noisy")
matrix_paths = sorted(matrix_paths)

agent_gnn = Agent_GNN_Features(alpha, beta)
agent_gnn.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/gnn_features/trained_net_double.py")

agent_feat = Agent_Features_Fixed(alpha, beta)
agent_feat.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/feature/trained_net_fixed_double.py")

results = []

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

    results.append({'Method': "FCNN-3", 'Num_Mut': n_mut, 'Perf': perf_feat})
    results.append({'Method': "GNN-4", 'Num_Mut': n_mut, 'Perf': perf_gnn})
    results.append({'Method': "Greedy", 'Num_Mut': n_mut, 'Perf': perf_greedy})
    results.append({'Method': "SCITE", 'Num_Mut': n_mut, 'Perf': perf_scite})

df = pd.DataFrame(results)
df.to_csv("/home/mi/maxa55/Master_Thesis/Results/Plots/Num_Mut_Double.csv", index=False)


plt.figure(figsize=(10, 6))
sns.boxplot(x='Num_Mut', y='Perf', hue='Method', data=df)
plt.title('Performance Comparison by Number of Mutations')
plt.ylabel('Performance')
plt.xlabel('Number of Mutations')
plt.legend(title='Method')
plt.tight_layout()
plt.savefig("/home/mi/maxa55/Master_Thesis/Results/Plots/Num_Mut_Double.png")
plt.show()