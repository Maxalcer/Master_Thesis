import sys
sys.path.append('Network')
sys.path.append('Tree')
sys.path.append('GNN_Features')
sys.path.append('features')
sys.path.append('Enviroment')
from helper import *
from greedy import solve_greedy
from SCITE_solve import solve_scite
from SiFit_solve import solve_sifit
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

all_data = read_data("/home/mi/maxa55/Master_Thesis/Data/test/beta/20", noisy=True, validation= True)
all_newick = read_newick("/home/mi/maxa55/Master_Thesis/Data/test/beta/20", validation= True)

matrix_paths = glob.glob("/home/mi/maxa55/Master_Thesis/Data/test/beta/20/valid*.noisy")
matrix_paths = sorted(matrix_paths)

agent_gnn_feat = Agent_GNN_Features(alpha, beta)
agent_gnn_feat.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/gnn_features/trained_net_swap.py")

agent_feat_fix = Agent_Features_Fixed(alpha, beta)
agent_feat_fix.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/feature/trained_net_fixed_noisy_mix.py")

results = []

for i in range(len(all_data)):
    n_mut = all_data[i].shape[0]
    n_cells = all_data[i].shape[1]

    gt_tree = MutationTree(n_mut, n_cells, all_newick[i])
    gt_llh = round(gt_tree.conditional_llh(all_data[i], 0.01, 0.2), 4)

    start_llh, end_llh = agent_gnn_feat.solve_tree(all_data[i], 24)
    perf_gnn_feat = min(1, round(abs(end_llh - start_llh)/abs(gt_llh - start_llh), 4))

    start_llh, end_llh = agent_feat_fix.solve_tree(all_data[i], 24)
    perf_feat_fix = min(1, round(abs(end_llh - start_llh)/abs(gt_llh - start_llh), 4))

    start_llh, end_llh = solve_greedy(all_data[i], alpha, beta)
    perf_greedy = min(1, round(abs(end_llh - start_llh)/abs(gt_llh - start_llh), 4))

    start_llh, end_llh = solve_scite(all_data[i], matrix_paths[i], alpha, beta)
    perf_scite = min(1, round(abs(end_llh - start_llh)/abs(gt_llh - start_llh), 4))

    start_llh, end_llh = solve_sifit(all_data[i], alpha, beta)
    perf_sifit = min(1, round(abs(end_llh - start_llh)/abs(gt_llh - start_llh), 4))

    results.append({'Method': "FCNN-3", 'Perf': perf_feat_fix})
    results.append({'Method': "GNN-4", 'Perf': perf_gnn_feat})
    results.append({'Method': "Greedy", 'Perf': perf_greedy})
    results.append({'Method': "SCITE", 'Perf': perf_scite})
    results.append({'Method': "SiFit", 'Perf': perf_sifit})


df = pd.DataFrame(results)
df.to_csv("/home/mi/maxa55/Master_Thesis/Results/Plots/Method.csv", index=False)


plt.figure(figsize=(10, 6))
sns.boxplot(x='Method', y='Perf', hue='Method', data=df)
plt.title('Performance Comparison by Tree Inference Method')
plt.ylabel('Performance')
plt.xlabel('Method')
#plt.legend(title='Method')
plt.tight_layout()
plt.savefig("/home/mi/maxa55/Master_Thesis/Results/Plots/Method.png")
plt.show()