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
betas = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]

results = []

for beta in betas:

    all_data, _ = read_data(f"/home/mi/maxa55/Master_Thesis/Data/test/beta/{int(beta*100)}", noisy=True, validation= True)
    all_newick = read_newick(f"/home/mi/maxa55/Master_Thesis/Data/test/beta/{int(beta*100)}", validation= True)

    agent_gnn = Agent_GNN_Features(alpha, beta)
    agent_gnn.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/gnn_features/trained_net_swap.py")

    agent_feat = Agent_Features_Fixed(alpha, beta)
    agent_feat.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/feature/trained_net_fixed_noisy_mix.py")

    matrix_paths = glob.glob(f"/home/mi/maxa55/Master_Thesis/Data/test/beta/{int(beta*100)}/valid*.noisy")
    matrix_paths = sorted(matrix_paths)

    for i in range(len(all_data)):
        n_mut = all_data[i].shape[0]
        n_cells = all_data[i].shape[1]

        gt_tree = MutationTree(n_mut, n_cells, all_newick[i])
        gt_llh = round(gt_tree.conditional_llh(all_data[i], 0.01, 0.2), 4)

        start_llh, end_llh = agent_gnn.solve_tree(all_data[i], n_mut*2)
        perf_gnn = round((end_llh - gt_llh), 4)
        #if perf_gnn < -10: print("start_llh:", start_llh, "end_llh:", end_llh, "gt_llh:", gt_llh)

        start_llh, end_llh = agent_feat.solve_tree(all_data[i], n_mut*2)
        perf_feat = round((end_llh - gt_llh), 4)

        start_llh, end_llh = solve_greedy(all_data[i], alpha, beta)
        perf_greedy = round((end_llh - gt_llh), 4)

        start_llh, end_llh = solve_scite(all_data[i], matrix_paths[i], alpha, beta)
        perf_scite = round((end_llh - gt_llh), 4)

        results.append({'Method': "FCNN-3", 'Beta': beta, 'Perf': perf_feat})
        results.append({'Method': "GNN-4", 'Beta': beta, 'Perf': perf_gnn})
        results.append({'Method': "Greedy", 'Beta': beta, 'Perf': perf_greedy})
        results.append({'Method': "SCITE", 'Beta': beta, 'Perf': perf_scite})

df = pd.DataFrame(results)
df.to_csv("/home/mi/maxa55/Master_Thesis/Results/Plots/CSVs/beta_diff_abs.csv", index=False)


plt.figure(figsize=(10, 6))
sns.boxplot(x='Beta', y='Perf', hue='Method', data=df)
#plt.title('Difference to ground truth LLH Comparison by beta')
plt.ylabel('signed absolute error')
plt.xlabel('beta')
plt.legend(title='Method')
plt.tight_layout()
plt.savefig("/home/mi/maxa55/Master_Thesis/Results/Plots/beta_diff_abs.png")
plt.show()