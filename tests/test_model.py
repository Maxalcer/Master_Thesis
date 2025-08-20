import sys
sys.path.append('Network')
sys.path.append('Tree')
sys.path.append('GNN_Features')
sys.path.append('features')
sys.path.append('Enviroment')
sys.path.append('n2_output_nodes')
sys.path.append('GNN')
sys.path.append('Agent')
from helper import *
from mutation_tree import MutationTree
from agent_GNN_Features import Agent_GNN_Features
from agent_features_fixed import Agent_Features_Fixed
from agent_n2 import Agent_N2_Nodes
from agent_features import Agent_Features
from agent_GNN_1 import Agent_GNN_1
from agent_GNN_2 import Agent_GNN_2
from agent_GNN_3 import Agent_GNN_3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#import torch

alpha = 0.01
beta = 0.2

all_data = read_data("/home/mi/maxa55/Master_Thesis/Data/test/model_comp", noisy=True, validation= True)
all_newick = read_newick("/home/mi/maxa55/Master_Thesis/Data/test/model_comp", validation= True)

matrix_paths = glob.glob("/home/mi/maxa55/Master_Thesis/Data/test/model_comp/valid*.noisy")
matrix_paths = sorted(matrix_paths)

agent_n2 = Agent_N2_Nodes(5, 10, alpha, beta)
agent_n2.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/n2/trained_net_5.py") 

agent_feat = Agent_Features(5, 10, alpha, beta)
agent_feat.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/feature/trained_net_5.py")

agent_gnn_1 = Agent_GNN_1(5, 10, alpha, beta)
agent_gnn_1.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/gnn/trained_net_1_5.py")

agent_gnn_2 = Agent_GNN_2(5, 10, alpha, beta)
agent_gnn_2.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/gnn/trained_net_2_5.py")

agent_gnn_3 = Agent_GNN_3(5, 10, alpha, beta)
agent_gnn_3.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/gnn/trained_net_3_5.py")

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

    start_llh, end_llh = agent_n2.solve_tree(all_data[i])
    perf_n2 = min(1, round(abs(end_llh - start_llh)/abs(gt_llh - start_llh), 4))

    start_llh, end_llh = agent_feat.solve_tree(all_data[i])
    perf_feat = min(1, round(abs(end_llh - start_llh)/abs(gt_llh - start_llh), 4))

    start_llh, end_llh = agent_gnn_1.solve_tree(all_data[i])
    perf_gnn_1 = min(1, round(abs(end_llh - start_llh)/abs(gt_llh - start_llh), 4))

    start_llh, end_llh = agent_gnn_2.solve_tree(all_data[i])
    perf_gnn_2 = min(1, round(abs(end_llh - start_llh)/abs(gt_llh - start_llh), 4))

    start_llh, end_llh = agent_gnn_3.solve_tree(all_data[i])
    perf_gnn_3 = min(1, round(abs(end_llh - start_llh)/abs(gt_llh - start_llh), 4))

    start_llh, end_llh = agent_gnn_feat.solve_tree(all_data[i])
    perf_gnn_feat = min(1, round(abs(end_llh - start_llh)/abs(gt_llh - start_llh), 4))

    start_llh, end_llh = agent_feat_fix.solve_tree(all_data[i])
    perf_feat_fix = min(1, round(abs(end_llh - start_llh)/abs(gt_llh - start_llh), 4))

    results.append({'Model': "FCNN-1", 'Perf': perf_n2})
    results.append({'Model': "FCNN-2", 'Perf': perf_feat})
    results.append({'Model': "FCNN-3", 'Perf': perf_feat_fix})
    results.append({'Model': "GNN-1", 'Perf': perf_gnn_1})
    results.append({'Model': "GNN-2", 'Perf': perf_gnn_2})
    results.append({'Model': "GNN-3", 'Perf': perf_gnn_3})
    results.append({'Model': "GNN-4", 'Perf': perf_gnn_feat})


df = pd.DataFrame(results)
df.to_csv("/home/mi/maxa55/Master_Thesis/Results/Plots/Model.csv", index=False)


plt.figure(figsize=(10, 6))
sns.boxplot(x='Model', y='Perf', hue='Model', data=df)
plt.title('Performance Comparison by Q-Learning Model')
plt.ylabel('Performance')
plt.xlabel('Model')
#plt.legend(title='Model')
plt.tight_layout()
plt.savefig("/home/mi/maxa55/Master_Thesis/Results/Plots/Model.png")
plt.show()