import sys
sys.path.append('Network')
sys.path.append('Tree')
sys.path.append('GNN_Features')
sys.path.append('features')
sys.path.append('Enviroment')
from helper import *
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


agent = Agent_Features_Fixed(alpha, beta)
agent.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/feature/trained_net_fixed_mix_15.py")

agent_extr = Agent_Features_Fixed(alpha, beta)
agent_extr.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/feature/trained_net_fixed_noisy_mix.py")

results = []

for i in range(len(all_data)):
    n_mut = all_data[i].shape[0]
    n_cells = all_data[i].shape[1]

    gt_tree = MutationTree(n_mut, n_cells, all_newick[i])
    gt_llh = round(gt_tree.conditional_llh(all_data[i], 0.01, 0.2), 4)

    start_llh, end_llh = agent.solve_tree(all_data[i])
    perf = min(1, round(abs(end_llh - start_llh)/abs(gt_llh - start_llh), 4))

    start_llh, end_llh = agent_extr.solve_tree(all_data[i])
    perf_extr = min(1, round(abs(end_llh - start_llh)/abs(gt_llh - start_llh), 4))

    results.append({'Method': "Normal", 'Num_Mut': n_mut, 'Perf': perf})
    results.append({'Method': "Extrapolated", 'Num_Mut': n_mut, 'Perf': perf_extr})

df = pd.DataFrame(results)
df.to_csv("/home/mi/maxa55/Master_Thesis/Results/Plots/extr_feat.csv", index=False)


plt.figure(figsize=(10, 6))
sns.boxplot(x='Num_Mut', y='Perf', hue='Method', data=df)
plt.title('Extraploation Strength')
plt.ylabel('Performance')
plt.xlabel('Number of Mutations')
plt.legend(title='Method')
plt.tight_layout()
plt.savefig("/home/mi/maxa55/Master_Thesis/Results/Plots/extr_feat.png")
plt.show()