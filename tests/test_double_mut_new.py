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
random
#import torch

alpha = 0.01
beta = 0.2

results = []

agent = Agent_GNN_Features(alpha, beta)
agent.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/gnn_features/trained_net_swap.py")

#agent = Agent_Features_Fixed(alpha, beta)
#agent.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/feature/trained_net_fixed_noisy_mix.py")


all_data, matrix_paths, double_muts = read_data(f"/home/mi/maxa55/Master_Thesis/Data/test/double_mut_new", noisy=True, validation= True)
all_newick = read_newick(f"/home/mi/maxa55/Master_Thesis/Data/test/double_mut_new", validation= True)

for d in range(2):

    all_data, matrix_paths, double_muts = read_data(f"/home/mi/maxa55/Master_Thesis/Data/test/double_mut/{d}", noisy=True, validation= True)
    all_newick = read_newick(f"/home/mi/maxa55/Master_Thesis/Data/test/double_mut/{d}", validation= True)

    for i in range(len(all_data)):
        
        data = all_data[i]
        n_mut = data.shape[0]
        n_cells = data.shape[1]

        start_llh, end_llh = agent.solve_tree(data, n_mut*2)
        results.append({'double_mut_tree': "No", 'double_mut': d, 'end_llh': end_llh})

        if d == 0: data = np.vstack([all_data[i], all_data[i][random.randint(0, 11)]])
        elif d == 1: data = np.vstack([all_data[i], all_data[i][double_muts[i]]])
        n_mut = data.shape[0]
        n_cells = data.shape[1]

        start_llh, end_llh = agent.solve_tree(data, n_mut*2)
        results.append({'double_mut_tree': "Yes", 'double_mut': d, 'end_llh': end_llh})
        


df = pd.DataFrame(results)
df.to_csv("/home/mi/maxa55/Master_Thesis/Results/Plots/CSVs/double_mut_new_gnn.csv", index=False)


plt.figure(figsize=(10, 6))
sns.boxplot(x='double_mut', y='end_llh', hue='double_mut_tree', data=df)
#plt.title('Performance Comparison by Doublet Rate')
plt.ylabel('Final LLH')
plt.xlabel('Double Mutations in Data')
plt.legend(title='Double Mutation in Tree')
plt.tight_layout()
plt.savefig("/home/mi/maxa55/Master_Thesis/Results/Plots/double_mut_new_gnn.png")
plt.show()
