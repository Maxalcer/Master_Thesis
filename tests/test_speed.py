import sys
sys.path.append('../')
sys.path.append('../Network')
sys.path.append('../Tree')
sys.path.append('../GNN_Features')
sys.path.append('../features')
sys.path.append('../Enviroment')
from helper import *
from greedy import solve_greedy
from SCITE_solve import speed_scite
from mutation_tree import MutationTree
from agent_GNN_Features import Agent_GNN_Features
from agent_features_fixed import Agent_Features_Fixed

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
#import torch

alpha = 0.01
beta = 0.2

all_data = read_data(f"/home/mi/maxa55/Master_Thesis/Data/test/beta/{int(beta*100)}", noisy=True, validation= True)
all_newick = read_newick(f"/home/mi/maxa55/Master_Thesis/Data/test/beta/{int(beta*100)}", validation= True)

agent_gnn = Agent_GNN_Features(alpha, beta)
agent_gnn.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/gnn_features/trained_net_swap.py")

agent_feat = Agent_Features_Fixed(alpha, beta)
agent_feat.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/feature/trained_net_fixed_noisy_mix.py")

matrix_paths = glob.glob(f"/home/mi/maxa55/Master_Thesis/Data/test/beta/{int(beta*100)}/valid*.noisy")
matrix_paths = sorted(matrix_paths)

results = []

for i in range(len(all_data)):
    n_mut = all_data[i].shape[0]
    n_cells = all_data[i].shape[1]

    start = time.time()
    tree = agent_gnn.solve_tree_fast(all_data[i])
    end = time.time()
    gnn_time = (end - start)

    start = time.time()
    tree = agent_feat.solve_tree_fast(all_data[i])
    end = time.time()
    fcnn_time = (end - start)

    start = time.time()
    llh_s, llh_e = solve_greedy(all_data[i], alpha, beta)
    end = time.time()
    greedy_time = (end - start)

    scite_time = speed_scite(all_data[i], matrix_paths[i], alpha, beta)
    
    results.append({'Method': "FCNN-3", 'Time': fcnn_time})
    results.append({'Method': "GNN-4",'Time': gnn_time})
    results.append({'Method': "Greedy", 'Time': greedy_time})
    results.append({'Method': "SCITE", 'Time': scite_time})
    

df = pd.DataFrame(results)
df.to_csv("/home/mi/maxa55/Master_Thesis/Results/Plots/CSVs/speed.csv", index=False)


plt.figure(figsize=(10, 6))
sns.boxplot(x='Method', y='Time', hue='Method', data=df)
plt.title('Speed Comparison')
plt.ylabel('Time [sec]')
plt.xlabel('Method')
#plt.legend(title='Method')
plt.tight_layout()
plt.savefig("/home/mi/maxa55/Master_Thesis/Results/Plots/speed.png")
plt.show()