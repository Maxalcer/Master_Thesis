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
import glob
from sklearn.impute import KNNImputer
#import torch

def read_data_miss(rate):
    path = f"/home/mi/maxa55/Master_Thesis/Data/test/miss_vals/valid_*.noisy{rate}pct*"
    files = glob.glob(path)
    files = sorted(files)
    output_list = []
    for f in files:
        file = open(f, "r")
        input = file.read()
        rows = input.split('\n')
        rows = rows[0:len(rows)-1]
        matrix = [row.split(' ') for row in rows]
        matrix = np.array(matrix, dtype=float)
        matrix[matrix == 3] = np.nan
        imputer = KNNImputer(n_neighbors=1, metric="nan_euclidean")
        matrix_imputed = imputer.fit_transform(matrix.T).T
        matrix_imputed = np.rint(matrix_imputed).astype(int)
        matrix_imputed = sort_matrix(matrix_imputed)
        output_list.append(matrix_imputed)
    return output_list, files

alpha = 0.01
beta = 0.2
miss_val_rates = [0, 0.01, 0.05, 0.1, 0.2]

results = []
all_newick = read_newick("/home/mi/maxa55/Master_Thesis/Data/test/miss_vals", validation= True)

agent_gnn = Agent_GNN_Features(alpha, beta)
agent_gnn.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/gnn_features/trained_net_15.py")

agent_feat = Agent_Features_Fixed(alpha, beta)
agent_feat.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/feature/trained_net_fixed_mix_15.py")

for miss in miss_val_rates:
    all_data, matrix_paths = read_data_miss(int(miss*100))
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

        results.append({'Method': "FCNN-3", 'miss_val': miss, 'Perf': perf_feat})
        results.append({'Method': "GNN-4", 'miss_val': miss, 'Perf': perf_gnn})
        results.append({'Method': "Greedy", 'miss_val': miss, 'Perf': perf_greedy})
        results.append({'Method': "SCITE", 'miss_val': miss, 'Perf': perf_scite})

df = pd.DataFrame(results)
df.to_csv("/home/mi/maxa55/Master_Thesis/Results/Plots/CSVs/miss_vals.csv", index=False)


plt.figure(figsize=(10, 6))
sns.boxplot(x='miss_val', y='Perf', hue='Method', data=df)
plt.title('Performance Comparison by Missing Values')
plt.ylabel('Performance')
plt.xlabel('Missing Value Rate')
plt.legend(title='Method')
plt.tight_layout()
plt.savefig("/home/mi/maxa55/Master_Thesis/Results/Plots/miss_vals.png")
plt.show()