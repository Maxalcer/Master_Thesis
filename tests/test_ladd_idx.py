import sys
sys.path.append('../')
sys.path.append('../Network')
sys.path.append('../Tree')
sys.path.append('../GNN_Features')
sys.path.append('../features')
sys.path.append('../Enviroment')
from helper import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from mutation_tree import MutationTree
from agent_GNN_Features import Agent_GNN_Features
from agent_features_fixed import Agent_Features_Fixed

all_data = read_data("/home/mi/maxa55/Master_Thesis/Data/12x24", noisy=True, validation=True)
all_newick = read_newick("/home/mi/maxa55/Master_Thesis/Data/12x24", validation= True)

agent_feat = Agent_Features_Fixed(0.01, 0.2)
agent_feat.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/feature/trained_net_fixed_noisy_mix.py")
agent_feat.policy_net.eval()

agent_gnn = Agent_GNN_Features(0.01, 0.2)
agent_gnn.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/gnn_features/trained_net_swap.py")
agent_gnn.policy_net.eval()

results = []

trees_bad_gnn = []
trees_good_gnn = []

trees_bad_feat = []
trees_good_feat = []

for i in range(len(all_data)):
    gt_tree = MutationTree(12, 24, all_newick[i])
    gt_llh = gt_tree.conditional_llh(all_data[i], 0.01, 0.2)
    ladd_indx = gt_tree.ladderization_index()

    start_llh, end_llh = agent_gnn.solve_tree(all_data[i], 24)
    perf_gnn = min(1, abs(end_llh - start_llh)/abs(gt_llh - start_llh))

    if abs(end_llh - start_llh)/abs(gt_llh - start_llh) < 0.8: trees_bad_gnn.append(gt_tree)
    else: trees_good_gnn.append(gt_tree)

    start_llh, end_llh = agent_feat.solve_tree(all_data[i], 24)
    perf_feat = min(1, abs(end_llh - start_llh)/abs(gt_llh - start_llh))

    if abs(end_llh - start_llh)/abs(gt_llh - start_llh) < 0.8: trees_bad_feat.append(gt_tree)
    else: trees_good_feat.append(gt_tree)

    results.append({'ladd_idx': ladd_indx, 'perf_gnn': perf_gnn, 'perf_feat': perf_feat})

print("Mean, >= 0.8, FCNN:", np.mean(np.array([tree.ladderization_index() for tree in trees_good_feat])))
print("Mean, < 0.8, FCNN:", np.mean(np.array([tree.ladderization_index() for tree in trees_bad_feat])))

print("Mean, >= 0.8, GNN:", np.mean(np.array([tree.ladderization_index() for tree in trees_good_gnn])))
print("Mean, < 0.8, GNN:", np.mean(np.array([tree.ladderization_index() for tree in trees_bad_gnn])))


df = pd.DataFrame(results)
df.to_csv("/home/mi/maxa55/Master_Thesis/Results/Plots/CSVs/ladd_idx.csv", index=False)

rho, _ = spearmanr(df['perf_feat'], df['ladd_idx'])
print(f"Spearman rank correlation, FCNN: {rho:.2f}")

rho, _ = spearmanr(df['perf_gnn'], df['ladd_idx'])
print(f"Spearman rank correlation, GNN: {rho:.2f}")

plt.scatter(x='perf_gnn', y='ladd_idx', alpha=0.6, data=df)
plt.xlabel("Perfromance")
plt.ylabel("Ladderization Index")
plt.title("Correlation Performance with Ladderization Index: GNN-4")
plt.grid(True)
plt.savefig("/home/mi/maxa55/Master_Thesis/Results/Plots/ladd_idx_gnn.png")
plt.show()    

plt.scatter(x='perf_feat', y='ladd_idx', alpha=0.6, data=df)
plt.xlabel("Perfromance")
plt.ylabel("Ladderization Index")
plt.title("Correlation Performance with Ladderization Index: FCNN-3")
plt.grid(True)
plt.savefig("/home/mi/maxa55/Master_Thesis/Results/Plots/ladd_idx_feat.png")
plt.show()   