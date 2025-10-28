import sys
sys.path.append('../')
sys.path.append('../Network')
sys.path.append('../Tree')
sys.path.append('../GNN_Features')
sys.path.append('../features')
sys.path.append('../Enviroment')
from helper import *
from agent_features_fixed import Agent_Features_Fixed
from mutation_tree import MutationTree

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
#import torch

alpha = 0.01
beta = 0.2

all_data, _ = read_data("/home/mi/maxa55/Master_Thesis/Data/test/100x200", noisy=True, validation= True)
all_newick = read_newick("/home/mi/maxa55/Master_Thesis/Data/test/100x200", validation= True)

agent_feat = Agent_Features_Fixed(alpha, beta)
agent_feat.load_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/feature/trained_net_fixed_mix_15.py")

matrix = all_data[0]
newick = all_newick[0]

n_mut = matrix.shape[0]
n_cells = matrix.shape[1]

gt_tree = MutationTree(n_mut, n_cells, newick)
gt_llh = round(gt_tree.conditional_llh(matrix, 0.01, 0.2), 4)
print("start time:", datetime.datetime.now())
print("ground truth llh:", gt_llh)

start_llh, end_llh = agent_feat.solve_tree(matrix, n_mut*2)

print("start llh:", start_llh)
print("end llh:", end_llh)
print("end time:", datetime.datetime.now())

