import sys
sys.path.append('Tree')
from mutation_tree import MutationTree
from helper import *

import numpy as np

all_data = read_data("/home/mi/maxa55/Master_Thesis/Data/12x24", noisy=False, validation= True)
all_newick = read_newick("/home/mi/maxa55/Master_Thesis/Data/12x24", validation= True)

total_perf = 0
last_perc = -1

for i in range(len(all_data)):

    perc = int(100*i/len(all_data))
    if (perc != last_perc):
        print(perc)
        last_perc = perc

    n_mut = all_data[i].shape[0]
    n_cells = all_data[i].shape[1]
    tree = MutationTree(n_mut, n_cells)
    pvec = np.repeat(n_mut, n_mut + 1)
    pvec[-1] = -1
    tree.use_parent_vec(pvec, n_mut)
    start_llh = round(tree.conditional_llh(all_data[i], 0.01, 0.2), 4)
    current_llh = start_llh

    while(True):
        all_spr = np.argwhere(tree.all_possible_spr == 1)
        next_llhs_spr = np.full(len(all_spr), -np.inf)
        for j in range(len(all_spr)):
            new_tree = MutationTree(n_mut, n_cells)
            new_tree.copy_structure(tree)
            new_tree.perf_spr(all_spr[j, 0], all_spr[j, 1])
            next_llhs_spr[j] = new_tree.conditional_llh(all_data[i], 0.01, 0.2)
        
        all_swaps = np.where((tree.parent_vec != (tree.n_vtx - 1)))[0][0:-1]
        next_llhs_swap = np.full(len(all_swaps), -np.inf)
        for k in range(len(all_swaps)):
            new_tree = MutationTree(n_mut, n_cells)
            new_tree.copy_structure(tree)
            new_tree.swap(all_swaps[k])
            next_llhs_swap[k] = new_tree.conditional_llh(all_data[i], 0.01, 0.2)
        
        max_spr_llh = round(np.max(next_llhs_spr), 4)
        if len(next_llhs_swap) == 0: max_swap_llh = -np.inf
        else: max_swap_llh = round(np.max(next_llhs_swap), 4)
        if (max(max_spr_llh, max_spr_llh) <= current_llh): break

        if max_spr_llh > max_swap_llh:
            spr = all_spr[np.argmax(next_llhs_spr)]
            tree.perf_spr(spr[0], spr[1])
        else:
            swap = all_swaps[np.argmax(next_llhs_swap)]
            tree.swap(swap)

        current_llh = round(tree.conditional_llh(all_data[i], 0.01, 0.2), 4)
    
    gt_tree = MutationTree(n_mut, n_cells, all_newick[i])
    gt_llh = round(gt_tree.conditional_llh(all_data[i], 0.01, 0.2), 4)
    total_perf += (abs(current_llh - start_llh)/abs(gt_llh - start_llh))

print(round(total_perf/len(all_data), 4))