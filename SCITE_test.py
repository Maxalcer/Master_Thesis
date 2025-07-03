import sys
sys.path.append('Tree')
from mutation_tree import MutationTree
from helper import *

import glob
import numpy as np
import subprocess

def get_pvec(filename):
    with open(filename, 'r') as f:
        content = f.read().strip()  # read the entire content and strip whitespace
    # Split by whitespace and convert each to int
    numbers = list(map(int, content.split()))
    numbers.append(-1)
    # Convert list to numpy array
    arr = np.array(numbers)
    return arr

matrix_paths = glob.glob("/home/mi/maxa55/Master_Thesis/Data/10x20/valid*.noisy")
matrix_paths = sorted(matrix_paths)

all_data = read_data("/home/mi/maxa55/Master_Thesis/Data/10x20", noisy=True, validation=True)
all_newick = read_newick("/home/mi/maxa55/Master_Thesis/Data/10x20", validation=True)
alpha = 0.01
beta = 0.2
perf = 0
for i in range(len(all_data)):
    subprocess.run(['/home/mi/maxa55/Master_Thesis/SCITE/./scite', 
                    '-i', matrix_paths[i],
                    '-n', '10',
                    '-m', '20',
                    '-r', '1',
                    '-l', '10000',
                    '-fd', str(alpha),
                    '-ad', str(beta),
                    '-o', '/home/mi/maxa55/Master_Thesis/SCITE/test_results/other'], 
                   capture_output=False, text=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    start_pvec = get_pvec('/home/mi/maxa55/Master_Thesis/SCITE/test_results/start.txt')
    #end_pvec = get_pvec('/home/mi/maxa55/Master_Thesis/SCITE/test_results/end.txt')
    start_tree = MutationTree(10, 20)
    start_tree.use_parent_vec(start_pvec)
    start_llh = start_tree.conditional_llh(all_data[i], alpha, beta)
    with open("/home/mi/maxa55/Master_Thesis/SCITE/test_results/other_ml0.newick", 'r') as f:
        newick = f.readline().strip()
    end_tree = MutationTree(10, 20, newick)
    #end_tree.use_parent_vec(end_pvec)
    end_llh = end_tree.conditional_llh(all_data[i], alpha, beta)
    gt_tree = MutationTree(10, 20, all_newick[i])
    gt_llh = gt_tree.conditional_llh(all_data[i], alpha, beta)
    perf += (end_llh - start_llh)/(gt_llh - start_llh)
    subprocess.run(['rm', '-r', '/home/mi/maxa55/Master_Thesis/SCITE/test_results/*'], 
                   capture_output=False, text=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print(perf/len(all_data))