import numpy as np
import glob

def read_data(path):
    files = glob.glob(path+"/*.noisy")
    output_list = []
    for f in files:
        file = open(f, "r")
        input = file.read()
        rows = input.split('\n')
        rows = rows[0:len(rows)-1]
        matrix = [row.split(' ') for row in rows]
        output_list.append(np.array(matrix, dtype=int))
    return output_list

def read_newick(path):
    files = glob.glob(path+"/*.newick")
    output_list = []
    for f in files:
        file = open(f, "r")
        content = file.read()
        output_list.append(content.split("\n")[0])
    return output_list

