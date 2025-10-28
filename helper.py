import numpy as np
import math
import random
from collections import namedtuple, deque
import glob
import time
import re

def convert(bitlist):
    out = 0
    for bit in bitlist:
      out = (out << 1) | int(bit)
    return(out)

def sort_matrix(mat):
    int_cols = [convert(mat[:,j]) for j in range(np.shape(mat)[1])]
    indices = np.argsort(int_cols)
    mat = mat[:, indices]
    return mat

def read_data(path, noisy = False, validation = False):
    a = "train_"
    b = ".data"
    if validation: a = "valid_" 
    if noisy: b = ".noisy"
    pattern = path+"/"+a+"*"+b
    files = glob.glob(pattern)
    files = sorted(files)
    output_list = []
    d_list = []
    for f in files:
        match = re.search(r"_d(\d+)", f)
        if match:
            d_list.append(int(match.group(1)))
        file = open(f, "r")
        input = file.read()
        rows = input.split('\n')
        rows = rows[0:len(rows)-1]
        matrix = [row.split(' ') for row in rows]
        matrix = sort_matrix(np.array(matrix, dtype=int))
        output_list.append(matrix)
    return output_list, files, d_list

def stochastic_choice(m, target_rate):

    exact = m * target_rate
    d_floor, d_ceil = int(np.floor(exact)), int(np.ceil(exact))
    if d_floor == d_ceil:
        return d_floor
    p_ceil = exact - d_floor
    return np.random.choice([d_floor, d_ceil], p=[1-p_ceil, p_ceil])

def parse_filename(filename):
    """
    Returns: (idx, n, m, dmut, doublets)
    dmut = None if not present
    """
    # Match optional _dX_ before the final _<doublets>doublets
    match = re.search(
        r"_(\d+)__n(\d+)_m(\d+).*?(?:_d(\d+))?_(\d+)doublets",
        filename
    )
    if not match:
        raise ValueError(f"Could not parse {filename}")
    
    idx, n, m, dmut, doublets = match.groups()
    return int(idx), int(n), int(m), (int(dmut) if dmut else None), int(doublets)

def read_data_double(path, doublet_rate, noisy = False, validation = False):
    a = "train_"
    b = ".data"
    if validation: a = "valid_" 
    if noisy: b = ".noisy"

    pattern = path + "/" + a + "*" + b
    files = sorted(glob.glob(pattern))

    # Group files by (idx, n, m, dmut)
    grouped = {}
    for f in files:
        idx, n, m, dmut, doublets = parse_filename(f)
        grouped.setdefault((idx, n, m, dmut), []).append((doublets, f))

    output_list = []
    file_list = []
    for (idx, n, m, dmut), dfiles in grouped.items():
        # stochastic target
        target_d = stochastic_choice(m, doublet_rate)
        # find candidates with correct doublet count
        candidates = [f for d, f in dfiles if d == target_d]
        if not candidates:
            # fallback: use closest available
            closest_d = min(dfiles, key=lambda x: abs(x[0]-target_d))[0]
            candidates = [f for d, f in dfiles if d == closest_d]
        chosen_file = random.choice(candidates)
        file_list.append(chosen_file)
        # load chosen file
        with open(chosen_file, "r") as fh:
            rows = fh.read().splitlines()
        matrix = [row.split(' ') for row in rows if row.strip()]
        matrix = sort_matrix(np.array(matrix, dtype=int))
        output_list.append(matrix)

    return output_list, file_list

def read_newick(path, validation = False):
    a = "train_"
    if validation: a = "valid_"
    pattern = path+"/"+a+"*.newick"
    files = glob.glob(pattern)
    files = sorted(files)
    output_list = []
    for f in files:
        file = open(f, "r")
        content = file.read()
        output_list.append(content.split("\n")[0])
    return output_list

Transition = namedtuple('Transition', ('state', 'matrix', 'action', 'next_state', 'next_actions', 'reward', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Scheduler:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        self.steps = 0

    def step(self):
        self.steps += 1

    def get_instance(self):
        return self.end + (self.start - self.end) * math.exp(-self.steps / self.decay)

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.4f} seconds")
        return result
    return wrapper