import numpy as np
import math
import random
from collections import namedtuple, deque
import glob

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
    for f in files:
        file = open(f, "r")
        input = file.read()
        rows = input.split('\n')
        rows = rows[0:len(rows)-1]
        matrix = [row.split(' ') for row in rows]
        matrix = sort_matrix(np.array(matrix, dtype=int))
        output_list.append(matrix)
    return output_list

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