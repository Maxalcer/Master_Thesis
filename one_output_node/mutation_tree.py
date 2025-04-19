import numpy as np
import graphviz
import warnings
from Bio import Phylo
from io import StringIO

class MutationTree():
    def __init__(self, n_mut=2, n_cells=0):
        if n_mut < 2:
            warnings.warn('Mutation tree too small, nothing to explore.', RuntimeWarning)
        
        self.n_mut = n_mut
        self.n_cells = n_cells

        self._pvec = np.ones(n_mut + 1, dtype=int) * -1 # parent vector
        self._clist = [[] for i in range(n_mut + 1)] + [list(range(n_mut + 1))]
        self._main_root = 0

        self._anc_mat = np.zeros((self.n_mut, self.n_mut + 1))

        self.flipped = np.zeros(self.n_vtx, dtype=bool)

        self.reroot(self.wt)
        self.random_mutation_tree()

    ######## Tree properties ########
    @property
    def roots(self):
        return self._clist[-1]
    
    @property
    def main_root(self):
        return self._main_root
    
    def reroot(self, new_main_root):
        if not self.isroot(new_main_root):
            self.prune(new_main_root)
        self._main_root = new_main_root

    @property
    def n_vtx(self):
        return len(self._pvec)

    @property
    def parent_vec(self):
        return self._pvec.copy()
    
    @property
    def ancestor_matrix(self):
        return self._anc_mat.copy()
    
    @property
    def wt(self):
        return self.n_mut

    @property
    def n_cells(self):
        return len(self.cell_loc)

    @n_cells.setter
    def n_cells(self, n_cells):
        self.cell_loc = np.ones(n_cells, dtype=int) * -1

    def use_parent_vec(self, new_parent_vec, main_root=None):
        if len(new_parent_vec) != self.n_vtx:
            raise ValueError('Parent vector must have the same length as number of vertices.')

        self._pvec = new_parent_vec
        self._clist = [[] for i in range(self.n_vtx)] + [[]]

        for vtx in range(self.n_vtx):
            self._clist[new_parent_vec[vtx]].append(vtx)

        if main_root is None:
            self._main_root = self.roots[0]
        elif self._pvec[main_root] != -1:
            raise ValueError('Provided main root is not a root.')
        else:
            self._main_root = main_root
        self._anc_mat = self.get_ancestor_matrix()

    def use_newick_string(self, newick):
        tree = Phylo.read(StringIO(newick), "newick")

        for node in tree.find_clades():
            if node.confidence is not None and node.name is None:
                node.name = str(int(node.confidence))

        node_list = list(tree.find_clades(order="level"))
        node_to_index = {node: int(node.name) - 1 for node in node_list}

        parent_vector = [-1] * len(node_to_index)
        for node in node_list:
            for child in node.clades:
                parent_vector[node_to_index[child]] = node_to_index[node]
        
        self.use_parent_vec(parent_vector, len(parent_vector) - 1)


    def copy_structure(self, other):
        self.use_parent_vec(other.parent_vec, other.main_root)

    def leaves(self, subroot):
        for vtx in self.dfs(subroot):
            if self.isleaf(vtx):
                yield vtx
    
    def get_ancestor_matrix(self):
        A_T = np.zeros((self.n_mut, self.n_mut + 1))

        for i in range(self.n_mut):
            for j in range(self.n_mut):
                if ((i == j) or self.isdescendant(j, i)): A_T[i,j] = 1           
        return A_T

    ######## Single-vertex properties ########
    def parent(self, vtx):
        return self._pvec[vtx]

    def children(self, vtx):
        return self._clist[vtx]

    def isroot(self, vtx):
        return self._pvec[vtx] == -1
    
    def isleaf(self, vtx):
        return len(self._clist[vtx]) == 0

    def isdescendant(self, vtx, ancestor):
        return ancestor in self.ancestors(vtx)
    
    def ancestors(self, vtx):
        ''' Traverse all ancestors of a vertex, NOT including itself '''
        if not self.isroot(vtx):
            yield self._pvec[vtx]
            yield from self.ancestors(self._pvec[vtx])

    ######## Methods to manipulate tree structure ########

    def random_mutation_tree(self):
        # Randomly choose one mutation to have self.wt as its parent
        root_assigned = np.random.randint(0, self.n_mut - 1)
        self.assign_parent(root_assigned, self.wt)

        # Assign each mutation a random parent ensuring it's a valid tree
        for vtx in range(self.n_mut):
            if vtx == root_assigned:
                continue  # Skip the already assigned root mutation

            # Assign a random parent from the set of already assigned mutations or self.wt
            potential_parents = [i for i in range(self.n_mut) if i != vtx and self._pvec[i] != -1]
            parent = np.random.choice(potential_parents+[self.wt])
            self.assign_parent(vtx, parent)
        self._anc_mat = self.get_ancestor_matrix()

    def assign_parent(self, vtx, new_parent):
        ''' Designate new_parent as the new parent of vtx. If new_parent is already the parent of vtx, nothing happens. '''

        self._clist[self._pvec[vtx]].remove(vtx)
        self._pvec[vtx] = new_parent
        self._clist[new_parent].append(vtx)

    def prune(self, subroot):
        ''' Prune the subtree whose root is subroot. If subroot is already a root, nothing happens. '''
        self.assign_parent(subroot, -1)

    def perf_spr(self, subroot, target):
        self.prune(subroot)
        self.assign_parent(subroot, target)
        self._anc_mat = self.get_ancestor_matrix()

    
    def all_spr(self):
        all_moves = []
        for i in range(self.n_vtx-1):
            for j in range(self.n_vtx):
                if ((self.parent(i) != j) & (not self.isdescendant(j, i)) & (i != j)): all_moves.append([i,j])
        return all_moves

    ######## Methods for Liklihood calculation ########

    def cell_attatchment(self, data, alpha, beta):
        p = np.log([1-alpha, beta, alpha, 1-beta])

        sig = np.zeros(self.n_cells, dtype=int)

        for j in range(self.n_cells):
            llh = np.zeros(self.n_mut+1)
            for k in range(self.n_mut+1):
                for i in range(self.n_mut):
                    llh[k] += p[int((data[i,j] * 2) + (self._anc_mat[i, k]))]            
            sig[j] = np.argmax(llh)

        return sig

    def conditional_llh(self, data, alpha, beta):
        p = np.log([1-alpha, beta, alpha, 1-beta])
        sig = self.cell_attatchment(data, alpha, beta)

        llh = 0

        for i in range(self.n_mut):
            for j in range(self.n_cells):
                llh += p[int((data[i,j] * 2) + (self._anc_mat[i, sig[j]]))]
        return llh

    def to_graphviz(self, filename=None, engine='dot'): 
        dgraph = graphviz.Digraph(filename=filename, engine=engine)
        
        dgraph.node(str(self.wt), label='wt', shape='rectangle', color='gray')
        for vtx in range(self.n_mut):
            dgraph.node(str(vtx), shape='rectangle', style='filled', color='gray')
            if self.isroot(vtx):
                # for root, create a corresponding void node
                dgraph.node(f'void_{vtx}', label='', shape='point')
                dgraph.edge(f'void_{vtx}', str(vtx))
            else:
                dgraph.node(str(vtx), shape='rectangle', style='filled', color='gray')
                dgraph.edge(str(self.parent(vtx)), str(vtx))
        
        for i in range(self.n_cells):
            name = 'c' + str(i)
            dgraph.node(name, shape = 'circle')
            dgraph.edge(str(self.cell_loc[i]), name)
        
        return dgraph