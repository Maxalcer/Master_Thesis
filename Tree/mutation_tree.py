import numpy as np
import graphviz
import warnings
from Bio import Phylo
from io import StringIO
from itertools import combinations

class MutationTree():
    def __init__(self, n_mut, n_cells, newick = None):
        self.n_mut = n_mut
        self.n_cells = n_cells
        self._pvec = np.ones(n_mut + 1, dtype=int) * -1 # parent vector
        self._clist = [[] for i in range(n_mut + 1)] + [list(range(n_mut + 1))]
        self._main_root = 0
        self._anc_mat = None
        self._edge_idx = None
        self._all_spr = None
        self.flipped = np.zeros(self.n_vtx, dtype=bool)
        self.reroot(self.wt)

        if newick is None: self.random_mutation_tree()
        else: self.use_newick_string(newick)

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
    def edge_index(self):
        return self._edge_idx.copy()

    @property
    def all_possible_spr(self):
        return self._all_spr.copy()
    
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
        self._edge_idx = self.get_edge_index()
        self._all_spr = self.get_all_spr()

    def use_newick_string(self, newick):
        tree = Phylo.read(StringIO(newick), "newick")

        for node in tree.find_clades():
            if node.confidence is not None and node.name is None:
                node.name = str(int(node.confidence))

        node_list = list(tree.find_clades(order="level"))
        node_to_index = {node: int(node.name) - 1 for node in node_list}

        parent_vector = [-1] * len(node_to_index)
        parent_vector = np.array(parent_vector)
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

    def n_leaves(self, vtx):
        return sum(1 for _ in self.leaves(vtx))

    def get_ancestor_matrix(self):
        A_T = np.zeros((self.n_mut, self.n_mut + 1))

        for i in range(self.n_mut):
            for j in range(self.n_mut):
                if ((i == j) or self.isdescendant(j, i)): A_T[i,j] = 1           
        return A_T
    
    def get_edge_index(self):
        child_indices = np.arange(self.n_vtx)
        mask = self.parent_vec >= 0

        parent_nodes = self.parent_vec[mask]
        child_nodes = child_indices[mask]

        edge_index = np.vstack([parent_nodes, child_nodes])
        return edge_index
    
    def get_depth(self, node):
        depth = 0
        while not self.isroot(node): 
            node = self.parent(node)
            depth += 1
        return depth

    def max_depth(self):
        max_depth = 1
        for i in range(self.n_mut):
            depth = self.get_depth(i)
            if depth > max_depth: max_depth = depth
        return max_depth

    def branching_stats(self):
        n_children = [len(self.children(i)) for i in range(self.n_vtx)]
        n_children = np.array(n_children)
        return np.mean(n_children), np.std(n_children)

    def max_subclone_size(self, data, alpha, beta, sig = None):
        if sig is None: sig = self.cell_attatchment(data, alpha, beta)
        bins = np.bincount(sig)
        indx = bins.argmax()
        return indx, bins[indx]

    def colless_index(self):
        total = 0
        for node in range(self.n_vtx):
            children = self._clist[node]
            if len(children) < 2:
                continue 
            leaf_counts = [self.n_leaves(child) for child in children]
            for a, b in combinations(leaf_counts, 2):
                total += abs(a - b)
        return total

    def ladderization_index(self):
        ladder_nodes = 0
        internal_nodes = 0
        for node in range(self.n_vtx):
            children = self._clist[node]
            if len(children) >= 2:
                internal_nodes += 1
                branching_children = sum(1 for c in children if any(self.leaves(c)))
                if branching_children == 1:
                    ladder_nodes += 1
        return ladder_nodes / internal_nodes if internal_nodes > 0 else 0

    def tree_features(self, data, alpha, beta):
        mean_chld, std_chld = self.branching_stats()
        n_leaves = self.n_leaves(self.n_mut)
        max_node, max_size = self.max_subclone_size(data, alpha, beta)
        feature_vector = np.array([self.n_mut,
                                   self.n_cells,
                                   n_leaves,
                                   self.max_depth(),
                                   (n_leaves/(self.n_vtx - n_leaves)),
                                   self.colless_index(),
                                   self.ladderization_index(),
                                   mean_chld,
                                   std_chld,
                                   max_node,
                                   max_size])
        return feature_vector

    def spr_features(self, spr, max_node):
        in_subtree = int(self.isdescendant(max_node, spr[0]))
        depth_subroot = self.get_depth(spr[0])
        depth_target = self.get_depth(spr[1])
        dist = self.distance(spr[0], spr[1], depth_subroot, depth_target)
        feature_vector = np.array([self.n_leaves(spr[0]),
                                   in_subtree,
                                   depth_subroot,
                                   depth_target,
                                   dist,
                                   spr[0],
                                   spr[1]])
        return feature_vector

    def feature_vector(self, data, alpha, beta, spr):
        tree_features = self.tree_features(data, alpha, beta)
        spr_fearures = self.spr_features(spr, tree_features[9])
        return np.concatenate((tree_features, spr_fearures))

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

    def lca(self, vtx_u, vtx_v):
        ancestors_u = set(self.ancestors(vtx_u))  
        while vtx_v not in ancestors_u:
            vtx_v = self._pvec[vtx_v]
        return vtx_v

    def distance(self, vtx_u, vtx_v, depth_u = None, depth_v = None):
        if depth_u is None: depth_u = self.get_depth(vtx_u)
        if depth_v is None: depth_v = self.get_depth(vtx_v)
        lca = self.lca(vtx_u, vtx_v)
        return depth_u + depth_v - 2 * self.get_depth(lca)
    
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
        self._edge_idx = self.get_edge_index()
        self._all_spr = self.get_all_spr()

    def assign_parent(self, vtx, new_parent):
        ''' Designate new_parent as the new parent of vtx. If new_parent is already the parent of vtx, nothing happens. '''

        self._clist[self._pvec[vtx]].remove(vtx)
        self._pvec[vtx] = new_parent
        self._clist[new_parent].append(vtx)

    def prune(self, subroot):
        ''' Prune the subtree whose root is subroot. If subroot is already a root, nothing happens. '''
        self.assign_parent(subroot, -1)

    def perf_spr(self, subroot, target):
        if self._all_spr[subroot, target] == 1:
            self.prune(subroot)
            self.assign_parent(subroot, target)
            self._anc_mat = self.get_ancestor_matrix()
            self._edge_idx = self.get_edge_index()
            self._all_spr = self.get_all_spr()

    def get_all_spr(self):
        all_moves = np.zeros((self.n_vtx-1, self.n_vtx))
        for i in range(self.n_vtx-1):
            for j in range(self.n_vtx):
                if ((self.parent(i) != j) & (not self.isdescendant(j, i)) & (i != j)): all_moves[i,j] = 1
        return all_moves

    ######## Methods for Liklihood calculation ########

    def cell_attatchment(self, data, alpha, beta):
        if (alpha == 0) & (beta == 0):
            cell_profiles = data.T
            inherited = self.ancestor_matrix.T
            diffs = np.abs(cell_profiles[:, None, :] - inherited[None, :, :])
            hamming = np.sum(diffs, axis=2)
            sig = np.argmin(hamming, axis=1)

            return sig
        else:
            p = np.log([1-alpha, beta, alpha, 1-beta])

            data_exp = data[:, :, None]
            anc_exp = self._anc_mat[:, None, :]
            indices = (data_exp * 2 + anc_exp).astype(int)
            log_probs = p[indices]
            llh = np.sum(log_probs, axis=0)
            sig = np.argmax(llh, axis=1)

            return sig

    def conditional_llh(self, data, alpha, beta, sig = None):
        if sig is None: sig = self.cell_attatchment(data, alpha, beta)
        if (alpha == 0) & (beta == 0):
            reconstructed = self.ancestor_matrix[:, sig]
            matches = (reconstructed == data).sum()
            return matches/(self.n_mut * self.n_cells)
        else:
            p = np.log([1-alpha, beta, alpha, 1-beta])

            anc_vals = self._anc_mat[np.arange(self.n_mut)[:, None], sig]

            index = (data * 2 + anc_vals).astype(int)

            llh = np.sum(p[index])
            return llh

    def dfs(self, subroot):
        ''' Traverse the subtree rooted at subroot, in DFS order '''
        yield subroot
        for child in self._clist[subroot]:
            yield from self.dfs(child)

    def to_graphviz(self, filename=None, engine='dot'): 
        dgraph = graphviz.Digraph(filename=filename, engine=engine)
        
        dgraph.node(str(self.wt), label='rt', shape='circle', style='solid', color='blue', fillcolor='white')
        for vtx in range(self.n_mut):
            dgraph.node(str(vtx), shape='circle', style='solid', color='blue', fillcolor='white')
            if self.isroot(vtx):
                # for root, create a corresponding void node
                dgraph.node(f'void_{vtx}', label='', shape='point')
                dgraph.edge(f'void_{vtx}', str(vtx))
            else:
                dgraph.node(str(vtx), shape='circle', style='solid', color='blue', fillcolor='white')
                dgraph.edge(str(self.parent(vtx)), str(vtx))
        
        return dgraph