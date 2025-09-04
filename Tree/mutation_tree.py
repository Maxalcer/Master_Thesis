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
        for vtx in range(self.n_vtx):
            if len(self._clist[vtx]) == 1: ladder_nodes += 1
            if len(self._clist[vtx]) >= 1: internal_nodes += 1
        return ladder_nodes/internal_nodes

    def tree_features(self, data, alpha, beta, both_1, a1_b0, a0_b1):
        mean_chld, std_chld = self.branching_stats()
        n_leaves = self.n_leaves(self.n_mut)
        max_node, max_size = self.max_subclone_size(data, alpha, beta)

        return np.array([
            self.n_mut,
            self.n_cells,
            n_leaves/self.n_vtx,
            self.max_depth()/self.n_vtx,
            n_leaves / (self.n_vtx - n_leaves),
            self.colless_index(),
            self.ladderization_index(),
            mean_chld/self.n_vtx,
            std_chld/self.n_vtx,
            max_node/self.n_vtx,
            max_size/self.n_cells,
            np.mean(both_1)/self.n_cells, np.std(both_1)/self.n_cells,
            np.mean(a1_b0)/self.n_cells, np.std(a1_b0)/self.n_cells,
            np.mean(a0_b1)/self.n_cells, np.std(a0_b1)/self.n_cells,
            self.conditional_llh(data, alpha, beta)
        ])
    
    def tree_features_sub(self, data, alpha, beta):
        mean_chld, std_chld = self.branching_stats()
        n_leaves = self.n_leaves(self.n_mut)
        max_node, max_size = self.max_subclone_size(data, alpha, beta)

        return np.array([
            self.n_mut,
            self.n_cells,
            n_leaves/self.n_vtx,
            self.max_depth()/self.n_vtx,
            n_leaves / (self.n_vtx - n_leaves),
            self.colless_index(),
            self.ladderization_index(),
            mean_chld/self.n_vtx,
            std_chld/self.n_vtx,
            max_node/self.n_vtx,
            max_size/self.n_cells
        ])


    def spr_features(self, spr, max_node, both_1, a1_b0, a0_b1):
        src, tgt = spr
        depth_src = self.get_depth(src)
        depth_tgt = self.get_depth(tgt)
        return np.array([
            self.n_leaves(src)/self.n_vtx,
            int(self.isdescendant(max_node, src)),
            depth_src/self.n_vtx,
            depth_tgt/self.n_vtx,
            self.distance(src, tgt, depth_src, depth_tgt)/self.n_vtx,
            both_1/self.n_cells,
            a1_b0/self.n_cells,
            a0_b1/self.n_cells,
            src,
            tgt
        ])
    
    def spr_features_sub(self, spr, max_node):
        src, tgt = spr
        depth_src = self.get_depth(src)
        depth_tgt = self.get_depth(tgt)
        return np.array([
            self.n_leaves(src)/self.n_vtx,
            int(self.isdescendant(max_node, src)),
            depth_src/self.n_vtx,
            depth_tgt/self.n_vtx,
            self.distance(src, tgt, depth_src, depth_tgt)/self.n_vtx,
            src,
            tgt
        ])

    def feature_vectors(self, data, alpha, beta):
        # Preprocess and expand data
        data_exp = np.vstack([data, np.ones((1, data.shape[1]), dtype=bool)])
        parent_matrix = data_exp[self._pvec]

        both_1 = np.sum(data_exp & parent_matrix, axis=1)
        a1_b0  = np.sum(data_exp & ~parent_matrix, axis=1)
        a0_b1  = np.sum(~data_exp & parent_matrix, axis=1)

        # Tree-level features
        tree_feat = self.tree_features(data, alpha, beta, both_1, a1_b0, a0_b1)
        max_node = int(tree_feat[9]*self.n_vtx)
        # SPR features
        spr_indices = np.argwhere(self.all_possible_spr == 1)
        spr_feats = []
        swap_feats = []
        llhs = []

        for src, tgt in spr_indices:
            d_src, d_tgt = data_exp[src], data_exp[tgt]
            spr_feats.append(self.spr_features(
                [src, tgt], max_node,
                np.sum(d_src & d_tgt),
                np.sum(d_src & ~d_tgt),
                np.sum(~d_src & d_tgt)
            ))
            swap_feats.append([-1, -1, -1, -1])
            new_tree = MutationTree(self.n_mut, self.n_cells)
            new_tree.copy_structure(self)
            new_tree.perf_spr(src, tgt)
            llhs.append(new_tree.conditional_llh(data, alpha, beta))

        swaps_idx = np.where((self._pvec != (self.n_vtx - 1)))[0][0:-1]
        
        for si in swaps_idx:
            spr_feats.append([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
            d_i = data_exp[si]
            d_p = data_exp[self.parent(si)]
            swap_feats.append([si,
                               np.sum(d_i & d_p),
                               np.sum(d_i & ~d_p),
                               np.sum(~d_i & d_p)])
            new_tree = MutationTree(self.n_mut, self.n_cells)
            new_tree.copy_structure(self)
            new_tree.swap(si)
            llhs.append(new_tree.conditional_llh(data, alpha, beta))

        spr_feats = np.array(spr_feats)
        swap_feats = np.array(swap_feats)
        llhs = np.array(llhs).reshape(-1, 1)
        tree_feat_repeated = np.tile(tree_feat, (spr_feats.shape[0], 1))
        return np.hstack((tree_feat_repeated, spr_feats, swap_feats, llhs))
    
    def feature_vectors_sub(self, data, alpha, beta):

        # Tree-level features
        tree_feat = self.tree_features_sub(data, alpha, beta)
        max_node = int(tree_feat[9]*self.n_vtx)
        # SPR features
        spr_indices = np.argwhere(self.all_possible_spr == 1)
        spr_feats = []

        for src, tgt in spr_indices:
            spr_feats.append(self.spr_features_sub([src, tgt], max_node))

        spr_feats = np.array(spr_feats)
        tree_feat_repeated = np.tile(tree_feat, (spr_feats.shape[0], 1))
        return np.hstack((tree_feat_repeated, spr_feats))
    
    def node_features(self, data, alpha, beta):
        cell_attach = self.cell_attatchment(data, alpha, beta)
        data = np.vstack((data, np.ones((1, data.shape[1]), dtype=data.dtype))).astype(bool)
        spr = self.all_possible_spr
        spr = np.argwhere(spr == 1)
        height = self.max_depth()

        node_features = np.empty((self.n_vtx, 10), dtype=np.float32)

        node_features[:, 0] = np.arange(self.n_vtx) / self.n_vtx
        node_features[:, 1] = np.array([self.isroot(i) for i in range(self.n_vtx)], dtype=np.float32)
        node_features[:, 2] = np.array([self.isleaf(i) for i in range(self.n_vtx)], dtype=np.float32)
        node_features[:, 3] = np.array([self.get_depth(i) for i in range(self.n_vtx)], dtype=np.float32) / height
        node_features[:, 4] = np.array([len(self.children(i)) for i in range(self.n_vtx)], dtype=np.float32) / self.n_vtx

        node_features[:, 5] = np.bincount(cell_attach, minlength=self.n_vtx) / self.n_cells

        node_features[:, 6] = np.sum(data, axis=1) / self.n_cells

        parent_matrix = data[self._pvec]

        both_1 = np.sum(data & parent_matrix, axis=1)
        a1_b0 = np.sum(data & (~parent_matrix), axis=1)
        a0_b1 = np.sum((~data) & parent_matrix, axis=1)
        """
        descendants = self.get_all_descendants()

        mean_both_1 = np.array([np.mean(both_1[d]) if len(d) > 0 else -1.0 for d in descendants])
        std_both_1 = np.array([np.std(both_1[d]) if len(d) > 0 else -1.0 for d in descendants])

        mean_a1_b0 = np.array([np.mean(a1_b0[d]) if len(d) > 0 else -1.0 for d in descendants])
        std_a1_b0 = np.array([np.std(a1_b0[d]) if len(d) > 0 else -1.0 for d in descendants])

        mean_a0_b1 = np.array([np.mean(a0_b1[d]) if len(d) > 0 else -1.0 for d in descendants])
        std_a0_b1 = np.array([np.std(a0_b1[d]) if len(d) > 0 else -1.0 for d in descendants])
        """
        node_features[:, 7] = both_1 / self.n_cells
        node_features[:, 8] = a1_b0 / self.n_cells
        node_features[:, 9] = a0_b1 / self.n_cells
        """
        node_features[:, 10] = mean_both_1 / self.n_cells
        node_features[:, 11] = std_both_1 / self.n_cells
        node_features[:, 12] = mean_a1_b0 / self.n_cells
        node_features[:, 13] = std_a1_b0 / self.n_cells
        node_features[:, 14] = mean_a0_b1 / self.n_cells
        node_features[:, 15] = std_a0_b1 / self.n_cells 
        """    
        return node_features
    
    def spr_node_features(self, data):
        spr = self.all_possible_spr
        spr = np.argwhere(spr == 1)
        n_spr = len(spr)
        data = np.vstack((data, np.ones((1, data.shape[1]), dtype=data.dtype))).astype(bool)
        spr_features = np.zeros((n_spr, self.n_vtx, 6), dtype=np.float32)
        spr_features[:, :, 2:] = -1
        for idx, (subroot, target) in enumerate(spr):
            dist = self.distance(subroot, target)

            both_1 = np.sum(data[subroot] & data[target]) 
            a1_b0 = np.sum(data[subroot] & ~data[target])
            a0_b1 = np.sum(~data[subroot] & data[target]) 

            spr_features[idx, subroot, 0] = 1  
            spr_features[idx, target, 1] = 1   

            for i in [subroot, target]:
                spr_features[idx, i, 2] = dist / self.n_vtx
                spr_features[idx, i, 3] = both_1 / self.n_cells
                spr_features[idx, i, 4] = a1_b0 / self.n_cells
                spr_features[idx, i, 5] = a0_b1 / self.n_cells
        return spr_features

    def all_node_features(self, data, alpha, beta):
        node_features = self.node_features(data, alpha, beta)
        node_features = node_features[None, :, :]
        spr_features = self.spr_node_features(data)
        swaps_idx = np.where((self._pvec != (self.n_vtx - 1)))[0][0:-1]
        spr_dim = spr_features.shape[0]
        swaps_dim = len(swaps_idx)
        spr_features = np.concatenate([spr_features, np.full((swaps_dim, self.n_vtx, 6), -1, dtype=np.float32)])
        node_features = np.repeat(node_features, spr_dim + swaps_dim, axis=0)
        swaps = np.zeros((spr_dim + swaps_dim, self.n_vtx, 1), dtype=np.float32)
        if swaps_dim !=0:
            swap_batch_indices = np.arange(spr_dim, spr_dim + swaps_dim)
            swaps[swap_batch_indices, swaps_idx, 0] = 1
        return np.concatenate([node_features, spr_features, swaps], axis=2) 


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

    def descendants(self, vtx):
        stack = list(self.children(vtx))
        while stack:
            node = stack.pop()
            yield node
            stack.extend(self.children(node))

    def get_all_descendants(self):

        descendants = [[] for _ in range(self.n_vtx)]

        def post_order(v):
            for child in self.children(v):
                post_order(child)
                descendants[v].append(child)
                descendants[v].extend(descendants[child])

        post_order(self.main_root)

        return descendants

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

    def swap(self, vtx):
        p = self.parent(vtx)
        if p != (self.n_vtx - 1):
            new_pvec = self._pvec.copy()
            childr_vtx = np.where(self._pvec == vtx)
            childr_p = np.where(self._pvec == p)
            new_pvec[childr_vtx] = p
            new_pvec[childr_p] = vtx
            new_pvec[vtx] = self._pvec[p]
            new_pvec[p] = vtx
            self.use_parent_vec(new_pvec, len(new_pvec) - 1)  

    def perf_spr(self, subroot, target):
        if self._all_spr[subroot, target] == 1:
            self.prune(subroot)
            self.assign_parent(subroot, target)
            self._anc_mat = self.get_ancestor_matrix()
            self._edge_idx = self.get_edge_index()
            self._all_spr = self.get_all_spr()

    def get_all_spr(self):
        n = self.n_vtx
        all_moves = np.zeros((n - 1, n), dtype=np.uint8)
    
        for i in range(n - 1):
            pi = self.parent(i)
            for j in range(n):
                if i == j or pi == j:
                    continue
                if self.isdescendant(j, i):
                    continue
                if self.distance(i, j) >= 5:
                    continue
                all_moves[i, j] = 1
    
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