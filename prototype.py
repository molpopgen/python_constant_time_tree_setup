# Efficient computation of the pairwise divergence matrix.
import time
import itertools

import msprime
import tskit
import numba
import numpy as np


spec = [
    ("parent", numba.int32[:]),
    ("left_sib", numba.int32[:]),
    ("right_sib", numba.int32[:]),
    ("left_child", numba.int32[:]),
    ("right_child", numba.int32[:]),
    ("left_sample", numba.int32[:]),
    ("right_sample", numba.int32[:]),
    ("next_sample", numba.int32[:]),
    ("sample_index_map", numba.int32[:]),
    ("nodes_time", numba.float64[:]),
    ("samples", numba.int32[:]),
    ("edges_left", numba.float64[:]),
    ("edges_right", numba.float64[:]),
    ("edges_parent", numba.int32[:]),
    ("edges_child", numba.int32[:]),
    ("edge_insertion_order", numba.int32[:]),
    ("edge_removal_order", numba.int32[:]),
    ("sequence_length", numba.float64),
    ("mrca_start", numba.float64[:, :]),
    ("divergence", numba.float64[:, :]),
    # New spec items by KRT
    ("edge_insertion_index", numba.int32[:]),
    ("edge_removal_index", numba.int32[:]),
    ("tree_left", numba.float64[:]),
]


@numba.experimental.jitclass(spec)
class TreeIndex:
    def __init__(self,
                 edge_insertion_index,
                 edge_removal_index,
                 tree_left):
        self.edge_insertion_index = edge_insertion_index
        self.edge_removal_index = edge_removal_index
        self.tree_left = tree_left


def build_indexes(ts: tskit.TreeSequence) -> TreeIndex:
    edge_insertion_index = np.full(ts.num_trees, -1, dtype=np.int32)
    edge_removal_index = np.full(ts.num_trees, -1, dtype=np.int32)
    tree_left = np.full(ts.num_trees, -1, dtype=np.float64)

    insertion = 0
    removal = 0
    for (i, e) in enumerate(ts.edge_diffs()):
        edge_insertion_index[i] = insertion
        edge_removal_index[i] = removal
        tree_left[i] = e.interval.left
        removal += len([i for i in e.edges_out])
        insertion += len([i for i in e.edges_in])

    return TreeIndex(edge_insertion_index, edge_removal_index, tree_left)


@numba.experimental.jitclass(spec)
class Tree:
    def __init__(
        self,
        num_nodes,
        samples,
        nodes_time,
        edges_left,
        edges_right,
        edges_parent,
        edges_child,
        edge_insertion_order,
        edge_removal_order,
        sequence_length,
    ):
        # Quintuply linked tree
        self.parent = np.full(num_nodes, -1, dtype=np.int32)
        self.left_sib = np.full(num_nodes, -1, dtype=np.int32)
        self.right_sib = np.full(num_nodes, -1, dtype=np.int32)
        self.left_child = np.full(num_nodes, -1, dtype=np.int32)
        self.right_child = np.full(num_nodes, -1, dtype=np.int32)
        # Singly-linked sample list
        self.left_sample = np.full(num_nodes, -1, dtype=np.int32)
        self.right_sample = np.full(num_nodes, -1, dtype=np.int32)
        self.next_sample = np.full(num_nodes, -1, dtype=np.int32)
        # Sample lists refer to sample *index*
        self.sample_index_map = np.full(num_nodes, -1, dtype=np.int32)
        # Edges and indexes
        self.edges_left = edges_left
        self.edges_right = edges_right
        self.edges_parent = edges_parent
        self.edges_child = edges_child
        self.edge_insertion_order = edge_insertion_order
        self.edge_removal_order = edge_removal_order
        self.sequence_length = sequence_length
        self.nodes_time = nodes_time
        self.samples = samples

        n = samples.shape[0]
        for j in range(n):
            u = samples[j]
            self.sample_index_map[u] = j
            self.left_sample[u] = j
            self.right_sample[u] = j

    def initialize_to_index(self, indexes: TreeIndex, index: int):
        raise NotImplementedError

    # Doesn't work in compiled mode, but handy for debugging
    # def __str__(self):
    #     fmt = "{:<5}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}\n"
    #     s = fmt.format(
    #         "node",
    #         "parent",
    #         "lsib",
    #         "rsib",
    #         "lchild",
    #         "rchild",
    #         "nsamp",
    #         "lsamp",
    #         "rsamp",
    #     )
    #     for u in range(self.parent.shape[0]):
    #         s += fmt.format(
    #             u,
    #             self.parent[u],
    #             self.left_sib[u],
    #             self.right_sib[u],
    #             self.left_child[u],
    #             self.right_child[u],
    #             self.next_sample[u],
    #             self.left_sample[u],
    #             self.right_sample[u],
    #         )
    #     # Strip off trailing newline
    #     s += str(self.mrca_start)
    #     return s

    def remove_edge(self, p, c):
        lsib = self.left_sib[c]
        rsib = self.right_sib[c]
        if lsib == -1:
            self.left_child[p] = rsib
        else:
            self.right_sib[lsib] = rsib
        if rsib == -1:
            self.right_child[p] = lsib
        else:
            self.left_sib[rsib] = lsib
        self.parent[c] = -1
        self.left_sib[c] = -1
        self.right_sib[c] = -1

    def insert_edge(self, p, c):
        assert self.parent[c] == -1, "contradictory edges"
        self.parent[c] = p
        u = self.right_child[p]
        if u == -1:
            self.left_child[p] = c
            self.left_sib[c] = -1
            self.right_sib[c] = -1
        else:
            self.right_sib[u] = c
            self.left_sib[c] = u
            self.right_sib[c] = -1
        self.right_child[p] = c

    # def update_sample_list(self, parent):
    #     # This can surely be done more efficiently and elegantly. We are iterating
    #     # up the tree and iterating over all the siblings of the nodes we visit,
    #     # rebuilding the links as we go. This results in visiting the same nodes
    #     # over again, which if we have nodes with many siblings will surely be
    #     # expensive. Another consequence of the current approach is that the
    #     # next pointer contains an arbitrary value for the rightmost sample of
    #     # every root. This should point to NULL ideally, but it's quite tricky
    #     # to do in practise. It's easier to have a slightly uglier iteration
    #     # over samples.
    #     #
    #     # In the future it would be good have a more efficient version of this
    #     # algorithm using next and prev pointers that we keep up to date at all
    #     # times, and which we use to patch the lists together more efficiently.
    #     u = parent
    #     while u != -1:
    #         sample_index = self.sample_index_map[u]
    #         if sample_index != -1:
    #             self.right_sample[u] = self.left_sample[u]
    #         else:
    #             self.right_sample[u] = -1
    #             self.left_sample[u] = -1
    #         v = self.left_child[u]
    #         while v != -1:
    #             if self.left_sample[v] != -1:
    #                 assert self.right_sample[v] != -1
    #                 if self.left_sample[u] == -1:
    #                     self.left_sample[u] = self.left_sample[v]
    #                     self.right_sample[u] = self.right_sample[v]
    #                 else:
    #                     self.next_sample[self.right_sample[u]
    #                                      ] = self.left_sample[v]
    #                     self.right_sample[u] = self.right_sample[v]
    #             v = self.right_sib[v]
    #         u = self.parent[u]

    # def _samples(self, u):
    #     ret = []
    #     x = self.left_sample[u]
    #     if x != -1:
    #         while True:
    #             ret.append(x)
    #             if x == self.right_sample[u]:
    #                 break
    #             x = self.next_sample[x]
    #     return ret

    # def _sibs(self, u):
    #     sibs = []
    #     p = self.parent[u]
    #     if p != -1:
    #         v = self.left_child[p]
    #         while v != -1:
    #             if v != u:
    #                 sibs.append(v)
    #             v = self.right_sib[v]
    #     return sibs

    # def start_mrca_pairs(self, u, v, pos):
    #     for w in self._samples(u):
    #         for x in self._samples(v):
    #             assert x != w
    #             self.mrca_start[x, w] = pos
    #             self.mrca_start[w, x] = pos

    # def start_mrcas(self, edge_parent, edge_child, pos):
    #     u = edge_child
    #     v = edge_parent
    #     while v != -1:
    #         for sib in self._sibs(u):
    #             self.start_mrca_pairs(edge_child, sib, pos)
    #         u = v
    #         v = self.parent[v]

    # def flush_mrca_pairs(self, mrca, u, v, pos):
    #     t = self.nodes_time
    #     s = self.samples
    #     for w in self._samples(u):
    #         for x in self._samples(v):
    #             start = self.mrca_start[x, w]
    #             distance = pos - start
    #             path_bl = 2 * t[mrca] - t[s[x]] - t[s[w]]
    #             area = path_bl * distance
    #             self.divergence[w, x] += area
    #             self.divergence[x, w] += area
    #             assert x != w

    # def flush_mrcas(self, edge_parent, edge_child, pos):
    #     u = edge_child
    #     v = edge_parent
    #     while v != -1:
    #         for sib in self._sibs(u):
    #             self.flush_mrca_pairs(v, edge_child, sib, pos)
    #         u = v
    #         v = self.parent[v]

    # def run(self):
    #     sequence_length = self.sequence_length
    #     M = self.edges_left.shape[0]
    #     in_order = self.edge_insertion_order
    #     out_order = self.edge_removal_order
    #     edges_left = self.edges_left
    #     edges_right = self.edges_right
    #     edges_parent = self.edges_parent
    #     edges_child = self.edges_child

    #     j = 0
    #     k = 0
    #     left = 0

    #     while j < M or left <= sequence_length:
    #         while k < M and edges_right[out_order[k]] == left:
    #             p = edges_parent[out_order[k]]
    #             c = edges_child[out_order[k]]
    #             self.flush_mrcas(p, c, left)
    #             self.remove_edge(p, c)
    #             self.update_sample_list(p)
    #             k += 1
    #         # TODO not sure if this is necessary or correct here, needs
    #         # to be validated. Note <= sequence_length above for left also.
    #         if k == M:
    #             break
    #         while j < M and edges_left[in_order[j]] == left:
    #             p = edges_parent[in_order[j]]
    #             c = edges_child[in_order[j]]
    #             self.insert_edge(p, c)
    #             self.update_sample_list(p)
    #             self.start_mrcas(p, c, left)
    #             j += 1
    #         right = sequence_length
    #         if j < M:
    #             right = min(right, edges_left[in_order[j]])
    #         if k < M:
    #             right = min(right, edges_right[out_order[k]])
    #         # print(left, right)
    #         # print(self)
    #         # yield left, right
    #         left = right
    #     return self.divergence


def make_tree_at_given_index(ts: tskit.TreeSequence, indexes: TreeIndex, tree_index: int) -> Tree:
    tree = Tree(
        ts.num_nodes,
        samples=ts.samples(),
        nodes_time=ts.nodes_time,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        edges_parent=ts.edges_parent,
        edges_child=ts.edges_child,
        edge_insertion_order=ts.indexes_edge_insertion_order,
        edge_removal_order=ts.indexes_edge_removal_order,
        sequence_length=ts.sequence_length)

    insertion = ts.indexes_edge_insertion_order
    pos = indexes.tree_left[tree_index]

    for i in insertion:
        if pos >= tree.edges_left[i] and pos < tree.edges_right[i]:
            tree.insert_edge(tree.edges_parent[i], tree.edges_child[i])

    return tree


def compare_perf():

    seed = 1234
    for n in [10, 100, 250, 500, 1000]:
        ts = msprime.sim_ancestry(
            n,
            ploidy=1,
            population_size=10**4,
            sequence_length=1e6,
            recombination_rate=1e-8,
            random_seed=seed,
        )
        indexes = build_indexes(ts)

        for i in range(ts.num_trees):
            tsktree = tskit.Tree(ts)
            # I did not know about this!
            tsktree.seek_index(i)
            assert tsktree.index == i

            tree = make_tree_at_given_index(ts, indexes, i)
            assert len(tsktree.parent_array) == len(tree.parent) + 1
            assert np.array_equal(
                tree.parent, tsktree.parent_array[:ts.num_nodes]), \
                f"{tree.parent} != {tsktree.parent_array}"
            assert np.array_equal(
                tree.left_child, tsktree.left_child_array[:ts.num_nodes]), \
                f"{tree.left_child} != {tsktree.left_child_array}"
            assert np.array_equal(
                tree.right_child, tsktree.right_child_array[:ts.num_nodes]), \
                f"{tree.right_child} != {tsktree.right_child_array}"


if __name__ == "__main__":

    np.set_printoptions(linewidth=500, precision=4)
    # verify()
    compare_perf()
