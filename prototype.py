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
class Tree:
    def __init__(
        self,
        num_nodes,
        num_trees,
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

        # Keep the index internal to the Tree.
        # In real life, we'd probably keep this in the treeseq
        # for reuse.
        self.edge_insertion_index = np.full(
            num_trees, np.nan, dtype=np.int32)
        self.edge_removal_index = np.full(num_trees, np.nan, dtype=np.int32)
        self.tree_left = np.full(num_trees, np.nan, dtype=np.float64)

        self.index()

        n = samples.shape[0]
        for j in range(n):
            u = samples[j]
            self.sample_index_map[u] = j
            self.left_sample[u] = j
            self.right_sample[u] = j

    def seek_to_index(self, tree_index: int):
        insertion = self.edge_insertion_order
        edges_left = self.edges_left
        edges_right = self.edges_right
        edges_parent = self.edges_parent
        edges_child = self.edges_child
        pos = self.tree_left[tree_index]

        for i in insertion:
            if pos >= edges_left[i] and pos < edges_right[i]:
                self.insert_edge(edges_parent[i], edges_child[i])

    def index(self):
        edge_insertion_index = self.edge_insertion_index
        edge_removal_index = self.edge_removal_index
        tree_left = self.tree_left
        sequence_length = self.sequence_length
        M = self.edges_left.shape[0]
        in_order = self.edge_insertion_order
        out_order = self.edge_removal_order
        edges_left = self.edges_left
        edges_right = self.edges_right

        j = 0
        k = 0
        left = 0.0
        tree_index = 0

        while j < M or left < sequence_length:
            edge_insertion_index[tree_index] = j
            edge_removal_index[tree_index] = k
            tree_left[tree_index] = left
            while k < M and edges_right[out_order[k]] == left:
                k += 1
            # if k == M:
            #     return
            while j < M and edges_left[in_order[j]] == left:
                j += 1
            right = sequence_length
            if j < M:
                right = min(right, edges_left[in_order[j]])
            if k < M:
                right = min(right, edges_right[out_order[k]])
            left = right
            tree_index += 1

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

    def advance(self, tree_index: int):
        sequence_length = self.sequence_length
        M = self.edges_left.shape[0]
        in_order = self.edge_insertion_order
        out_order = self.edge_removal_order
        edges_left = self.edges_left
        edges_right = self.edges_right
        edges_parent = self.edges_parent
        edges_child = self.edges_child

        j = self.edge_insertion_index[tree_index]
        k = self.edge_removal_index[tree_index]
        left = self.tree_left[tree_index]

        if j < M or left <= sequence_length:
            while k < M and edges_right[out_order[k]] == left:
                p = edges_parent[out_order[k]]
                c = edges_child[out_order[k]]
                self.remove_edge(p, c)
                # self.update_sample_list(p)
                k += 1
            # TODO not sure if this is necessary or correct here, needs
            # to be validated. Note <= sequence_length above for left also.
            if k == M:
                return
            while j < M and edges_left[in_order[j]] == left:
                p = edges_parent[in_order[j]]
                c = edges_child[in_order[j]]
                self.insert_edge(p, c)
                # self.update_sample_list(p)
                j += 1
            right = sequence_length
            if j < M:
                right = min(right, edges_left[in_order[j]])
            if k < M:
                right = min(right, edges_right[out_order[k]])
            # print(left, right)
            # print(self)
            # yield left, right
            left = right


def make_tree_at_given_index(ts: tskit.TreeSequence, tree_index: int) -> Tree:
    tree = Tree(
        ts.num_nodes,
        ts.num_trees,
        samples=ts.samples(),
        nodes_time=ts.nodes_time,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        edges_parent=ts.edges_parent,
        edges_child=ts.edges_child,
        edge_insertion_order=ts.indexes_edge_insertion_order,
        edge_removal_order=ts.indexes_edge_removal_order,
        sequence_length=ts.sequence_length)
    tree.seek_to_index(tree_index)
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
        # indexes = build_indexes(ts)

        for i in range(ts.num_trees):
            tsktree = tskit.Tree(ts)
            # I did not know about this!
            before = time.perf_counter()
            tsktree.seek_index(i)
            time_lib = time.perf_counter() - before
            assert tsktree.index == i

            before = time.perf_counter()
            tree = make_tree_at_given_index(ts, i)
            time_prototype = time.perf_counter() - before
            # print(i, ts.num_trees, time_lib, time_prototype)
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

            # FIXME: the laziness here is painful
            dummy = i + 1
            while tsktree.next():
                tree.advance(dummy)
                assert np.array_equal(
                    tree.parent, tsktree.parent_array[:ts.num_nodes]), \
                    f"{tree.parent} != {tsktree.parent_array}"
                assert np.array_equal(
                    tree.left_child, tsktree.left_child_array[:ts.num_nodes]), \
                    f"{tree.left_child} != {tsktree.left_child_array}"
                assert np.array_equal(
                    tree.right_child, tsktree.right_child_array[:ts.num_nodes]), \
                    f"{tree.right_child} != {tsktree.right_child_array}"
                dummy += 1


if __name__ == "__main__":

    np.set_printoptions(linewidth=500, precision=4)
    # verify()
    compare_perf()
