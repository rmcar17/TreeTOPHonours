import sys
from copy import copy
from itertools import permutations, product

from cogent3 import make_tree
from munkres import Munkres  # https://pypi.org/project/munkres3/
from numpy import array, empty, logical_not, zeros
from tree import Tree


def convert_tree_to_vectors(tree, tip_names):
    ref_tip = tip_names[0]
    name_set = set(tip_names)
    name_index = {n: i for i, n in enumerate(tip_names)}
    # we get the tree as a set of splits.
    splits = tree.subsets()
    rows = []
    for split in splits:
        row = zeros(len(tip_names), int)
        # Cogent only returns one side of a
        # split, so we build the other side
        if ref_tip in split:
            names = list(split)
        else:
            names = list(name_set - split)
        indices = [name_index[n] for n in names]
        row[indices] = True
        rows.append(row)
    rows = array(rows)
    return rows


def Hamming(V1, V2):
    return (V1 != V2).sum()


def W(V1, V2):
    return min(Hamming(V1, V2), Hamming(V1, 1 - V2))


def bipartite_graph(T1, T2):
    if not len(T1) == len(T2):
        print("number of edges must be equal")
        print(T1)
        print(T1.shape)
        print()
        print(T2)
        print(T2.shape)
        raise RuntimeError

    B = empty([len(T1)] * 2, int)
    for i, j in product(*[range(len(T1))] * 2):
        B[i, j] = W(T1[i], T2[j])
    return B


def matched_distance(T1, T2):
    B = bipartite_graph(T1, T2)
    m = Munkres()
    # print("START")
    matching = m.compute(copy(B))  # compute overwrites B
    # print("END")
    return B[tuple(zip(*matching))].sum()


def main():
    a = make_tree(treestring="(1,(((2,3),4),(5,((6,(7,(8,9))),(10,11)))),12);")
    b = make_tree(treestring="(1,((((2,3),4),5),((6,7),((8,9),(10,11)))),12);")
    names = a.get_tip_names()
    assert set(names) == set(b.get_tip_names()), "leaves must match"
    names.sort()
    T1 = convert_tree_to_vectors(a, names)
    T2 = convert_tree_to_vectors(b, names)
    print(type(T2))  # python 2 syntax!
    print("expected: 8")
    r = matched_distance(T1, T2)
    print("calculated:", r)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Included as may be useful for writing tests


def min_weight_matching(B):
    min_weight = B.sum()
    for p in permutations(range(B.shape[0])):
        weight = 0
        candidate = zip(range(B.shape[0]), p)
        for i, j in candidate:
            weight += B[i, j]
        if weight < min_weight:
            min_weight = weight
            matching = candidate
    return min_weight, matching


def calculate_matching_distance(tree1: Tree, tree2: Tree) -> int:
    a = make_tree(str(tree1) + ";")
    b = make_tree(str(tree2) + ";")

    names = a.get_tip_names()
    assert set(names) == set(b.get_tip_names()), "leaves must match"
    names.sort()

    T1 = convert_tree_to_vectors(a, names)
    T2 = convert_tree_to_vectors(b, names)

    return matched_distance(T1, T2)
