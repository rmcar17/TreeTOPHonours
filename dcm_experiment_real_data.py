import pickle
import sys

import cogent3
from cherry_picker import cherry_picker_names
from dcm_parallel import DCM_Config, dcm_parallel
from resolver import TripleResolver
from tree import Tree


def make_random_tree(names) -> Tree:
    return Tree.construct_from_tuple(cherry_picker_names(names, []))


def make_names(size: int):
    return set([f"x{i}" for i in range(size)])


def cogent_to_tuple(t):
    if t.is_tip():
        return t.get_tip_names()[0]
    tree = []
    for child in t:
        tree.append(cogent_to_tuple(child))
    return tuple(tree)


def run_experiment(out_file, name_size, subproblem_size, max_iters, processes=16):
    print("Loading", name_size)
    seq_path = "data/aln_data/" + str(name_size) + ".pkl"
    with open(seq_path, "rb") as f:
        aln = pickle.load(f)
    print("Loaded", name_size)

    print("Loading Tree")
    zt = cogent3.load_tree("data/sim_align/zhu-tree-rooted-resolved-molclock.nwk")
    print("Loaded Tree")
    st = zt.get_sub_tree(aln.names)

    names = set(aln.names)

    ground_truth = Tree.construct_from_tuple(cogent_to_tuple(st))
    guide_tree = make_random_tree(names)

    config = DCM_Config(
        subproblem_size,
        2,
        0,
        max_iters=max_iters,
        processes=processes,
        verbosity=0,
        out_file=out_file,
    )

    resolver = TripleResolver(seq_path=seq_path)

    dcm_parallel(config, guide_tree, resolver, ground_truth=ground_truth)


def run_experiments(
    trials, start, stop, step, out_file, subproblem_size, max_iters, processes=16
):
    for trial in range(trials):
        for size in range(start, stop, step):
            run_experiment(out_file, size, subproblem_size, max_iters, processes)


if __name__ == "__main__":
    print("STARTING", sys.argv)
    run_experiments(
        1,
        int(sys.argv[1]),
        int(sys.argv[2]),
        int(sys.argv[3]),
        "dcm_real_actual.txt",
        140,
        100,
        processes=12,
    )
