from cherry_picker import cherry_picker_names
from dcm import construct_random_tree
from dcm_parallel import DCM_Config, dcm_parallel
from resolver import GroundTruthCherryResolver
from tree import Tree


def make_random_tree(names) -> Tree:
    return Tree.construct_from_tuple(cherry_picker_names(names, []))


def make_names(size: int):
    return set([f"x{i}" for i in range(size)])


def run_experiment(out_file, name_size, subproblem_size, max_iters, processes=16):
    ground_truth = construct_random_tree(name_size, name_size)
    guide_tree = construct_random_tree(name_size, name_size)

    config = DCM_Config(
        subproblem_size,
        2,
        0,
        max_iters=max_iters,
        processes=processes,
        verbosity=1,
        out_file=out_file,
    )

    resolver = GroundTruthCherryResolver(ground_truth)

    dcm_parallel(config, guide_tree, resolver, ground_truth=ground_truth)


def run_experiments(
    trials, start, stop, step, out_file, subproblem_size, max_iters, processes=16
):
    for trial in range(trials):
        for size in range(start, stop, step):
            run_experiment(out_file, size, subproblem_size, max_iters, processes)


if __name__ == "__main__":
    print("STARTING")
    run_experiments(
        3,
        500,
        10001,
        500,
        "dcm_random.txt",
        100,
        200,
        processes=12,
    )
    # run_experiment(
    #     "/mnt/data/dayhoff/home/u6956078/experiments/dcm_random.txt",
    #     10000,
    #     100,
    #     200,
    # )
    # print("STARTING")

    # result = dcm_parallel(
    #     guide_tree,
    #     80,
    #     TripleResolver(seq_path=seq_path),
    #     2,
    #     ground_truth=ground_truth,
    #     max_iters=1000,
    # )
    # print("ground truth " + str(ground_truth))
    # print("result " + str(result))
