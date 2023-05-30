import random
import time
from typing import Iterable

import mincut_supertree
import spectral_cut
import timeout_decorator
from cherry_picker import cherry_picker_names
from resolver import GroundTruthCherryResolver
from short_subtree_graph import (
    compute_short_subtree_graph,
    compute_short_subtrees,
    convert_partition_to_tuple_tree,
    decide_partition_method,
)
from tree import Tree, jaccard_distance
from tuple_resolver import RandomTupleResolver


def split_tree(guide_tree: Tree):
    short_subtrees = compute_short_subtrees(guide_tree)
    short_subtree_graph = compute_short_subtree_graph(short_subtrees)

    partition = decide_partition_method(guide_tree, short_subtrees, short_subtree_graph)

    subtrees = list(
        map(
            lambda x: Tree.construct_from_tuple(
                convert_partition_to_tuple_tree(x, guide_tree)
            ),
            partition,
        )
    )

    return subtrees


def disk_covering_method(
    guide_tree: Tree,
    max_problem_size: int,
    resolver: GroundTruthCherryResolver,
    depth=[0],
    supertree_experiment=False,
) -> Tree:
    depth[0] += 1
    subtrees = split_tree(guide_tree)

    if len(subtrees) == 1:
        return subtrees[0]
        # return resolver.resolve(subtrees[0].get_descendants())

    result = []

    for subtree in subtrees:
        if len(subtree) > max_problem_size:
            result.append(
                disk_covering_method(
                    subtree,
                    max_problem_size,
                    resolver,
                    supertree_experiment=supertree_experiment,
                )
            )
        else:
            result.append(resolver.resolve(subtree.get_descendants()))

    res = merge_trees(
        result,
        resolver=RandomTupleResolver(),
        supertree_experiment=supertree_experiment,
    )
    depth[0] -= 1
    return res


@timeout_decorator.timeout(15 * 60)
def mincut_wrapper(trees, resolver):
    start = time.time()
    r = mincut_supertree.mincut_supertree(trees, resolver=resolver)
    end = time.time()

    return r, end - start


def merge_trees(
    trees: Iterable[Tree],
    resolver=RandomTupleResolver(),
    supertree_experiment=False,
) -> Tree:
    if supertree_experiment:
        all = set()

        for tree in trees:
            all.update(tree.get_descendants())

        print("DOING M", str(len(all)) + ";")

        try:
            r, t = mincut_wrapper(trees, resolver)
            print("M;" + str(len(all)) + ";" + str(t))
            with open("supertree_experiment.txt", "a") as f:
                f.write("M;")
                f.write(str(len(all)) + ";")
                f.write(str(t) + "\n")
        except timeout_decorator.timeout_decorator.TimeoutError:
            with open("supertree_experiment.txt", "a") as f:
                print("M;" + str(len(all)) + ";" + "TIMEOUT")
                f.write("M;")
                f.write(str(len(all)) + ";")
                f.write("TIMEOUT" + "\n")

        print("DOING S", str(len(all)) + ";")

        start = time.time()
        r = spectral_cut.mincut_supertree(trees, resolver=resolver)
        end = time.time()
        with open("supertree_experiment.txt", "a") as f:
            print("S;" + str(len(all)) + ";" + str(end - start))
            f.write("S;")
            f.write(str(len(all)) + ";")
            f.write(str(end - start) + "\n")
    else:
        r = spectral_cut.mincut_supertree(trees, resolver=resolver)
    # print(r)
    return Tree.construct_from_tuple(r)


def recursive_resolve(in_tuple):
    if not isinstance(in_tuple, tuple):
        return in_tuple
    resolved_parts = list(map(recursive_resolve, in_tuple))
    while len(resolved_parts) > 2:
        resolved_parts.append(
            (
                resolved_parts.pop(random.randrange(len(resolved_parts))),
                resolved_parts.pop(random.randrange(len(resolved_parts))),
            )
        )
    return tuple(resolved_parts)


def construct_random_tree(min_size, max_size):
    names = ["x" + str(i) for i in range(random.randint(min_size, max_size))]
    while len(names) > 1:
        names.append(
            (
                names.pop(random.randrange(len(names))),
                names.pop(random.randrange(len(names))),
            )
        )
    return Tree.construct_from_tuple(names[0])


def has_converged(distances, convergence_size, convergence_threshold):
    if len(distances) < convergence_size:
        return False
    return all(map(lambda x: x <= convergence_threshold, distances))


def dcm(
    guide_tree,
    max_problem_size,
    resolver,
    convergence_size: int,
    convergence_threshold: float,
):
    previous_solution = None
    distances = []

    while not has_converged(distances, convergence_size, convergence_threshold):
        print("\nSTART", distances)
        print()
        guide_tree = disk_covering_method(guide_tree, max_problem_size, resolver)

        if previous_solution is not None:
            distances.append(jaccard_distance(guide_tree, previous_solution))
            if len(distances) > convergence_size:
                distances.pop(0)
        previous_solution = guide_tree

    return guide_tree


def run_once(ground_truth_tree, subproblem_size, supertree_experiment=False):
    guide_tree = Tree.construct_from_tuple(
        cherry_picker_names(ground_truth_tree.get_descendants(), [])
    )
    iteration = disk_covering_method(
        guide_tree,
        subproblem_size,
        GroundTruthCherryResolver(ground_truth_tree),
        supertree_experiment=supertree_experiment,
    )

    # start = time.time()
    # iters = 1
    # relative_distances = []
    # true_distances = [jaccard_distance(guide_tree, ground_truth_tree)]
    # try:
    #     previous = guide_tree
    #     iteration = disk_covering_method(
    #         guide_tree,
    #         subproblem_size,
    #         GroundTruthCherryResolver(ground_truth_tree),
    #         supertree_experiment=supertree_experiment,
    #     )
    #     distance = jaccard_distance(previous, iteration)
    #     relative_distances.append(distance)
    #     true_distances.append(jaccard_distance(iteration, ground_truth_tree))
    #     while distance > 0:
    #         # print("SO FAR", iteration)
    #         # print("GT", ground_truth_tree)
    #         # print(
    #         #     f"\nIterations: {iters} Distance From Previous: {distance}\n"  # {calculate_matching_distance(previous, iteration)} Distance From GT: {calculate_matching_distance(ground_truth_tree, iteration)}\n"
    #         # )
    #         previous = iteration
    #         iteration = disk_covering_method(
    #             iteration,
    #             subproblem_size,
    #             GroundTruthCherryResolver(ground_truth_tree),
    #             supertree_experiment=supertree_experiment,
    #         )
    #         iters += 1
    #         distance = jaccard_distance(previous, iteration)
    #         relative_distances.append(distance)
    #         true_distances.append(jaccard_distance(iteration, ground_truth_tree))
    #     end = time.time()
    #     # print(
    #     #     f"\nIterations: {iters} Distance From Previous: {distance}\n"  # {calculate_matching_distance(previous, iteration)} Distance From GT: {calculate_matching_distance(ground_truth_tree, iteration)}\n"
    #     # )

    # except timeout_decorator.timeout_decorator.TimeoutError:
    #     end = time.time()
    #     with open("dcm_results.txt", "a") as f:
    #         f.write(
    #             f"FAIL,{len(ground_truth_tree)},{subproblem_size},{iters},{end-start}\n"
    #         )
    # else:
    #     with open("dcm_results.txt", "a") as f:
    #         f.write(
    #             f"SUCCESS,{len(ground_truth_tree)},{subproblem_size},{iters},{end-start}\n"
    #         )
    #     with open("distances.txt", "a") as f:
    #         f.write(
    #             f"{len(ground_truth_tree)};{subproblem_size};"
    #             + ",".join(map(str, relative_distances))
    #             + ";"
    #             + ",".join(map(str, true_distances))
    #             + "\n"
    #         )
    #     print(f"Took {iters} iterations, {end-start} seconds\n")


def dcm_experiment(min_size, max_size, subproblem_size):
    while True:
        ground_truth_tree = construct_random_tree(min_size, max_size)
        run_once(ground_truth_tree, subproblem_size)


def ramp_up(start, stop, step, subproblem_size, repeats=1, supertree_experiment=False):
    for done in range(repeats):
        for size in range(start, stop, step):
            print("Doing", done, size)
            # try:
            ground_truth_tree = construct_random_tree(size, size)
            run_once(
                ground_truth_tree,
                subproblem_size,
                supertree_experiment=supertree_experiment,
            )
            # except:
            #     with open("failure.txt", "a") as f:
            #         f.write(str(ground_truth_tree) + "\n")


if __name__ == "__main__":
    # ramp_up(1500,5001,500,65,2)
    # run_once(construct_random_tree(3000, 3000), 100, supertree_experiment=True)
    ramp_up(1000, 10001, 500, 100, supertree_experiment=True, repeats=30)
