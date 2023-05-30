import os
import random
import time
from typing import List

from cherry_picker import cherry_picker
from cogent3 import make_tree
from matched_distance import convert_tree_to_vectors, matched_distance
from tree import Tree
from triple import Triple


def corrupt_triple(triple: Triple) -> Triple:
    """Given a triple, returns a new triple formed by randomly swapping the root with one of the pairs

    Args:
        triple (Triple): The triple the corruption is based off of

    Returns:
        Triple: A new corrupted triple
    """
    pair = list(triple.pair)
    swap = random.randint(0, 1)
    return Triple(pair[swap], (triple.root, pair[1 - swap]), weight=triple.weight)


def corrupt_triples(
    triples: List[Triple], corruption_percentage: float = 0
) -> List[Triple]:
    """Given a list of triples and a percentage of them to be constructed corrupt that many
    triples by randomly swapping the roots with one of the pairs

    Args:
        triples (List[Triple]): The list of triples to be corrupted
        corruption_percentage (float, optional): The percentage of triples that will be corrupted. Defaults to 0.

    Returns:
        List[Triple]: A new list of corrupted triples
    """
    new_triples = triples.copy()
    corrupt_indices = list(range(len(triples)))
    random.shuffle(corrupt_indices)
    corrupt_indices = corrupt_indices[
        : int(corruption_percentage * len(corrupt_indices))
    ]

    for index in corrupt_indices:
        new_triples[index] = corrupt_triple(new_triples[index])
    return new_triples


def compare_distance(a, b) -> int:
    """Uses matching distance to find the distance between two trees

    Args:
        a, b: The two trees to find the distance betwee

    Returns:
        int: The matching distance between the two trees
    """
    names = a.get_tip_names()
    assert set(names) == set(b.get_tip_names()), "leaves must match"
    names.sort()
    T1 = convert_tree_to_vectors(a, names)
    T2 = convert_tree_to_vectors(b, names)
    r = matched_distance(T1, T2)
    return r


def run_method(method: str, triples: List[Triple], ground_tree):
    start = time.time()
    result = cherry_picker(triples, method=method)
    end = time.time()

    distance = compare_distance(make_tree(str(result)), ground_tree)

    return end - start, distance


def run_percentage(
    percentage: int, methods: List[str], ground_triples: List[Triple], ground_tree
):
    corrupted_triples = set(corrupt_triples(list(ground_triples), percentage / 100))
    results = []
    for method in methods:
        results.append(run_method(method, corrupted_triples, ground_tree))
    return results


def run_experiment(out_path, percentages: List[int], methods: List[str], iters: int):
    if not os.path.exists(out_path):
        with open(out_path, "w") as f:
            f.write("pct,size")
            for method in methods:
                for part in ["time", "dist"]:
                    f.write("," + method + "_" + part)
            f.write("\n")

    for _ in range(iters):
        tree = Tree.construct_random_tree(4, 9, 0.2, 0.7)
        ground_triples = tree.generate_triples()
        ground_tree = make_tree(str(tree))

        for percentage in percentages:
            results = run_percentage(percentage, methods, ground_triples, ground_tree)

            with open(out_path, "a") as f:
                all_results = []
                for result in results:
                    all_results.extend(result)
                f.write(
                    str(percentage)
                    + ","
                    + str(len(tree))
                    + ","
                    + ",".join(map(str, all_results))
                    + "\n"
                )


if __name__ == "__main__":
    run_experiment(
        "corruption.csv",
        [0, 5, 10, 15, 25],
        ["max", "average", "geometric", "harmonic"],
        1000,
    )
    # Perform an experiment which measure the time it takes to reconstruct randomly generate
    # phylogenetic trees with varying percentages of corruption
    # percentages = [0, 5, 10, 15, 25]
    # while True:
    #     print("MAKING TREE")
    #     tree = Tree.construct_random_tree(4, 8, 0.2, 0.7)
    #     print("TREE HAS", len(tree), "VARIABLES")
    #     print("THERE WILL BE", math.comb(len(tree), 3), "TRIPLES")
    #     print(tree)
    #     print("GENERATING TRIPLES")
    #     ground_triples = tree.generate_triples()
    #     print("GENERATED", len(ground_triples), "TRIPLES")

    #     print("GENERATING GROUND SOLUTION")
    #     true_solution = Tree.construct_from_tuple(cherry_picker(ground_triples))
    #     assert true_solution == tree
    #     print("GENERATED GROUND SOLUTION")

    #     ground_tree = make_tree(str(true_solution))
    #     for percent in percentages:
    #         print("INITIALISING FOR", percent)
    #         triples = set(corrupt_triples(list(ground_triples), percent / 100))

    #         print("CHERRY SOLVING")
    #         cherry_start = time.time()
    #         cherry_tree = Tree.construct_from_tuple(
    #             cherry_picker(triples, method="max")
    #         )
    #         cherry_time = time.time() - cherry_start
    #         print(cherry_tree)

    #         print("CHEERY AVERAGE SOLVING")
    #         cherry_avg_start = time.time()
    #         cherry_avg_tree = Tree.construct_from_tuple(
    #             cherry_picker(triples, method="average")
    #         )
    #         cherry_avg_time = time.time() - cherry_avg_start

    #         triples.clear()

    #         print("RECONSTRUCTING")
    #         cherry_recreated_tree = make_tree(str(cherry_tree))
    #         cherry_avg_recreated_tree = make_tree(str(cherry_avg_tree))
    #         cherry_distance = compare_distance(ground_tree, cherry_recreated_tree)
    #         cherry_avg_distance = compare_distance(
    #             ground_tree, cherry_avg_recreated_tree
    #         )
    #         print(
    #             "CHECKING",
    #             "TOOK",
    #             round(cherry_time, 2),
    #             cherry_distance,
    #             round(cherry_avg_time, 2),
    #             cherry_avg_distance,
    #         )
    #         # print(recreated_tree)
    #         with open("results_percentage.txt", "a") as f:
    #             f.write(
    #                 str(percent)
    #                 + ","
    #                 + str(len(tree))
    #                 + ","
    #                 + str(cherry_distance)
    #                 + ","
    #                 + str(cherry_time)
    #                 + ","
    #                 + str(cherry_avg_distance)
    #                 + ","
    #                 + str(cherry_avg_time)
    #                 + "\n"
    #             )
    #     ground_triples.clear()
    #     break
