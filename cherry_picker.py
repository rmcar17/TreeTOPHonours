from typing import Dict, FrozenSet, Iterable, Set, Tuple, Union

import sampler
from taxa import Taxa
from tree import Tree
from triple import Triple

ALLOWED_METHODS = ["max", "average", "geometric", "harmonic"]


def count_initial_cherries(
    triples: Iterable[Triple],
) -> Dict[FrozenSet[Union[Tuple, Taxa]], int]:
    cherry_count = {}
    for triple in triples:
        cherry_count[triple.pair] = cherry_count.get(triple.pair, 0) + 1
    return cherry_count


def get_names(triples: Iterable[Triple]) -> Set[Union[Tuple, Taxa]]:
    names = set()
    for triple in triples:
        names.update(triple.get_names_set())
    return names


def merge(names_set: Set[Union[Taxa, Tuple]], cherry: FrozenSet[Taxa]) -> None:
    """_summary_

    names_set =  {"a", "b", "c", "d"}
    cherry = {"b", "d"}
    resulting names_set =  {"a", ("b", "d"), "c"}

    Args:
        names_set (Set[Union[str, Tuple]]): _description_
        cherry (FrozenSet[str]): _description_
    """
    names_set.difference_update(cherry)
    # print(names_set)
    names_set.update((tuple([part for part in cherry]),))


def arbitrary_merge(names_set: Set[Union[Taxa, Tuple]]) -> None:
    names_set.update(((names_set.pop(), names_set.pop()),))


def get_best_cherry(
    cherries: Dict[FrozenSet[Union[Tuple, Taxa]], int]
) -> FrozenSet[Union[Tuple, Taxa]]:
    return max(cherries, key=lambda x: cherries[x])


def update_cherries(
    cherries: Dict[FrozenSet[Union[Tuple, Taxa]], int],
    merge_cherry: FrozenSet[Union[Tuple, Taxa]],
    method: str,
) -> None:
    assert method in ALLOWED_METHODS
    # print("\nCherries Dict",cherries)
    # print(merge_cherry)
    del cherries[merge_cherry]
    relevant_keys = {}
    # {a,b}
    # {a,c},{b,c}
    # {(a, b), c}
    for name in merge_cherry:
        relevant_keys[name] = tuple(
            filter(lambda x: name in x and x != merge_cherry, cherries)
        )
    # print("Relevant Keys", relevant_keys)
    if method == "average" or method == "geometric" or method == "harmonic":
        dividers = {}
    # print(divider, tuple(merge_cherry))
    # dividers = {}
    for name in merge_cherry:
        for cherry in relevant_keys[name]:
            count = cherries.pop(cherry)
            new_cherry = cherry.difference(set((name,))).union((tuple(merge_cherry),))
            if method == "max":
                cherries[new_cherry] = max(cherries.get(new_cherry, 0), count)
            elif method == "average":
                cherries[new_cherry] = cherries.get(new_cherry, 0) + count
                dividers[new_cherry] = dividers.get(new_cherry, 0) + 1
            elif method == "geometric":
                cherries[new_cherry] = cherries.get(new_cherry, 1) * count
                dividers[new_cherry] = dividers.get(new_cherry, 0) + 1
            elif method == "harmonic":
                cherries[new_cherry] = cherries.get(new_cherry, 0) + 1 / count
                dividers[new_cherry] = dividers.get(new_cherry, 0) + 1

    if method == "average":
        for divider in dividers:
            cherries[divider] /= dividers[divider]
    elif method == "geometric":
        for divider in dividers:
            cherries[divider] = cherries[divider] ** (1 / dividers[divider])
    elif method == "harmonic":
        for divider in dividers:
            cherries[divider] = dividers[divider] * (1 / cherries[divider])
    # print("\nNew Cherries Dict", cherries)


def cherry_picker(triples: Iterable[Triple], method="max"):
    assert method in ALLOWED_METHODS
    cherry_count = count_initial_cherries(triples)
    # print(cherry_count)
    tree = get_names(triples)
    # print(tree)
    best_cherry = get_best_cherry(cherry_count)
    # print(best_cherry)
    merge(tree, best_cherry)
    # print("After", tree)

    while len(tree) > 2:
        if len(cherry_count) > 0:
            update_cherries(cherry_count, best_cherry, method)
            best_cherry = get_best_cherry(cherry_count)
            merge(tree, best_cherry)
        else:
            # In reality, we probably want to add more samples.
            # Based off the items yet to be merged (get names and
            # sample between). For now, we merge randomly
            arbitrary_merge(tree)
    return tuple(tree)


def cherry_picker_names(
    names: Iterable[Taxa], triples: Iterable[Triple], method="average"
):
    assert method in ALLOWED_METHODS
    cherry_count = count_initial_cherries(triples)
    # print(cherry_count)
    tree = set(names)
    # print(tree)
    if len(cherry_count) > 0:
        best_cherry = get_best_cherry(cherry_count)
        # print(best_cherry)
        merge(tree, best_cherry)
        # print("After", tree)
    else:
        arbitrary_merge(tree)

    # update_cherries(cherry_count, best_cherry, method)
    while len(tree) > 2:
        # print(tree)
        if len(cherry_count) > 0:
            update_cherries(cherry_count, best_cherry, method)
            if len(cherry_count) > 0:
                best_cherry = get_best_cherry(cherry_count)
                merge(tree, best_cherry)
        else:
            # In reality, we probably want to add more samples.
            # Based off the items yet to be merged (get names and
            # sample between). For now, we merge randomly
            # print("ARBITRARY")
            arbitrary_merge(tree)
    return tuple(tree)


def cherry_picker_names_with_counts(
    names: Iterable[Taxa],
    cherry_count: Dict[FrozenSet[Union[Tuple, Taxa]], int],
    method="average",
):
    assert method in ALLOWED_METHODS

    tree = set(names)
    # print(tree)
    if len(cherry_count) > 0:
        best_cherry = get_best_cherry(cherry_count)
        # print(best_cherry)
        merge(tree, best_cherry)
        # print("After", tree)
    else:
        arbitrary_merge(tree)

    # update_cherries(cherry_count, best_cherry, method)
    while len(tree) > 2:
        # print(tree)
        if len(cherry_count) > 0:
            update_cherries(cherry_count, best_cherry, method)
            if len(cherry_count) > 0:
                best_cherry = get_best_cherry(cherry_count)
                merge(tree, best_cherry)
        else:
            # In reality, we probably want to add more samples.
            # Based off the items yet to be merged (get names and
            # sample between). For now, we merge randomly
            # print("ARBITRARY")
            arbitrary_merge(tree)
    return tuple(tree)


def sample_cherry_picker(
    names: Iterable[Taxa], sample_tree: Tree, sample_percent: float
):
    triple_names = sampler.sample_triple_names_percentage(names, sample_percent)
    triples = sample_tree.generate_triples_from_triple_names(triple_names)
    return cherry_picker(triples)


def reconstruct_sampled_tree(tree: Tree, sample_percent: float):
    names = tree.get_descendants()
    reconstructed = sample_cherry_picker(names, tree, sample_percent)
    return reconstructed


if __name__ == "__main__":
    import time

    print("GENERATING TREE")
    # tree = Tree.construct_random_tree(9, 11, 0.2, 0.7)
    # import math

    # print(
    #     "GENERATING TRIPLES for tree with",
    #     len(tree),
    #     "LEAVES",
    #     "EXPECTED",
    #     math.comb(len(tree), 3),
    #     "TRIPLES",
    # )
    # triples = tree.generate_triples()
    # print("SOLVING")
    # start = time.time()
    # solution = my_cherry_picker(triples, method="average")
    # sol_time = time.time() - start
    # print("CONVERTING")
    # reconstructed = Tree.construct_from_tuple(solution)
    # print(tree == reconstructed, sol_time)

    # tree = Tree.construct_random_tree(12, 14, 0.2, 0.7)
    # print(len(tree))
    # start = time.time()
    # reconstructed = reconstruct_sampled_tree(tree, 50000 / math.comb(len(tree), 3))
    # print(time.time() - start)
    t = True
    while t:
        tree = Tree.construct_random_tree(3, 5, 0.2, 0.7)
        triples = tree.generate_triples()
        # print(triples)
        print(tree)
        reconstructed = Tree.construct_from_tuple(
            cherry_picker(triples, method="average")
        )
        print(reconstructed)
        print(tree == reconstructed)
        t = tree == reconstructed
