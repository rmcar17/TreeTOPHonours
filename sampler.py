import math
import random
from typing import FrozenSet, Iterable, List, Optional


def linear_congruential_generator(stop: int, n_samples: Optional[int] = None):
    if n_samples is None:
        n_samples = stop

    m = 2 ** math.ceil(math.log2(stop))
    c = 2 * random.randrange((m + 1) // 2) + 1
    a = 4 * (2 * random.randrange(max(1, (m + 2) // 8)) + 1) + 1
    # print(a, c, m)
    x = random.randrange(m)
    samples = 0
    while samples < n_samples:
        if x < stop:
            yield x
            samples += 1
        x = (a * x + c) % m


def combinatorial_indexes(position: int, choices: int) -> List[int]:
    # TODO Replace with binary search
    if choices == 0:
        return []
    n = choices - 1
    smallest = math.comb(n, choices)
    while smallest <= position:
        n += 1
        smallest = math.comb(n, choices)
    n -= 1
    return combinatorial_indexes(position - math.comb(n, choices), choices - 1) + [n]


def combinatorial_indexes_binary_search(position: int, choices: int) -> List[int]:
    print("START")
    if choices == 0:
        return []
    n = choices - 1
    smallest = math.comb(n, choices)
    while smallest <= position:
        n = max(2 * n, 1)

        smallest = math.comb(n, choices)
    l = n // 2
    r = n
    print(f"L={l}, R={r}, {math.comb(l, choices)}, {math.comb(r, choices)}, {position}")
    while l != r:
        m = (l + r) // 2
        comb = math.comb(m, choices)
        print(
            f"L={l}, R={r},m={m},c={comb}, {math.comb(l, choices)}, {math.comb(r, choices)}, {position}"
        )
        if comb > position:
            r = m - 1
        elif comb < position:
            l = m + 1
        else:
            return combinatorial_indexes_binary_search(position - comb, choices - 1) + [
                m
            ]
    solution = l
    return combinatorial_indexes_binary_search(
        position - math.comb(solution, choices), choices - 1
    ) + [solution]
    return combinatorial_indexes(position - math.comb(n, choices), choices - 1) + [n]


def sample_triple_names_percentage(names: Iterable[str], sample_pct: float):
    return sample_triple_names(names, round(sample_pct * math.comb(len(names), 3)))


def sample_triple_names(
    names: Iterable[str], n_samples: Optional[int] = None
) -> List[FrozenSet[str]]:
    """
    [a,b,c,d]
    2

    4 choose 3 possible things to sample
    Mapping from [0, 4 choose 3] -> Triples (a,b,c), (a,b,d),...
    """

    names = list(names)
    random.shuffle(names)

    if n_samples is None:
        n_samples = math.comb(len(names), 3)
    assert n_samples <= math.comb(len(names), 3), "Too many samples"

    triples_size = math.comb(len(names), 3)

    combinatorial_generator = linear_congruential_generator(triples_size, n_samples)

    triples = []
    for position in combinatorial_generator:
        triple_names = []
        for name_index in combinatorial_indexes(position, 3):
            triple_names.append(names[name_index])
        triples.append(frozenset(triple_names))
    return triples
