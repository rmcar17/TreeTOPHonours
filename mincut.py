import math
import random
from copy import deepcopy
from typing import Any, Dict, FrozenSet, Set, Tuple

import networkx as nx


def networkx_mincut(vertices: Set[Any], edge_weights: Dict[FrozenSet[Any], int]):
    G = nx.Graph()
    G.add_nodes_from(vertices)
    for edge, weight in edge_weights.items():
        u, v = tuple(edge)
        G.add_edge(u, v, weight=weight)
    # print("THIS ONE?")
    try:
        edges = nx.minimum_edge_cut(G)
    except nx.exception.NetworkXError as e:
        print(e)
        return None, 0
    total_weight = 0
    for edge in edges:
        total_weight += edge_weights[frozenset(edge)]
    # print("RETURN VAL",edges)
    # print(G.edges())
    # print("Here Please", edges)
    return edges, total_weight


def flatten_set(s: FrozenSet):
    result = set()
    for v in s:
        if isinstance(v, set) or isinstance(v, frozenset):
            result.update(flatten_set(v))
        else:
            result.add(v)
    return result


def karger(vertices: Set[Any], edge_weights: Dict[FrozenSet[Any], int], iters=10):
    v_copy, e_copy = vertices.copy(), edge_weights.copy()
    contract(v_copy, e_copy, 2)

    best_e = e_copy
    best_w = sum(e_copy.values())
    # print("K", best_w, len(v_copy))

    for _ in range(iters - 1):
        v_copy, e_copy = vertices.copy(), edge_weights.copy()
        contract(v_copy, e_copy, 2)

        w = sum(e_copy.values())
        # print("K", w, len(v_copy))
        # print(w)
        if w < best_w:
            best_e = e_copy
            best_w = w

    assert len(best_e.keys()) == 1
    # print("HERE", best_e, "\n")
    # print(tuple(best_e.keys())[0], len(tuple(best_e.keys())[0]))
    l, r = tuple(tuple(best_e.keys())[0])
    if isinstance(l, int):
        l = frozenset([l])
    if isinstance(r, int):
        r = frozenset([r])

    # print(l, r)
    l = flatten_set(l)
    r = flatten_set(r)

    to_remove = set()
    for u in l:
        for v in r:
            if frozenset((u, v)) in edge_weights:
                to_remove.add(frozenset((u, v)))

    return to_remove, best_w


def karger_stein(
    vertices: Set[Any], edge_weights: Dict[FrozenSet[Any], int]
) -> Tuple[Set[FrozenSet[Any]], int]:
    if len(vertices) <= 6:
        return networkx_mincut(vertices, edge_weights)
    t = math.ceil(1 + len(vertices) / (2))

    vertices_2 = vertices.copy()
    edge_weights_2 = edge_weights.copy()

    contract(vertices, edge_weights, t)
    contract(vertices_2, edge_weights_2, t)

    sol_edges, sol_weight = karger_stein(vertices, edge_weights)
    sol_edges_2, sol_weight_2 = karger_stein(vertices_2, edge_weights_2)

    if sol_weight < sol_weight_2:
        return sol_edges, sol_weight
    return sol_edges_2, sol_weight


def contract(
    vertices: Set[Any], edge_weights: Dict[FrozenSet[Any], int], t: int
) -> None:
    while len(vertices) > t:
        edges = list(edge_weights.keys())
        new_vertex = random.choices(
            edges, weights=list(map(lambda x: edge_weights[x], edges))
        )[0]
        new_vertex = random.choices(edges)[0]

        del edge_weights[new_vertex]
        for v in new_vertex:
            vertices.remove(v)

            for edge in tuple(edge_weights.keys()):
                if v in edge:
                    weight = edge_weights.pop(edge)

                    u1, u2 = tuple(edge)

                    new_edge = frozenset([u1 if v == u2 else u2, new_vertex])
                    edge_weights[new_edge] = edge_weights.get(new_edge, 0) + weight
        vertices.add(new_vertex)
    # print(vertices)


def karger_2(
    vertices: Set[Any],
    edges: Dict[Any, Set[Any]],
    edge_weights: Dict[FrozenSet[Any], int],
    iters=10,
):
    v_copy, e_copy, ew_copy = vertices.copy(), deepcopy(edges), edge_weights.copy()
    contract_2(v_copy, e_copy, ew_copy, 2)

    best_e = e_copy
    best_w = sum(ew_copy.values())
    best_ew = ew_copy
    # print("K", best_w, len(v_copy))

    for _ in range(iters - 1):
        v_copy, e_copy, ew_copy = vertices.copy(), deepcopy(edges), edge_weights.copy()
        contract_2(v_copy, e_copy, ew_copy, 2)

        w = sum(ew_copy.values())
        # print("K", w, len(v_copy))
        # print(w)
        if w < best_w:
            best_e = e_copy
            best_ew = ew_copy
            best_w = w

    assert len(best_ew.keys()) == 1
    # print("HERE", best_e, "\n")
    # print(tuple(best_e.keys())[0], len(tuple(best_e.keys())[0]))
    l, r = tuple(tuple(best_ew.keys())[0])
    if isinstance(l, int):
        l = frozenset([l])
    if isinstance(r, int):
        r = frozenset([r])

    # print(l, r)
    l = flatten_set(l)
    r = flatten_set(r)

    to_remove = set()
    for u in l:
        for v in r:
            if frozenset((u, v)) in edge_weights:
                to_remove.add(frozenset((u, v)))

    return to_remove, best_w


def karger_stein_2(
    vertices: Set[Any],
    edges: Dict[Any, Set[Any]],
    edge_weights: Dict[FrozenSet[Any], int],
) -> Tuple[Set[FrozenSet[Any]], int]:
    # print("START?", len(vertices))
    if len(vertices) <= 6:
        # print("NETWORKING?")
        return networkx_mincut(vertices, edge_weights)
    # print("CONTINUING?", len(vertices))
    t = math.ceil(1 + len(vertices) / math.sqrt(2))

    vertices_2 = vertices.copy()
    edges_2 = deepcopy(edges)
    edge_weights_2 = deepcopy(edge_weights)

    contract_2(vertices, edges, edge_weights, t)
    contract_2(vertices_2, edges_2, edge_weights_2, t)

    # pool = ThreadPool(processes=2)
    # result = pool.starmap(
    #     karger_stein_2,
    #     [
    #         (vertices, edges, edge_weights),
    #         (vertices_2, edges_2, edge_weights_2),
    #     ],
    # )
    # a, b = result
    # sol_edges, sol_weight = a
    # sol_edges_2, sol_weight_2 = b
    sol_edges, sol_weight = karger_stein_2(vertices, edges, edge_weights)
    sol_edges_2, sol_weight_2 = karger_stein_2(vertices_2, edges_2, edge_weights_2)

    if sol_weight < sol_weight_2:
        return sol_edges, sol_weight
    return sol_edges_2, sol_weight


def contract_2(
    vertices: Set[Any],
    edges: Dict[Any, Set[Any]],
    edge_weights: Dict[FrozenSet[Any], int],
    t: int,
) -> None:
    # print("BEGINNING")
    while len(vertices) > t:
        # print("START ITERS", vertices, len(vertices))
        # print("EDGES", edges)
        # print("EDGE WEIGHTS", edge_weights)
        edges_list = list(edge_weights.keys())
        # new_vertex = random.choices(
        #     edges_list, weights=list(map(lambda x: edge_weights[x], edges_list))
        # )[0]
        new_vertex = random.choices(edges_list)[0]
        # print("NEW VERTEX", new_vertex)

        del edge_weights[new_vertex]
        a, b = tuple(new_vertex)

        edges[a].remove(b)
        edges[b].remove(a)

        # print(vertices)
        vertices.add(new_vertex)
        edges[new_vertex] = set()
        for v in new_vertex:
            vertices.remove(v)

            for other in edges.get(v, []):
                edges[new_vertex].add(other)
                edges[other].remove(v)
                edges[other].add(new_vertex)

                new_edge = frozenset([other, new_vertex])
                weight = edge_weights.pop(frozenset([v, other]))
                edge_weights[new_edge] = edge_weights.get(new_edge, 0) + weight

            del edges[v]

        # print("END ITER", vertices, len(vertices))


if __name__ == "__main__":
    n = 400
    g = [i for i in range(n)]
    ew = {}
    for i in range(1, n):
        ew[frozenset([g[i - 1], g[i]])] = 1
        for j in range(i):
            w = random.randint(0, 5)
            if w != 0:
                ew[frozenset([g[i], g[j]])] = 1  # w
    es = {}
    for v in g:
        es[v] = set()
    for e in ew:
        a, b = tuple(e)
        es[a].add(b)
        es[b].add(a)
    # print(g, ew)
    # print("START")
    print("N", networkx_mincut(set(g), ew)[1])
    # print(karger_stein_2(set(g), es, deepcopy(ew)))
    print(karger_2(set(g), es, deepcopy(ew), 40))
    # print("COMPLETE")
    # print(karger(set(g), ew, 40))
    # for i in range(10):
    #     print("KS", karger_stein(set(g), ew.copy())[1])
    # print(stoer_wagner_mincut(set(g), ew)[1])
