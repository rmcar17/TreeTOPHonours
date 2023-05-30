from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from mincut import networkx_mincut
from taxa import Taxa
from tree import Tree
from tuple_resolver import RandomTupleResolver

MCNode = Union[Taxa, FrozenSet["MCNode"]]


def flatten_set(s: FrozenSet):
    result = set()
    for v in s:
        if isinstance(v, set) or isinstance(v, frozenset):
            result.update(flatten_set(v))
        else:
            result.add(v)
    return result


class MCGraph:
    def __init__(self) -> None:
        self.vertices: Set[MCNode] = set()
        self.edges: Dict[MCNode, Set[MCNode]] = {}

    def add_vertices(self, vertices: Set[MCNode]):
        self.vertices.update(vertices)
        for vertex in vertices:
            if vertex not in self.edges:
                self.edges[vertex] = set()

    def add_edge(self, u: MCNode, v: MCNode):
        self.edges[u].add(v)
        self.edges[v].add(u)

    def remove_edges(
        self,
        to_remove: Any,
    ):
        for edge in to_remove:
            left, right = tuple(edge)
            if isinstance(left, frozenset) or isinstance(left, set):
                left = flatten_set(left)
            else:
                left = set([left])
            if isinstance(right, frozenset) or isinstance(right, set):
                right = flatten_set(right)
            else:
                right = set([right])
            for l in left:
                for r in right:
                    self.edges[l].remove(r)
                    self.edges[r].remove(l)

    def get_components(self) -> List[Set[MCNode]]:
        components = []

        nodes = self.vertices.copy()
        while nodes:
            frontier = [nodes.pop()]
            component = set(frontier)
            while frontier:
                current_node = frontier.pop()
                for neighbour in self.edges[current_node]:
                    if neighbour not in component:
                        frontier.append(neighbour)
                        component.add(neighbour)
            components.append(component)
            nodes.difference_update(component)
        return components


class WeightedMCGraph(MCGraph):
    def __init__(self, old_graph: MCGraph) -> None:
        super().__init__()
        self.vertices = old_graph.vertices.copy()
        for node in old_graph.edges:
            self.edges[node] = old_graph.edges[node].copy()
        self.edge_weights: Dict[FrozenSet[MCNode], int] = {}
        for node in self.edges:
            for neighbour in self.edges[node]:
                edge = frozenset((node, neighbour))
                self.edge_weights[edge] = None

    def calculate_all_weights(
        self, induced_trees: Iterable[Tree], tree_weights: Iterable[int]
    ):
        for proper_cluster in self.edge_weights:
            weight = 0
            for tree, tree_weight in zip(induced_trees, tree_weights):
                if tree.is_in_proper_cluster(proper_cluster):
                    weight += tree_weight
            self.add_edge_weight(proper_cluster, weight)

        self.check_weights()

    def remove_edges(self, to_remove: Any):
        for edge in to_remove:
            left, right = tuple(edge)
            # if isinstance(left, frozenset) or isinstance(left, set):
            #     pass
            # else:
            #     left = set([left])
            # if isinstance(right, frozenset) or isinstance(right, set):
            #     pass
            # else:
            #     right = set([right])
            # print("THIS IS WHERE")
            # print(left, right)
            del self.edge_weights[edge]
            self.edges[left].remove(right)
            self.edges[right].remove(left)
            # for l in left:
            #     for r in right:
            #         self.edges[l].remove(r)
            #         self.edges[r].remove(l)

    def contract_max_weights(
        self, induced_trees: Iterable[Tree], tree_weights: Iterable[int]
    ):
        max_possible_weight = sum(tree_weights)

        temp_graph = MCGraph()

        for edge in self.edge_weights:
            if self.edge_weights[edge] == max_possible_weight:
                temp_graph.add_vertices(edge)
                temp_graph.add_edge(*edge)

        contractions = temp_graph.get_components()

        vertex_to_contraction = {}
        for contraction in contractions:
            for vertex in contraction:
                vertex_to_contraction[vertex] = frozenset(contraction)

        new_edge_weights = {}
        for contraction in contractions:
            self.vertices.difference_update(contraction)
            new_vertex = frozenset(contraction)
            for vertex in contraction:
                for neighbour in self.edges[vertex]:
                    if neighbour in new_vertex:
                        if frozenset((vertex, neighbour)) in self.edge_weights:
                            del self.edge_weights[frozenset((vertex, neighbour))]
                        continue

                    new_edge_pair = frozenset(
                        (new_vertex, vertex_to_contraction.get(neighbour, neighbour))
                    )
                    if new_edge_pair not in new_edge_weights:
                        new_edge_weights[new_edge_pair] = [
                            self.edge_weights[frozenset((vertex, neighbour))]
                        ]
                    else:
                        new_edge_weights[new_edge_pair].append(
                            self.edge_weights[frozenset((vertex, neighbour))]
                        )
                    self.edges[neighbour].remove(vertex)
                    del self.edge_weights[frozenset((vertex, neighbour))]
                del self.edges[vertex]
        for contraction in contractions:
            self.add_vertices([frozenset(contraction)])
        for edge in new_edge_weights:
            self.add_edge(*edge)
            if len(new_edge_weights[edge]) == 1:
                self.add_edge_weight(edge, new_edge_weights[edge][0])
            else:
                weight = 0
                names = flatten_set(edge)
                for tree, tree_weight in zip(induced_trees, tree_weights):
                    if tree.is_in_proper_cluster(names):
                        weight += tree_weight
                self.add_edge_weight(edge, weight)

    def add_edge_weight(self, edge: FrozenSet[MCNode], weight: int):
        self.edge_weights[edge] = weight

    def check_weights(self):
        for node in self.edges:
            for neighbour in self.edges[node]:
                pair = frozenset((node, neighbour))
                assert pair in self.edge_weights
                assert self.edge_weights[pair] is not None
                assert self.edge_weights[pair] > 0


def mincut_supertree(
    trees: Iterable[Tree], weights: Optional[Iterable[int]] = None, resolver=None
) -> Tuple[Taxa]:
    if weights is None:
        weights = [1 for _ in range(len(trees))]
    print("FINDING S")
    s = construct_s(trees)

    if len(s) <= 2:
        print("DONE")
        return tuple(s) if len(s) != 1 else s

    print("INDUCING")
    induced_trees = generate_induced_trees(s, trees)

    print("CONSTRUCTING SF")
    sf = construct_sf(s, induced_trees)

    print("FIDING COMPONENTS")
    components = sf.get_components()

    if len(components) > 1:
        # If the graph is disconnected list components
        print("FOUND COMPONENTS")
        # print("ALREADY DISCONNECTED", components)
        pass
        # print("Y")
    else:
        print("FINDING SF_E_MAX")
        sf_e_max = construct_sf_e_max(sf, induced_trees, weights)

        print("FINDING CONTRADICTIONS")
        contradicted_edges = find_contradicted_edges(sf_e_max, induced_trees)

        print("REMOVING CONTRADICTIONS")
        sf_e_max.remove_edges(list(map(lambda x: x[0], contradicted_edges)))

        print("FINDING COMPONENTS 2")
        components = sf_e_max.get_components()

        print("NUMBER OF COMPONENTS IS...", len(components))
        # print(components)
        if len(components) > 1:
            pass
            print("YAY" * 100)
        print("WEIGHTING")
        for e, w in contradicted_edges:
            sf_e_max.add_edge_weight(e, w)

        mincut_edges = find_mincut_edges(sf_e_max)
        print("REMOVING MINCUT")
        sf.remove_edges(mincut_edges)
        print("FINDING COMPONENTS 3")
        components = sf.get_components()
    print("FINDING SUBTREES")
    subtrees = []
    for component in components:
        if len(component) <= 2:
            print("A")
            subtrees.append(
                tuple(component) if len(component) != 1 else next(iter(component))
            )
        else:
            print("GENERATING INDUCED SUB")
            new_induced_trees, new_weights = generate_induced_trees_with_weights(
                component, trees, weights
            )
            print("RECURSING")
            subtrees.append(mincut_supertree(new_induced_trees, new_weights, resolver))

    subtrees = tuple(subtrees)
    if resolver is not None:
        print("RESOLVING")
        subtrees = resolver.resolve(subtrees)
    print("RETURN")
    return subtrees


def construct_s(trees: Iterable[Tree]) -> Set[Taxa]:
    s = set()
    for tree in trees:
        s.update(tree.get_descendants())
    return s


def generate_induced_trees(s: Set[Taxa], trees: Iterable[Tree]) -> List[Tree]:
    induced_trees = []
    for tree in trees:
        induced_tuple = tree.generate_induced_tree_tuple(s)
        induced_trees.append(Tree.construct_from_tuple(induced_tuple))
    return induced_trees


def construct_sf(s: Set[Taxa], induced_trees: Iterable[Tree]) -> MCGraph:
    graph = MCGraph()
    graph.add_vertices(s)

    for tree in induced_trees:
        for l in (
            list(tree.left.get_descendants()),
            list(tree.right.get_descendants()),
        ):
            for i in range(1, len(l)):
                for j in range(i):
                    graph.add_edge(l[i], l[j])
    return graph


def construct_sf_e_max(
    sf: MCGraph, induced_trees: Iterable[Tree], tree_weights: Iterable[int]
) -> WeightedMCGraph:
    sf_e_max = WeightedMCGraph(sf)

    sf_e_max.calculate_all_weights(induced_trees, tree_weights)
    sf_e_max.contract_max_weights(induced_trees, tree_weights)

    return sf_e_max


def find_mincut_edges(
    sf_e_max_f: WeightedMCGraph,
    min_cut_method: Callable[
        [Set[MCNode]], Dict[FrozenSet[MCNode], int]
    ] = networkx_mincut,
) -> Tuple[Taxa]:
    print("STARTING MINCUT")
    min_cut, min_cut_weight = min_cut_method(
        sf_e_max_f.vertices, sf_e_max_f.edge_weights
    )
    # min_cut, min_cut_weight = karger_stein_2(
    #     sf_e_max_f.vertices.copy(),
    #     sf_e_max_f.edges.copy(),
    #     sf_e_max_f.edge_weights.copy(),
    # )

    print("FINISHED MINCUT")

    return set(map(frozenset, min_cut))

    mincut_edges = set()
    done = 1
    for edge in sf_e_max_f.edge_weights:
        # print(f"TRYING EDGE {done}/{len(sf_e_max_f.edge_weights)}", edge)
        done += 1
        new_vertices = sf_e_max_f.vertices.copy()
        new_edge_weights = sf_e_max_f.edge_weights.copy()
        edge_weight = new_edge_weights.pop(edge)
        new_edge_weights[edge] = 0

        new_min_cut, new_min_cut_weight = stoer_wagner_mincut(
            new_vertices, new_edge_weights
        )
        # print(
        #     "MIN CUT WITHOUT EDGE",
        #     edge,
        #     "IS",
        #     new_min_cut,
        #     "WITH WEIGHT",
        #     new_min_cut_weight,
        # )
        if new_min_cut_weight + edge_weight == min_cut_weight:
            mincut_edges.add(edge)
    return mincut_edges


def generate_induced_trees_with_weights(
    s: Set[Taxa], trees: Iterable[Tree], weights: Iterable[int]
):
    induced_trees = []
    new_weights = []
    for tree, weight in zip(trees, weights):
        if len(s.intersection(tree.get_descendants())) <= 1:
            continue
        induced_tuple = tree.generate_induced_tree_tuple(s)
        induced_trees.append(Tree.construct_from_tuple(induced_tuple))
        new_weights.append(weight)
    return induced_trees, new_weights


def find_contradicted_edges(
    sf_e_max: WeightedMCGraph, induced_subtrees: Iterable[Tree]
):
    contradicted_edges = []

    for edge in sf_e_max.edge_weights:
        l, r = tuple(edge)
        if not isinstance(l, frozenset):
            left = [l]
        else:
            left = list(l)
        if not isinstance(r, frozenset):
            right = [r]
        else:
            right = list(r)

        cluster = set(left).union(right)
        # print(cluster)
        for tree in induced_subtrees:
            if cluster.issubset(tree.get_descendants()):
                if not tree.is_in_proper_cluster(cluster):
                    contradicted_edges.append((edge, sf_e_max.edge_weights[edge]))
                    break
    # print("CONTRADICTIONS", contradicted_edges)
    return contradicted_edges


if __name__ == "__main__":
    t1 = Tree.construct_from_tuple(((("a", "b"), "c"), ("d", "e")))
    t2 = Tree.construct_from_tuple((("a", "b"), ("c", "d")))
    print(mincut_supertree([t1, t2], resolver=RandomTupleResolver()))
