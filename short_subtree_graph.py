from typing import List, Set, Tuple, Union

from taxa import Taxa
from tree import Leaf, Tree


class ShortSubtreeGraph:
    def __init__(self) -> None:
        self.nodes = set()
        self.edges = {}

    def add_node(self, node):
        self.nodes.add(node)
        self.edges[node] = set()

    def add_edge(self, u, v):
        if u not in self.nodes:
            self.add_node(u)
        if v not in self.nodes:
            self.add_node(v)

        self.edges[u].add(v)
        self.edges[v].add(u)

    def compute_components(self, separator: Set[Taxa]) -> List[Set[Taxa]]:
        nodes_backup = self.nodes.copy()
        self.nodes.difference_update(separator)

        edges_backup = {}
        for node, connections in self.edges.items():
            edges_backup[node] = connections.copy()

        for node in separator:
            del self.edges[node]

        for edge in self.edges:
            self.edges[edge].difference_update(separator)

        components = []
        while self.nodes:
            node = self.nodes.pop()
            new_component = set([node])

            frontier = set(self.edges[node])
            while frontier:
                node = frontier.pop()
                new_component.add(node)
                self.nodes.remove(node)

                frontier.update(self.edges[node].difference(new_component))
            components.append(new_component)
        self.nodes = nodes_backup
        self.edges = edges_backup
        return components


def is_edge_touching_leaf(node: Tree, left_edge: bool) -> bool:
    if node.parent is None:
        return isinstance(node.left, Leaf) or isinstance(node.right, Leaf)
    if left_edge:
        return isinstance(node.left, Leaf)
    else:
        return isinstance(node.right, Leaf)


def compute_short_subtrees(tree: Tree) -> List[Set[Taxa]]:
    short_subtrees = []
    if not is_edge_touching_leaf(tree, None):
        short_subtrees.append(calculate_short_subtree(tree, None))

    subtrees = [tree.left, tree.right]
    while subtrees:
        subtree = subtrees.pop()
        if isinstance(subtree, Leaf):
            continue
        for side in (True, False):
            if not is_edge_touching_leaf(subtree, side):
                short_subtrees.append(calculate_short_subtree(subtree, side))
        subtrees.extend(subtree.get_children())

    return short_subtrees


def calculate_short_subtree(node: Tree, left_edge: bool) -> Set[Taxa]:
    if node.parent is not None:
        if left_edge:
            child, other = node.get_children()
        else:
            other, child = node.get_children()
        down_right = bfs_short_subtree([(child.right, "parent")])
        down_left = bfs_short_subtree([(child.left, "parent")])
        down_other = bfs_short_subtree([(other, "parent")])
        up = bfs_short_subtree([(node.parent, node.get_side())])

        return down_right.union(down_left).union(down_other).union(up)

    right_right = bfs_short_subtree([(node.right.right, "parent")])
    right_left = bfs_short_subtree([(node.right.left, "parent")])
    left_right = bfs_short_subtree([(node.left.right, "parent")])
    left_left = bfs_short_subtree([(node.left.left, "parent")])

    return right_right.union(right_left).union(left_right).union(left_left)


def bfs_short_subtree(nodes: List[Tuple[Tree, str]]) -> Set[Taxa]:
    leaves = set()
    new_nodes = []
    while len(leaves) == 0:
        # print("ITERATING", nodes)
        for node, banned in nodes:
            # print(node, banned)
            if isinstance(node, Leaf):
                leaves.add(node.value)
                continue
            if banned == "parent":
                new_nodes.append((node.left, "parent"))
                new_nodes.append((node.right, "parent"))
            elif banned == "right":
                new_nodes.append((node.left, "parent"))
                if node.parent is not None:
                    new_nodes.append((node.parent, node.get_side()))
            elif banned == "left":
                new_nodes.append((node.right, "parent"))
                if node.parent is not None:
                    new_nodes.append((node.parent, node.get_side()))
        nodes = new_nodes
        new_nodes = []
    return leaves


def compute_short_subtree_graph(short_subtrees: List[Set[Taxa]]) -> ShortSubtreeGraph:
    short_subtree_graph = ShortSubtreeGraph()
    for short_subtree in short_subtrees:
        ss = list(short_subtree)
        for i in range(len(ss)):
            for j in range(i + 1, len(ss)):
                short_subtree_graph.add_edge(ss[i], ss[j])
    return short_subtree_graph


def decide_partition_method(
    tree: Tree, short_subtrees: List[Set[Taxa]], graph: ShortSubtreeGraph
) -> List[Set[Taxa]]:
    heuristic_partition = find_heuristic_partition(tree)

    components = graph.compute_components(heuristic_partition)

    if len(components) <= 1:
        return find_optimal_partition(short_subtrees, graph)

    for component in components:
        component.update(heuristic_partition)
    return components


def find_heuristic_partition(tree: Tree) -> Set[Taxa]:
    left = len(tree.left)
    right = len(tree.right)

    difference = abs(left - right)

    if left > right:
        return recursive_find_heuristic_partition(tree.left, right, difference)
    return recursive_find_heuristic_partition(tree.right, left, difference)


def recursive_find_heuristic_partition(
    tree: Tree, old_other: int, previous: int
) -> Set[Taxa]:
    if isinstance(tree, Leaf):
        return calculate_short_subtree(tree.parent, tree.get_side() == "left")
    left = len(tree.left)
    right = len(tree.right)

    left_choice = abs(left - (right + old_other))
    right_choice = abs(right - (left + old_other))

    if min(left_choice, right_choice) > previous:
        return calculate_short_subtree(tree.parent, tree.get_side() == "left")
    if left_choice < right_choice:
        return recursive_find_heuristic_partition(
            tree.left, right + old_other, left_choice
        )
    return recursive_find_heuristic_partition(
        tree.right, left + old_other, right_choice
    )


def find_optimal_partition(
    short_subtrees: List[Set[Taxa]], graph: ShortSubtreeGraph
) -> List[Set[Taxa]]:
    possible_partitions = []
    for sep in short_subtrees:
        components = graph.compute_components(sep)
        if len(components) <= 1:
            continue

        for component in components:
            component.update(sep)
        possible_partitions.append(components)
    if len(possible_partitions) == 0:
        return [graph.nodes]
    return min(possible_partitions, key=lambda partition: max(map(len, partition)))


def convert_partition_to_tuple_tree(
    partition: Set[Taxa], guide_tree: Tree
) -> Tuple[Union[Tuple, Taxa]]:
    if len(partition) == 1:
        return next(iter(partition))
    if len(partition) == 2:
        return tuple(partition)

    left_desc = guide_tree.left.get_descendants()
    right_desc = guide_tree.right.get_descendants()

    left_names = set()
    right_names = set()
    for name in partition:
        if name in left_desc:
            left_names.add(name)
        elif name in right_desc:
            right_names.add(name)
        else:
            assert False

    if len(right_names) == 0:
        return convert_partition_to_tuple_tree(left_names, guide_tree.left)
    if len(left_names) == 0:
        return convert_partition_to_tuple_tree(right_names, guide_tree.right)

    return (
        convert_partition_to_tuple_tree(left_names, guide_tree.left),
        convert_partition_to_tuple_tree(right_names, guide_tree.right),
    )
