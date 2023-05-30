from __future__ import annotations

import random
from typing import (
    FrozenSet,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from taxa import Taxa
from triple import Triple

Node = Union["Tree", "Leaf"]


class Tree:
    def __init__(
        self,
        left: Node,
        right: Node,
        left_weight: Optional[int] = None,
        right_weight: Optional[int] = None,
    ):
        self.left: Node = left
        self.right: Node = right

        self.left_weight = left_weight
        self.right_weight = right_weight

        self.parent: Optional[Tree] = None

        self.descendants: FrozenSet[Taxa] = self.left.get_descendants().union(
            self.right.get_descendants()
        )

    def get_children(self) -> Tuple[Node, Node]:
        return self.left, self.right

    def get_descendants(self) -> FrozenSet[Taxa]:
        return self.descendants

    def get_subsets(self) -> Set[FrozenSet[Taxa]]:
        subsets = set() if self.parent is None else set([self.get_descendants()])
        if isinstance(self.left, Tree):
            subsets.update(self.left.get_subsets())
        if isinstance(self.right, Tree):
            subsets.update(self.right.get_subsets())
        return subsets

    def get_side(self) -> str:
        if self.parent.right is self:
            return "right"
        elif self.parent.left is self:
            return "left"
        assert False

    def is_in_proper_cluster(self, group: Set[Taxa]) -> bool:
        return group.issubset(self.left.get_descendants()) or group.issubset(
            self.right.get_descendants()
        )

    def set_parent(self, parent: Tree) -> None:
        self.parent = parent

    def generate_triples(self) -> List[Triple]:
        triples = []

        left_desc = tuple(self.left.get_descendants())
        right_desc = tuple(self.right.get_descendants())

        for l in range(len(left_desc)):
            for r1 in range(len(right_desc)):
                for r2 in range(r1 + 1, len(right_desc)):
                    triples.append(
                        Triple(left_desc[l], (right_desc[r1], right_desc[r2]))
                    )

        for r in range(len(right_desc)):
            for l1 in range(len(left_desc)):
                for l2 in range(l1 + 1, len(left_desc)):
                    triples.append(
                        Triple(right_desc[r], (left_desc[l1], left_desc[l2]))
                    )

        if isinstance(self.left, Tree):
            triples.extend(self.left.generate_triples())
        if isinstance(self.right, Tree):
            triples.extend(self.right.generate_triples())

        return triples

    def generate_triples_from_triple_names(
        self, triple_names: Iterable[FrozenSet[Taxa]]
    ) -> List[Triple]:
        triples = []
        left_desc = self.left.get_descendants()
        right_desc = self.right.get_descendants()

        todo_left = []
        todo_right = []
        for triple_name in triple_names:
            left_count = 0
            for name in triple_name:
                if name in left_desc:
                    left_count += 1
                else:
                    assert name in right_desc
            if left_count == 3:
                todo_left.append(triple_name)
            elif left_count == 0:
                todo_right.append(triple_name)
            elif left_count == 1:
                left_name = left_desc.intersection(triple_name)
                right_names = triple_name.difference(left_name)
                triples.append(Triple(next(iter(left_name)), right_names))
            elif left_count == 2:
                right_name = right_desc.intersection(triple_name)
                left_names = triple_name.difference(right_name)
                triples.append(Triple(next(iter(right_name)), left_names))
            else:
                assert False, "Invalid count"
        if len(todo_left) > 0:
            triples.extend(self.left.generate_triples_from_triple_names(todo_left))
        if len(todo_right) > 0:
            triples.extend(self.right.generate_triples_from_triple_names(todo_right))
        return triples

    def generate_induced_tree_tuple(self, s: Set[Taxa]) -> Tuple:
        if len(s) == 0:
            return tuple()

        left = s.intersection(self.left.get_descendants())
        right = s.intersection(self.right.get_descendants())

        if len(left) == 0 and len(right) == 0:
            assert False, "No induced tree possible"
        if len(left) == 0:
            return self.right.generate_induced_tree_tuple(right)
        if len(right) == 0:
            return self.left.generate_induced_tree_tuple(left)
        return (
            self.left.generate_induced_tree_tuple(left),
            self.right.generate_induced_tree_tuple(right),
        )

    @staticmethod
    def construct_random_tree(
        min_depth: Tree,
        max_depth: int,
        cutoff_chance: float = 0.0,
        symmetric_chance: float = 1.0,
        prefix: str = "x",
        var_gen: Optional[Generator[str, None, None]] = None,
        rng: Optional[random.Random] = None,
    ) -> Tree:
        if var_gen is None:

            def var_generator():
                count = 0
                while True:
                    yield prefix + str(count)
                    count += 1

            var_gen = var_generator()

        if rng is None:
            rng = random.Random()

        if max_depth <= 1 or (min_depth <= 1 and rng.random() < cutoff_chance):
            tree = Tree(Leaf(next(var_gen)), Leaf(next(var_gen)))
        elif rng.random() < symmetric_chance:
            tree = Tree(
                Tree.construct_random_tree(
                    min_depth - 1,
                    max_depth - 1,
                    cutoff_chance,
                    symmetric_chance,
                    prefix,
                    var_gen,
                    rng,
                ),
                Tree.construct_random_tree(
                    min_depth - 1,
                    max_depth - 1,
                    cutoff_chance,
                    symmetric_chance,
                    prefix,
                    var_gen,
                    rng,
                ),
            )
        else:
            tree = Tree(
                Leaf(next(var_gen)),
                Tree.construct_random_tree(
                    min_depth - 1,
                    max_depth - 1,
                    cutoff_chance,
                    symmetric_chance,
                    prefix,
                    var_gen,
                    rng,
                ),
            )
        for subtree in tree:
            subtree.set_parent(tree)
        return tree

    @staticmethod
    def construct_from_tuple(tree: Union[Tuple, Taxa]) -> Tree:
        if isinstance(tree, tuple):
            assert len(tree) == 2, str(tree)
            result_tree = Tree(*map(lambda t: Tree.construct_from_tuple(t), tree))
            for subtree in result_tree:
                subtree.set_parent(result_tree)
            return result_tree
        return Leaf(tree)

    def __str__(self) -> str:
        return f"({str(self.left)}, {str(self.right)})"

    def __repr__(self) -> str:
        return f"Tree({str(self.left)}, {str(self.right)})"

    def __iter__(self) -> Iterator[Node]:
        yield self.left
        yield self.right

    def __len__(self) -> int:
        return len(self.descendants)

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Tree) and (
            (self.left == __o.left and self.right == __o.right)
            or (self.left == __o.right and self.right == __o.left)
        )

    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)


class Leaf:
    def __init__(self, value: Taxa) -> None:
        self.value = value

        self.parent: Optional[Tree] = None

    def get_descendants(self) -> FrozenSet[Taxa]:
        return frozenset({self.value})

    def set_parent(self, parent: Tree) -> None:
        self.parent = parent

    def generate_induced_tree_tuple(self, s: Set[str]) -> Tuple:
        assert len(s) == 1
        v = next(iter(s))
        assert v == self.value
        return v

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"Leaf({repr(self.value)})"

    def __len__(self):
        return 1

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Leaf) and self.value == __o.value

    def __ne__(self, __o: object) -> bool:
        return isinstance(__o, Leaf) and self.value == __o.value


def jaccard_distance(tree1: Tree, tree2: Tree):
    subsets1 = tree1.get_subsets()
    subsets2 = tree2.get_subsets()
    return 1 - len(subsets1 & subsets2) / len(subsets1 | subsets2)


if __name__ == "__main__":
    print(
        Tree.construct_from_tuple(
            (
                (
                    (
                        (
                            ("x22", "x6"),
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        ("x2", "x24"),
                                                        (
                                                            (
                                                                (
                                                                    "x44",
                                                                    (
                                                                        (
                                                                            "x21",
                                                                            (
                                                                                "x11",
                                                                                "x25",
                                                                            ),
                                                                        ),
                                                                        "x4",
                                                                    ),
                                                                ),
                                                                "x31",
                                                            ),
                                                            (
                                                                ("x15", ("x33", "x1")),
                                                                (("x13", "x17"), "x40"),
                                                            ),
                                                        ),
                                                    ),
                                                    "x49",
                                                ),
                                                (),
                                            ),
                                            ("x28", "x41"),
                                        ),
                                        "x5",
                                    ),
                                    (
                                        ("x26", ("x12", "x39")),
                                        ((("x32", "x18"), "x46"), "x47"),
                                    ),
                                ),
                                ("x8", "x35"),
                            ),
                        ),
                        ("x42", "x34"),
                    ),
                    "x23",
                ),
                (
                    (
                        ((("x9", ("x19", "x48")), ("x30", ("x45", "x20"))), "x27"),
                        ((((("x29", "x0"), "x37"), "x3"), "x14"), "x16"),
                    ),
                    "x10",
                ),
            )
        )
    )
