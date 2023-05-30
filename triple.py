from __future__ import annotations

from typing import FrozenSet, Generator, Iterable, Tuple

from taxa import Taxa


class Triple:
    def __init__(self, root: Taxa, pair: Iterable[str], weight: int = 1) -> None:
        self.root = root
        self.pair = frozenset(pair)
        self.weight = weight

    def get_names_set(self) -> FrozenSet[str]:
        return self.pair.union((self.root,))

    def get_opposite_pairs(self) -> Generator[Tuple[str, str], None, None]:
        for p in self.pair:
            yield (self.root, p) if self.root < p else (p, self.root)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Triple):
            return False
        return self.root == __o.root and self.pair == __o.pair

    def __lt__(self, __o: Triple):
        this = sorted(self.get_names_set())
        other = sorted(__o.get_names_set())
        for t, o in zip(this, other):
            if t != o:
                return t < o
        return False

    def __hash__(self) -> int:
        return hash((self.root, self.pair))

    def __str__(self) -> str:
        pair_str = ", ".join(self.pair)
        return f"({self.root}, ({pair_str}))"

    def __repr__(self) -> str:
        pair_str = '"' + '", "'.join(map(str, self.pair)) + '"'
        return f'Triple("{str(self.root)}", ({pair_str}))'
