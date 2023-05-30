import importlib
import pickle
from abc import ABC, abstractmethod
from typing import Iterable, Set

from cherry_picker import cherry_picker_names
from sampler import sample_triple_names_percentage
from taxa import Taxa
from tree import Tree
from triple import Triple


class Resolver(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def initialise(self) -> None:
        pass

    def resolve_triples(
        self, names: Set[Taxa], triples: Iterable[Triple], method="average"
    ):
        return Tree.construct_from_tuple(cherry_picker_names(names, triples, method))

    @abstractmethod
    def resolve(self, names: Set[Taxa]):
        pass


class GroundTruthCherryResolver(Resolver):
    def __init__(self, ground_truth: Tree) -> None:
        super().__init__()
        self.ground_truth = ground_truth

    def initialise(self) -> None:
        return super().initialise()

    def resolve(self, names: Set[Taxa]) -> Tree:
        triples = self.ground_truth.generate_triples_from_triple_names(
            sample_triple_names_percentage(names, 1)
        )

        return self.resolve_triples(names, triples)


class TripleResolver(Resolver):
    def __init__(self, seq_aln=None, seq_path=None) -> None:
        super().__init__()
        assert seq_aln is not None or seq_path is not None
        self.seq_aln = seq_aln
        self.seq_path = seq_path

    def resolve(self, names: Set[Taxa]):
        triples = self.generate_triples_for_alignment(self.seq_aln.take_seqs(names))
        return self.resolve_triples(names, triples)

    def initialise(self) -> None:
        self.run_import()
        if self.seq_aln is None:
            with open(self.seq_path, "rb") as f:
                self.seq_aln = pickle.load(f)

    def run_import(self):
        self.cr = importlib.import_module("clock_rooting")

    def generate_triples_for_alignment(self, aln):
        for name in ["paralinear", "tn93", "jc69", "hamming", "percent"]:
            try:
                distances = aln.distance_matrix(calc=name)
                break
            except ArithmeticError:
                # print("FAIL", name)
                pass

        names = distances.names

        triples = []

        for i in range(2, len(names)):
            for j in range(1, i):
                for k in range(j):
                    ct = self.cr.clocked_triple(
                        distances.take_dists([names[i], names[j], names[k]])
                    )
                    triples.append(Triple(ct.outgroup, ct.ingroups))

        return triples
