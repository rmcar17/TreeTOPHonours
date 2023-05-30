from dataclasses import dataclass
from functools import lru_cache

from cogent3.evolve.fast_distance import DistanceMatrix
from sim_align import sim_alignment


@dataclass
class clocked_triple:
    """uses mid-point rooting to identify the outgroup from 3 species"""

    # dists is a symmetric pairwise-distance matrix
    dists: DistanceMatrix
    _dist_dict = None

    def __post_init__(self):
        assert len(self.dists.names) == 3

    def __hash__(self):
        return hash(tuple(self._as_dict.items()))

    @property
    def _as_dict(self) -> str:
        # this is so we can hash the object for lru_caching to work
        if self._dist_dict is None:
            self._dist_dict = self.dists.to_dict()
        return self._dist_dict

    @property
    @lru_cache
    def outgroup(self) -> str:
        """outgroup is the seq with largest summed distance"""
        # we just get the sum of the columns
        totals = self.dists.col_sum().to_dict()
        return max(totals, key=lambda x: totals[x])

    @property
    @lru_cache
    def ingroups(self) -> frozenset:
        return tuple(sorted(frozenset(self.dists.names) - frozenset([self.outgroup])))

    @property
    @lru_cache
    def ingroup1(self) -> frozenset:
        """the first ingroup sequence"""
        return self.ingroups[0]

    @property
    @lru_cache
    def ingroup2(self) -> frozenset:
        """the second ingroup sequence"""
        return self.ingroups[1]

    @property
    def rooted_triple(self) -> frozenset:
        return frozenset([self.outgroup, self.ingroups])

    @property
    @lru_cache
    def outgroup_length(self):
        """the position of root from outgroup tip"""
        index = self.dists.names.index(self.outgroup)
        return self.dists.array[index].sum() / 4

    @property
    @lru_cache
    def _length_inner_node_to_outgroup(self):
        "length from common ancestor of ingroup to outgroup"
        a, b = self.ingroups
        og = self.outgroup
        return (self.dists[a, og] + self.dists[b, og] - self.dists[a, b]) / 2

    @property
    @lru_cache
    def internal_length(self):
        """returns length from root node to internal node (common ancestor to ingroups)"""
        # this may prove a useful property
        return self._length_inner_node_to_outgroup - self.outgroup_length

    @property
    @lru_cache
    def ingroup1_length(self):
        """length from internal node to ingroup1 tip"""
        # this may prove a useful property
        return (
            self.dists[self.ingroup1, self.outgroup]
            - self._length_inner_node_to_outgroup
        )

    @property
    @lru_cache
    def ingroup2_length(self):
        """length from internal node to ingroup2 tip"""
        # this may prove a useful property
        return (
            self.dists[self.ingroup2, self.outgroup]
            - self._length_inner_node_to_outgroup
        )


def test_clocked_triple():
    from cogent3 import load_aligned_seqs

    aln = load_aligned_seqs("~/repos/Cogent3/tests/data/brca1.fasta", moltype="dna")

    # it's possible to build the
    # taking 3 sequences -- Rhesus is a monkey, others are great apes
    # taking the most variable positions
    subaln = aln[2::3]
    # non-canonical characters are omitted in the pairwise distance calculation
    # getting the pairwise distances using the paralinear distance measure
    all_dists = subaln.distance_matrix(calc="paralinear")
    # you can create the different groups of names, hard-coding this example
    all_dists = all_dists.drop_invalid()

    dists = all_dists.take_dists(["Human", "Chimpanzee", "Rhesus"])

    ct = clocked_triple(dists)

    # outgroup is the edge with the largest sum of pairwise distances
    assert ct.outgroup == "Rhesus"
    expect = frozenset({"Rhesus", ("Chimpanzee", "Human")})
    assert ct.rooted_triple == expect, (ct.rooted_triple, expect)


def test_reconstructed_lengths():
    from cogent3.evolve.fast_distance import DistanceMatrix

    # actually computing some edges lengths because these might useful
    # for establishing how reliable a grouping set is,
    # numbers above edges are distancs for constructing example
    #                4
    #           /---------------O
    # -root----|             2
    #          |     1    /----A
    #           \edge.0--|    4
    #                     \--------B

    dists = {("A", "O"): 4 + 1 + 2, ("A", "B"): 4 + 2, ("B", "O"): 4 + 1 + 4}
    for a, b in list(dists):
        dists[(b, a)] = dists[(a, b)]
        dists[(a, a)] = 0.0
        dists[(b, b)] = 0.0
    dists = DistanceMatrix(dists)
    ct = clocked_triple(dists)
    assert ct.outgroup == "O"
    assert ct.rooted_triple == frozenset({"O", ("A", "B")})
    assert ct.outgroup_length == 4
    assert ct.internal_length == 1
    assert ct.ingroup1_length == 2
    assert ct.ingroup2_length == 4


def get_triples(aln):
    dists = aln.distance_matrix()

    names = dists.names

    triples = []

    for i in range(2, len(names)):
        for j in range(1, i):
            for k in range(j):
                sub_d = dists.take_dists([names[i], names[j], names[k]])
                triples.append(clocked_triple(sub_d).rooted_triple)
                # print(triples[-1])

    # print(triples)
    return triples


if __name__ == "__main__":
    import time

    print("START")
    # test_clocked_triple()
    # test_reconstructed_lengths()
    print("ALIGNING")
    a1 = time.time()
    a = sim_alignment(num_seqs=80)
    at = time.time() - a1
    print("ALIGNED")

    t1 = time.time()
    t = get_triples(a)
    tt = time.time() - t1

    print(at, tt)


# with open("delme.json", "w") as out:
#         out.write(dists.to_json())
# from cogent3.util.deserialise import deserialise_object
# got = deserialise_object("delme.json")
