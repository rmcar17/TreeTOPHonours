import random


class RandomTupleResolver:
    def __init__(self) -> None:
        pass

    def resolve(self, tuples):
        t = list(tuples)
        while len(t) > 2:
            t.append((t.pop(random.randrange(len(t))), t.pop(random.randrange(len(t)))))
            assert len(t[-1]) != 0 and len(t[-2]) != 0, str(t) + " " + str(tuples)
        return tuple(t)
