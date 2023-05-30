import multiprocessing as mp
import time
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Any, Dict, Iterable, List, Optional, Tuple

from cherry_picker import cherry_picker_names
from dcm import construct_random_tree, merge_trees, split_tree
from resolver import GroundTruthCherryResolver, Resolver
from tree import Tree, jaccard_distance


@dataclass
class DCM_Config:
    max_problem_size: int
    convergence_size: int
    convergence_threshold: float
    max_iters: Optional[int] = None
    processes: Optional[int] = 1
    verbosity: Optional[int] = 0
    out_file: Optional[str] = None


def dcm_parallel(
    config: DCM_Config,
    guide_tree: Tree,
    resolver: Resolver,
    ground_truth=None,
):
    from dcm import dcm

    if config.processes == 1:
        return dcm(guide_tree, config.max_problem_size, resolver, 1, 0)

    worker_count = config.processes - 1

    controller_connections, worker_connections = [], []

    for _ in range(worker_count):
        conn1, conn2 = mp.Pipe()
        controller_connections.append(conn1)
        worker_connections.append(conn2)

    if config.verbosity >= 2:
        print("Starting workers")
    workers = []
    for i in range(worker_count):
        if config.verbosity >= 2:
            print("Starting worker", i)
        workers.append(
            mp.Process(
                target=dcm_worker,
                args=(
                    i,
                    worker_connections[i],
                    config.max_problem_size,
                    resolver,
                    config.verbosity,
                ),
            )
        )
        workers[i].start()
    if config.verbosity >= 2:
        print("workers started")
    solution = dcm_controller(
        controller_connections,
        guide_tree,
        config.convergence_size,
        config.convergence_threshold,
        ground_truth=ground_truth,
        max_iters=config.max_iters,
        verbosity=config.verbosity,
        out_file=config.out_file,
    )

    for p in workers:
        p.terminate()

    return solution


@dataclass(order=True)
class PrioritizedItem:
    priority: Tuple  # Usually depth, tiebreak (size), ..., stage
    item: Any = field(compare=False)


def dcm_controller(
    connections,
    guide_tree: Tree,
    convergence_size: int,
    convergence_threshold: float,
    ground_truth=None,
    max_iters=None,
    verbosity=0,
    out_file=None,
):
    available = list(connections)
    not_available = []

    split_queue = PriorityQueue()
    solve_queue = PriorityQueue()
    merge_queue = PriorityQueue()

    stage_counter = 0

    depths: Dict[int, int] = {0: 0}
    parents: Dict[int, int] = {}
    required_solutions: Dict[int, int] = {}
    partial_solutions: Dict[int, List[Tree]] = {}

    split_queue.put(
        PrioritizedItem(
            (depths[stage_counter], -len(guide_tree), stage_counter), guide_tree
        )
    )

    iters = 0
    previous_solution = None
    previous_distances = []
    all_distances = []
    gt_distances = []
    best_so_far = [float("inf"), None, None, 0]
    if ground_truth is not None:
        gt_distances.append(jaccard_distance(guide_tree, ground_truth))

    start_time = time.time()
    while True:
        # Send data
        while len(available) > 0 and not split_queue.empty():
            conn = available.pop()

            pItem = split_queue.get_nowait()
            conn.send(("split", pItem.priority[-1], pItem.item))

            not_available.append(conn)

        while len(available) > 0 and not solve_queue.empty():
            conn = available.pop()

            pItem = solve_queue.get_nowait()
            conn.send(("solve", pItem.priority[-1], pItem.item))

            not_available.append(conn)

        while len(available) > 0 and not merge_queue.empty():
            conn = available.pop()

            pItem = merge_queue.get_nowait()
            conn.send(("merge", pItem.priority[-1], pItem.item))

            not_available.append(conn)

        # Receive data
        receives = mp.connection.wait(not_available)
        found_solution = None
        for conn in receives:
            mode, stage, data = conn.recv()
            if verbosity >= 2:
                print("controller", mode, stage, len(available) + len(receives))

            if mode == "split":
                new_depth = depths[stage] + 1

                required_solutions[stage] = len(data)
                partial_solutions[stage] = []

                for should_solve, subtree in data:
                    stage_counter += 1
                    parents[stage_counter] = stage
                    depths[stage_counter] = new_depth

                    if should_solve:
                        solve_queue.put(
                            PrioritizedItem(
                                (new_depth, -len(guide_tree), stage_counter), subtree
                            )
                        )
                    else:
                        split_queue.put(
                            PrioritizedItem(
                                (new_depth, -len(guide_tree), stage_counter), subtree
                            )
                        )
            elif mode == "solve":
                parent = parents.pop(stage)
                del depths[stage]

                partial_solutions[parent].append(data)

                if required_solutions[parent] == len(partial_solutions[parent]):
                    if required_solutions[parent] != 1:
                        all_descendants = set()
                        for t in partial_solutions[parent]:
                            all_descendants.update(t.get_descendants())
                        merge_queue.put(
                            PrioritizedItem(
                                (-depths[parent], -len(all_descendants), parent),
                                partial_solutions[parent],
                            )
                        )
                    else:
                        # Two cases, the parent's parent must either be a merge or none existent (we are at the top)
                        if parent != 0:
                            # We are not the root
                            grandparent = parents.pop(parent)
                            del depths[parent]

                            partial_solutions[grandparent].append(data)
                            if required_solutions[grandparent] == len(
                                partial_solutions[grandparent]
                            ):
                                assert required_solutions[grandparent] != 1, (
                                    str(stage)
                                    + " parent "
                                    + str(parent)
                                    + " "
                                    + str(required_solutions)
                                    + " "
                                    + str(parents)
                                    + " gp "
                                    + str(grandparent)
                                )  # We must be at a proper merge

                                all_descendants = set()

                                for t in partial_solutions[grandparent]:
                                    all_descendants.update(t.get_descendants())

                                merge_queue.put(
                                    PrioritizedItem(
                                        (
                                            -depths[grandparent],
                                            -len(all_descendants),
                                            grandparent,
                                        ),
                                        partial_solutions[grandparent],
                                    )
                                )

                                del required_solutions[grandparent]
                                del partial_solutions[grandparent]
                        else:
                            # We are the root, need to move onto next iteration
                            found_solution = data
                    del required_solutions[parent]
                    del partial_solutions[parent]

            elif mode == "merge":
                if stage != 0:
                    parent = parents.pop(stage)
                    del depths[stage]

                    partial_solutions[parent].append(data)

                    if required_solutions[parent] == len(partial_solutions[parent]):
                        assert required_solutions[parent] > 1
                        all_descendants = set()
                        for t in partial_solutions[parent]:
                            all_descendants.update(t.get_descendants())
                        merge_queue.put(
                            PrioritizedItem(
                                (-depths[parent], -len(all_descendants), parent),
                                partial_solutions[parent],
                            )
                        )

                        del required_solutions[parent]
                        del partial_solutions[parent]

                else:
                    # We are the root, need to move onto next iteration
                    found_solution = data
            else:
                assert False

            available.append(conn)
            for i in range(len(not_available)):
                if not_available[i] is conn:
                    not_available.pop(i)
                    break
        if found_solution is not None:
            assert len(not_available) == 0
            assert len(available) == len(connections)
            assert depths == {0: 0}
            assert len(parents) == 0
            assert len(required_solutions) == 0
            assert len(partial_solutions) == 0
            assert split_queue.empty()
            assert solve_queue.empty()
            assert merge_queue.empty()

            iters += 1
            # print(iters)

            if ground_truth is not None:
                gt_distances.append(jaccard_distance(found_solution, ground_truth))
                if gt_distances[-1] < best_so_far[0]:
                    best_so_far[0] = gt_distances[-1]
                    best_so_far[1] = found_solution
                    best_so_far[2] = time.time() - start_time
                    best_so_far[3] = 0
                else:
                    best_so_far[3] += 1

            if previous_solution is not None:
                previous_distances.append(
                    jaccard_distance(previous_solution, found_solution)
                )
                all_distances.append(previous_distances[-1])
            if verbosity >= 1:
                print(iters)
                print("Prev Rel distance", previous_distances)
                print("All Rel distance", all_distances)
                if ground_truth is not None:
                    print("GT Distance", gt_distances)
            if len(previous_distances) == convergence_size:
                if all(map(lambda x: x <= convergence_threshold, previous_distances)):
                    end_time = time.time()
                    if out_file is not None:
                        with open(out_file, "a") as f:
                            f.write(str(len(found_solution)) + ";")
                            f.write(str(iters) + ";")
                            f.write(str(end_time - start_time) + ";")
                            f.write(str(best_so_far[2]) + ";")
                            f.write(str(best_so_far[3]) + ";")
                            f.write(",".join(map(str, all_distances)) + ";")
                            if ground_truth is not None:
                                f.write(",".join(map(str, gt_distances)) + ";")
                            f.write(str(found_solution) + ";")
                            f.write(str(best_so_far[1]) + "\n")

                    return found_solution
                previous_distances.pop(0)

            previous_solution = found_solution

            stage_counter = 0
            split_queue.put(
                PrioritizedItem(
                    (depths[stage_counter], -len(found_solution), stage_counter),
                    found_solution,
                )
            )

            if iters >= max_iters or best_so_far[3] > 5:
                end_time = time.time()
                if out_file is not None:
                    with open(out_file, "a") as f:
                        f.write(str(len(found_solution)) + ";")
                        f.write(str(iters) + ";")
                        f.write(str(end_time - start_time) + ";")
                        f.write(str(best_so_far[2]) + ";")
                        f.write(str(best_so_far[3]) + ";")
                        f.write(",".join(map(str, all_distances)) + ";")
                        if ground_truth is not None:
                            f.write(",".join(map(str, gt_distances)) + ";")
                        f.write(str(found_solution) + ";")
                        f.write(str(best_so_far[1]) + "\n")
                return found_solution


def dcm_worker(pid, connection, max_problem_size, resolver, verbosity):
    if verbosity >= 2:
        print("worker", pid, "INITIALISING")
    resolver.initialise()
    if verbosity >= 2:
        print("worker", pid, "BEGIN")
    while True:
        mode, stage, data = receive_data(connection)
        if verbosity >= 2:
            print("worker", pid, mode, stage)
        if mode == "split":
            process_split(connection, stage, max_problem_size, data)
        elif mode == "solve":
            process_solve(connection, stage, resolver, data)
        elif mode == "merge":
            process_merge(connection, stage, data)
        else:
            assert False


def process_merge(connection, stage: int, trees: Iterable[Tree]):
    supertree = merge_trees(trees)
    connection.send(("merge", stage, supertree))


def process_solve(connection, stage: int, resolver: Resolver, tree: Tree):
    resolved_tree = resolver.resolve(tree.get_descendants())
    connection.send(("solve", stage, resolved_tree))


def process_split(connection, stage: int, max_problem_size: int, tree: Tree):
    split_result = generate_split_result(max_problem_size, tree)
    connection.send(("split", stage, split_result))


def generate_split_result(max_problem_size: int, tree: Tree):
    subtrees = split_tree(tree)

    if len(subtrees) == 1:
        return [(True, subtrees[0])]

    split_result = []
    for subtree in subtrees:
        split_result.append((len(subtree) <= max_problem_size, subtree))

    return split_result


def receive_data(connection):
    connection.poll(timeout=None)
    mode, stage, data = connection.recv()
    return mode, stage, data


if __name__ == "__main__":
    import time

    while True:
        print("START")
        ground_truth = construct_random_tree(200, 200)
        print("Expected", ground_truth)
        guide_tree = Tree.construct_from_tuple(
            cherry_picker_names(ground_truth.get_descendants(), [])
        )
        resolver = GroundTruthCherryResolver(ground_truth)
        start = time.time()
        t = dcm_parallel(guide_tree, 30, resolver, 12)
        end = time.time()
        print("Got", t)
        print("Distance", jaccard_distance(ground_truth, t))
        print("Time", end - start)
