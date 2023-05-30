import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sns.set_theme()
sns.set_style("ticks")
sns.set_context("poster", font_scale=1.25)
sns.set_palette("muted")


def load_data(file_path):
    data = []
    with open(file_path) as f:
        for line in f:
            (
                size,
                iterations,
                total_time,
                best_time,
                best_prev,
                rel_dists,
                gt_dists,
                found_sol,
                best_sol,
            ) = line.strip().split(";")
            size = int(size)
            iterations = int(iterations)
            total_time = float(total_time)
            best_time = float(best_time)
            best_prev = int(best_prev)
            rel_dists = list(map(float, rel_dists.split(",")))
            gt_dists = list(map(float, gt_dists.split(",")))

            data.append(
                [
                    size,
                    iterations,
                    total_time,
                    best_time,
                    best_prev,
                    rel_dists,
                    gt_dists,
                ]
            )
    return data


def plot_times(data):
    sizes = []
    times = []
    iters = []

    for d in data:
        sizes.append(d[0])
        times.append(d[3])
        iters.append(d[1] - d[4])

    sizes = np.array(sizes)
    times = np.array(times)
    iters = np.array(iters)

    fig = plt.figure(figsize=(16, 9))
    plt.scatter(
        sizes,
        times / 60 / 60,
    )
    plt.xlabel("Number of Taxa")
    plt.ylabel("Time to Resolve (Hours)")
    plt.title("Time to Reconstruct for Simulated DNA Sequences")

    fig.savefig("dcm_real_total_time.pdf", bbox_inches='tight')

    fig = plt.figure(figsize=(16, 9))
    plt.scatter(
        sizes,
        times / iters / 60,
    )
    plt.xlabel("Number of Taxa")
    plt.ylabel("Time per Iteration (Minutes)")
    plt.title("Time per Iteration for Simulated DNA Sequences")

    fig.savefig("dcm_real_iteration_time.pdf", bbox_inches='tight')


def plot_dists(data):
    fig = plt.figure(figsize=(16, 9))
    seen = set()
    for d in data:
        if d[0] in seen or d[0] % 1000 != 0:
            continue
        seen.add(d[0])
        plt.plot(np.arange(len(d[6])), d[6], label=str(d[0]))

    plt.legend(title="Taxa", ncol=2)  # ,bbox_to_anchor=(1,1))
    plt.xlabel("Iterations")
    plt.ylabel("Jaccard Distance from Ground Truth")
    plt.title("Distance from Ground Truth Tree Each Iteration")

    fig.savefig("dcm_real_gt_dist.pdf")


if __name__ == "__main__":
    print("Loading Data")
    data = load_data("dcm_simulated.txt")
    print("Plotting")
    plot_times(data)
    print("Done")
    plot_dists(data)
