import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme()
sns.set_style("ticks")
sns.set_context("poster", font_scale=1.125)
sns.set_palette("muted")


def process_file(file_path):
    mincut_results = []
    spectral_results = []
    with open(file_path, "r") as f:
        for line in f:
            mode, size, time = line.strip().split(";")
            size = int(size)
            if time == "TIMEOUT":
                time = None
            else:
                time = float(time)

            if mode == "M":
                mincut_results.append((size, time))
            elif mode == "S":
                spectral_results.append((size, time))
    return mincut_results, spectral_results


def filter_timeouts(results):
    new_results = []
    for result in results:
        if result[1] is not None:
            new_results.append(result)
    return new_results


def get_x(results):
    return list(map(lambda x: x[0], results))


def get_y(results):
    return list(map(lambda x: x[1], results))


def plot_times(mincut_results, spectral_results):
    print("A")
    fig = plt.figure(figsize=(16, 9))
    # gs = fig.add_gridspec(1, 2)

    # ax = fig.add_subplot(gs[0, 0])
    mincut_results = filter_timeouts(mincut_results)
    plt.scatter(
        get_x(mincut_results),
        get_y(mincut_results),
        label="Min-Cut Supertree",
        alpha=0.5,
    )

    # ax = fig.add_subplot(gs[0, 1])
    plt.scatter(
        get_x(spectral_results),
        get_y(spectral_results),
        label="Spectral Cluster Supertree",
        alpha=0.5,
    )
    plt.legend()

    plt.xlabel("Taxa in Supertree")
    plt.ylabel("Time (Seconds)")
    plt.title("Time to Successfully Resolve Supertrees")

    print("B")
    fig.tight_layout()
    plt.savefig("supertree_time.pdf")
    print("C")


def plot_timeouts(mincut_results, bucket_width=500):
    buckets = [(i + 1, i + bucket_width) for i in range(0, 10000, bucket_width)]
    timeouts = np.zeros(len(buckets))
    total = np.zeros(len(buckets))

    for size, time in mincut_results:
        index = None
        for i, (b_l, b_u) in enumerate(buckets):
            if b_l <= size <= b_u:
                index = i
                break
        assert index is not None
        if time is None:
            timeouts[index] += 1
        total[index] += 1

    percentage = 100 * timeouts / total
    xs = [(a + b) / 2 for a, b in buckets]
    labels = [f"{a}-{b}" for a, b in buckets]

    fig = plt.figure(figsize=(16, 9))
    plt.bar(xs, percentage, width=bucket_width, label=labels)

    plt.xlabel("Taxa in Supertree")
    plt.xticks([i for i in range(0, 10000, bucket_width)])
    plt.ylabel("Percetage of Problems Timed Out (%)")
    plt.title("Percentage of Problems where Min-Cut Supertree Times Out (>10 min)")

    fig.tight_layout()
    plt.savefig("supertree_timeouts.pdf")


if __name__ == "__main__":
    print("Loading Data")
    m, s = process_file("supertree_experiment.txt")
    print("Plotting")
    plot_times(m, s)
    plot_timeouts(m, bucket_width=1000)
    print("Done")
