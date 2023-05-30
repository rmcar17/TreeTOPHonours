import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sns.set_theme()
sns.set_style("ticks")
sns.set_context("poster", font_scale=1.125)
sns.set_palette("muted")


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def plot_dist(df, bin_width=50):
    scaling = 1.5
    fig = plt.figure(figsize=(16 * scaling, 9 * scaling))

    gs = fig.add_gridspec(2, 2)

    for i, (pct, data) in enumerate(df.groupby("pct")):
        if pct == 0:
            continue

        ax = fig.add_subplot(gs[(i - 1) // 2, (i - 1) % 2])

        # sns.scatterplot(data=data, x="size", y="max_dist", label="Max", alpha=0.25)
        # sns.scatterplot(
        #     data=data, x="size", y="average_dist", label="Arithmetic Mean", alpha=0.25
        # )
        # sns.scatterplot(
        #     data=data, x="size", y="geometric_dist", label="Geometric Mean", alpha=0.25
        # )
        # sns.scatterplot(
        #     data=data, x="size", y="harmonic_dist", label="Harmonic Mean", alpha=0.25
        # )

        print(pct)
        data["bin"] = pd.cut(
            data["size"], bins=list(range(0, data["size"].max(), bin_width))
        )

        xs = []
        ys = []
        for i, (b, d) in enumerate(data.groupby("bin")):
            xs.append((i + 1) * bin_width - bin_width / 2)
            # xs.append(0)
            ys.append(
                [
                    d["max_dist"].mean(),
                    d["average_dist"].mean(),
                    d["geometric_dist"].mean(),
                    d["harmonic_dist"].mean(),
                ]
            )
        labels = ["Max", "Arithmetic Mean", "Geometric Mean", "Harmonic Mean"]
        for i in range(4):
            plt.bar(
                list(map(lambda x: x + bin_width / 4 * i - bin_width * 3 / 8, xs)),
                list(map(lambda y: y[i], ys)),
                width=bin_width / 4,
                label=labels[i],
            )
        plt.legend()
        plt.xlabel("Number of Taxa")
        plt.xticks(
            list(range(bin_width // 2, int(max(xs)) + bin_width, bin_width)),
            [f"{int(x)-bin_width//2+1}-{int(x)+bin_width//2}" for x in xs],
        )
        plt.ylabel("Average Matching Distance")
        plt.title(f"Average Matching Distance with {pct}% of Triples Corrupted")
        print(xs, bin_width)
    fig.suptitle("Effect of Incorrect Triples on Distance from Ground Truth Tree")
    fig.tight_layout()
    fig.savefig("corruption_dist.pdf")


def plot_time(data, col="average_time"):
    scaling = 1
    fig = plt.figure(figsize=(16 * scaling, 9 * scaling))
    data = data[data["size"] < 251]
    sns.scatterplot(data, x="size", y=col)
    plt.xlabel("Number of Taxa")
    plt.ylabel("Reconstruction Time (Seconds)")
    plt.title(
        "TripleTree: Time to Reconstruct Phylogenetic Tree (Pre-Computed Triples)"
    )

    fig.savefig("corruption_time.pdf")


if __name__ == "__main__":
    data = load_data("corruption.csv")
    print(data.mean())

    # plot_dist(data)
    # plot_time(data)
    # for row in data:
    #     print(row)
    #     break
