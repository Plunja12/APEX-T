"""Plot the M-score analysis for a cluster or a synthetic example."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pileup.io.clog_reader import get_frame, cluster_to_local_matrix
from pileup.multiple_tracks import score_m
from viewer import select_file


def synthetic_cluster(kind):
    """Synthetic examples: straight, cross, branch, bent, blob."""
    coordinates = set()
    if kind == "straight":
        for x in range(10, 41):
            coordinates.add((x, round(0.7 * x + 8)))
    elif kind == "cross":
        for x in range(10, 41):
            coordinates.add((x, round(0.7 * x + 8)))
            coordinates.add((x, round(-0.7 * x + 43)))
    elif kind == "branch":
        for y in range(10, 29):
            coordinates.add((25, y))
        for y in range(28, 46):
            coordinates.add((round(25 + 0.8 * (y - 28)), y))
            coordinates.add((round(25 - 0.8 * (y - 28)), y))
    elif kind == "bent":
        for x in range(10, 41):
            coordinates.add((x, round(12 + 0.055 * (x - 25)**2)))
    else:  # blob
        for x in range(15, 36):
            for y in range(15, 36):
                if (x - 25)**2 + (y - 25)**2 <= 7**2:
                    coordinates.add((x, y))
    return [{"x": x, "y": y, "value": 1.0, "extra": None}
            for x, y in sorted(coordinates)]


def plot_cluster(cluster, output, title):
    result = score_m(cluster)
    matrix, xmin, ymin, xmax, ymax = cluster_to_local_matrix(cluster)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    for ax in (ax1, ax2):
        ax.imshow(matrix, origin="lower", interpolation="nearest", cmap="viridis",
                  extent=(xmin - 0.5, xmax + 0.5, ymin - 0.5, ymax + 0.5))
        ax.set_xlim(xmin - 3, xmax + 3)
        ax.set_ylim(ymin - 3, ymax + 3)
        ax.set_aspect("equal")
        ax.set_xlabel("x [pixel]")
        ax.set_ylabel("y [pixel]")

    ax1.set_title("Original cluster")
    ax2.set_title(f"M={result['M']:.3f} | {result['status']}")
    pts = np.array([[p["x"], p["y"]] for p in cluster], dtype=float)
    for line, color in zip(result["lines"], ("cyan", "orange")):
        center, direction = line["center"], line["direction"]
        along = (pts - center) @ direction
        t = np.array([along.min(), along.max()])
        line_points = center + t[:, None] * direction
        ax2.plot(line_points[:, 0], line_points[:, 1], color=color, lw=2.5)
    if result["intersection"] is not None:
        ax2.plot(*result["intersection"], marker="x", color="red", ms=10, mew=2)

    fig.suptitle(f"{title} | pixels={len(cluster)} | angle={result['angle_deg']:.1f}°"
                 f" | coverage={result['coverage']:.2f}")
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
        print(f"Saved figure: {output}")
    print({k: v for k, v in result.items() if k not in ("lines", "intersection")})
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=int, default=2)
    parser.add_argument("--frame", type=int, default=186)
    parser.add_argument("--cluster", type=int, default=0)
    parser.add_argument("--file")
    parser.add_argument("--synthetic", choices=["straight", "cross", "branch", "bent", "blob"])
    parser.add_argument("--save")
    args = parser.parse_args()

    if args.synthetic:
        cluster = synthetic_cluster(args.synthetic)
        title = f"Synthetic: {args.synthetic}"
    else:
        source = select_file(args.dataset, args.file)
        clusters = get_frame(source, args.frame)
        if not 0 <= args.cluster < len(clusters):
            raise ValueError(f"Cluster index 0..{len(clusters)-1} required")
        cluster = clusters[args.cluster]
        if not cluster:
            raise ValueError("Selected cluster is empty")
        title = f"Dataset {args.dataset}, frame {args.frame}, cluster {args.cluster}"
    plot_cluster(cluster, args.save, title)


if __name__ == "__main__":
    main()
