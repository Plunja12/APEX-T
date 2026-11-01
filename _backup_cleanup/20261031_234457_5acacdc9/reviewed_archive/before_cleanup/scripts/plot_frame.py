
#vykreslování grafů a ukládání jako png
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from pileup.io.clog_reader import read_frames


def get_frame(path, frame_number):
    """Return clusters belonging to a selected frame."""
    for frame_id, clusters in read_frames(path):
        if frame_id == frame_number:
            return clusters

    raise ValueError(f"Frame {frame_number} not found")


def plot_frame(clusters, frame_number, cluster_number):
    """Plot detector occupancy and pixel values."""

    if not 0 <= cluster_number < len(clusters):
        raise ValueError("Selected cluster does not exist")

    occupancy = np.zeros((256, 256), dtype=int)

    for cluster in clusters:
        for x, y, value, extra in cluster:
            occupancy[y, x] = 1

    cluster = clusters[cluster_number]

    xs = [pixel[0] for pixel in cluster]
    ys = [pixel[1] for pixel in cluster]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    width = xmax - xmin + 1
    height = ymax - ymin + 1

    values = np.full((height, width), np.nan)

    for x, y, value, extra in cluster:
        row = y - ymin
        col = x - xmin

        if not np.isnan(values[row, col]):
            raise ValueError("Duplicate pixel in cluster")

        values[row, col] = value

    # Prepare plots
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(14, 7),
        constrained_layout=True
    )

    # Plot 1: full detector
    ax1.imshow(
        occupancy,
        origin="lower",
        cmap="Greys",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    ax1.set_title(f"Frame {frame_number}")
    ax1.set_xlabel("X coordinate")
    ax1.set_ylabel("Y coordinate")

    # Plot 2: selected cluster values
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f0f0f0")

    image = ax2.imshow(
        values,
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        extent=(
            xmin - 0.5, xmax + 0.5,
            ymin - 0.5, ymax + 0.5
        )
    )

    # Write third-column values inside hit pixels
    for x, y, value, extra in cluster:
        ax2.text(
            x, y, f"{value:g}",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            path_effects=[
                pe.withStroke(
                    linewidth=2,
                    foreground="black"
                )
            ]
        )

    ax2.set_title(
        f"Frame {frame_number} | Cluster {cluster_number}"
    )
    ax2.set_xlabel("X coordinate")
    ax2.set_ylabel("Y coordinate")

    # Show coordinates for smaller clusters
    if width <= 20:
        ax2.set_xticks(range(xmin, xmax + 1))

    if height <= 20:
        ax2.set_yticks(range(ymin, ymax + 1))

    ax2.set_xticks(
        np.arange(xmin - 0.5, xmax + 1, 1),
        minor=True
    )
    ax2.set_yticks(
        np.arange(ymin - 0.5, ymax + 1, 1),
        minor=True
    )
    ax2.grid(which="minor", alpha=0.3)

    fig.colorbar(
        image,
        ax=ax2,
        label="Third-column value (unit unknown)"
    )

    return fig


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("file", help="CLOG file path")
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--cluster", type=int, default=0)

    args = parser.parse_args()

    clusters = get_frame(args.file, args.frame)

    fig = plot_frame(
        clusters,
        args.frame,
        args.cluster
    )

    # Save result
    output = Path("results")
    output.mkdir(exist_ok=True)

    filename = (
        f"frame_{args.frame}_"
        f"cluster_{args.cluster}.png"
    )

    fig.savefig(output / filename, dpi=200)

    print(f"Saved: {output / filename}")

    plt.show()


if __name__ == "__main__":
    main()
