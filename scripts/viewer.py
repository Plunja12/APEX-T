
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from pileup.io.clog_reader import (
    iter_frames,
    get_frame,
    frame_to_matrix,
    cluster_to_local_matrix
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"


def find_data_files():
    """Find likely measurement files in data/raw."""

    files = []

    for path in DATA_DIR.rglob("*"):
        if not path.is_file():
            continue

        name = path.name.lower()
        extension = path.suffix.lower()

        if extension in {".zip", ".png", ".jpg", ".jpeg"}:
            continue

        if extension == ".clog" or name.startswith(
            ("tot_", "itot", "cnt_")
        ):
            files.append(path)

    return sorted(files)


def select_file(dataset, custom_file=None):
    """Select a measurement file."""

    if custom_file is not None:
        path = Path(custom_file)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if not path.is_file():
            raise FileNotFoundError(path)

        return path

    prefix = f"{dataset:02d} "

    matches = [
        path for path in find_data_files()
        if path.relative_to(DATA_DIR).parts[0].startswith(prefix)
    ]

    if not matches:
        raise FileNotFoundError(
            f"No measurement file found for dataset {dataset}. "
            "Use --list to inspect available files."
        )

    if len(matches) > 1:
        raise ValueError(
            "Multiple files found. Select one with --file:\n"
            + "\n".join(str(path) for path in matches)
        )

    return matches[0]

def print_frame_stats(source, limit=200):
    """Analyze pixel and cluster counts in the first frames."""

    stats = []
    total_clusters = 0
    total_pixels = 0
    total_duplicates = 0

    for frame_id, clusters in iter_frames(source):

        pixel_count = sum(len(cluster) for cluster in clusters)
        cluster_count = len(clusters)

        coordinates = [
            (pixel["x"], pixel["y"])
            for cluster in clusters
            for pixel in cluster
        ]

        unique_pixels = len(set(coordinates))
        duplicates = pixel_count - unique_pixels

        stats.append((frame_id, cluster_count, pixel_count, duplicates))

        total_clusters += cluster_count
        total_pixels += pixel_count
        total_duplicates += duplicates

        if len(stats) >= limit:
            break

    print(f"\nFile: {source.name}")
    print(f"Frames analyzed: {len(stats)}")
    print(f"Total clusters: {total_clusters}")
    print(f"Total pixel records: {total_pixels}")
    print(f"Duplicate coordinates: {total_duplicates}")

    print("\nFive frames with most pixel records:")

    densest = sorted(
        stats,
        key=lambda item: item[2],
        reverse=True
    )[:5]

    for frame_id, clusters, pixels, duplicates in densest:
        print(
            f"Frame {frame_id}: "
            f"{clusters} clusters, "
            f"{pixels} pixels, "
            f"{duplicates} duplicates"
        )

def create_viewer(source, frame_id, first_cluster):
    """Create an interactive full-frame and cluster viewer."""

    if frame_id is None:

        # Find the first non-empty frame automatically.
        for candidate_id, candidate_clusters in iter_frames(source):

            if any(candidate_clusters):
                frame_id = candidate_id
                clusters = candidate_clusters
                break

        else:
            raise ValueError("No non-empty frames found")

    else:
        # Load a specific frame when requested.
        clusters = get_frame(source, frame_id)

    if not clusters:
        raise ValueError("Selected frame has no clusters")

    print(f"Selected frame: {frame_id}")

    frame_matrix = frame_to_matrix(clusters)

    # Map detector coordinates to cluster numbers.
    pixel_lookup = {}

    for cluster_id, cluster in enumerate(clusters):
        for pixel in cluster:
            pixel_lookup[(pixel["x"], pixel["y"])] = cluster_id

    # Use the same color scale in both plots.
    values = frame_matrix[np.isfinite(frame_matrix)]

    if len(values) == 0:
        raise ValueError("No valid pixel values")

    vmin = float(values.min())
    vmax = float(values.max())

    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5

    norm = Normalize(vmin=vmin, vmax=vmax)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f0f0f0")

    fig, (ax_full, ax_detail) = plt.subplots(
        1, 2,
        figsize=(15, 7),
        constrained_layout=True
    )

    # Left plot: the complete detector.
    full_image = ax_full.imshow(
        frame_matrix,
        origin="lower",
        cmap=cmap,
        norm=norm,
        interpolation="nearest"
    )

    ax_full.set_title(f"Full detector — frame {frame_id}")
    ax_full.set_xlabel("X pixel")
    ax_full.set_ylabel("Y pixel")

    fig.colorbar(
        full_image,
        ax=[ax_full, ax_detail],
        label="Third-column value (unit unknown)",
        shrink=0.8
    )

    selected = [0]
    highlight = [None]

    def show_cluster(cluster_id):
        """Update the detail plot and highlight the selected area."""

        cluster = clusters[cluster_id]

        if not cluster:
            print(f"Cluster {cluster_id} is empty")
            return

        selected[0] = cluster_id

        local_matrix, xmin, ymin, xmax, ymax = (
            cluster_to_local_matrix(cluster)
        )

        ax_detail.clear()

        ax_detail.imshow(
            local_matrix,
            origin="lower",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            extent=(
                xmin - 0.5, xmax + 0.5,
                ymin - 0.5, ymax + 0.5
            )
        )

        width = xmax - xmin + 1
        height = ymax - ymin + 1

        fontsize = max(5, min(11, 140 / max(width, height)))

        # Write the third-column value inside each hit pixel.
        for pixel in cluster:
            ax_detail.text(
                pixel["x"],
                pixel["y"],
                f'{pixel["value"]:g}',
                ha="center",
                va="center",
                fontsize=fontsize,
                color="white",
                path_effects=[
                    pe.withStroke(
                        linewidth=2,
                        foreground="black"
                    )
                ]
            )

        if width <= 20:
            ax_detail.set_xticks(range(xmin, xmax + 1))

        if height <= 20:
            ax_detail.set_yticks(range(ymin, ymax + 1))

        if width <= 40 and height <= 40:
            ax_detail.set_xticks(
                np.arange(xmin - 0.5, xmax + 1, 1),
                minor=True
            )
            ax_detail.set_yticks(
                np.arange(ymin - 0.5, ymax + 1, 1),
                minor=True
            )
            ax_detail.grid(which="minor", alpha=0.3)

        pixel_values = np.array(
            [pixel["value"] for pixel in cluster]
        )

        total = pixel_values.sum()
        minimum = pixel_values.min()
        maximum = pixel_values.max()

        ax_detail.set_title(
            f"Cluster {cluster_id} | "
            f"N={len(cluster)} | "
            f"Sum={total:g}\n"
            f"Min={minimum:g} | Max={maximum:g}"
        )

        ax_detail.set_xlabel("X pixel")
        ax_detail.set_ylabel("Y pixel")

        # Highlight the selected cluster on the full detector.
        if highlight[0] is not None:
            highlight[0].remove()

        rectangle = Rectangle(
            (xmin - 0.5, ymin - 0.5),
            width,
            height,
            fill=False,
            edgecolor="red",
            linewidth=1.8
        )

        ax_full.add_patch(rectangle)
        highlight[0] = rectangle

        fig.canvas.draw_idle()

    def on_click(event):
        """Select a cluster by clicking a hit pixel."""

        if event.inaxes != ax_full:
            return

        if event.xdata is None or event.ydata is None:
            return

        x = round(event.xdata)
        y = round(event.ydata)

        cluster_id = pixel_lookup.get((x, y))

        if cluster_id is not None:
            show_cluster(cluster_id)

    def save_view():
        """Save the current visualization."""

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        filename = (
            f"{source.stem}_frame_{frame_id}_"
            f"cluster_{selected[0]}.png"
        )

        destination = RESULTS_DIR / filename

        fig.savefig(destination, dpi=200)

        print(f"Saved: {destination}")

    def on_key(event):
        """Navigate between clusters or save the figure."""

        if event.key == "right":
            next_id = (selected[0] + 1) % len(clusters)
            show_cluster(next_id)

        elif event.key == "left":
            previous_id = (selected[0] - 1) % len(clusters)
            show_cluster(previous_id)

        elif event.key == "p":
            save_view()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    if not 0 <= first_cluster < len(clusters):
        raise ValueError(
            f"Cluster must be between 0 and {len(clusters)-1}"
        )

    show_cluster(first_cluster)

    print(f"File: {source}")
    print(f"Frame: {frame_id}")
    print(f"Clusters: {len(clusters)}")
    print(f"Pixel records: {sum(len(c) for c in clusters)}")
    print("Click a hit pixel to select its cluster.")
    print("Left/Right: change cluster | P: save PNG")

    save_view()
    plt.show()


def main():
    """Read command-line arguments and start the viewer."""

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=int, default=1)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--cluster", type=int, default=0)
    parser.add_argument("--file", type=str)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--stats", action="store_true")

    args = parser.parse_args()

    if args.list:
        files = find_data_files()

        if not files:
            print("No measurement files found")

        for path in files:
            print(path.relative_to(PROJECT_ROOT))

        return

    source = select_file(args.dataset, args.file)

    if args.stats:
        print_frame_stats(source)
        return

    create_viewer(
        source=source,
        frame_id=args.frame,
        first_cluster=args.cluster
    )


if __name__ == "__main__":
    main()
