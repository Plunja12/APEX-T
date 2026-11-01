 #pouze diagnostický skript pro kontrolu že se data načítají správně
from pileup.io.clog_reader import iter_frames
from viewer import select_file


def inspect_dataset(dataset_id):
    """Find the first five non-empty frames."""

    file_path = select_file(dataset_id)

    print(f"\nDATASET {dataset_id}")
    print(f"File: {file_path.name}")
    print("-" * 45)

    found = 0
    scanned = 0

    for frame_id, clusters in iter_frames(file_path):
        scanned += 1

        if not clusters:
            continue

        pixel_count = sum(
            len(cluster) for cluster in clusters
        )

        print(
            f"Frame {frame_id}: "
            f"{len(clusters)} clusters, "
            f"{pixel_count} pixels"
        )

        found += 1

        if found == 5:
            break

    if found == 0:
        print("No non-empty frames found.")

    print(f"Frames scanned: {scanned}")


def main():
    for dataset_id in (1, 2, 3):
        inspect_dataset(dataset_id)


if __name__ == "__main__":
    main()
