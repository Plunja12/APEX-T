
from pathlib import Path
import re

import numpy as np


# Find the frame number in a header.
FRAME_RE = re.compile(r"^Frame\s+(\d+)\b")

# Find pixel records inside square brackets.
PIXEL_RE = re.compile(r"\[([^\[\]]+)\]")


def iter_frames(file_path):
    """Read a CLOG file, yielding one frame at a time."""

    file_path = Path(file_path)

    frame_id = None
    clusters = []

    with file_path.open("r", encoding="utf-8-sig") as file:

        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            # A new frame starts here.
            if line.startswith("Frame "):
                match = FRAME_RE.match(line)

                if match is None:
                    raise ValueError(
                        f"Invalid frame header at line {line_number}"
                    )

                if frame_id is not None:
                    yield frame_id, clusters

                frame_id = int(match.group(1))
                clusters = []
                continue

            if frame_id is None:
                raise ValueError(
                    f"Pixel data before frame header at line {line_number}"
                )

            # Each data line represents one exported cluster.
            records = PIXEL_RE.findall(line)

            remaining = PIXEL_RE.sub("", line).strip()

            if not records or remaining:
                raise ValueError(
                    f"Invalid cluster at line {line_number}: {line[:80]}"
                )

            cluster = []

            for record in records:
                parts = [part.strip() for part in record.split(",")]

                if len(parts) not in (3, 4):
                    raise ValueError(
                        f"Invalid pixel at line {line_number}"
                    )

                x = int(parts[0])
                y = int(parts[1])
                value = float(parts[2])

                extra = (
                    float(parts[3])
                    if len(parts) == 4
                    else None
                )

                if not (0 <= x < 256 and 0 <= y < 256):
                    raise ValueError(
                        f"Pixel outside detector: ({x}, {y})"
                    )

                cluster.append({
                    "x": x,
                    "y": y,
                    "value": value,
                    "extra": extra
                })

            clusters.append(cluster)

    if frame_id is not None:
        yield frame_id, clusters


def get_frame(file_path, target_frame):
    """Find and return the clusters of one frame."""

    for frame_id, clusters in iter_frames(file_path):
        if frame_id == target_frame:
            return clusters

    raise ValueError(f"Frame {target_frame} not found")


def frame_to_matrix(clusters):
    """Convert all clusters into a 256x256 matrix."""

    matrix = np.full((256, 256), np.nan)

    for cluster in clusters:
        for pixel in cluster:
            x = pixel["x"]
            y = pixel["y"]
            value = pixel["value"]

            if not np.isnan(matrix[y, x]):
                raise ValueError(
                    f"Duplicate pixel at ({x}, {y})"
                )

            matrix[y, x] = value

    return matrix


def cluster_to_local_matrix(cluster):
    """Create a small matrix containing one cluster."""

    if not cluster:
        raise ValueError("Cannot plot an empty cluster")

    xs = [pixel["x"] for pixel in cluster]
    ys = [pixel["y"] for pixel in cluster]

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    width = xmax - xmin + 1
    height = ymax - ymin + 1

    matrix = np.full((height, width), np.nan)

    for pixel in cluster:
        x = pixel["x"] - xmin
        y = pixel["y"] - ymin

        if not np.isnan(matrix[y, x]):
            raise ValueError("Duplicate pixel inside cluster")

        matrix[y, x] = pixel["value"]

    return matrix, xmin, ymin, xmax, ymax
