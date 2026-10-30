
import re
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import label

# ==========================================
# File select
# ==========================================

FILE = "itot__r0000000006.clog"

# ==========================================
# File parsing
# ==========================================

frame_re = re.compile(
    r"Frame\s+(\d+).*?\(([\d.]+)"
)

data_re = re.compile(
    r"\[(\d+),\s*(\d+),\s*(\d+)\]"
)

frames = []
current_frame = None

with open(FILE, "r", encoding="utf-8") as f:

    for line in f:

        line = re.sub(r"<[^>]+>", "", line)

        frame_match = frame_re.search(line)

        if frame_match:

            frame_id = int(frame_match.group(1))
            timestamp = float(frame_match.group(2))

            current_frame = {
                "id": frame_id,
                "timestamp": timestamp,
                "data": []
            }

            frames.append(current_frame)
            continue

        if current_frame is None:
            continue

        for match in data_re.finditer(line):

            x, y, value = map(
                int,
                match.groups()
            )

            current_frame["data"].append(
                (x, y, value)
            )

print(f"Found {len(frames)} frames")

# ==========================================
# Create coordinate system
# ==========================================

all_x = list(range(256))
all_y = list(range(256))

x_map = {
    value: value
    for value in all_x
}

y_map = {
    value: value
    for value in all_y
}

# ==========================================
# Converting frames to matrices
# ==========================================
#--------------------------------
#minimal particle size in pixels
#--------------------------------
MIN_CLUSTER_SIZE = 10

frame_matrices = []

for frame in frames:

    matrix = np.full(
        (256, 256),
        np.nan
    )

    for x, y, value in frame["data"]:

        if 0 <= x < 256 and 0 <= y < 256:

            matrix[
                y,
                x
            ] = value
    # -----------------------------
    # deleting of x% lowest values
    # -----------------------------
    values = matrix[~np.isnan(matrix)]

    if len(values) == 0:

        frame_matrices.append(matrix)
        continue

    threshold = np.percentile(
        values,
        40
    )

    matrix[matrix < threshold] = np.nan

    binary = ~np.isnan(matrix)

    labels, n_clusters = label(binary)

    filtered_matrix = np.full(
        matrix.shape,
        np.nan
    )

    for cluster_id in range(
        1,
        n_clusters + 1
    ):

        mask = labels == cluster_id

        pixel_count = np.sum(mask)

        if pixel_count >= MIN_CLUSTER_SIZE:

            filtered_matrix[mask] = matrix[mask]

    frame_matrices.append(
        filtered_matrix
    )

# ==========================================
# Delete empty frames
# ==========================================
#-------------------------------------
# Minimal number of pixels in frame
#-------------------------------------
MIN_PIXELS_IN_FRAME = 10

valid_frames = []
valid_matrices = []

for frame, matrix in zip(frames, frame_matrices):

    valid_pixels = np.count_nonzero(
        ~np.isnan(matrix)
    )

    if valid_pixels < MIN_PIXELS_IN_FRAME:

        print(
            f"Deleted frame ID={frame['id']} "
            f"(pixels={valid_pixels})"
        )

        continue

    valid_frames.append(frame)
    valid_matrices.append(matrix)

frames = valid_frames
frame_matrices = valid_matrices

if len(frames) == 0:

    raise RuntimeError(
        "No non-empty frames left."
    )

print(
    f"After filtering remaining "
    f"{len(frames)} frames."
)

# ==========================================
# GUI
# ==========================================

current_index = 0

fig, ax = plt.subplots(
    figsize=(14, 8)
)

plt.subplots_adjust(
    bottom=0.15
)

first_matrix = frame_matrices[0]

cmap = plt.cm.jet.copy()
cmap.set_bad("white")

img = ax.imshow(
    first_matrix,
    cmap=cmap,
    origin="lower",
    extent=[0, 255, 0, 255],
    vmin=np.nanmin(first_matrix),
    vmax=np.nanmax(first_matrix)
)

cbar = plt.colorbar(img)
cbar.set_label("Value")

ax.set_xlabel("X")
ax.set_ylabel("Y")

ax.set_xticks([0, 255])
ax.set_yticks([0, 255])

title = ax.set_title("")

# ==========================================
# Refresh display
# ==========================================

def refresh():

    frame = frames[current_index]

    matrix = frame_matrices[current_index]

    frame_min = np.nanmin(matrix)
    frame_max = np.nanmax(matrix)

    if frame_min == frame_max:
        frame_max += 1

    img.set_data(matrix)

    img.set_clim(
        frame_min,
        frame_max
    )

    dt = datetime.fromtimestamp(
        frame["timestamp"]
    )

    title.set_text(
        f"Frame {frame['id']} | "
        f"{dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} | "
        f"min={frame_min:.0f} "
        f"max={frame_max:.0f} | "
        f"{current_index + 1}/{len(frames)}"
    )

    fig.canvas.draw_idle()

# ==========================================
# Slider
# ==========================================

slider_ax = plt.axes(
    [0.15, 0.05, 0.7, 0.03]
)

slider = Slider(
    slider_ax,
    "Frame",
    0,
    len(frames) - 1,
    valinit=0,
    valstep=1
)

slider_busy = False

def slider_changed(val):

    global current_index

    if slider_busy:
        return

    current_index = int(val)

    refresh()

slider.on_changed(
    slider_changed
)

# ==========================================
# Frame switching
# ==========================================

def set_frame(idx):

    global current_index
    global slider_busy

    idx = max(
        0,
        min(
            len(frames) - 1,
            idx
        )
    )

    current_index = idx

    slider_busy = True
    slider.set_val(idx)
    slider_busy = False

    refresh()

# ==========================================
# Mouse wheel
# ==========================================

def on_scroll(event):

    if event.button == "up":

        set_frame(
            current_index + 1
        )

    elif event.button == "down":

        set_frame(
            current_index - 1
        )

fig.canvas.mpl_connect(
    "scroll_event",
    on_scroll
)

# ==========================================
# Keyboard
# ==========================================

def on_key(event):

    if event.key == "right":

        set_frame(
            current_index + 1
        )

    elif event.key == "left":

        set_frame(
            current_index - 1
        )

    elif event.key == "pageup":

        set_frame(
            current_index + 100
        )

    elif event.key == "pagedown":

        set_frame(
            current_index - 100
        )

    elif event.key == "home":

        set_frame(0)

    elif event.key == "end":

        set_frame(
            len(frames) - 1
        )

fig.canvas.mpl_connect(
    "key_press_event",
    on_key
)

# ==========================================
# First render
# ==========================================

refresh()

print("\nControls")
print("--------------------------------")
print("Mouse wheel     : next/previous frame")
print("← →              : next/previous frame")
print("PgUp / PgDn      : ±100 frames")
print("Home             : první frame")
print("End              : last frame")
print("--------------------------------\n")

plt.show()
