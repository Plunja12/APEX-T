"""APEX-T interactive M3 labeler (curved skeleton trajectories + signal statistics).

Run from the project root (after pip install -e .):
    python scripts/label_clusters.py --dataset 2 --min-m 40

Original CLOG data is opened read-only; labeled results go into results/labels.
Scores are cached in results/cache and loaded before the Matplotlib window opens.
No expensive candidate search takes place during clicks.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import random
import sqlite3
import time

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.widgets import Button
import numpy as np

from pileup.io.clog_reader import iter_frames, cluster_to_local_matrix
from pileup.multiple_tracks import score_m, VERSION
from viewer import select_file, PROJECT_ROOT

# Old columns remain unchanged so existing CSVs can be safely appended to.
BASE_FIELDS = [
    "timestamp_utc", "dataset", "source", "frame", "cluster", "label",
    "n_pixels", "signal_sum", "signal_max", "M", "m_status",
    "angle_deg", "coverage", "balance", "arm_balance",
]
NEW_FIELDS = BASE_FIELDS + [
    "m_version", "geometry", "n_tracks", "separation_px", "gap_ratio",
    "E", "signal_density", "signal_peak_ratio", "path_smoothness", "score_combined",
    "E_density", "E_maximum"
]


def existing_labels(csv_path, source):
    completed = set()
    if not csv_path.exists():
        return completed
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("source") == str(source.resolve()):
                try:
                    completed.add((int(row["frame"]), int(row["cluster"])))
                except (KeyError, TypeError, ValueError):
                    pass
    return completed


def read_clusters(source, min_pixels):
    """Keep original IDs and all eligible clusters without changing the file."""
    items = []
    values = []
    for frame_id, clusters in iter_frames(source):
        for cluster_id, cluster in enumerate(clusters):
            if len(cluster) < min_pixels:
                continue
            items.append((frame_id, cluster_id, cluster))
            values.extend(p["value"] for p in cluster)
    return items, values




def energy_percentiles(items):
    """Dataset-relative proxy E from signal density and strongest pixel.

    Compare only within logarithmic cluster-size bins. Do NOT interpret raw
    pixel values as calibrated keV or E as a probability of a pile-up.
    """
    summaries = {}
    buckets = {}
    for frame, cid, cluster in items:
        values = np.asarray([p["value"] for p in cluster], dtype=float)
        valid = bool(len(values) and np.isfinite(values).all() and
                     (values >= 0).all())
        density = float(values.mean()) if valid else float('nan')
        maximum = float(values.max()) if valid else float('nan')
        bin_id = int(np.log2(max(1, len(cluster))))
        summaries[(frame, cid)] = (bin_id, density, maximum)
        if valid:
            buckets.setdefault(bin_id, []).append((density, maximum))

    all_values = np.asarray([(a, b) for _, a, b in summaries.values()
                             if np.isfinite(a) and np.isfinite(b)], dtype=float)
    if len(all_values) < 30:
        return {key: (0.0, density, maximum, 0.0, 0.0)
                for key, (_, density, maximum) in summaries.items()}
    global_refs = (np.sort(all_values[:, 0]), np.sort(all_values[:, 1]))
    refs = {}
    for key, vals in buckets.items():
        arr = np.asarray(vals)
        refs[key] = ((np.sort(arr[:,0]), np.sort(arr[:,1]))
                     if len(arr) >= 30 else global_refs)

    def tail_rank(reference, value, starting_percentile):
        pct = np.searchsorted(reference, value, side='right') / len(reference)
        return float(np.clip((pct-starting_percentile) /
                             (1-starting_percentile), 0, 1))

    result = {}
    for key, (b, density, maximum) in summaries.items():
        if not (np.isfinite(density) and np.isfinite(maximum)):
            result[key] = (0.0, density, maximum, 0.0, 0.0)
            continue
        densities, maxima = refs[b]
        e_density = tail_rank(densities, density, 0.80)
        e_maximum = tail_rank(maxima, maximum, 0.90)
        # Density gets most weight; a single hot pixel is weaker evidence.
        E = 0.8 * e_density + 0.2 * e_maximum
        result[key] = (float(E), density, maximum, e_density, e_maximum)
    return result


def source_signature(source):
    """Invalidate cached scores whenever source file or algorithm changes."""
    st = source.stat()
    identity = f"{source.resolve()}|{st.st_size}|{st.st_mtime_ns}|{VERSION}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def screen_candidates(source, dataset, items, min_m, min_score,
                      include_high_e, energy, completed):
    """Score ONCE; save scores in SQLite for next run / threshold changes."""
    cache = PROJECT_ROOT / "results" / "cache" / f"m_scores_dataset_{dataset}.sqlite3"
    cache.parent.mkdir(parents=True, exist_ok=True)
    signature = source_signature(source)
    conn = sqlite3.connect(cache)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS m_scores ("
                     "signature TEXT NOT NULL, frame INTEGER NOT NULL, "
                     "cluster INTEGER NOT NULL, score REAL NOT NULL, "
                     "PRIMARY KEY(signature, frame, cluster))")
        cached = {(int(f), int(c)): float(s) for f, c, s in
                  conn.execute("SELECT frame,cluster,score FROM m_scores "
                               "WHERE signature = ?", (signature,))}
        hits = sum((f, c) in cached for f, c, _ in items)
        missing = len(items) - hits
        print(f"M version: {VERSION} | Cache: {hits} hits, {missing} to compute", flush=True)
        start = time.perf_counter()
        batch = []
        for index, (frame, cid, cluster) in enumerate(items, start=1):
            if (frame, cid) not in cached:
                try:
                    score = float(score_m(cluster)["M"])
                    if not np.isfinite(score):
                        score = 0.0
                except (ValueError, np.linalg.LinAlgError) as e:
                    print(f"M warning: frame={frame}, cluster={cid}: {e}")
                    score = 0.0
                cached[frame, cid] = score
                batch.append((signature, frame, cid, score))
            if len(batch) >= 300:
                with conn:
                    conn.executemany("INSERT OR REPLACE INTO m_scores VALUES (?,?,?,?)", batch)
                batch.clear()
            if index % 1000 == 0:
                print(f"Scanned {index}/{len(items)} clusters...", flush=True)
        if batch:
            with conn:
                conn.executemany("INSERT OR REPLACE INTO m_scores VALUES (?,?,?,?)", batch)
        elapsed = time.perf_counter() - start
        def matches(f, c):
            if (f, c) in completed:
                return False
            m, e = cached[f, c], energy[f, c][0]
            geometry_ok = (100*m >= min_m and
                           (min_score is None or
                            100*(0.8*m + 0.2*e) >= min_score))
            # Optional discovery of rare high-signal candidates even if their
            # tracks lie on top of one another, hence geometry M is small.
            energy_ok = include_high_e is not None and 100*e >= include_high_e
            return geometry_ok or energy_ok

        candidates = [(f, c, cl) for f, c, cl in items if matches(f, c)]
        print(f"Screening completed in {elapsed:.1f}s; "
              f"matching unreviewed clusters: {len(candidates)}", flush=True)
        return candidates
    finally:
        conn.close()


class ClusterLabeler:
    def __init__(self, *, candidates, source, dataset, csv_path, min_m,
                 scale, seed, dataset_values, energy):
        self.candidates = list(candidates)
        random.Random(seed).shuffle(self.candidates)
        self.source = source.resolve()
        self.dataset = dataset
        self.csv_path = csv_path
        self.min_m = min_m
        self.energy = energy
        self.current = None
        self.result = None
        self.saved_count = 0

        self.global_norm = None
        if scale == "global":
            vals = np.asarray(dataset_values, dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                lo = float(min(0.0, vals.min()))
                hi = float(np.percentile(vals, 99.5))
                self.global_norm = Normalize(vmin=lo, vmax=max(lo + 1e-9, hi),
                                             clip=True)

        self.fig, (self.ax_raw, self.ax_fit) = plt.subplots(
            1, 2, figsize=(13, 7.4))
        self.fig.subplots_adjust(left=0.06, right=0.96, top=0.82,
                                 bottom=0.25, wspace=0.18)
        self.cmap = plt.get_cmap("viridis").copy()
        self.cmap.set_bad("#eeeeee")
        self.buttons = []
        for x, label, color, value in [
            (0.10, "PILE-UP [1]", "#bce5c3", "pileup"),
            (0.39, "NOT PILE-UP [2]", "#f2cccc", "not_pileup"),
            (0.68, "UNCERTAIN [3]", "#f8e8b5", "uncertain"),
        ]:
            btn_ax = self.fig.add_axes([x, 0.105, 0.23, 0.075])
            btn = Button(btn_ax, label, color=color, hovercolor="#f5f5f5")
            btn.on_clicked(lambda event, v=value: self.label(v))
            self.buttons.append(btn)

        self.info = self.fig.text(0.5, 0.93, "", ha="center", fontsize=11)
        self.progress = self.fig.text(0.5, 0.87, "", ha="center", fontsize=10)
        self.fig.text(0.5, 0.045,
                      "1: pile-up | 2: not pile-up | 3: uncertain | "
                      "Space: skip | Q: exit", ha="center", fontsize=10)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.next_cluster()

    def next_cluster(self):
        """O(1) selection: all filtering was performed before GUI startup."""
        self.current = None
        self.result = None
        if not self.candidates:
            self.ax_raw.clear()
            self.ax_fit.clear()
            self.ax_raw.text(0.5, 0.5, "No more matching clusters",
                             ha="center", va="center",
                             transform=self.ax_raw.transAxes)
            self.ax_fit.set_axis_off()
            self.info.set_text("Finished — no unreviewed clusters above threshold")
            self.progress.set_text(f"Saved this session: {self.saved_count}")
            self.fig.canvas.draw_idle()
            return
        self.current = self.candidates.pop()
        self.result = score_m(self.current[2])  # fast recomputation for drawing
        self.draw_current()

    def draw_current(self):
        frame, cid, cluster = self.current
        result = self.result
        matrix, xmin, ymin, xmax, ymax = cluster_to_local_matrix(cluster)
        values = np.array([p["value"] for p in cluster], dtype=float)
        if self.global_norm is None:
            finite = values[np.isfinite(values)]
            lo = float(finite.min()) if len(finite) else 0.0
            hi = float(finite.max()) if len(finite) else 1.0
            norm = Normalize(vmin=lo, vmax=max(lo + 1e-9, hi))
        else:
            norm = self.global_norm

        for ax in (self.ax_raw, self.ax_fit):
            ax.clear()
            ax.imshow(matrix, cmap=self.cmap, norm=norm,
                      origin="lower", interpolation="nearest",
                      extent=(xmin-.5, xmax+.5, ymin-.5, ymax+.5))
            padding = max(2.0, .07 * max(xmax-xmin+1, ymax-ymin+1))
            ax.set_xlim(xmin-padding, xmax+padding)
            ax.set_ylim(ymin-padding, ymax+padding)
            ax.set_aspect("equal")
            ax.set_xlabel("x [pixel]")
            ax.set_ylabel("y [pixel]")
        self.ax_raw.set_title("Original cluster")
        self.ax_fit.set_title(
            f"M {result['M']*100:.1f}/100 | {result['geometry']} | "
            f"curves: {result['n_tracks']}", fontsize=11)

        # Draw actual curved skeleton trajectories, NOT RANSAC line fragments.
        colors = ["#00f5ff", "#ff9c30", "#f542c5", "#a9fd39"]
        for curve, color in zip(result["paths"], colors):
            xy = np.asarray(curve, dtype=float)
            self.ax_fit.plot(xy[:, 0], xy[:, 1], lw=2.3,
                             color=color, alpha=0.94)
            if len(xy) > 5:
                self.ax_fit.scatter(xy[[0,-1], 0], xy[[0,-1], 1],
                                    s=18, color=color, edgecolors='black')
        if result["junction"] is not None:
            x, y = result["junction"]
            self.ax_fit.plot(x, y, marker='x', color='red',
                             ms=10, mew=2.0)

        E, density, max_signal, Ed, Em = self.energy.get(
            (frame, cid), (0.0, float('nan'), float('nan'), 0.0, 0.0))
        combined = 0.8 * result['M'] + 0.2 * E
        self.info.set_text(
            f"Dataset {self.dataset} | Frame {frame} | Cluster {cid} | "
            f"Pixels {len(cluster)} | Sum {np.nansum(values):.5g} | "
            f"Max {np.nanmax(values):.5g} | signal density {density:.2f}")
        self.progress.set_text(
            f"M {result['M']*100:.0f} | E {E*100:.0f} | "
            f"Combined {combined*100:.0f} (0.8M+0.2E) | "
            f"Remaining: {len(self.candidates)} | Saved: {self.saved_count}")
        self.fig.canvas.draw_idle()

    def label(self, value):
        if self.current is None:
            return
        frame, cid, cluster = self.current
        r = self.result
        vals = np.asarray([p["value"] for p in cluster], dtype=float)
        E, density, max_signal, Ed, Em = self.energy.get(
            (frame, cid), (0.0, float('nan'), float('nan'), 0.0, 0.0))
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": self.dataset, "source": str(self.source),
            "frame": frame, "cluster": cid, "label": value,
            "n_pixels": len(cluster), "signal_sum": float(np.nansum(vals)),
            "signal_max": float(np.nanmax(vals)), "M": r["M"],
            "m_status": f"{VERSION}:{r['status']}",
            "angle_deg": "", "coverage": r["coverage"],
            "balance": "", "arm_balance": "",
            "m_version": VERSION, "geometry": r["geometry"],
            "n_tracks": r["n_tracks"],
            "separation_px": "", "gap_ratio": "",
            "E": E, "signal_density": density,
            "signal_peak_ratio": r["signal_peak_ratio"],
            "path_smoothness": r["path_smoothness"],
            "score_combined": 0.8 * r["M"] + 0.2 * E,
            "E_density": Ed, "E_maximum": Em,
        }
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        if is_new:
            fields = NEW_FIELDS
        else:
            # Use the original header to avoid corrupting pre-existing CSVs.
            with self.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                fields = csv.DictReader(f).fieldnames
            if not fields or not set(BASE_FIELDS).issubset(fields):
                raise ValueError("Existing label CSV has an unexpected header. "
                                 "Use --output to choose a new CSV filename.")
        with self.csv_path.open("a", newline="",
                                encoding="utf-8-sig" if is_new else "utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if is_new:
                writer.writeheader()
            writer.writerow(row)
        self.saved_count += 1
        print(f"Saved frame {frame} cluster {cid}: {value}, "
              f"M3={r['M']*100:.1f} E={E*100:.1f} [{r['geometry']}]", flush=True)
        self.next_cluster()

    def on_key(self, event):
        if event.key == "1":
            self.label("pileup")
        elif event.key == "2":
            self.label("not_pileup")
        elif event.key == "3":
            self.label("uncertain")
        elif event.key == " ":
            self.next_cluster()
        elif event.key in ("q", "escape"):
            plt.close(self.fig)


def main():
    parser = argparse.ArgumentParser(description="APEX-T M3 curved trajectories + relative signal labeler")
    parser.add_argument("--dataset", type=int, default=2)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--min-pixels", type=int, default=16)
    parser.add_argument("--min-m", type=float, default=40.0,
                        help="Minimum geometric M score 0..100")
    parser.add_argument("--min-score", type=float, default=None,
                        help="Optional minimum combined score 0.8M + 0.2E (0..100)")
    parser.add_argument("--include-high-e", type=float, default=None,
                        help="Also include high-signal clusters with E >= threshold, even if M low")
    parser.add_argument("--scale", choices=["global", "local"], default="global")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.min_pixels < 1:
        parser.error("--min-pixels must be >= 1")
    if not np.isfinite(args.min_m) or not 0 <= args.min_m <= 100:
        parser.error("--min-m must be 0..100")
    if args.min_score is not None and (not np.isfinite(args.min_score) or
                                       not 0 <= args.min_score <= 100):
        parser.error("--min-score must be 0..100")
    if args.include_high_e is not None and (
            not np.isfinite(args.include_high_e) or
            not 0 <= args.include_high_e <= 100):
        parser.error("--include-high-e must be 0..100")
    source = select_file(args.dataset, args.file)
    csv_path = args.output or (PROJECT_ROOT / "results" / "labels" /
                               f"dataset_{args.dataset}_m3.csv")
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path
    completed = existing_labels(csv_path, source)
    if args.output is None:
        # Don't relabel items from the older M2 training table.
        old = PROJECT_ROOT / "results" / "labels" / f"dataset_{args.dataset}.csv"
        completed |= existing_labels(old, source)
    items, dataset_values = read_clusters(source, args.min_pixels)
    print(f"Input: {source} | {len(items)} eligible clusters "
          f"| already labeled {len(completed)}", flush=True)
    energy = energy_percentiles(items)
    candidates = screen_candidates(source, args.dataset, items,
                                   args.min_m, args.min_score,
                                   args.include_high_e, energy, completed)
    print(f"M geometry threshold >= {args.min_m:g}/100; E is a dataset-relative "
          "uncalibrated signal-density anomaly (NOT probability).")
    if args.min_score is not None:
        print(f"Combined score threshold >= {args.min_score:g}/100")
    if args.include_high_e is not None:
        print(f"Additional high-signal candidates: E >= {args.include_high_e:g}/100")
    print(f"Label CSV: {csv_path}")
    if not candidates:
        print("No matches. Try lowering --min-m (e.g. 20) or --min-pixels.")
        return
    app = ClusterLabeler(candidates=candidates, source=source, dataset=args.dataset,
                         csv_path=csv_path, min_m=args.min_m, scale=args.scale,
                         seed=args.seed, dataset_values=dataset_values,
                         energy=energy)
    plt.show()
    print(f"Session ended, labeled {app.saved_count} clusters")

if __name__ == "__main__":
    main()
