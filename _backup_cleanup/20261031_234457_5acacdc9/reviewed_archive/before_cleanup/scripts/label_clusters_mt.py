"""APEX-T: label candidate clusters with T skeleton and existing M side by side.

From repo root: python scripts/label_clusters_mt.py --dataset 2 --min-t 40 --min-m 40
Requirements: python -m pip install numpy matplotlib scipy scikit-image
This script does not modify the original data or current label_clusters.py.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import random
import sqlite3
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from pileup.io.clog_reader import iter_frames, cluster_to_local_matrix
from pileup.multiple_tracks import score_m, VERSION as M_VERSION
from pileup.topology import score_t, VERSION_T

FIELDS = ['timestamp_utc', 'dataset', 'source', 'frame', 'cluster', 'label',
          'n_pixels', 'M', 'T', 'm_version', 't_version', 'm_status', 't_status',
          't_endpoints', 't_junctions', 'notes']


def select_measurement(dataset, custom):
    if custom:
        p = Path(custom).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f'Not a measurement file: {p}')
        return p
    data_dir = ROOT / 'data' / 'raw'
    folders = [p for p in data_dir.iterdir()
               if p.is_dir() and p.name.startswith(f'{dataset:02d} ')]
    if len(folders) != 1:
        raise RuntimeError(f'Expected exactly one dataset {dataset} folder, found {len(folders)}')
    files = [p for p in folders[0].rglob('*') if p.is_file() and
             (p.suffix.lower() == '.clog' or p.name.lower().startswith(('itot', 'tot_', 'cnt_')))]
    if len(files) != 1:
        raise RuntimeError(f'Expected exactly one measurement file, found {len(files)}: {files}')
    return files[0]


def read_candidates(source, min_pixels):
    items = []
    for frame_id, clusters in iter_frames(source):
        for cid, cl in enumerate(clusters):
            if len(cl) >= min_pixels:
                items.append((frame_id, cid, cl))
    return items


def existing_labels(path, source):
    completed = set()
    if not path.exists():
        return completed
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise RuntimeError(f'CSV schema differs: {path}; use a different --output path')
        for row in reader:
            if row['source'] == str(source.resolve()):
                completed.add((int(row['frame']), int(row['cluster'])))
    return completed


def score_candidates(source, items, cache_path, min_m, min_t, done):
    st = source.stat()
    sig = hashlib.sha256(f'{source.resolve()}:{st.st_size}:{st.st_mtime_ns}:{M_VERSION}:{VERSION_T}'.encode()).hexdigest()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cache_path)
    try:
        conn.execute('CREATE TABLE IF NOT EXISTS scores (signature TEXT, frame INTEGER, cluster INTEGER, m REAL, t REAL, PRIMARY KEY(signature,frame,cluster))')
        cached = {(f, c): (m, t) for f, c, m, t in conn.execute(
            'SELECT frame, cluster, m, t FROM scores WHERE signature = ?', (sig,))}
        print(f'Cached scores: {len(cached)} | Total eligible: {len(items)}', flush=True)
        insert = []
        for index, (frame, cid, cluster) in enumerate(items, 1):
            if (frame, cid) not in cached:
                try:
                    m = float(score_m(cluster)['M'])
                    t = float(score_t(cluster)['T'])
                except (ValueError, np.linalg.LinAlgError) as exc:
                    print(f'Score issue frame={frame} cluster={cid}: {exc}', flush=True)
                    m, t = 0.0, 0.0
                cached[(frame, cid)] = (m, t)
                insert.append((sig, frame, cid, m, t))
            if len(insert) >= 200:
                with conn:
                    conn.executemany('INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?)', insert)
                insert.clear()
            if index % 1000 == 0:
                print(f'Analyzed {index}/{len(items)}', flush=True)
        if insert:
            with conn:
                conn.executemany('INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?)', insert)
        return [(frame, cid, cl) for frame, cid, cl in items
                if (frame, cid) not in done and
                (100*cached[(frame, cid)][0] >= min_m or
                 100*cached[(frame, cid)][1] >= min_t)]
    finally:
        conn.close()


class Labeler:
    def __init__(self, items, source, output, dataset, seed):
        self.items = list(items)
        random.Random(seed).shuffle(self.items)
        self.source = source.resolve()
        self.output = output
        self.dataset = dataset
        self.saved = 0
        self.current = None
        self.m_result = None
        self.t_result = None
        self.fig, (self.raw, self.topo, self.multi) = plt.subplots(1, 3, figsize=(16, 7.4))
        self.fig.subplots_adjust(left=.055, right=.985, top=.82, bottom=.24, wspace=.19)
        self.title = self.fig.text(.5, .945, '', ha='center', fontsize=13, fontweight='bold')
        self.meta = self.fig.text(.5, .88, '', ha='center', fontsize=10)
        self.help = self.fig.text(.5, .035,
            '1: pile-up    2: not pile-up    3: uncertain    Space: skip    Q: exit',
            ha='center', fontsize=10)
        self.buttons = []
        for x, label, color, key in [
            (.09, 'PILE-UP [1]', '#bce5c3', 'pileup'),
            (.385, 'NOT PILE-UP [2]', '#f2cccc', 'not_pileup'),
            (.68, 'UNCERTAIN [3]', '#f8e8b5', 'uncertain')]:
            b = Button(self.fig.add_axes([x, .105, .235, .075]), label,
                       color=color, hovercolor='#eeeeee')
            b.on_clicked(lambda evt, value=key: self.record(value))
            self.buttons.append(b)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.next_cluster()

    def next_cluster(self):
        if not self.items:
            self.current = None
            for ax in (self.raw, self.topo, self.multi):
                ax.clear()
                ax.set_axis_off()
            self.title.set_text('All selected clusters reviewed')
            self.meta.set_text(f'Saved this session: {self.saved}')
            self.fig.canvas.draw_idle()
            return
        self.current = self.items.pop()
        f, cid, cluster = self.current
        self.m_result = score_m(cluster)
        self.t_result = score_t(cluster)
        self.draw()

    def draw(self):
        frame, cid, cluster = self.current
        m, t = self.m_result, self.t_result
        matrix, xmin, ymin, xmax, ymax = cluster_to_local_matrix(cluster)
        vals = np.asarray([p['value'] for p in cluster], dtype=float)
        vals = vals[np.isfinite(vals)]
        lo, hi = (float(vals.min()), float(vals.max())) if vals.size else (0., 1.)
        norm = Normalize(lo, hi if hi > lo else lo + 1)
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad('#efefef')
        padding = max(2, .08*max(xmax-xmin+1, ymax-ymin+1))
        for ax in (self.raw, self.topo, self.multi):
            ax.clear()
            ax.imshow(matrix, origin='lower', cmap=cmap, norm=norm,
                      interpolation='nearest', extent=(xmin-.5, xmax+.5, ymin-.5, ymax+.5))
            ax.set_xlim(xmin-padding, xmax+padding)
            ax.set_ylim(ymin-padding, ymax+padding)
            ax.set_aspect('equal')
            ax.set_xlabel('x [pixel]')
            ax.set_ylabel('y [pixel]')
        self.raw.set_title('Original pixels / signal')
        self.topo.set_title(f'T = {t["T"]*100:.0f}/100 | ends={t["n_endpoints_t"]} | junctions={t["n_junctions_t"]}', fontsize=11)
        self.multi.set_title(f'M = {m["M"]*100:.0f}/100 | {m.get("geometry", "unresolved")}', fontsize=11)
        sk = np.asarray(t['skeleton_t'], dtype=float).reshape(-1,2)
        if len(sk):
            self.topo.scatter(sk[:,0], sk[:,1], s=12, c='#fff145', marker='s',
                              edgecolors='none', zorder=5)
        ep = np.asarray(t['endpoints_t'], dtype=float).reshape(-1,2)
        if len(ep):
            self.topo.scatter(ep[:,0], ep[:,1], c='#12c8fc', s=47,
                              edgecolors='#15232b', zorder=6, label='Endpoints')
        jp = np.asarray(t['junctions_t'], dtype=float).reshape(-1,2)
        if len(jp):
            self.topo.scatter(jp[:,0], jp[:,1], c='#ff4e44', s=100,
                              marker='x', linewidths=2.5, zorder=7, label='Junctions')
        # Existing M2: show the estimated fitted lines, if available.
        for i, line in enumerate(m.get('lines', [])):
            try:
                center = np.asarray(line['center'], dtype=float)
                direct = np.asarray(line['direction'], dtype=float)
                points = np.asarray([(p['x'], p['y']) for p in cluster], dtype=float)
                indices = np.asarray(line.get('indices', []), dtype=int)
                if len(indices) < 2:
                    continue
                projections = (points[indices]-center) @ direct
                p1 = center + projections.min()*direct
                p2 = center + projections.max()*direct
                self.multi.plot([p1[0], p2[0]], [p1[1], p2[1]],
                                lw=2.5, color=['#00eaff','#ffbc39','#ff62de','#a1ff45'][i % 4], zorder=7)
            except (IndexError, KeyError, ValueError):
                pass
        # Existing M3: visualize curved tracks instead of lines, if supplied.
        for i, path in enumerate(m.get('paths', [])):
            xy = np.asarray(path, dtype=float)
            if xy.ndim == 2 and len(xy):
                self.multi.plot(xy[:,0], xy[:,1], lw=2.5,
                                color=['#00eaff','#ffbc39','#ff62de','#a1ff45'][i % 4], zorder=7)
        self.title.set_text(f'Dataset {self.dataset} | Frame {frame} | Cluster {cid} | T {t["T"]*100:.0f} / M {m["M"]*100:.0f}')
        self.meta.set_text(f'{len(cluster)} pixels | T: {t["status_t"]} | M: {m.get("status", "unknown")} | Remaining {len(self.items)} | Saved {self.saved}')
        self.fig.canvas.draw_idle()

    def record(self, label_value):
        if self.current is None:
            return
        frame, cid, cluster = self.current
        t, m = self.t_result, self.m_result
        row = {'timestamp_utc':datetime.now(timezone.utc).isoformat(),
               'dataset':self.dataset,'source':str(self.source),
               'frame':frame,'cluster':cid,'label':label_value,
               'n_pixels':len(cluster),'M':m['M']*100,'T':t['T']*100,
               'm_version':M_VERSION,'t_version':VERSION_T,
               'm_status':m.get('status',''),'t_status':t['status_t'],
               't_endpoints':t['n_endpoints_t'],'t_junctions':t['n_junctions_t'],
               'notes':''}
        self.output.parent.mkdir(parents=True, exist_ok=True)
        new = not self.output.exists()
        with self.output.open('a', encoding='utf-8-sig' if new else 'utf-8', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=FIELDS)
            if new:
                writer.writeheader()
            writer.writerow(row)
        self.saved += 1
        print(f'Frame {frame} cluster {cid}: {label_value} | T={row["T"]:.0f} | M={row["M"]:.0f}', flush=True)
        self.next_cluster()

    def on_key(self, event):
        if event.key == '1': self.record('pileup')
        elif event.key == '2': self.record('not_pileup')
        elif event.key == '3': self.record('uncertain')
        elif event.key == ' ': self.next_cluster()
        elif event.key in ('q','escape'): plt.close(self.fig)


def main():
    parser = argparse.ArgumentParser(description='APEX-T T skeleton + existing M scoring, interactive labeling')
    parser.add_argument('--dataset', type=int, default=2)
    parser.add_argument('--file', type=Path)
    parser.add_argument('--min-pixels', type=int, default=16)
    parser.add_argument('--min-m', type=float, default=40,
                        help='Show cluster if M >= this threshold (0..100)')
    parser.add_argument('--min-t', type=float, default=40,
                        help='OR T >= this threshold (0..100)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Show at most N randomly chosen matching clusters')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not (0<=args.min_m<=100 and 0<=args.min_t<=100):
        parser.error('--min-m/--min-t must be between 0 and 100')
    if args.min_pixels < 1 or args.limit is not None and args.limit < 1:
        parser.error('--min-pixels and --limit must be positive')
    source = select_measurement(args.dataset, args.file)
    output = args.output or ROOT/'results'/'labels'/f'dataset_{args.dataset}_mt.csv'
    if not output.is_absolute():
        output = ROOT/output
    print(f'Measurement: {source}\nM version: {M_VERSION}, T version: {VERSION_T}', flush=True)
    items = read_candidates(source, args.min_pixels)
    done = existing_labels(output, source)
    print(f'Clusters >= {args.min_pixels} pixels: {len(items)} | already reviewed: {len(done)}', flush=True)
    candidates = score_candidates(source, items, ROOT/'results'/'cache'/f'mt_dataset_{args.dataset}.sqlite3', args.min_m, args.min_t, done)
    print(f'Matching T >= {args.min_t} OR M >= {args.min_m}: {len(candidates)}', flush=True)
    if args.limit is not None and len(candidates) > args.limit:
        candidates = random.Random(args.seed).sample(candidates, args.limit)
    if not candidates:
        print('No candidates. Try --min-t 0 --min-m 0 to inspect all clusters.')
        return
    print(f'Labels will be saved to: {output}', flush=True)
    Labeler(candidates, source, output, args.dataset, args.seed)
    plt.show()

if __name__ == '__main__':
    main()
