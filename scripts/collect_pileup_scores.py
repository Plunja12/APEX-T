"""APEX-T: index once, freeze a sampling batch, review and audit.

Run from the project root:
    python scripts/collect_pileup_scores.py collect --dataset 2
    python scripts/collect_pileup_scores.py report --dataset 2
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from pileup import score_collection as collection


def draw_pixels(ax, pixels, *, full=False, norm=None):
    """Draw every active pixel, including missing/zero signals; no signal cut."""
    import numpy as np
    from matplotlib.colors import Normalize, ListedColormap
    if not pixels:
        ax.text(.5, .5, 'No valid pixels for display', transform=ax.transAxes,
                ha='center', va='center')
        return
    xy = np.array([(p['x'], p['y']) for p in pixels], dtype=int)
    xmin, ymin = (0, 0) if full else xy.min(axis=0)
    xmax, ymax = (255, 255) if full else xy.max(axis=0)
    shape = (int(ymax-ymin+1), int(xmax-xmin+1))
    occupancy = np.zeros(shape, dtype=bool)
    values = np.full(shape, np.nan)
    # If separate input clusters share a pixel, the overview shows maximum
    # finite signal; original records stay separate in the database.
    for p in pixels:
        x, y = p['x']-xmin, p['y']-ymin
        occupancy[y, x] = True
        if p['value'] is not None:
            if not np.isfinite(values[y, x]) or p['value'] > values[y, x]:
                values[y, x] = p['value']
    extent = (xmin-.5, xmax+.5, ymin-.5, ymax+.5)
    ax.imshow(np.ma.masked_where(~occupancy, occupancy), origin='lower',
              interpolation='nearest', extent=extent, cmap=ListedColormap(['#a0a0a0']))
    finite = values[np.isfinite(values)]
    if norm is None:
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0., 1.)
        norm = Normalize(lo, hi if hi > lo else lo+1.)
    image = ax.imshow(np.ma.masked_invalid(values), origin='lower', cmap='viridis',
                      norm=norm, interpolation='nearest', extent=extent)
    pad = 0 if full else max(2, .08*max(shape))
    ax.set_xlim(xmin-.5-pad, xmax+.5+pad)
    ax.set_ylim(ymin-.5-pad, ymax+.5+pad)
    ax.set_aspect('equal')
    ax.set_facecolor('#f4f4f4')
    ax.set_xlabel('x [pixel]')
    ax.set_ylabel('y [pixel]')
    return image


class Labeler:
    def __init__(self, conn, batch, output):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, TextBox
        self.conn, self.batch, self.output = conn, batch, output
        self.items = [r for r in collection.queue_rows(conn, batch) if not r['label']]
        self.position, self.diagnostics, self.current = 0, False, None
        self.signal_view = False
        self.export_error = ''
        self.frame_cache = {}
        self.fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        self.axes = list(axes.flat)
        self.fig.subplots_adjust(left=.06, right=.96, bottom=.25, top=.87, hspace=.70, wspace=.20)
        self.title = self.fig.suptitle('APEX-T manual review', fontsize=14)
        self.status = self.fig.text(.06, .19, '', fontsize=9, va='top')
        self.buttons = []
        for i, (text, action, color) in enumerate([
            ('1  Pile-up', lambda _: self.record('pileup'), '#d8ecdb'),
            ('2  Neni pile-up', lambda _: self.record('not_pileup'), '#eadfdc'),
            ('3  Nejiste', lambda _: self.record('uncertain'), '#ede6c9'),
            ('D  Geometrie', self.toggle, '#dce3ec'),
            ('E  Signal', self.toggle_signal, '#e3ddf0')]):
            button = Button(self.fig.add_axes([.06+i*.18, .035, .17, .045]), text, color=color)
            button.on_clicked(action)
            self.buttons.append(button)
        self.notes = TextBox(self.fig.add_axes([.12, .10, .78, .028]), 'Notes: ')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.show_current()

    def show_current(self):
        import numpy as np
        from matplotlib.patches import Rectangle
        from matplotlib.collections import LineCollection
        for ax in self.axes:
            ax.clear()
            ax.set_axis_on()
        self.notes.set_val('')
        if self.position >= len(self.items):
            self.current = None
            self.title.set_text('Vybrany vzorek je hotovy; Q zavre okno')
            self.status.set_text('Stitky ulozeny. Mezernik preskakuje jen v teto relaci; preskocene zustavaji bez stitku.')
            for ax in self.axes:
                ax.set_axis_off()
            self.fig.canvas.draw_idle()
            return
        self.current = self.items[self.position]
        self.diagnostics = False  # Reset visibility, retain permanent exposure flag.
        self.signal_view = False
        row = self.conn.execute('SELECT * FROM collection_records WHERE run_id=? AND frame=? AND cluster=?',
                                (self.batch['run_id'], self.current['frame'], self.current['cluster'])).fetchone()
        self.pixels = json.loads(row['pixels_json'])
        self.features = json.loads(row['features_json'])
        self.detail = json.loads(row['diagnostics_json'])
        frame = self.current['frame']
        if frame not in self.frame_cache:
            pixels = []
            for r in self.conn.execute('SELECT pixels_json FROM collection_records WHERE run_id=? AND frame=?',
                                       (self.batch['run_id'], frame)):
                pixels.extend(json.loads(r[0]))
            # Bound memory without rereading measurement or recalculating scores.
            if len(self.frame_cache) >= 4:
                self.frame_cache.pop(next(iter(self.frame_cache)))
            self.frame_cache[frame] = pixels
        full, raw = self.axes[:2]
        draw_pixels(full, self.frame_cache[frame], full=True)
        draw_pixels(raw, self.pixels)
        full.set_title(f'Frame {frame} | signal (maximum for shared pixels)', fontsize=10)
        raw.set_title(f'Original cluster {self.current["cluster"]} | N={row["n_pixels"]}', fontsize=10)
        if self.pixels:
            xy = np.array([(p['x'], p['y']) for p in self.pixels])
            lo, hi = xy.min(axis=0), xy.max(axis=0)
            full.add_patch(Rectangle(lo-.5, *(hi-lo+1), fill=False, edgecolor='red', lw=1.4))
        self.title.set_text(f'APEX-T | dataset {self.batch["dataset"]} | '
                            f'{self.position+1}/{len(self.items)} | frame {frame} / cluster {self.current["cluster"]}')
        self.render_diagnostics()

    def render_diagnostics(self):
        import numpy as np
        from matplotlib.collections import LineCollection
        topo, multi = self.axes[2:]
        for ax in (topo, multi):
            ax.clear()
            ax.set_axis_on()
        if not self.diagnostics:
            for ax, text in [(topo, 'Nejprve posud originalni pixely.'),
                             (multi, 'D: geometrie | E: signal a jeho rozlozeni\nZobrazeni se zaznamena pro vyhodnoceni.')]:
                ax.set_axis_off()
                ax.text(.5, .5, text, ha='center', va='center', transform=ax.transAxes, fontsize=11)
        else:
            for ax in (topo, multi):
                draw_pixels(ax, self.pixels)
            t, m = self.detail.get('T', {}), self.detail.get('M', {})
            sk = np.asarray(t.get('skeleton_t', []), dtype=int).reshape(-1, 2)
            if len(sk):
                low = sk.min(axis=0)
                local = sk-low
                mask = np.zeros((int(local[:, 1].max()+1), int(local[:, 0].max()+1)), dtype=bool)
                mask[local[:, 1], local[:, 0]] = True
                graph = collection.topology._graph(mask)
                edges = [[(a[1]+low[0], a[0]+low[1]), (b[1]+low[0], b[0]+low[1])]
                         for a, ns in graph.items() for b in ns if a < b]
                topo.add_collection(LineCollection(edges, colors='#fff45a', linewidths=2))
            for name, marker, color in [('endpoints_t', 'o', '#00bfff'), ('junctions_t', 'x', '#ff4338')]:
                xy = np.asarray(t.get(name, []), dtype=float).reshape(-1, 2)
                if len(xy):
                    topo.scatter(xy[:, 0], xy[:, 1], marker=marker, color=color, s=40, zorder=5)
            points = np.array([(p['x'], p['y']) for p in self.pixels], dtype=float)
            for i, line in enumerate(m.get('lines', [])):
                indices = np.asarray(line.get('indices', []), dtype=int)
                if not len(indices):
                    continue
                center, direction = np.asarray(line['center']), np.asarray(line['direction'])
                projections = (points[indices]-center) @ direction
                segment = center + np.outer([projections.min(), projections.max()], direction)
                multi.plot(segment[:, 0], segment[:, 1], lw=2.4,
                           color=['#00cfff', '#ff9a32', '#ff65d7', '#8fd136'][i % 4])
            val = lambda k: '--' if self.features.get(k) is None else f'{self.features[k]:.1f}'
            topo.set_title(f'T1.1 = {val("T")} | {self.features["T_status"]}', fontsize=10)
            multi.set_title(f'M2 = {val("M")} | {self.features["M_status"]}\n'
                            'Barevne usecky jsou geometricke hypotezy', fontsize=10)
            if self.signal_view:
                multi.clear()
                values = np.array([p['value'] for p in self.pixels if p['value'] is not None],dtype=float)
                if values.size:
                    multi.hist(values,bins=min(24,max(1,int(np.sqrt(len(values))))),
                               color='#7155a6',edgecolor='white')
                    multi.axvline(np.mean(values),color='#e1771d',lw=1.5,label='mean')
                    multi.legend(loc='upper left',fontsize=8)
                unit = self.features.get('signal_unit','unknown')
                multi.set_xlabel(f'Pixel signal [{unit}]')
                multi.set_ylabel('Pixel count')
                multi.set_aspect('auto')
                multi.set_title(f'E = {val("E")} | {self.features.get("E_status","legacy: no E")}\n'
                                f'Reference: {self.features.get("e_reference_n",0)} clusters in size bin',fontsize=10)
                numbers = ['signal_sum','signal_mean','signal_max','signal_junction_ratio']
                text = '\n'.join(f'{key.replace("signal_","")}: '
                                  + ('--' if self.features.get(key) is None else f'{self.features[key]:.3g}')
                                  for key in numbers)
                multi.text(.98,.98,text,transform=multi.transAxes,ha='right',va='top',fontsize=8,
                           bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
        warning = self.features.get('coordinate_error', '')
        if not warning:
            unit = self.features.get('signal_unit','unknown')
            health = self.features.get('signal_status','legacy')
            warning = f'Signal [{unit}] | {health} | chybejici hodnoty: {self.features["signal_missing"]}.'
        extra = ''
        if self.diagnostics:
            g_text = '--' if self.features['G'] is None else f'{self.features["G"]:.1f}'
            e_text = '--' if self.features.get('E') is None else f'{self.features["E"]:.1f}'
            extra = f' G={g_text}, E={e_text} | skupina={self.current["stratum"]} | '
            extra += f'komponenty={self.features["n_components"]}'
        self.status.set_text(f'1/2/3 oznacit | Space preskocit | D geometrie | E signal | Q konec\n{warning}{extra}\n{self.export_error}')
        self.fig.canvas.draw_idle()

    def toggle(self, event=None):
        if self.current is None:
            return
        if self.signal_view:
            self.signal_view = False
            self.diagnostics = True
            self.render_diagnostics()
            return
        self.diagnostics = not self.diagnostics
        if self.diagnostics:
            collection.mark_diagnostics_seen(self.conn, self.batch, self.current)
        self.render_diagnostics()

    def toggle_signal(self, event=None):
        if self.current is None:
            return
        self.signal_view = not (self.diagnostics and self.signal_view)
        self.diagnostics = True
        collection.mark_diagnostics_seen(self.conn,self.batch,self.current)
        self.render_diagnostics()

    def record(self, label_value):
        if self.current is None:
            return
        if self.features.get('coordinate_error') and label_value != 'uncertain':
            self.export_error = 'Neplatne souradnice: pouzij Nejiste a zkontroluj vstupni data.'
            self.render_diagnostics()
            return
        try:
            collection.save_label(self.conn, self.batch, self.current, label_value, self.notes.text)
        except (sqlite3.Error, ValueError) as exc:
            self.export_error = f'Label NOT saved: {exc}'
            self.render_diagnostics()
            return
        try:
            collection.export_labels(self.conn, self.batch, self.output)
            self.export_error = ''
        except OSError as exc:
            self.export_error = f'Saved in database; CSV export failed: {exc}. Close Excel and run report.'
            print(self.export_error, flush=True)
        print(f'Frame {self.current["frame"]}, cluster {self.current["cluster"]}: {label_value}', flush=True)
        self.position += 1
        self.show_current()

    def on_key(self, event):
        import matplotlib.pyplot as plt
        # Never save a label while the reviewer types into the notes box.
        if event.inaxes == self.notes.ax or self.notes.capturekeystrokes:
            return
        if event.key in {'1', '2', '3'}:
            self.record({'1':'pileup', '2':'not_pileup', '3':'uncertain'}[event.key])
        elif event.key == 'd':
            self.toggle()
        elif event.key == 'e':
            self.toggle_signal()
        elif event.key == ' ' and self.current is not None:
            self.position += 1
            self.show_current()
        elif event.key in {'q', 'escape'}:
            plt.close(self.fig)


def import_legacy(conn, batch, csv_path, legacy_source=None):
    """Explicit source mapping; never guesses a file by basename."""
    path = Path(csv_path)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {'source', 'frame', 'cluster', 'label'}.issubset(rows[0]):
        raise ValueError('Legacy CSV needs source,frame,cluster,label columns')
    expected = legacy_source or batch['source_path']
    selected = [r for r in rows if r['source'].replace('\\', '/') == expected.replace('\\', '/')]
    if not selected:
        sources = sorted({r['source'] for r in rows})
        raise ValueError('No exact source match. Use --legacy-source with the exact old path. '
                         f'Sources in CSV: {sources}')
    valid = {(r['frame'], r['cluster']): r for r in collection.load_features(conn, batch['run_id'])}
    proposed = {}
    for row in selected:
        key = int(row['frame']), int(row['cluster'])
        if row['label'] not in collection.LABELS or key not in valid:
            raise ValueError(f'Invalid label or absent cluster in legacy CSV: {key}')
        if row.get('n_pixels') and int(row['n_pixels']) != valid[key]['n_pixels']:
            raise ValueError(f'Pixel count mismatch: {key}; verify the measurement mapping')
        if key in proposed and proposed[key]['label'] != row['label']:
            raise ValueError(f'Conflicting legacy labels at {key}; import cancelled')
        old = conn.execute('SELECT label FROM collection_labels WHERE source_sha=? AND frame=? AND cluster=?',
                           (batch['source_sha'], *key)).fetchone()
        if old and old['label'] != row['label']:
            raise ValueError(f'Existing label conflicts at {key}; import cancelled')
        proposed[key] = row
    # The whole import is one transaction; conflicts cannot cause a partial import.
    count = 0
    with conn:
        for key, row in proposed.items():
            cur = conn.execute('INSERT OR IGNORE INTO collection_labels VALUES (?,?,?,?,?,?,?,?)',
                               (batch['source_sha'], *key, row['label'], row.get('timestamp_utc') or collection.now(),
                                'legacy:' + str(path.resolve()), 1,
                                'Explicit legacy source mapping; visually scored provenance unknown. ' + row.get('notes', '')))
            count += cur.rowcount
    print(f'Imported {count} legacy labels; retained provenance; excluded from new blind control metrics.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', nargs='?', default='collect',
                        choices=['collect', 'index', 'sample', 'label', 'report', 'import'])
    parser.add_argument('--dataset', type=int, default=2)
    parser.add_argument('--file', type=Path)
    parser.add_argument('--db', type=Path, default=ROOT/'results'/'collection'/'scores.sqlite3')
    parser.add_argument('--output', type=Path, default=ROOT/'results'/'collection')
    parser.add_argument('--batch', help='Exact batch id, unique prefix, or explicit latest')
    parser.add_argument('--control', type=int, default=60)
    parser.add_argument('--t-samples', type=int, default=40)
    parser.add_argument('--m-samples', type=int, default=20)
    parser.add_argument('--g-samples', type=int, default=0)
    parser.add_argument('--e-samples', type=int, default=20,
                        help='Extra high-signal candidates with measured M/T below their thresholds')
    parser.add_argument('--size-samples', type=int, default=0)
    parser.add_argument('--threshold-t', type=float, default=40)
    parser.add_argument('--threshold-m', type=float, default=40)
    parser.add_argument('--threshold-g', type=float, default=35)
    parser.add_argument('--threshold-e', type=float, default=50)
    parser.add_argument('--signal-unit', choices=sorted(collection.signal_features.VALID_UNITS),default='unknown',
                        help='Metadata assertion for already stored values; performs NO calibration')
    parser.add_argument('--signal-ceiling', type=float,
                        help='Known instrument/export clipping value; flagged clusters do not enter E ranking')
    parser.add_argument('--e-min-reference',type=int,default=30,
                        help='Minimum number of usable clusters per log2(N) reference bin')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--csv', type=Path, help='Legacy CSV to import; originals remain unchanged')
    parser.add_argument('--legacy-source', help='Exact old source path in the legacy CSV, explicitly mapped to this measurement')
    args = parser.parse_args(argv)
    if any(getattr(args, k) < 0 for k in ['control', 't_samples', 'm_samples', 'g_samples', 'e_samples', 'size_samples']):
        parser.error('Sample counts must be >= 0')
    if any(not 0 <= getattr(args, 'threshold_' + k) <= 100 for k in 'mtge'):
        parser.error('Thresholds must be in 0..100')
    if args.e_min_reference < 2:
        parser.error('--e-min-reference must be >= 2')
    if args.signal_ceiling is not None:
        import math
        if not math.isfinite(args.signal_ceiling) or args.signal_ceiling <= 0:
            parser.error('--signal-ceiling must be finite and > 0')
    config = {k: getattr(args, k) for k in ['control', 't_samples', 'm_samples', 'g_samples',
                                          'e_samples', 'size_samples', 'threshold_t', 'threshold_m',
                                          'threshold_g', 'threshold_e', 'seed']}
    conn = collection.open_db(args.db)
    try:
        if args.command in {'collect', 'index', 'sample'}:
            try:
                from pileup.io.clog_reader import iter_frames
            except ImportError as exc:
                raise RuntimeError('Copy the add-on into the existing APEX-T project containing '
                                   'src/pileup/io/clog_reader.py') from exc
            source = collection.select_measurement(ROOT, args.dataset, args.file)
            run_id = collection.index_source(conn, source, args.dataset, iter_frames,
                                              signal_unit=args.signal_unit,signal_ceiling=args.signal_ceiling,
                                              e_min_reference=args.e_min_reference)
            if args.command == 'index':
                return
            if args.batch:
                if args.command == 'sample':
                    parser.error('sample creates a batch from the sample settings; use label --batch to resume')
                batch = collection.resolve_batch(conn, args.dataset, args.batch)
                if batch['run_id'] != run_id:
                    raise ValueError('Selected batch belongs to different data/method versions; use label --batch '
                                     'to review its stored records, or omit --batch for a new collection')
            else:
                batch_id = collection.create_batch(conn, run_id, config)
                batch = collection.resolve_batch(conn, args.dataset, batch_id)
        else:
            batch = collection.resolve_batch(conn, args.dataset, args.batch)
        if args.command == 'import':
            if args.csv is None:
                parser.error('import requires --csv')
            import_legacy(conn, batch, args.csv, args.legacy_source)
        paths = collection.export_batch(conn, batch, args.output)
        print('Export:\n' + '\n'.join(map(str, paths)), flush=True)
        if args.command in {'collect', 'label'}:
            import matplotlib.pyplot as plt
            app = Labeler(conn, batch, args.output)
            plt.show()
            collection.export_batch(conn, batch, args.output)
        report = collection.build_report(conn, batch)
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    finally:
        conn.close()


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
