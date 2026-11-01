"""Reproducible collection of spatial pile-up features and manual labels.

M/T/G are frozen heuristics, not calibrated probabilities. The random control
is sampled from ALL nonempty clusters before candidates are selected. Its
membership is fixed; uncertain, failed and skipped examples remain visible
in the report. Raw measurements and previous label files are never written.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import tempfile
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import scipy
from scipy.ndimage import label as component_label

from .collection_baseline import multiple_tracks, topology, geometry
from . import signal_features

VERSION = 'collection-1.1-signal'
LEGACY_COLLECTOR_SHA = '5df202fab55013ac5f2bf2e0c6392a7f10443c0d2ac327cfa1e0a3b5d399781c'
LABELS = {'pileup', 'not_pileup', 'uncertain'}
SCORE_KEYS = ['M', 'T', 'G', 'S_MT']


def now():
    return datetime.now(timezone.utc).isoformat()


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dumps(value):
    return json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(',', ':'))


def hashed(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def rules_for(reader, *, signal_unit='unknown', signal_ceiling=None, e_min_reference=30):
    """Bind the cache to actual code, defaults and numerical dependencies."""
    modules = dict(M=multiple_tracks, T=topology, G=geometry)
    return dict(collection=VERSION,
                collector_sha=digest_file(__file__),
                reader_sha=digest_file(inspect.getsourcefile(reader)),
                methods={k: dict(sha=digest_file(m.__file__),
                                defaults=str(inspect.signature(getattr(m, 'score_' + k.lower()))))
                         for k, m in modules.items()},
                versions=dict(M=multiple_tracks.VERSION, T=topology.VERSION_T,
                              G=geometry.VERSION_G, numpy=np.__version__, scipy=scipy.__version__),
                signal=dict(version=signal_features.VERSION_SIGNAL,
                            sha=digest_file(signal_features.__file__), unit=signal_unit,
                            ceiling=signal_ceiling, min_reference=e_min_reference,
                            reference='same source; log2(N) bins; complete positive non-clipped signals',
                            E='100*clip((midrank(mean_signal)-0.90)/0.10,0,1)'),
                score_units='0..100; not probability',
                ranking='S_MT=max(available M,T); G and size excluded')


def open_db(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys=ON')
    conn.executescript('''
      CREATE TABLE IF NOT EXISTS collection_runs (
        run_id TEXT PRIMARY KEY, dataset INTEGER NOT NULL,
        source_sha TEXT NOT NULL, source_path TEXT NOT NULL,
        rules_json TEXT NOT NULL, complete INTEGER NOT NULL DEFAULT 0,
        n_frames INTEGER NOT NULL DEFAULT 0, empty_clusters INTEGER NOT NULL DEFAULT 0,
        created_utc TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS collection_records (
        run_id TEXT NOT NULL REFERENCES collection_runs(run_id),
        frame INTEGER NOT NULL, cluster INTEGER NOT NULL, n_pixels INTEGER NOT NULL,
        pixels_json TEXT NOT NULL, features_json TEXT NOT NULL,
        diagnostics_json TEXT NOT NULL,
        PRIMARY KEY(run_id,frame,cluster));
      CREATE TABLE IF NOT EXISTS collection_batches (
        batch_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES collection_runs(run_id),
        config_json TEXT NOT NULL, created_utc TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS collection_queue (
        batch_id TEXT NOT NULL REFERENCES collection_batches(batch_id),
        position INTEGER NOT NULL, frame INTEGER NOT NULL, cluster INTEGER NOT NULL,
        stratum TEXT NOT NULL, diagnostics_seen INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(batch_id,frame,cluster), UNIQUE(batch_id,position));
      CREATE TABLE IF NOT EXISTS collection_labels (
        source_sha TEXT NOT NULL, frame INTEGER NOT NULL, cluster INTEGER NOT NULL,
        label TEXT NOT NULL CHECK(label IN ('pileup','not_pileup','uncertain')),
        timestamp_utc TEXT NOT NULL, origin_batch TEXT NOT NULL,
        diagnostics_seen INTEGER NOT NULL, notes TEXT NOT NULL,
        PRIMARY KEY(source_sha,frame,cluster));
      CREATE TABLE IF NOT EXISTS collection_exposure (
        source_sha TEXT NOT NULL, frame INTEGER NOT NULL, cluster INTEGER NOT NULL,
        PRIMARY KEY(source_sha,frame,cluster));
    ''')
    return conn


def select_measurement(root, dataset, custom=None):
    if custom:
        source = Path(custom).expanduser()
        source = source if source.is_absolute() else Path(root) / source
        if not source.is_file():
            raise FileNotFoundError(source)
        return source.resolve()
    raw = Path(root) / 'data' / 'raw'
    matches = [p for p in raw.rglob('*') if p.is_file()
               and p.relative_to(raw).parts[0].startswith(f'{dataset:02d} ')
               and p.suffix.lower() not in {'.zip', '.png', '.jpg', '.jpeg'}
               and (p.suffix.lower() == '.clog'
                    or p.name.lower().startswith(('tot_', 'itot', 'cnt_')))]
    if len(matches) != 1:
        raise ValueError(f'Expected one measurement for dataset {dataset}; found {len(matches)}. '
                         'Specify --file.\n' + '\n'.join(map(str, matches)))
    return matches[0].resolve()


def prepare_pixels(cluster):
    """Retain x/y and uncalibrated signal for display; discard extra/time."""
    pixels, seen = [], set()
    for p in cluster:
        x, y = float(p['x']), float(p['y'])
        if not (math.isfinite(x) and math.isfinite(y) and x.is_integer() and y.is_integer()
                and 0 <= x < 256 and 0 <= y < 256):
            raise ValueError('Coordinates must be integers in detector range 0..255')
        xy = (int(x), int(y))
        if xy in seen:
            raise ValueError('Duplicate pixel coordinates; no silent deduplication')
        seen.add(xy)
        try:
            value = float(p.get('value', float('nan')))
        except (ValueError, TypeError):
            value = float('nan')
        pixels.append(dict(x=xy[0], y=xy[1], value=value if math.isfinite(value) else None))
    return pixels


def calculate(cluster):
    """Calculate each method independently; missing/error is never score zero."""
    features = dict(n_pixels=len(cluster), M=None, T=None, G=None, S_MT=None, S_MT_status='unavailable',
                    M_status='not_applicable', T_status='not_applicable', G_status='not_applicable',
                    n_components=None, coordinate_error='', signal_missing=0,
                    m_geometry='', m_lines=None, m_coverage=None, m_angle_deg=None,
                    t_endpoints=None, t_junctions=None, g_width=None, g_concavity=None,
                    g_solidity=None, g_median_width=None, g_p90_width=None)
    diagnostics = {}
    try:
        pixels = prepare_pixels(cluster)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        features['coordinate_error'] = f'{type(exc).__name__}: {exc}'
        for key in 'MTG':
            features[key + '_status'] = 'invalid_coordinates'
        return enrich_signal([], features, diagnostics)
    features['signal_missing'] = sum(p['value'] is None for p in pixels)
    if pixels:
        pts = np.array([(p['x'], p['y']) for p in pixels])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        features.update(bbox_width=int(hi[0]-lo[0]+1), bbox_height=int(hi[1]-lo[1]+1))
        local = pts - lo + 1
        mask = np.zeros((int(hi[1]-lo[1]+3), int(hi[0]-lo[0]+3)), dtype=bool)
        mask[local[:, 1], local[:, 0]] = True
        features['n_components'] = int(component_label(mask, structure=np.ones((3, 3)))[1])
    # Only x/y reach the geometric scorers; arbitrary value/extra cannot affect scores.
    spatial = [dict(x=p['x'], y=p['y']) for p in pixels]
    for key, fn, minimum in [('T', topology.score_t, 8),
                             ('M', multiple_tracks.score_m, 16),
                             ('G', geometry.score_g, 8)]:
        if len(pixels) < minimum:
            continue
        try:
            if key == 'G':
                if 'T' not in diagnostics:
                    raise ValueError('Skeleton T unavailable; G width cannot be evaluated')
                result = fn(spatial, topology=diagnostics['T'])
            else:
                result = fn(spatial)
            score = float(result[key])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f'{key} returned invalid score {score}')
            features[key] = score * 100
            features[key + '_status'] = result.get('status' if key == 'M' else 'status_' + key.lower(), '')
            diagnostics[key] = result
        except Exception as exc:
            features[key + '_status'] = f'ERROR {type(exc).__name__}: {exc}'
    m, t, g = (diagnostics.get(k, {}) for k in 'MTG')
    features.update(m_geometry=m.get('geometry', ''), m_lines=m.get('n_tracks'),
                    m_coverage=m.get('coverage'), m_angle_deg=m.get('angle_deg'),
                    t_endpoints=t.get('n_endpoints_t'), t_junctions=t.get('n_junctions_t'),
                    g_width=g.get('width_component'), g_concavity=g.get('concavity_component'),
                    g_solidity=g.get('solidity'), g_median_width=g.get('median_width'),
                    g_p90_width=g.get('p90_width'))
    available = [features[k] for k in 'MT' if features[k] is not None]
    features['S_MT_status'] = 'complete' if len(available) == 2 else 'partial' if available else 'unavailable'
    if available:
        features['S_MT'] = max(available)
    return enrich_signal(pixels, features, diagnostics)


def enrich_signal(pixels, features, diagnostics, *, unit='unknown', ceiling=None):
    """Append signal evidence without modifying any geometry score."""
    features.update(signal_features.describe_signal(pixels, topology=diagnostics.get('T'),
                                                     unit=unit, ceiling=ceiling))
    if features.get('coordinate_error'):
        features.update(signal_status='invalid_coordinates', E_status='invalid_coordinates')
    return pixels, json_safe(features), json_safe(diagnostics)


def rank_run_signal(conn, run_id, *, min_reference=30):
    records = load_features(conn, run_id)
    ranked = signal_features.rank_signal(records, min_reference=min_reference)
    with conn:
        conn.executemany('UPDATE collection_records SET features_json=? '
                         'WHERE run_id=? AND frame=? AND cluster=?',
                         [(dumps({k:v for k,v in row.items() if k not in {'frame','cluster'}}),
                           run_id, row['frame'], row['cluster']) for row in ranked])
    valid = sum(row.get('E') is not None for row in ranked)
    print(f'Signal reference ready: E available for {valid}/{len(ranked)} clusters', flush=True)


def compatible_geometry_run(conn, source_sha, dataset, rules):
    """Reuse only known collector schemas with identical geometry/reader/runtime.

    Signal settings may change. A changed M/T/G implementation, parser or
    numerical dependency MUST force geometric recalculation.
    """
    for row in conn.execute('SELECT * FROM collection_runs WHERE source_sha=? AND dataset=? '
                             'AND complete=1 ORDER BY created_utc DESC', (source_sha,dataset)):
        old = json.loads(row['rules_json'])
        if old.get('collection') not in {'collection-1.0', VERSION}:
            continue
        approved = LEGACY_COLLECTOR_SHA if old['collection'] == 'collection-1.0' else rules['collector_sha']
        if old.get('collector_sha') != approved:
            continue
        if all(old.get(key) == rules.get(key) for key in ['reader_sha','methods','versions']):
            return row
    return None


def copy_geometry_with_signal(conn, old_run, run_id, *, unit, ceiling):
    """Enrich cached pixels in chunks; interruption never marks the new run complete."""
    rows = conn.execute('SELECT * FROM collection_records WHERE run_id=? ORDER BY frame,cluster',
                         (old_run['run_id'],)).fetchall()
    for start in range(0,len(rows),100):
        with conn:
            for row in rows[start:start+100]:
                pixels = json.loads(row['pixels_json'])
                _, features, diagnostics = enrich_signal(pixels,json.loads(row['features_json']),
                                                          json.loads(row['diagnostics_json']),unit=unit,ceiling=ceiling)
                conn.execute('INSERT OR REPLACE INTO collection_records VALUES (?,?,?,?,?,?,?)',
                             (run_id,row['frame'],row['cluster'],row['n_pixels'],row['pixels_json'],
                              dumps(features),dumps(diagnostics)))
        if start % 1000 == 0:
            print(f'Adding signal to cached geometry: {min(start+100,len(rows))}/{len(rows)}',flush=True)


def index_source(conn, source, dataset, reader, *, signal_unit='unknown', signal_ceiling=None, e_min_reference=30):
    source = Path(source).resolve()
    before = source.stat()
    source_sha = digest_file(source)
    if signal_unit not in signal_features.VALID_UNITS or e_min_reference < 2:
        raise ValueError('Invalid signal unit or reference size')
    if signal_ceiling is not None and (not math.isfinite(signal_ceiling) or signal_ceiling <= 0):
        raise ValueError('Signal ceiling must be finite and > 0')
    rules = rules_for(reader, signal_unit=signal_unit, signal_ceiling=signal_ceiling,
                       e_min_reference=e_min_reference)
    run_id = hashed(dict(source_sha=source_sha, dataset=dataset, rules=rules))
    row = conn.execute('SELECT * FROM collection_runs WHERE run_id=?', (run_id,)).fetchone()
    if row and row['complete']:
        with conn:
            conn.execute('UPDATE collection_runs SET source_path=? WHERE run_id=?', (str(source), run_id))
        print(f'Cached complete index: {run_id[:12]}', flush=True)
        return run_id
    with conn:
        conn.execute('INSERT OR IGNORE INTO collection_runs '
                     '(run_id,dataset,source_sha,source_path,rules_json,created_utc) VALUES (?,?,?,?,?,?)',
                     (run_id, dataset, source_sha, str(source), dumps(rules), now()))
    reusable = compatible_geometry_run(conn, source_sha, dataset, rules)
    if reusable:
        print(f'Reusing geometry from {reusable["run_id"][:12]}; M/T/G will not be recalculated.',flush=True)
        copy_geometry_with_signal(conn,reusable,run_id,unit=signal_unit,ceiling=signal_ceiling)
        rank_run_signal(conn,run_id,min_reference=e_min_reference)
        after = source.stat()
        if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
            raise RuntimeError('Measurement changed during indexing; index remains incomplete')
        with conn:
            conn.execute('UPDATE collection_runs SET complete=1,n_frames=?,empty_clusters=? WHERE run_id=?',
                         (reusable['n_frames'],reusable['empty_clusters'],run_id))
        return run_id
    existing = {(r['frame'], r['cluster']) for r in conn.execute(
        'SELECT frame,cluster FROM collection_records WHERE run_id=?', (run_id,))}
    seen_frames, n, empty = set(), 0, 0
    for frame, clusters in reader(source):
        if isinstance(frame, bool) or int(frame) != frame:
            raise ValueError(f'Non-integer frame id: {frame!r}')
        frame = int(frame)
        if frame in seen_frames:
            raise ValueError(f'Duplicate frame id {frame}; reader/data must be checked')
        seen_frames.add(frame)
        with conn:
            for cid, cluster in enumerate(clusters):
                if not cluster:
                    empty += 1
                    continue
                n += 1
                if (frame, cid) in existing:
                    continue
                pixels, features, diagnostics = calculate(cluster)
                if signal_unit != 'unknown' or signal_ceiling is not None:
                    pixels,features,diagnostics = enrich_signal(pixels,features,diagnostics,
                                                                unit=signal_unit,ceiling=signal_ceiling)
                conn.execute('INSERT INTO collection_records VALUES (?,?,?,?,?,?,?)',
                             (run_id, frame, cid, len(cluster), dumps(pixels),
                              dumps(features), dumps(diagnostics)))
                if n % 250 == 0:
                    print(f'Calculating: {n} clusters; frame {frame}', flush=True)
        print(f'Indexed frame {frame}; nonempty clusters {n}; resumed {len(existing)}', flush=True)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Measurement changed during indexing; index remains incomplete')
    if not n:
        raise ValueError('No nonempty clusters were returned by the reader')
    rank_run_signal(conn,run_id,min_reference=e_min_reference)
    with conn:
        conn.execute('UPDATE collection_runs SET complete=1,n_frames=?,empty_clusters=? WHERE run_id=?',
                     (len(seen_frames), empty, run_id))
    print(f'Complete: {n} nonempty clusters, {empty} empty clusters; run {run_id[:12]}', flush=True)
    return run_id


def load_features(conn, run_id):
    return [dict(frame=r['frame'], cluster=r['cluster'], **json.loads(r['features_json']))
            for r in conn.execute('SELECT frame,cluster,features_json FROM collection_records '
                                  'WHERE run_id=? ORDER BY frame,cluster', (run_id,))]


def size_percentiles(records):
    """Dataset-specific empirical midrank, not a pile-up anomaly score A."""
    counts = Counter(r['n_pixels'] for r in records)
    below, percentiles = 0, {}
    for n in sorted(counts):
        percentiles[n] = 100 * (below + .5 * counts[n]) / max(1, len(records))
        below += counts[n]
    return percentiles


def sample_records(records, already_labelled, config):
    """Control first from all records, including small/failed/previously labelled.

    Previously labelled controls keep provenance and are excluded from a new
    blind audit. Removing them before sampling would change the population.
    """
    rng = random.Random(config['seed'])
    all_rows = sorted(records, key=lambda r: (r['frame'], r['cluster']))
    key = lambda r: (r['frame'], r['cluster'])
    controls = rng.sample(all_rows, min(config['control'], len(all_rows)))
    chosen = [(r, 'control_random') for r in controls]
    used = {key(r) for r in controls} | set(already_labelled)

    def add(name, count, predicate, largest=False):
        pool = [r for r in all_rows if key(r) not in used and predicate(r)]
        if largest:
            pool.sort(key=lambda r: (-r['n_pixels'], key(r)))
        else:
            rng.shuffle(pool)
        for r in pool[:count]:
            chosen.append((r, name))
            used.add(key(r))

    def high(r, score):
        return r.get(score) is not None and r[score] >= config['threshold_' + score.lower()]

    add('T_high', config['t_samples'], lambda r: high(r, 'T'))
    def low(r, score):
        return r.get(score) is not None and r[score] < config['threshold_' + score.lower()]

    add('M_without_T', config['m_samples'], lambda r: high(r, 'M') and low(r, 'T'))
    # Test E on a distinct, explicitly geometry-negative group. Missing or
    # failed M/T are not silently interpreted as low scores.
    add('E_without_MT', config.get('e_samples',0),
        lambda r: r.get('E') is not None and r['E'] >= config.get('threshold_e',50)
        and low(r,'M') and low(r,'T'))
    add('G_without_MT', config['g_samples'], lambda r: high(r, 'G') and low(r, 'T') and low(r, 'M'))
    add('size_large', config['size_samples'], lambda r: not r.get('coordinate_error'), largest=True)
    rng.shuffle(chosen)
    return [(r['frame'], r['cluster'], stratum) for r, stratum in chosen]


def create_batch(conn, run_id, config):
    run = conn.execute('SELECT * FROM collection_runs WHERE run_id=?', (run_id,)).fetchone()
    if not run or not run['complete']:
        raise ValueError('Sampling requires a complete index')
    batch_id = hashed(dict(run_id=run_id, config=config))
    if conn.execute('SELECT 1 FROM collection_batches WHERE batch_id=?', (batch_id,)).fetchone():
        print(f'Resume frozen batch {batch_id[:12]}', flush=True)
        return batch_id
    done = {(r['frame'], r['cluster']) for r in conn.execute(
        'SELECT frame,cluster FROM collection_labels WHERE source_sha=?', (run['source_sha'],))}
    chosen = sample_records(load_features(conn, run_id), done, config)
    with conn:
        conn.execute('INSERT INTO collection_batches VALUES (?,?,?,?)', (batch_id, run_id, dumps(config), now()))
        conn.executemany('INSERT INTO collection_queue '
                         '(batch_id,position,frame,cluster,stratum) VALUES (?,?,?,?,?)',
                         [(batch_id, i, f, c, s) for i, (f, c, s) in enumerate(chosen)])
    print(f'Frozen batch {batch_id[:12]}: {dict(Counter(s for _, _, s in chosen))}', flush=True)
    return batch_id


def resolve_batch(conn, dataset, prefix=None):
    rows = conn.execute('SELECT b.*, r.source_sha, r.source_path, r.dataset FROM collection_batches b '
                         'JOIN collection_runs r ON b.run_id=r.run_id '
                         'WHERE r.dataset=? ORDER BY b.created_utc DESC', (dataset,)).fetchall()
    if prefix == 'latest' and rows:
        print(f'Explicit latest batch: {rows[0]["batch_id"][:12]}',flush=True)
        return dict(rows[0])
    matches = [r for r in rows if not prefix or r['batch_id'].startswith(prefix)]
    if len(matches) != 1:
        ids = ', '.join(r['batch_id'][:12] for r in rows) or '(none)'
        raise ValueError(f'Choose one batch using --batch. Available for dataset {dataset}: {ids}')
    return dict(matches[0])


def queue_rows(conn, batch):
    return [dict(r) for r in conn.execute('''
       SELECT q.*, l.label, l.origin_batch, l.diagnostics_seen AS label_diagnostics_seen,
              l.notes, l.timestamp_utc
       FROM collection_queue q LEFT JOIN collection_labels l
       ON l.source_sha=? AND l.frame=q.frame AND l.cluster=q.cluster
       WHERE q.batch_id=? ORDER BY q.position
    ''', (batch['source_sha'], batch['batch_id']))]


def mark_diagnostics_seen(conn, batch, item):
    with conn:
        conn.execute('UPDATE collection_queue SET diagnostics_seen=1 WHERE batch_id=? AND frame=? AND cluster=?',
                     (batch['batch_id'], item['frame'], item['cluster']))
        conn.execute('INSERT OR IGNORE INTO collection_exposure VALUES (?,?,?)',
                     (batch['source_sha'], item['frame'], item['cluster']))


def save_label(conn, batch, item, label_value, notes=''):
    if label_value not in LABELS:
        raise ValueError(label_value)
    stored = conn.execute('SELECT * FROM collection_queue WHERE batch_id=? AND frame=? AND cluster=?',
                          (batch['batch_id'], item['frame'], item['cluster'])).fetchone()
    if not stored:
        raise ValueError('Cluster does not belong to this frozen batch')
    exposed = bool(conn.execute('SELECT 1 FROM collection_exposure WHERE source_sha=? AND frame=? AND cluster=?',
                                (batch['source_sha'], item['frame'], item['cluster'])).fetchone())
    with conn:
        conn.execute('INSERT INTO collection_labels VALUES (?,?,?,?,?,?,?,?)',
                     (batch['source_sha'], item['frame'], item['cluster'], label_value,
                      now(), batch['batch_id'], int(exposed or stored['diagnostics_seen']), notes))


def wilson(success, total):
    if not total:
        return None
    p, z = success / total, 1.96
    den = 1 + z*z/total
    center = (p + z*z/(2*total)) / den
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / den
    return [max(0., center-half), min(1., center+half)]


def screening_metrics(records, score, threshold):
    """Metrics conditional on binary visual labels; missing scores abstain.

    Screening recall includes positive abstentions as not selected; they
    remain separate from negative numeric scores and from confusion counts.
    """
    counts = Counter(tp=0, fp=0, fn=0, tn=0, abstain=0, positive_abstain=0,
                     uncertain=0, unlabelled=0)
    for row in records:
        label_value = row.get('label')
        if label_value not in {'pileup', 'not_pileup'}:
            counts['uncertain' if label_value == 'uncertain' else 'unlabelled'] += 1
            continue
        positive = label_value == 'pileup'
        value = row.get(score)
        if value is None:
            counts['abstain'] += 1
            counts['positive_abstain'] += int(positive)
            continue
        selected = value >= threshold
        counts['tp' if selected and positive else 'fp' if selected else 'fn' if positive else 'tn'] += 1
    predicted = counts['tp'] + counts['fp']
    positives = counts['tp'] + counts['fn'] + counts['positive_abstain']
    return dict(counts, threshold=threshold,
                precision=counts['tp']/predicted if predicted else None,
                screening_recall=counts['tp']/positives if positives else None,
                precision_wilson95=wilson(counts['tp'], predicted),
                recall_wilson95=wilson(counts['tp'], positives))


def build_report(conn, batch):
    config = json.loads(batch['config_json'])
    features = {(r['frame'], r['cluster']): r for r in load_features(conn, batch['run_id'])}
    rows = [dict(features[r['frame'], r['cluster']], **r) for r in queue_rows(conn, batch)]
    control = [r for r in rows if r['stratum'] == 'control_random']
    blind = [r for r in control if r['origin_batch'] == batch['batch_id']
             and not r['label_diagnostics_seen']]
    binary_blind = sum(r['label'] in {'pileup', 'not_pileup'} for r in blind)
    by_group = {}
    for name in sorted({r['stratum'] for r in rows}):
        group = [r for r in rows if r['stratum'] == name]
        c = Counter(r['label'] or 'unlabelled' for r in group)
        by_group[name] = dict(selected=len(group), **c)
    thresholds = {k: config['threshold_' + k.lower()] for k in 'MTG'}
    has_e = any('E' in row for row in features.values())
    if has_e:
        thresholds['E'] = config.get('threshold_e',50)
    # The union uses the separately fixed M and T thresholds (not max with
    # a common threshold when the two thresholds differ).
    for r in blind:
        valid = [r[k] >= thresholds[k] for k in 'MT' if r[k] is not None]
        r['MT_union'] = 100. if any(valid) else 0. if len(valid) == 2 else None
    metrics = {k: screening_metrics(blind, k, thresholds[k]) for k in thresholds}
    metrics['MT_union'] = screening_metrics(blind, 'MT_union', 50)
    if has_e:
        for row in blind:
            available = [row[k] >= thresholds[k] for k in 'MTE' if row.get(k) is not None]
            row['MTE_union'] = 100. if any(available) else 0. if len(available) == 3 else None
        metrics['MTE_union'] = screening_metrics(blind,'MTE_union',50)
    # Incremental captures are measured on the same independent control.
    incremental = {k: sum(r['label'] == 'pileup' and r[k] is not None
                          and r[k] >= thresholds[k]
                          and r['T'] is not None and r['T'] < thresholds['T']
                          for r in blind) for k in 'MG'}
    return dict(batch_id=batch['batch_id'], run_id=batch['run_id'], dataset=batch['dataset'],
                config=config, groups=by_group,
                control=dict(selected=len(control), labelled=sum(bool(r['label']) for r in control),
                             blind_current_batch=len(blind), binary_blind=binary_blind,
                             uncertain_blind=sum(r['label'] == 'uncertain' for r in blind),
                             previously_labelled=sum(bool(r['label']) and r['origin_batch'] != batch['batch_id']
                                                      for r in control)),
                audit_fully_binary_and_blind=bool(control) and binary_blind == len(control),
                metrics_on_blind_control=metrics,
                additional_positive_captures_beyond_T=incremental,
                additional_E_captures_beyond_MT=sum(r['label'] == 'pileup' and r.get('E') is not None
                                                   and r['E'] >= thresholds.get('E',50)
                                                   and r['T'] is not None and r['T'] < thresholds['T']
                                                   and r['M'] is not None and r['M'] < thresholds['M']
                                                   for r in blind),
                signal_summary=dict(
                    E_available=sum(r.get('E') is not None for r in features.values()),
                    status_counts=dict(Counter(r.get('signal_status','legacy_unavailable') for r in features.values())),
                    E_status_counts=dict(Counter(r.get('E_status','legacy_unavailable') for r in features.values()))),
                additional_G_captures_beyond_MT=sum(r['label'] == 'pileup' and r['G'] is not None
                                                   and r['G'] >= thresholds['G']
                                                   and r['T'] is not None and r['T'] < thresholds['T']
                                                   and r['M'] is not None and r['M'] < thresholds['M']
                                                   for r in blind),
                n_coordinate_errors=sum(bool(r.get('coordinate_error')) for r in features.values()),
                limitations=['Visual labels are judgements, not physical ground truth.',
                             'E is in-source high-tail signal ranking, not energy calibration or a pile-up probability.',
                             'Incomplete/uncertain/unblinded control: descriptive pilot only.',
                             'Recall is conditional on visually resolvable labelled cases.',
                             'Wilson intervals are cluster-level pilot intervals, not frame-cluster robust.',
                             'Do not tune thresholds or train on this control and report it as an untouched test.'])


def atomic_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8-sig', newline='',
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def export_batch(conn, batch, output_dir):
    output_dir = Path(output_dir)
    records = load_features(conn, batch['run_id'])
    percentiles = size_percentiles(records)
    labels = {(r['frame'], r['cluster']): dict(r) for r in conn.execute(
        'SELECT * FROM collection_labels WHERE source_sha=?', (batch['source_sha'],))}
    queue = {(r['frame'], r['cluster']): r for r in queue_rows(conn, batch)}
    rows = []
    for r in records:
        key = r['frame'], r['cluster']
        label_row, q = labels.get(key, {}), queue.get(key, {})
        rows.append(dict(r, size_percentile=percentiles[r['n_pixels']],
                         run_id=batch['run_id'], source_sha=batch['source_sha'],
                         source=batch['source_path'], dataset=batch['dataset'],
                         batch_id=batch['batch_id'], stratum=q.get('stratum', ''),
                         label=label_row.get('label', ''), notes=label_row.get('notes', ''),
                         label_origin=label_row.get('origin_batch', ''),
                         label_diagnostics_seen=label_row.get('diagnostics_seen', ''),
                         timestamp_utc=label_row.get('timestamp_utc', '')))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    prefix = batch['batch_id'][:12]
    score_path = output_dir / f'{prefix}_scores.csv'
    label_path = output_dir / f'{prefix}_labels.csv'
    atomic_csv(score_path, rows, fields)
    atomic_csv(label_path, [r for r in rows if r['label']], fields)
    report = build_report(conn, batch)
    report_path = output_dir / f'{prefix}_report.json'
    # Machine report can always be regenerated from the committed database.
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    manifest_path = output_dir / f'{batch["run_id"][:12]}_methods.json'
    run = conn.execute('SELECT * FROM collection_runs WHERE run_id=?', (batch['run_id'],)).fetchone()
    manifest_path.write_text(json.dumps(dict(run), indent=2, ensure_ascii=False), encoding='utf-8')
    return score_path, label_path, report_path


def export_labels(conn, batch, output_dir):
    """Small per-click export; does not rescore or serialize the whole dataset."""
    query = '''SELECT r.frame,r.cluster,r.features_json,l.* FROM collection_labels l
       JOIN collection_records r ON r.run_id=? AND r.frame=l.frame AND r.cluster=l.cluster
       WHERE l.source_sha=? ORDER BY r.frame,r.cluster'''
    queue = {(r['frame'], r['cluster']): r for r in queue_rows(conn, batch)}
    rows = []
    for row in conn.execute(query, (batch['run_id'], batch['source_sha'])):
        q = queue.get((row['frame'], row['cluster']), {})
        rows.append(dict(json.loads(row['features_json']),
                         frame=row['frame'], cluster=row['cluster'],
                         dataset=batch['dataset'], source_sha=batch['source_sha'],
                         source=batch['source_path'], run_id=batch['run_id'],
                         batch_id=batch['batch_id'], label=row['label'],
                         timestamp_utc=row['timestamp_utc'], notes=row['notes'],
                         label_origin=row['origin_batch'],
                         label_diagnostics_seen=row['diagnostics_seen'], stratum=q.get('stratum', '')))
    fields = list(dict.fromkeys(['frame', 'cluster', 'label'] + [k for r in rows for k in r]))
    path = Path(output_dir) / f'{batch["batch_id"][:12]}_labels.csv'
    atomic_csv(path, rows, fields)
    return path
