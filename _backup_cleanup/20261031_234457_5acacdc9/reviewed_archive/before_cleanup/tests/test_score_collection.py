"""Meaningful synthetic, provenance, sampling, resume and GUI checks.

python -m unittest discover -s tests -p test_score_collection.py -v
Synthetic shapes test intended code behaviour, not real-data accuracy.
"""
from __future__ import annotations
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
os.environ['MPLBACKEND'] = 'Agg'
from pileup import score_collection as c
from collect_pileup_scores import Labeler, import_legacy, main


def points(coords):
    return [dict(x=x, y=y, value=1, extra=123) for x, y in sorted(set(coords))]


def shapes():
    return dict(
        line=points((x, 32) for x in range(10, 55)),
        thick=points((x, y) for x in range(10, 55) for y in range(30, 35)),
        L=points([(x, 10) for x in range(10, 50)] + [(10, y) for y in range(10, 50)]),
        X=points([(32+i, 32+i) for i in range(-20, 21)] + [(32+i, 32-i) for i in range(-20, 21)]),
        Y=points([(32, 32+i) for i in range(21)] + [(32-i, 32-i) for i in range(21)] +
                 [(32+i, 32-i) for i in range(21)]),
        parallel=points((x, y) for x in range(10, 55) for y in (29, 35)),
        tiny=points([(3, 3)]))


def fixture_reader(path):
    """Test adapter for the existing iter_frames API, deliberately not a CLOG parser."""
    for frame in json.loads(Path(path).read_text()):
        yield frame['frame'], frame['clusters']


def config(**kwargs):
    result = dict(control=2, t_samples=2, m_samples=2, g_samples=0, size_samples=0,
                  threshold_t=40., threshold_m=40., threshold_g=35., seed=2026)
    result.update(kwargs)
    return result


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.conn = c.open_db(self.root/'scores.sqlite3')
        self.data = shapes()
        self.source = self.root/'measurement.clog'
        self.source.write_text(json.dumps([
            dict(frame=0, clusters=[]),
            dict(frame=1, clusters=[[], self.data['tiny'], self.data['line'], self.data['X']]),
            dict(frame=2, clusters=[self.data['Y'], self.data['parallel'], self.data['thick']])]))

    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close('all')
        self.conn.close()
        self.temporary.cleanup()

    def index(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return c.index_source(self.conn, self.source, 2, fixture_reader)

    def batch(self, **kwargs):
        run_id = self.index()
        with contextlib.redirect_stdout(io.StringIO()):
            batch_id = c.create_batch(self.conn, run_id, config(**kwargs))
        return c.resolve_batch(self.conn, 2, batch_id)

    def test_shapes_and_independence_from_signal_and_time(self):
        expected = dict(line=(0, 0), thick=(0, 0), L=(0, 0), X=(98.7654321, 80), Y=(0, 45), parallel=(100, 0))
        for name, (m, t) in expected.items():
            _, first, _ = c.calculate(self.data[name])
            self.assertAlmostEqual(first['M'], m, places=5)
            self.assertAlmostEqual(first['T'], t)
            if name in {'L', 'X', 'Y'}:
                self.assertAlmostEqual(first['G'], 20.)
            changed = [dict(p, value=-99999*i, extra={'time':1e99}) for i, p in enumerate(self.data[name])]
            _, second, _ = c.calculate(changed)
            self.assertEqual([first[k] for k in c.SCORE_KEYS], [second[k] for k in c.SCORE_KEYS])

    def test_missing_and_invalid_are_not_zero(self):
        _, small, _ = c.calculate(self.data['tiny'])
        self.assertTrue(all(small[k] is None for k in c.SCORE_KEYS))
        duplicate = self.data['line'] + [self.data['line'][0]]
        _, bad, _ = c.calculate(duplicate)
        self.assertTrue(bad['coordinate_error'])
        self.assertTrue(all(bad[k] is None for k in c.SCORE_KEYS))
        with patch.object(c.multiple_tracks, 'score_m', side_effect=ValueError('test failure')):
            _, bad_m, _ = c.calculate(self.data['X'])
        self.assertIsNone(bad_m['M'])
        self.assertEqual(bad_m['T'], 80)
        self.assertIn('ERROR', bad_m['M_status'])
        self.assertEqual(bad_m['S_MT_status'], 'partial')

    def test_all_nonempty_clusters_indexed_ids_preserved_cache_and_move(self):
        before = self.source.read_bytes()
        run_id = self.index()
        records = c.load_features(self.conn, run_id)
        self.assertEqual(len(records), 6)
        self.assertEqual((records[0]['frame'], records[0]['cluster']), (1, 1))
        self.assertEqual(self.source.read_bytes(), before)
        moved = self.root/'moved.clog'
        moved.write_bytes(before)
        with patch.object(c, 'calculate', side_effect=AssertionError('must not rescore')):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(c.index_source(self.conn, moved, 2, fixture_reader), run_id)
        with patch.object(c.topology, 'VERSION_T', 'test-other-version'):
            self.assertNotEqual(self.index(), run_id)
        self.source.write_text(self.source.read_text() + '\n')
        self.assertNotEqual(self.index(), run_id)

    def test_interruption_leaves_incomplete_resumable_index(self):
        original = c.calculate
        calls = []
        def interrupted(cluster):
            calls.append(1)
            if len(calls) == 4:
                raise KeyboardInterrupt()
            return original(cluster)
        with patch.object(c, 'calculate', side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.index()
        run = self.conn.execute('SELECT * FROM collection_runs').fetchone()
        self.assertEqual(run['complete'], 0)
        self.assertEqual(self.conn.execute('SELECT COUNT(*) FROM collection_records').fetchone()[0], 3)
        with self.assertRaises(ValueError):
            c.create_batch(self.conn, run['run_id'], config())
        with patch.object(c, 'calculate', wraps=original) as calculate:
            self.assertEqual(self.index(), run['run_id'])
        self.assertEqual(calculate.call_count, 3)

    def test_sampling_control_first_includes_tiny_errors_and_known_labels(self):
        rows = [dict(frame=0, cluster=i, n_pixels=i+1, M=80. if i%3==0 else 0.,
                     T=80. if i%3==1 else 0., G=80. if i%3==2 else 0., coordinate_error='')
                for i in range(60)]
        all_control = c.sample_records(rows, {(0, 0)}, config(control=60, t_samples=0, m_samples=0))
        self.assertEqual({(f, cl) for f, cl, _ in all_control}, {(0, i) for i in range(60)})
        a = c.sample_records(rows, set(), config(control=10, t_samples=10, m_samples=10, g_samples=10))
        b = c.sample_records(list(reversed(rows)), set(), config(control=10, t_samples=10, m_samples=10, g_samples=10))
        self.assertEqual(a, b)
        self.assertEqual(len(a), len({(f, cl) for f, cl, _ in a}))
        self.assertEqual(sum(s == 'control_random' for _, _, s in a), 10)
        broken = [dict(frame=1, cluster=1, n_pixels=20, M=90., T=None, G=None)]
        self.assertEqual(c.sample_records(broken, set(), config(control=0)), [])

    def test_batch_is_frozen_and_exposure_sticky_across_batches(self):
        batch = self.batch(control=6, t_samples=0, m_samples=0)
        item = c.queue_rows(self.conn, batch)[0]
        c.mark_diagnostics_seen(self.conn, batch, item)
        second_id = c.create_batch(self.conn, batch['run_id'], config(control=6, t_samples=0, m_samples=0, seed=7))
        second = c.resolve_batch(self.conn, 2, second_id)
        second_item = next(r for r in c.queue_rows(self.conn, second) if
                           (r['frame'], r['cluster']) == (item['frame'], item['cluster']))
        c.save_label(self.conn, second, second_item, 'pileup')
        row = next(r for r in c.queue_rows(self.conn, second) if r['label'])
        self.assertEqual(row['label_diagnostics_seen'], 1)
        self.assertEqual(c.build_report(self.conn, second)['control']['blind_current_batch'], 0)
        self.assertEqual(c.create_batch(self.conn, batch['run_id'], config(control=6, t_samples=0, m_samples=0)), batch['batch_id'])
        with self.assertRaises(sqlite3.IntegrityError):
            c.save_label(self.conn, second, second_item, 'not_pileup')

    def test_abstentions_uncertain_and_incremental_report(self):
        metrics = c.screening_metrics([
            dict(label='pileup', T=80), dict(label='pileup', T=0),
            dict(label='pileup', T=None), dict(label='not_pileup', T=80),
            dict(label='uncertain', T=0), dict(label=None, T=0)], 'T', 40)
        self.assertEqual((metrics['tp'], metrics['fp'], metrics['fn'], metrics['positive_abstain']), (1, 1, 1, 1))
        self.assertEqual(metrics['screening_recall'], 1/3)
        self.assertEqual(metrics['precision'], .5)
        batch = self.batch(control=6, t_samples=0, m_samples=0)
        for row in c.queue_rows(self.conn, batch):
            label_value = 'pileup' if (row['frame'], row['cluster']) in {(1, 3), (2, 1)} else 'not_pileup'
            c.save_label(self.conn, batch, row, label_value)
        report = c.build_report(self.conn, batch)
        self.assertTrue(report['audit_fully_binary_and_blind'])
        self.assertEqual(report['additional_positive_captures_beyond_T']['M'], 1)
        self.assertEqual(report['metrics_on_blind_control']['T']['screening_recall'], .5)

    def test_csv_export_and_label_survive_reopen(self):
        batch = self.batch()
        row = c.queue_rows(self.conn, batch)[0]
        c.save_label(self.conn, batch, row, 'uncertain', 'test, note')
        paths = c.export_batch(self.conn, batch, self.root/'exports')
        c.export_labels(self.conn, batch, self.root/'exports')
        with paths[1].open(encoding='utf-8-sig', newline='') as stream:
            labels = list(csv.DictReader(stream))
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0]['notes'], 'test, note')
        self.assertEqual(labels[0]['label'], 'uncertain')
        self.assertEqual(c.size_percentiles([dict(n_pixels=7)]), {7:50.})
        self.conn.close()
        self.conn = c.open_db(self.root/'scores.sqlite3')
        self.assertEqual(sum(bool(r['label']) for r in c.queue_rows(self.conn, batch)), 1)

    def test_legacy_import_conflict_is_atomic(self):
        batch = self.batch()
        csv_path = self.root/'legacy.csv'
        fields = ['source', 'frame', 'cluster', 'label', 'n_pixels']
        rows = [dict(source=batch['source_path'], frame=1, cluster=2, label='pileup', n_pixels=45),
                dict(source=batch['source_path'], frame=1, cluster=2, label='not_pileup', n_pixels=45)]
        c.atomic_csv(csv_path, rows, fields)
        with self.assertRaises(ValueError):
            import_legacy(self.conn, batch, csv_path)
        self.assertEqual(self.conn.execute('SELECT COUNT(*) FROM collection_labels').fetchone()[0], 0)
        c.atomic_csv(csv_path, rows[:1], fields)
        import_legacy(self.conn, batch, csv_path)
        self.assertEqual(self.conn.execute('SELECT COUNT(*) FROM collection_labels').fetchone()[0], 1)
        self.assertEqual(c.build_report(self.conn, batch)['control']['blind_current_batch'], 0)

    def test_gui_uses_cached_diagnostics_and_allows_no_keyboard_label_in_notes(self):
        import matplotlib.pyplot as plt
        from types import SimpleNamespace
        batch = self.batch(control=6, t_samples=0, m_samples=0)
        output = self.root/'gui_exports'
        with patch.object(c, 'calculate', side_effect=AssertionError('GUI must use cache')):
            app = Labeler(self.conn, batch, output)
            app.toggle()
            app.toggle()
            app.notes.begin_typing()
            app.on_key(SimpleNamespace(inaxes=app.notes.ax, key='1'))
            self.assertEqual(self.conn.execute('SELECT COUNT(*) FROM collection_labels').fetchone()[0], 0)
            app.notes.stop_typing()
            app.record('uncertain')
            self.assertEqual(self.conn.execute('SELECT COUNT(*) FROM collection_labels').fetchone()[0], 1)
            app.fig.canvas.draw()
            plt.close(app.fig)

    def test_cli_sample_and_report_end_to_end_with_reader_adapter(self):
        from types import ModuleType
        reader = ModuleType('pileup.io.clog_reader')
        reader.iter_frames = fixture_reader
        modules = {'pileup.io': ModuleType('pileup.io'), 'pileup.io.clog_reader': reader}
        database = self.root/'cli.sqlite3'
        output = self.root/'cli_exports'
        shared = ['--dataset', '2', '--db', str(database), '--output', str(output)]
        with patch.dict(sys.modules, modules):
            with contextlib.redirect_stdout(io.StringIO()):
                main(['sample', '--file', str(self.source), '--control', '6', '--t-samples', '0',
                      '--m-samples', '0', *shared])
                main(['report', *shared])
        report_path, = output.glob('*_report.json')
        report = json.loads(report_path.read_text())
        self.assertEqual(report['control']['selected'], 6)
        self.assertFalse(report['audit_fully_binary_and_blind'])
        self.assertEqual(len(list(output.glob('*_scores.csv'))), 1)


if __name__ == '__main__':
    unittest.main()
