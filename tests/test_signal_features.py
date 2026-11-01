"""Signal semantics, reference statistics, migration and cached GUI tests."""
from __future__ import annotations
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
os.environ['MPLBACKEND']='Agg'
from pileup import score_collection as c
from pileup.signal_features import describe_signal,rank_signal
from pileup.io.clog_reader import iter_frames
from collect_pileup_scores import Labeler,main


def line(value=1.,n=16):
    return [dict(x=x+10,y=30,value=value,extra=999999) for x in range(n)]


def reference_rows(count=100,n=16):
    return [dict(frame=i,cluster=0,n_pixels=n,M=0.,T=0.,G=0.,coordinate_error='',
                 **describe_signal(line(i+1,n))) for i in range(count)]


class SignalFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.conn=c.open_db(self.root/'scores.sqlite3')
        self.source=self.root/'signal.clog'
        text='Frame 0\n'
        for value in range(1,101):
            text+=' '.join(f'[{p["x"]},{p["y"]},{value},999999]' for p in line())+'\n'
        self.source.write_text(text)

    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close('all')
        self.conn.close()
        self.temp.cleanup()

    def index(self,**kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return c.index_source(self.conn,self.source,2,iter_frames,**kwargs)

    def config(self,**kwargs):
        config=dict(control=10,t_samples=0,m_samples=0,e_samples=10,g_samples=0,size_samples=0,
                    threshold_t=40,threshold_m=40,threshold_g=35,threshold_e=50,seed=7)
        config.update(kwargs)
        return config

    def test_raw_descriptors_scale_and_time(self):
        pixels=line(2.)
        a=describe_signal(pixels)
        self.assertEqual(a['signal_sum'],32.)
        self.assertEqual(a['signal_mean'],2.)
        self.assertEqual(a['signal_cv'],0.)
        self.assertAlmostEqual(a['signal_top10_fraction'],2/16)
        self.assertAlmostEqual(a['signal_centroid_shift_px'],0.)
        b=describe_signal([dict(p,value=p['value']*50,extra=-1e30) for p in pixels])
        for key in ['signal_cv','signal_max_to_median','signal_top10_fraction',
                    'signal_centroid_shift_px','signal_peak_neighbor_ratio']:
            self.assertAlmostEqual(a[key],b[key])
        self.assertEqual(b['signal_sum'],50*a['signal_sum'])

    def test_missing_negative_zero_and_ceiling_are_explicit(self):
        for value,status in [(None,'missing_values'),(float('nan'),'missing_values'),
                             (float('inf'),'missing_values'),(-1.,'negative_values'),(0.,'zero_total')]:
            with self.subTest(status=status,value=value):
                result=describe_signal(line(value))
                self.assertEqual(result['signal_status'],status)
                self.assertIsNone(rank_signal([dict(n_pixels=16,**result)],min_reference=2)[0]['E'])
                if status=='missing_values':
                    self.assertIsNone(result['signal_sum'])
        result=describe_signal(line(5.),ceiling=5.)
        self.assertEqual(result['signal_ceiling_fraction'],1.)
        self.assertEqual(result['signal_status'],'at_declared_ceiling')
        with self.assertRaises(ValueError):
            describe_signal(line(),ceiling=-1)

    def test_junction_signal_ratio(self):
        coords=sorted(set([(32+i,32+i) for i in range(-12,13)]+
                          [(32+i,32-i) for i in range(-12,13)]))
        cluster=[dict(x=x,y=y,value=10. if max(abs(x-32),abs(y-32))<=2 else 1.) for x,y in coords]
        topo=c.topology.score_t(cluster)
        self.assertTrue(topo['junctions_t'])
        result=describe_signal(cluster,topology=topo)
        self.assertEqual(result['signal_junction_ratio'],10.)

    def test_midrank_size_bins_and_equal_signals(self):
        rows=rank_signal(reference_rows())
        self.assertAlmostEqual(rows[-1]['e_mean_percentile'],99.5)
        self.assertAlmostEqual(rows[-1]['E'],95.)
        self.assertEqual(rows[0]['E'],0.)
        equal=reference_rows()
        for row in equal:row['signal_mean']=12.
        self.assertTrue(all(r['E']==0. for r in rank_signal(equal)))
        # Very different size population must not change the small-bin rank.
        large=reference_rows(count=100,n=64)
        combined=rank_signal(reference_rows()+large)
        self.assertAlmostEqual(combined[99]['E'],rows[-1]['E'])
        self.assertEqual(rank_signal(reference_rows(10))[0]['E_status'],'insufficient_reference')

    def test_E_rank_invariant_to_gain_within_source(self):
        a=reference_rows()
        b=[dict(r,signal_mean=r['signal_mean']*1000) for r in a]
        self.assertEqual([r['E'] for r in rank_signal(a)],[r['E'] for r in rank_signal(b)])

    def test_real_reader_integration_and_E_candidate_group(self):
        run=self.index()
        rows=c.load_features(self.conn,run)
        self.assertEqual(len(rows),100)
        self.assertEqual(rows[-1]['signal_sum'],1600.)
        self.assertAlmostEqual(rows[-1]['E'],95.)
        self.assertTrue(all(r['M']==0. and r['T']==0. for r in rows))
        chosen=c.sample_records(rows,set(),self.config(control=0))
        self.assertEqual(len(chosen),5)
        self.assertTrue(all(group=='E_without_MT' for _,_,group in chosen))
        # Numerical/statistical signal oddity alone does not label a particle.
        self.assertEqual(self.conn.execute('SELECT COUNT(*) FROM collection_labels').fetchone()[0],0)

    def test_signal_changes_leave_geometry_unchanged(self):
        a=line(1.)
        b=[dict(p,value=200. if i==2 else 0.,extra=-1e200) for i,p in enumerate(a)]
        _,first,_=c.calculate(a);_,second,_=c.calculate(b)
        self.assertEqual([first[k] for k in c.SCORE_KEYS],[second[k] for k in c.SCORE_KEYS])
        self.assertNotEqual(first['signal_sum'],second['signal_sum'])

    def test_cached_geometry_reuse_and_label_provenance(self):
        old=self.index()
        old_rules=json.loads(self.conn.execute('SELECT rules_json FROM collection_runs WHERE run_id=?',(old,)).fetchone()[0])
        # Emulate an actual v1.0 run with geometry/pixels only.
        old_rules['collection']='collection-1.0';old_rules['collector_sha']=c.LEGACY_COLLECTOR_SHA
        old_rules.pop('signal')
        with self.conn:
            self.conn.execute('UPDATE collection_runs SET rules_json=? WHERE run_id=?',(c.dumps(old_rules),old))
            for row in c.load_features(self.conn,old):
                f={k:v for k,v in row.items() if not k.startswith(('signal_','e_'))
                   and k not in {'E','E_status','frame','cluster','touches_detector_edge'}}
                self.conn.execute('UPDATE collection_records SET features_json=? WHERE run_id=? AND frame=? AND cluster=?',
                                  (c.dumps(f),old,row['frame'],row['cluster']))
        with contextlib.redirect_stdout(io.StringIO()):
            batch_id=c.create_batch(self.conn,old,self.config())
        batch=c.resolve_batch(self.conn,2,batch_id)
        item=c.queue_rows(self.conn,batch)[0]
        c.save_label(self.conn,batch,item,'uncertain','keep me')
        with patch.object(c,'calculate',side_effect=AssertionError('must reuse geometry')):
            new=self.index(signal_unit='keV')
        self.assertNotEqual(new,old)
        self.assertEqual(self.conn.execute('SELECT COUNT(*) FROM collection_labels').fetchone()[0],1)
        self.assertAlmostEqual(c.load_features(self.conn,new)[-1]['E'],95.)
        self.assertEqual(c.load_features(self.conn,new)[-1]['signal_unit'],'keV')
        # Updating a geometry method invalidates this reuse path.
        with patch.object(c.topology,'VERSION_T','changed'):
            with patch.object(c,'calculate',wraps=c.calculate) as calculate:
                self.index(signal_unit='keV')
                self.assertEqual(calculate.call_count,100)

    def test_ceiling_omits_clipped_reference_and_report_extends_union(self):
        run=self.index(signal_ceiling=100.)
        rows=c.load_features(self.conn,run)
        self.assertIsNone(rows[-1]['E'])
        self.assertEqual(rows[-1]['signal_ceiling_fraction'],1.)
        self.assertEqual(rows[0]['e_reference_n'],99)
        with contextlib.redirect_stdout(io.StringIO()):
            bid=c.create_batch(self.conn,run,self.config(control=100,e_samples=0))
        batch=c.resolve_batch(self.conn,2,bid)
        for item in c.queue_rows(self.conn,batch):
            c.save_label(self.conn,batch,item,'pileup' if item['cluster']==98 else 'not_pileup')
        report=c.build_report(self.conn,batch)
        self.assertEqual(report['additional_E_captures_beyond_MT'],1)
        self.assertEqual(report['metrics_on_blind_control']['MTE_union']['tp'],1)
        self.assertEqual(report['signal_summary']['status_counts']['at_declared_ceiling'],1)

    def test_E_view_uses_cache_and_records_unblinding(self):
        import matplotlib.pyplot as plt
        run=self.index()
        with contextlib.redirect_stdout(io.StringIO()):
            bid=c.create_batch(self.conn,run,self.config())
        batch=c.resolve_batch(self.conn,2,bid)
        with patch.object(c,'calculate',side_effect=AssertionError('GUI must use cache')):
            app=Labeler(self.conn,batch,self.root/'exports')
            app.toggle_signal()
            self.assertTrue(app.signal_view)
            self.assertEqual(app.axes[3].get_ylabel(),'Pixel count')
            app.toggle()
            self.assertFalse(app.signal_view)
            self.assertTrue(app.diagnostics)
            app.record('uncertain')
            labelled=next(r for r in c.queue_rows(self.conn,batch) if r['label'])
            self.assertEqual(labelled['label_diagnostics_seen'],1)
            plt.close(app.fig)

    def test_explicit_latest_and_unknown_collector_recalculate(self):
        old=self.index()
        with contextlib.redirect_stdout(io.StringIO()):
            bid=c.create_batch(self.conn,old,self.config())
            self.assertEqual(c.resolve_batch(self.conn,2,'latest')['batch_id'],bid)
        rules=json.loads(self.conn.execute('SELECT rules_json FROM collection_runs WHERE run_id=?',(old,)).fetchone()[0])
        rules.update(collector_sha='unreviewed-wrapper',collection='collection-1.0')
        with self.conn:
            self.conn.execute('UPDATE collection_runs SET rules_json=? WHERE run_id=?',(c.dumps(rules),old))
        with patch.object(c,'calculate',wraps=c.calculate) as calculate:
            self.index(signal_unit='ToT')
            self.assertEqual(calculate.call_count,100)


if __name__=='__main__':
    unittest.main()
