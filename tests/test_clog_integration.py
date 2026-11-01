"""Integration of the uploaded CLOG reader with the restored collector."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pileup.io.clog_reader import iter_frames
from pileup import score_collection as c


class ClogIntegration(unittest.TestCase):
    def test_actual_reader_to_index_batch_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'sample.clog'
            coordinates = sorted(set([(32+i,32+i) for i in range(-20,21)] +
                                     [(32+i,32-i) for i in range(-20,21)]))
            text = 'Frame 0\nFrame 1\n[1,1,0,99999]\n'
            text += ' '.join(f'[{x},{y},12,888888]' for x,y in coordinates) + '\n'
            source.write_text(text)
            before = source.read_bytes()
            conn = c.open_db(root/'index.sqlite3')
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    run = c.index_source(conn,source,2,iter_frames)
                    batch_id = c.create_batch(conn,run,dict(control=2,t_samples=0,m_samples=0,
                        g_samples=0,size_samples=0,seed=42,threshold_t=40,threshold_m=40,threshold_g=35))
                rows = c.load_features(conn,run)
                self.assertEqual(len(rows),2)
                self.assertEqual(rows[0]['n_pixels'],1)
                self.assertIsNone(rows[0]['M'])
                self.assertEqual(rows[1]['T'],80.)
                batch = c.resolve_batch(conn,2,batch_id)
                paths = c.export_batch(conn,batch,root/'exports')
                self.assertTrue(all(path.is_file() for path in paths))
                self.assertEqual(source.read_bytes(),before)
            finally:
                conn.close()


if __name__ == '__main__':
    unittest.main()
