"""Run: python -m unittest discover -s tests -p 'test_geometry.py'

For convenience, you can also run from project root:
python tests/test_geometry.py
"""
import sys
from pathlib import Path
import unittest
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pileup.geometry import score_g, VERSION_G


def cluster(coords):
    return [dict(x=int(x), y=int(y), value=10, extra=None)
            for x, y in sorted(coords)]


class GeometryTests(unittest.TestCase):
    def test_straight_line_low(self):
        line = {(x, 20) for x in range(5, 55)}
        value = score_g(cluster(line))
        self.assertLess(value['G'], 0.05)

    def test_local_widening_higher(self):
        straight = {(x, y) for x in range(5, 55) for y in range(19, 22)}
        bulge = straight | {(x, y) for x in range(22, 38) for y in range(15, 26)}
        a, b = score_g(cluster(straight)), score_g(cluster(bulge))
        self.assertGreater(b['G'], a['G'] + 0.2)
        self.assertGreater(b['width_component'], 0.2)
        self.assertGreater(b['n_thick_spots_g'], 0)

    def test_curved_thin_track_not_high(self):
        curve = {(x, round(20 + 10 * math.sin(x / 8))) for x in range(5, 70)}
        value = score_g(cluster(curve))
        self.assertLess(value['G'], 0.35)

    def test_tiny_and_empty_safe(self):
        for coords in (set(), {(4, 4)}):
            value = score_g(cluster(coords))
            self.assertEqual(value['G'], 0)
        self.assertTrue(VERSION_G.startswith('G'))


if __name__ == '__main__':
    unittest.main()
