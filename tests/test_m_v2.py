"""Minimal regression checks; run: python tests/test_m_v2.py"""
from pileup.collection_baseline.multiple_tracks import score_m


def make_cluster(points):
    return [{"x": x, "y": y, "value": 1.0, "extra": None}
            for x, y in sorted(set(points))]


cases = {
    "X crossing": make_cluster(
        [(x, x // 2 + 10) for x in range(5, 60)] +
        [(x, 45 - x // 2) for x in range(5, 60)]),
    "2 parallel tracks": make_cluster(
        [(x, y) for x in range(4, 60) for y in (10, 16)] +
        [(30, y) for y in range(11, 16)]),
    "3 parallel tracks": make_cluster(
        [(x, y) for x in range(4, 60) for y in (10, 16, 22)] +
        [(30, y) for y in range(11, 22)]),
    "single line": make_cluster([(x, 10) for x in range(4, 60)]),
    "thick single line": make_cluster(
        [(x, y) for x in range(4, 60) for y in range(10, 14)]),
    "Y branch": make_cluster(
        [(x, 10) for x in range(4, 34)] +
        [(x, 10 + (x-33)//2) for x in range(34, 59)] +
        [(x, 10 - (x-33)//2) for x in range(34, 59)]),
}

for name, cluster in cases.items():
    result = score_m(cluster)
    print(f"{name:20s} M={result['M']*100:5.1f}  "
          f"{result['status']:25s} fitted={result['n_tracks']}")
    if name in ("X crossing", "2 parallel tracks", "3 parallel tracks"):
        assert result["M"] >= 0.4, name
    else:
        assert result["M"] < 0.4, name
print("ALL TESTS PASSED")
