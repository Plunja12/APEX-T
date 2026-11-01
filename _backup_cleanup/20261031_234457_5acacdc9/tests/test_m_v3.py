"""Run from APEX-T root: python tests/test_m_v3.py (no pytest needed)."""
from __future__ import annotations
import math
from pileup.multiple_tracks import score_m, VERSION


def make(points, value=12.0):
    return [{'x': x, 'y': y, 'value': value, 'extra': 90000}
            for x, y in sorted(set(points))]


def line(func, start=4, stop=62, thickness=0):
    return [(x, int(round(func(x))) + dy) for x in range(start, stop)
            for dy in range(-thickness, thickness + 1)]

cases = {
    'straight': (make(line(lambda x: 25)), False),
    'one curved': (make(line(lambda x: 25 + 5 * math.sin(x/11))), False),
    'thick curved': (make(line(lambda x: 25 + 5 * math.sin(x/11), thickness=2)), False),
    'single Y branch': (make([(x, 30) for x in range(2, 33)] +
                              line(lambda x: 30 + (x-32)*.55, 33, 60) +
                              line(lambda x: 30 - (x-32)*.55, 33, 60)), False),
    'large blob': (make([(x, y) for x in range(10, 22)
                               for y in range(10, 22)]), False),
    'X crossing': (make(line(lambda x: 10 + .65*x) +
                        line(lambda x: 51 - .65*x)), True),
    'two parallel': (make(line(lambda x: 21) + line(lambda x: 31) +
                          [(33, y) for y in range(21, 32)]), True),
    'three parallel': (make(line(lambda x: 14) + line(lambda x: 24) +
                            line(lambda x: 34) +
                            [(30, y) for y in range(14, 35)]), True),
    'two curved parallel': (make(line(lambda x: 14 + 3*math.sin(x/10)) +
                                 line(lambda x: 29 + 3*math.sin(x/10)) +
                                 [(29, y) for y in range(14, 29)]), True),
    'two curved crossing': (make(line(lambda x: 18+.60*x+5*math.sin(x/14), 4, 73) +
                                 line(lambda x: 65-.54*x+5*math.sin(x/14), 4, 73)), True),
}

for name, (cluster, expect) in cases.items():
    r = score_m(cluster)
    print(f'{name:24s}: M = {r["M"]*100:5.1f}  '
          f'curves={r["n_tracks"]}  {r["status"]}')
    assert r['version'] == VERSION
    assert 0 <= r['M'] <= 1
    assert (r['M'] >= 0.4) == expect, name

# The fourth column may contain ToA, not energy; ensure it changes nothing.
base = cases['X crossing'][0]
changed = [dict(p, extra=7.0e9) for p in base]
assert score_m(base)['M'] == score_m(changed)['M']

# Changing a few signals may alter only the signal bonus, not the number
# of skeleton trajectories; values are not used to add extra fake paths.
changed = [dict(p, value=(70.0 if p['x'] == 30 else 2.0)) for p in base]
assert score_m(base)['n_tracks'] == score_m(changed)['n_tracks']

print('ALL M3 TESTS PASSED')
