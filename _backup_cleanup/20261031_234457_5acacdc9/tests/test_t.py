"""From project root after copying topology.py: python tests/test_t.py"""
from pileup.topology import score_t

def cluster(points):
    return [{'x': x, 'y': y, 'value': 10.0, 'extra': 12345}
            for x, y in sorted(set(points))]

straight = cluster([(x, 30) for x in range(4, 58)])
y_shape = cluster([(x, 30) for x in range(4, 34)] +
                  [(x, 30 + x-33) for x in range(34, 49)] +
                  [(x, 30 - x+33) for x in range(34, 49)])
x_shape = cluster([(x, x) for x in range(5, 60)] +
                  [(x, 64-x) for x in range(5, 60)])
for name, c, lower, upper in [('straight', straight, 0, 0),
                               ('Y', y_shape, .4, .7),
                               ('X', x_shape, .7, 1)]:
    r = score_t(c)
    print(name, f'T={100*r["T"]:.0f}', r['status_t'])
    assert lower <= r['T'] <= upper, (name, r)
    changed = [dict(pixel, value=999, extra=-999) for pixel in c]
    assert score_t(changed)['T'] == r['T'], 'T must ignore signal and timestamp'
print('T1 synthetic tests OK')
