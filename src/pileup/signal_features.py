"""Pixel-signal descriptors and an exploratory, size-conditioned high-tail E.

No energy calibration is performed. Only the third CLOG field (value) is
used; extra/time is never inspected. E ranks mean signal within one source
and log2(N) size bins. It is NOT a pile-up probability or dE/dx.
"""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np

VERSION_SIGNAL = 'E1.0-size-bin-mean-tail'
VALID_UNITS = {'unknown', 'ToT', 'keV', 'MeV'}


def describe_signal(pixels, *, topology=None, unit='unknown', ceiling=None):
    """Accept validated, unique detector pixels. Preserve missing/negative flags.

    Raw totals are available only for complete, finite signals. Concentration
    and weighted positions require nonnegative samples and positive total.
    A configured ceiling marks possible clipping, not proven saturation.
    """
    if unit not in VALID_UNITS:
        raise ValueError(f'Unsupported declared signal unit: {unit}')
    if ceiling is not None and (not math.isfinite(ceiling) or ceiling <= 0):
        raise ValueError('Signal ceiling must be finite and > 0')
    n = len(pixels)
    result = dict(E=None, E_status='not_ranked', e_version=VERSION_SIGNAL,
                  e_size_bin=(n.bit_length()-1) if n else None,
                  e_reference_n=0, e_mean_percentile=None,
                  signal_unit=unit, signal_ceiling=ceiling, signal_status='empty',
                  signal_n_finite=0, signal_n_missing=n, signal_n_negative=0,
                  signal_zero_fraction=None, signal_ceiling_fraction=None,
                  signal_observed_sum=None, signal_sum=None, signal_mean=None,
                  signal_median=None, signal_max=None, signal_p90=None,
                  signal_cv=None, signal_max_to_median=None, signal_top10_fraction=None,
                  signal_centroid_shift_px=None, signal_centroid_shift_norm=None,
                  signal_peak_neighbor_ratio=None, signal_junction_ratio=None,
                  touches_detector_edge=any(p['x'] in (0,255) or p['y'] in (0,255) for p in pixels))
    if not n:
        result['E_status'] = 'empty'
        return result
    values = np.array([np.nan if p.get('value') is None else float(p['value']) for p in pixels])
    finite = np.isfinite(values)
    observed = values[finite]
    result.update(signal_n_finite=int(finite.sum()), signal_n_missing=int((~finite).sum()),
                  signal_n_negative=int((observed < 0).sum()))
    if observed.size:
        with np.errstate(over='ignore', invalid='ignore'):
            observed_sum = float(observed.sum())
        result['signal_observed_sum'] = observed_sum if math.isfinite(observed_sum) else None
    if not finite.all():
        result.update(signal_status='missing_values', E_status='missing_values')
        return result
    with np.errstate(over='ignore', invalid='ignore'):
        total = float(values.sum())
    if not math.isfinite(total):
        result.update(signal_status='overflow', E_status='overflow')
        return result
    mean, median = total/n, float(np.median(values))
    result.update(signal_sum=total, signal_mean=mean, signal_median=median,
                  signal_max=float(values.max()), signal_p90=float(np.percentile(values,90)),
                  signal_zero_fraction=float((values == 0).mean()),
                  signal_ceiling_fraction=float((values >= ceiling).mean()) if ceiling is not None else None)
    if (values < 0).any():
        result.update(signal_status='negative_values', E_status='negative_values')
        return result
    if total <= 0:
        result.update(signal_status='zero_total', E_status='zero_total')
        return result
    if mean <= 0:
        result.update(signal_status='numerical_underflow', E_status='numerical_underflow')
        return result
    # Work with relative values for scale-invariant descriptors and stable arithmetic.
    weights = values / total
    relative = values / mean
    xy = np.array([(p['x'],p['y']) for p in pixels], dtype=float)
    shift = float(np.linalg.norm(weights @ xy - xy.mean(axis=0)))
    diagonal = max(1., float(np.linalg.norm(np.ptp(xy,axis=0))))
    top_n = max(1, math.ceil(.1*n))
    result.update(signal_cv=float(np.std(relative)),
                  signal_max_to_median=float(values.max()/median) if median > 0 else None,
                  signal_top10_fraction=float(np.sort(weights)[-top_n:].sum()),
                  signal_centroid_shift_px=shift, signal_centroid_shift_norm=shift/diagonal)
    peak = int(np.argmax(values))
    distances = np.max(np.abs(xy-xy[peak]),axis=1)
    neighbors = values[(distances > 0) & (distances <= 2)]
    if len(neighbors) >= 3 and np.median(neighbors) > 0:
        result['signal_peak_neighbor_ratio'] = float(values[peak]/np.median(neighbors))
    # Junction enrichment is diagnostic, not part of E. Use the existing T
    # skeleton; multiple pixels in a junction group must not be counted as
    # multiple particles. T's representatives are approximate positions.
    topo = topology or {}
    junctions = np.asarray(topo.get('junctions_t',[]),dtype=float).reshape(-1,2)
    skeleton = topo.get('skeleton_t',[])
    if len(junctions) and skeleton:
        value_at = {(p['x'],p['y']):v for p,v in zip(pixels,values)}
        arm = [value_at[tuple(p)] for p in skeleton if tuple(p) in value_at
               and np.max(np.abs(junctions-np.asarray(p)),axis=1).min() >= 4]
        if len(arm) >= 5 and np.median(arm) > 0:
            ratios = []
            for junction in junctions:
                patch = values[np.max(np.abs(xy-junction),axis=1) <= 2]
                if len(patch) >= 3:
                    ratios.append(float(np.median(patch)/np.median(arm)))
            if ratios:
                result['signal_junction_ratio'] = max(ratios)
    status = 'at_declared_ceiling' if ceiling is not None and np.any(values >= ceiling) else 'ok'
    result.update(signal_status=status, E_status='not_ranked' if status == 'ok' else status)
    return result


def rank_signal(records, *, min_reference=30):
    """Return updated row copies; reference = same source, complete positive signals.

    Size bins are [1], [2..3], [4..7], ... . Midranks handle ties. Self is
    included: this is descriptive in-source ranking, never an independent
    normal-track calibration. E = 100*clip((percentile/100-.90)/.10,0,1).
    """
    if min_reference < 2:
        raise ValueError('At least two reference clusters are required')
    rows = [dict(r) for r in records]
    groups = defaultdict(list)
    for row in rows:
        row.update(E=None, e_mean_percentile=None, e_reference_n=0)
        if row.get('signal_status') == 'ok' and row.get('signal_mean') is not None:
            groups[row['e_size_bin']].append(row)
        else:
            row['E_status'] = row.get('signal_status','signal_unavailable')
    for group in groups.values():
        reference = np.sort([r['signal_mean'] for r in group])
        n = len(reference)
        for row in group:
            row['e_reference_n'] = n
            if n < min_reference:
                row['E_status'] = 'insufficient_reference'
                continue
            value = row['signal_mean']
            left = int(np.searchsorted(reference,value,side='left'))
            right = int(np.searchsorted(reference,value,side='right'))
            rank = (left + right)/(2*n)
            row.update(e_mean_percentile=100*rank,
                       E=float(np.clip((rank-.90)/.10,0,1)*100), E_status='ok')
    return rows
