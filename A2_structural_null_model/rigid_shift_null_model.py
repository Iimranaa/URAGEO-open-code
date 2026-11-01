# -*- coding: utf-8 -*-
"""A2 · Association between fault families / nodes and mining units, tested against a null model of rigid shifts.

For each unit (sector of pits, or old workings) the statistic is the share of its cells lying within THR metres of a
fault family (or of a node). The null distribution is obtained by translating the whole set of units rigidly by random
vectors (radius RAD) and recomputing the statistic: the shape and relative position of the units are preserved,
only their position relative to the faults changes. One-sided p = (#null >= observed + 1) / (n + 1).
Five tests (four families + nodes) are corrected with the Holm step-down procedure.

Runs on the synthetic layers by default (common/synthetic_gis.py). Output: outputs/A2/rigid_shift_results.json
Usage:  python A2_structural_null_model/rigid_shift_null_model.py
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import numpy as np
from gis_tools import *

D = load_layers(); S = load_sites()
FAM = ['N110-150', 'N70', 'NS', 'N30', 'Nodes']
units_sector = []
for s in sorted(S[S.kind == 'pit'].sector.unique()):                      # one unit per sector: union of its pits
    units_sector.append(np.argwhere(mask_of(S[(S.kind == 'pit') & (S.sector == s)])))
units_work = [np.argwhere(mask_of(S[S.kind == 'working']))]

def stat(units, arr, shift=(0, 0), thr=500):
    out = []
    for ix in units:
        r = ix[:, 0] + shift[0]; c = ix[:, 1] + shift[1]
        if r.min() < 0 or c.min() < 0 or r.max() >= H or c.max() >= W: return None
        out.append(np.mean(arr[r, c] <= thr))
    return float(np.mean(out))

def test(units, k, thr=500, rad=6000, N=2000, seed=7):
    rng = np.random.default_rng(seed); m = int(rad / PX); obs = stat(units, D[k], thr=thr); nul = []
    while len(nul) < N:
        v = stat(units, D[k], (int(rng.integers(-m, m + 1)), int(rng.integers(-m, m + 1))), thr)
        if v is not None: nul.append(v)
    nul = np.array(nul)
    return dict(obs=obs, null_mean=float(nul.mean()), ratio=float(obs / nul.mean()) if nul.mean() > 0 else None,
                p=float((np.sum(nul >= obs) + 1) / (len(nul) + 1)), n_null=len(nul), p95=float(np.percentile(nul, 95)))

res = {'n_sector_units': len(units_sector), 'primary': {k: test(units_sector, k) for k in FAM}}
res['holm5'] = {k: dict(p=v[0], threshold=v[1], reject=v[2]) for k, v in holm({k: res['primary'][k]['p'] for k in FAM}).items()}
res['sensitivity'] = {f'thr{thr}|rad{rad}': {k: test(units_sector, k, thr, rad, 1000)['p'] for k in ('N110-150', 'NS', 'N70', 'Nodes')}
                      for thr in (250, 500, 1000) for rad in (3000, 6000, 10000)}
res['old_workings'] = {k: test(units_work, k) for k in FAM}
o = OUT / 'A2'; o.mkdir(parents=True, exist_ok=True)
json.dump(res, open(o / 'rigid_shift_results.json', 'w'), indent=1, default=float)
for k, v in res['primary'].items(): print(f"{k:9s} obs={v['obs']:.3f} null={v['null_mean']:.3f} p={v['p']:.4f}  Holm reject={res['holm5'][k]['reject']}")
