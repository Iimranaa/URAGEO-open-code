# -*- coding: utf-8 -*-
"""A1 · URAGEO favourability model, reference implementation of the statistical steps on synthetic layers.

Steps: (1) AHP weights from the pairwise-comparison matrix of the model (geometric-mean method, Saaty consistency ratio),
(2) Monte-Carlo stability of the weights (each judgement moved by one step of the 1-9 scale), (3) weighted overlay of seven
illustrative factor classes (1-5), (4) AUC of the index against pit footprints, (5) ablation of one factor, (6) hit-rate of the pit
footprints in the top 10 % and top 20 % of the index, compared with a rigid-shift null model.

IMPORTANT: the construction of the real factor rasters (F1 to F4 and their classes) was done in ArcMap/QGIS (see the thesis and the
ArcMap guides). Here the factors are simple reclassifications of synthetic distance rasters, so that only the statistical
machinery is reproduced. F1 (granite source) uses distance to a fictitious massif; F4 (host continuity) a fictitious host band.

Output: outputs/A1/ahp_overlay_results.json     Usage: python A1_URAGEO_favourability/ahp_overlay_ablation.py
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import numpy as np
from gis_tools import *

# ---- 1. AHP (matrix of the model: rows/columns F4, N110-150, F3, N70, F1, NS, N30)
names = ['F4', 'N110-150', 'F3', 'N70', 'F1', 'NS', 'N30']
A = np.array([[1,1,2,3,5,5,7],[1,1,2,3,5,5,7],[1/2,1/2,1,2,4,4,6],[1/3,1/3,1/2,1,2,2,4],[1/5,1/5,1/4,1/2,1,1,3],[1/5,1/5,1/4,1/2,1,1,3],[1/7,1/7,1/6,1/4,1/3,1/3,1]])
RI = {3: .58, 4: .90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41}
def ahp(A):
    n = len(A); gm = np.prod(A, axis=1) ** (1 / n); w = gm / gm.sum(); lam = np.linalg.eigvals(A).real.max(); ci = (lam - n) / (n - 1)
    return w, lam, ci, ci / RI[n]
w, lam, ci, cr = ahp(A)
res = {'weights': {n: round(float(x), 4) for n, x in zip(names, w)}, 'lambda_max': float(lam), 'CI': float(ci), 'CR': float(cr)}
# ---- 2. Monte-Carlo on the judgements
ladder = np.array([1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2,1,2,3,4,5,6,7,8,9]); rng = np.random.default_rng(7)
iu = np.triu_indices(7, 1); Wm = []; CRm = []
for _ in range(5000):
    B = np.ones((7, 7))
    for i, j in zip(*iu):
        k = int(np.abs(ladder - A[i, j]).argmin()) + int(rng.choice([-1, 0, 1])); k = min(max(k, 0), 16); B[i, j] = ladder[k]; B[j, i] = 1 / ladder[k]
    w_, _, _, c_ = ahp(B); Wm.append(w_); CRm.append(c_)
Wm = np.array(Wm); CRm = np.array(CRm)
res['monte_carlo'] = dict(n=5000, share_CR_below_0_10=float((CRm < .1).mean()),
                          weights_2_5_50_97_5={n: np.percentile(Wm[:, i], [2.5, 50, 97.5]).round(4).tolist() for i, n in enumerate(names)})
# ---- 3. weighted overlay on synthetic factors
D = load_layers(); S = load_sites()
F = {'F4': reclass(D['host'], (1, 2000, 4000, 8000)), 'N110-150': reclass(D['N110-150']), 'F3': reclass(D['Nodes']), 'N70': reclass(D['N70']),
     'F1': reclass(D['granite'], (6000, 10000, 15000, 22000)), 'NS': reclass(D['NS']), 'N30': reclass(D['N30'])}
wd = dict(zip(names, w))
def overlay(drop=None):
    ws = {k: v for k, v in wd.items() if k != drop}; t = sum(ws.values()); return sum(ws[k] * F[k] for k in ws) / t
pits = S[S.kind == 'pit']; pos = mask_of(pits); zone = D['host'] <= 6000
full = overlay(); res['auc_full_index'] = round(auc(full, pos, (~pos) & zone), 3)
res['ablation_auc'] = {k: round(auc(overlay(k), pos, (~pos) & zone), 3) for k in names}
# ---- 6. hit-rate in top 10 % / 20 % with rigid-shift null
def hit_rate(I, m, q):
    thr = np.quantile(I[zone], 1 - q); return float((I[m] >= thr).mean() * 100)
units = [np.argwhere(mask_of(pits.iloc[[i]])) for i in range(len(pits))]
def shifted(I, q, shift):
    out = []
    thr = np.quantile(I[zone], 1 - q)
    for ix in units:
        r = ix[:, 0] + shift[0]; c = ix[:, 1] + shift[1]
        if r.min() < 0 or c.min() < 0 or r.max() >= H or c.max() >= W: return None
        out.append((I[r, c] >= thr).mean() * 100)
    return float(np.mean(out))
rng = np.random.default_rng(5); m = int(6000 / PX); res['hit_rate'] = {}
for nm, I in {'full': full, 'without_F4': overlay('F4')}.items():
    for q in (0.10, 0.20):
        obs = shifted(I, q, (0, 0)); nul = []
        while len(nul) < 2000:
            v = shifted(I, q, (int(rng.integers(-m, m + 1)), int(rng.integers(-m, m + 1))))
            if v is not None: nul.append(v)
        nul = np.array(nul)
        res['hit_rate'][f'{nm}_top{int(q*100)}'] = dict(obs=obs, null_mean=float(nul.mean()), p95=float(np.percentile(nul, 95)), p=float((np.sum(nul >= obs) + 1) / (len(nul) + 1)))
o = OUT / 'A1'; o.mkdir(parents=True, exist_ok=True)
json.dump(res, open(o / 'ahp_overlay_results.json', 'w'), indent=1, default=float)
print('weights', res['weights'], 'CR', round(cr, 3)); print('AUC full', res['auc_full_index'], 'ablation', res['ablation_auc']); print(res['hit_rate'])
