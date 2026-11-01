# -*- coding: utf-8 -*-
"""Synthetic borehole dataset that mimics the STRUCTURE of the Tarat borehole table (no real value).

All coordinates, thicknesses and indices are SIMULATED (fixed seed). Nothing here comes from TSUMCO.
The data contain a planted spatial structure (a smooth trend plus a correlated Gaussian field for thickness and depth,
almost pure noise for the redox indices) so that the scripts of A3 run end to end and show the *method*.
Results obtained on this file are NOT the results of the article and must not be interpreted geologically.

Usage:  python common/synthetic_boreholes.py            (writes data_synthetic/*.csv)
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist

SEED = 2026
OUT = Path(__file__).resolve().parents[1] / 'data_synthetic'
N = 731

def gaussian_field(XY, rng, sill, rang, nug_frac=0.1):
    D = cdist(XY, XY)
    C = sill * (1 - nug_frac) * np.exp(-3 * D / rang) + np.eye(len(XY)) * sill * nug_frac
    L = np.linalg.cholesky(C + 1e-9 * np.eye(len(XY)))
    return L @ rng.standard_normal(len(XY))

def make(seed=SEED):
    rng = np.random.default_rng(seed)
    # irregular, clustered design in a 3.6 x 3.4 km local block (synthetic local coordinates, origin 0,0)
    n_core = int(N * 0.78)
    cx, cy = rng.uniform(900, 2700, 6), rng.uniform(900, 2500, 6)
    pts = []
    for _ in range(n_core):
        k = rng.integers(0, 6); pts.append([rng.normal(cx[k], 420), rng.normal(cy[k], 380)])
    for _ in range(N - n_core):
        pts.append([rng.uniform(0, 3600), rng.uniform(0, 3400)])
    XY = np.clip(np.array(pts), 0, [3600, 3400]); XY += rng.normal(0, 3, XY.shape)
    x, y = XY[:, 0], XY[:, 1]
    u, v = (x - 1800) / 1800, (y - 1700) / 1700
    # thickness (m): quadratic trend + correlated field + noise
    th = 38 + 9 * u - 4 * v + 6 * u ** 2 - 3 * u * v + gaussian_field(XY, rng, 28, 1000)
    th = np.clip(th, 8, 62)
    roof = 50 + 6 * u + 7 * v + 3 * u ** 2 + gaussian_field(XY, rng, 22, 1300)
    roof = np.clip(roof, 22, 75)
    # semi-quantitative redox indices: zero-inflated, right-skewed, almost no spatial structure
    def index(p_zero, scale, struct=0.0):
        g = rng.standard_normal(N) + struct * gaussian_field(XY, rng, 1, 600)
        val = np.exp(rng.normal(np.log(scale), 0.9, N))
        val[rng.random(N) < p_zero] = 0.0
        return np.where(g > -3, val, 0.0)
    oxy = index(0.46, 0.2, 0.7) * (1 + 0.15 * (th - th.mean()) / th.std()).clip(0.2)
    sulf = index(0.44, 0.22)
    mo = index(0.59, 0.2) + 0.25 * sulf * (rng.random(N) < 0.5)
    carb = np.full(N, np.nan); idx = rng.choice(N, 31, replace=False); carb[idx] = rng.exponential(0.02, 31)
    # principal components of the three indices (computed here, as in the original table)
    Z = np.c_[oxy, sulf, mo]; Z = (Z - Z.mean(0)) / Z.std(0)
    w, V = np.linalg.eigh(np.cov(Z.T)); V = V[:, ::-1]; PC = Z @ V
    pre = rng.choice(['TMRI', 'TK', 'TO', 'COGE', 'TAMR'], N, p=[.94, .035, .019, .003, .003])
    hid = [f'{p}_{i:04d}' for i, p in enumerate(pre)]
    def cls(a):
        return np.where(a == 0, 'Faible', np.where(a < np.quantile(a[a > 0], .7), 'Modere', 'Eleve'))
    d = pd.DataFrame(dict(hole_id=hid, x=x.round(2), y=y.round(2), z=(417 + rng.normal(0, 2, N)).round(2), puiss_m=th.round(2),
                          acc_oxy=oxy, acc_carb=carb, acc_sulf=sulf, acc_mo=mo, prefixe=pre,
                          cl_oxy=cls(oxy), cl_sulf=cls(sulf), cl_mo=cls(mo),
                          PC1_731=PC[:, 0], PC2_731=PC[:, 1], PC3_731=PC[:, 2],
                          cl_reduc=np.where(PC[:, 0] > np.quantile(PC[:, 0], .88), 'Eleve', 'Faible'),
                          carb_ok=(~np.isnan(carb)).astype(int), toit_m=roof.round(2)))
    d['mur_m'] = (d.toit_m + d.puiss_m).round(2); d['epaisseur_check'] = d.puiss_m
    d['PC1_calc'] = d.PC1_731; d['PC2_calc'] = d.PC2_731; d['PC3_calc'] = d.PC3_731
    # internal units U1..U4 (codes 310, 320, 330, 340): shares of the total thickness; some units absent
    sh = np.c_[rng.dirichlet([4, 1.2, 5, 2.2], N)]
    sh[:, 0] += 0.04 * u * 0; absent = rng.random((N, 4)) < [0.026, 0.186, 0.0, 0.089]
    sh = np.where(absent, 0, sh); sh = sh / sh.sum(1, keepdims=True)
    rows = []
    for i in range(N):
        a = 0.0
        for k, code in enumerate([310, 320, 330, 340]):
            t = sh[i, k] * th[i]
            if t > 0: rows.append((hid[i], code, round(roof[i] + a, 2), round(roof[i] + a + t, 2))); a += t
    st = pd.DataFrame(rows, columns=['hole_id', 'code', 'from_m', 'to_m'])
    return d, st

if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    d, st = make()
    d.to_csv(OUT / 'tarat_boreholes_synthetic.csv', index=False)
    st.to_csv(OUT / 'strati_units_synthetic.csv', index=False)
    print('written', OUT, len(d), 'boreholes,', len(st), 'unit intervals')
