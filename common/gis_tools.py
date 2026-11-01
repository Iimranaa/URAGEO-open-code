# -*- coding: utf-8 -*-
"""Helpers shared by the A1, A2 and A4 scripts: loading the synthetic layers, rasterising sites, Holm correction, AUC."""
import os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
from shapely.geometry import Point
from rasterio.features import rasterize
from rasterio.transform import from_origin

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('URAGEO_GIS_DIR', REPO / 'data_synthetic'))
OUT = Path(os.environ.get('URAGEO_OUTPUT', REPO / 'outputs'))
H, W, PX = 200, 260, 100.0
TR = from_origin(0, H * PX, PX, PX)

def load_layers():
    """Distance rasters (m) keyed by family, plus host and granite distances. Replace by your own rasters (same grid) if needed."""
    z = np.load(DATA / 'gis_synthetic.npz')
    return {k: z[k].astype(float) for k in z.files}

def load_sites():
    return pd.read_csv(DATA / 'sites_synthetic.csv')

def mask_of(rows):
    """Boolean mask of the union of discs described by rows (x, y, r), in the local grid."""
    geoms = [(Point(r.x, r.y).buffer(r.r), 1) for r in rows.itertuples()]
    return rasterize(geoms, out_shape=(H, W), transform=TR, fill=0, all_touched=True).astype(bool)

def mask_units(rows):
    """One index array (row, col) per object."""
    return [np.argwhere(mask_of(rows.iloc[[i]])) for i in range(len(rows))]

def holm(pvals):
    """Holm step-down. pvals: dict name -> p. Returns dict name -> (p, threshold, reject)."""
    ps = sorted(pvals.items(), key=lambda t: t[1]); m = len(ps); stop = False; out = {}
    for i, (k, p) in enumerate(ps):
        thr = 0.05 / (m - i); ok = (p <= thr) and not stop
        if not ok: stop = True
        out[k] = (p, thr, bool(ok))
    return out

def auc(index, pos, neg):
    a = index[pos]; b = index[neg]; r = rankdata(np.r_[a, b])
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))

def reclass(dist, breaks=(250, 500, 1000, 2000)):
    """Illustrative 1-5 classes from a distance raster (5 = closest). The real thresholds come from the model (see the thesis and the ArcMap guides)."""
    c = np.ones_like(dist); 
    for score, b in zip((5, 4, 3, 2), breaks): c[(dist <= b) & (c == 1)] = score
    return c
