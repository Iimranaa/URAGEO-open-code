# -*- coding: utf-8 -*-
"""Synthetic GIS layers for the tests of A1, A2 and A4 (fault-family distances, nodes, host, granite, pits, sites).

EVERYTHING IS FICTITIOUS: fault traces, pit outlines and site positions are randomly generated in a local grid
(100 m cells, 260 columns x 200 rows, origin at the lower left, no coordinate system). They bear no relation to the
real Arlit or Tim Mersoi geometry and do not come from TSUMCO. Pits are placed with a planted preference for
cells close to N110-150 faults and to fault nodes, so that the tests have something to detect; the result of the
tests on this file is therefore illustrative of the METHOD only.

Usage: python common/synthetic_gis.py      (writes data_synthetic/gis_synthetic.npz and pits_synthetic.csv)
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from shapely.geometry import LineString, Point
from rasterio.features import rasterize
from rasterio.transform import from_origin

OUT = Path(__file__).resolve().parents[1] / 'data_synthetic'
H, W, PX = 200, 260, 100.0
TR = from_origin(0, H * PX, PX, PX)           # local grid, north up
FAMS = {'N110-150': (110, 150), 'N70': (60, 80), 'NS': (-10, 10), 'N30': (20, 40)}
NLINES = {'N110-150': 10, 'N70': 6, 'NS': 6, 'N30': 5}

def lines(rng, az_range, n):
    out = []
    for _ in range(n):
        x, y = rng.uniform(2000, W * PX - 2000), rng.uniform(2000, H * PX - 2000)
        az = np.deg2rad(rng.uniform(*az_range)); L = rng.uniform(3000, 9000)
        dx, dy = np.sin(az) * L / 2, np.cos(az) * L / 2
        out.append(LineString([(x - dx, y - dy), (x + dx, y + dy)]))
    return out

def dist(geoms):
    m = rasterize([(g, 1) for g in geoms], out_shape=(H, W), transform=TR, fill=0, all_touched=True).astype(bool)
    return ndi.distance_transform_edt(~m) * PX, m

def make(seed=2027):
    rng = np.random.default_rng(seed)
    L = {k: lines(rng, FAMS[k], NLINES[k]) for k in FAMS}
    D = {}
    for k in FAMS: D[k], _ = dist(L[k])
    # nodes: intersections between lines of different families
    nodes = []
    ks = list(FAMS)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            for g1 in L[a]:
                for g2 in L[b]:
                    p = g1.intersection(g2)
                    if p.geom_type == 'Point': nodes.append(p)
    D['Nodes'], _ = dist([n.buffer(PX) for n in nodes])
    # fictitious host formation (a band) and granite source (a massif on the east side)
    yy, xx = np.mgrid[0:H, 0:W] * PX
    host = np.abs((yy - 0.35 * xx) - 6000) < 4500
    D['host'] = ndi.distance_transform_edt(~host) * PX
    D['granite'] = np.hypot(xx - 25000, yy - 14000)
    # pits: sampling weights planted on N110-150 proximity and nodes, inside the host
    wgt = np.exp(-D['N110-150'] / 500) * (0.2 + np.exp(-D['Nodes'] / 900)) * (D['host'] == 0)
    wgt[:, :20] = 0; wgt[:, -20:] = 0; wgt[:20] = 0; wgt[-20:] = 0
    p = (wgt / wgt.sum()).ravel()
    rows = []
    sector_centres = []
    for s in range(6):                                    # six sectors, ~3 pits each
        c = rng.choice(len(p), p=p); cy, cx = divmod(c, W); sector_centres.append((cx, cy))
        for j in range(3 if s < 5 else 2):
            ox, oy = rng.normal(0, 900), rng.normal(0, 900)
            r = rng.uniform(260, 520)
            rows.append(dict(kind='pit', sector=f'S{s+1}', x=cx * PX + ox, y=(H - cy) * PX + oy, r=r))
    # three 'workings' of an old mine, planted near N70 faults
    wgt2 = np.exp(-D['N70'] / 500) * (D['host'] == 0); wgt2[:20] = 0; wgt2[-20:] = 0; wgt2[:, :20] = 0; wgt2[:, -20:] = 0
    p2 = (wgt2 / wgt2.sum()).ravel(); c = rng.choice(len(p2), p=p2); cy, cx = divmod(c, W)
    for j in range(3):
        rows.append(dict(kind='working', sector='M1', x=cx * PX + rng.normal(0, 500), y=(H - cy) * PX + rng.normal(0, 500), r=rng.uniform(180, 320)))
    # four 'external' sites (3 km circles) with a weaker planted signal, away from the pits
    for k, name in enumerate(['E1', 'E2', 'E3', 'E4']):
        for _ in range(100):
            x, y = rng.uniform(5000, W * PX - 5000), rng.uniform(5000, H * PX - 5000)
            if all(np.hypot(x - r['x'], y - r['y']) > 6000 for r in rows): break
        rows.append(dict(kind='external', sector=name, x=x, y=y, r=3000.0))
    sites = pd.DataFrame(rows)
    return D, sites

if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    D, sites = make()
    np.savez_compressed(OUT / 'gis_synthetic.npz', **{k: v.astype('float32') for k, v in D.items()})
    sites.to_csv(OUT / 'sites_synthetic.csv', index=False)
    print('written', OUT, {k: v.shape for k, v in D.items()}, len(sites), 'site objects')
