# -*- coding: utf-8 -*-
"""Rebuild the three regional index grids of article A4 (window 240-370 km E, 1920-2090 km N, EPSG:30732, 100 m cells).

Inputs (not distributed here; published sources, see README.md):
  URAGEO_FAULTS     regional fault base (SIGU), lines, EPSG:30732, field NOM_CLASSE
  URAGEO_GEOL_TM    geological map of the Air / Tim Mersoi area, polygons, fields TITRE and GROUPE
  URAGEO_GEOL_REG   regional geological map used for the binary host rule, polygons, fields TITRE and CLASSE
Outputs: the three GeoTIFFs of this folder (written to URAGEO_OUTPUT, default ./outputs).
Expected checks: 2 028 faults >= 500 m in the window + 20 km, 856 merged intersection nodes, host class 0 on 5.1 % of the window.
"""
import os, numpy as np, geopandas as gpd, shapely, rasterio
from rasterio import features
from rasterio.transform import from_origin
from scipy import ndimage as ndi
from scipy.cluster.hierarchy import fcluster, linkage
from shapely.strtree import STRtree
from shapely.ops import nearest_points
from shapely.geometry import box, Point

FAULTS, GEOL_TM, GEOL_REG = (os.environ[k] for k in ('URAGEO_FAULTS', 'URAGEO_GEOL_TM', 'URAGEO_GEOL_REG'))
OUT = os.environ.get('URAGEO_OUTPUT', 'outputs'); os.makedirs(OUT, exist_ok=True)
R = 100; LEFT, RIGHT, BOT, TOP = 240000, 370000, 1920000, 2090000
shp = ((TOP - BOT) // R, (RIGHT - LEFT) // R); tr = from_origin(LEFT, TOP, R, R)

def dist(geoms):
    m = features.rasterize([(g, 1) for g in geoms], out_shape=shp, transform=tr, all_touched=True, fill=0).astype(bool)
    return ndi.distance_transform_edt(~m) * R if m.any() else np.full(shp, 1e6)
def fam(a):
    a = a % 180; return 'NS' if (a >= 165 or a < 15) else 'N30' if a < 50 else 'N70' if a < 100 else 'N110_150'
def rc(d, t):
    return np.select([d <= t[0], d <= t[1], d <= t[2], d <= t[3]], [4, 3, 2, 1], 0)

# 1. faults >= 500 m, four azimuth families, intersection nodes merged within 150 m
g = gpd.read_file(FAULTS)
g = g[g.NOM_CLASSE.isin(['Faille (dont secondaire)', 'Faille, type non renseigné', 'Faille majeure', 'Faille supposée'])].explode(index_parts=False)
g = g[g.geom_type == 'LineString'].copy(); g['L'] = g.length
g['az'] = g.geometry.apply(lambda l: np.degrees(np.arctan2(l.coords[-1][0] - l.coords[0][0], l.coords[-1][1] - l.coords[0][1])) % 180)
Fw = g[(g.L >= 500) & g.intersects(box(LEFT, BOT, RIGHT, TOP).buffer(20000))]
D = {k: dist(list(Fw[Fw.az.apply(fam) == k].geometry)) for k in ['NS', 'N30', 'N70', 'N110_150']}
L = list(Fw.geometry); t = STRtree(L); pts = []
for i, a in enumerate(L):
    for j in t.query(a.buffer(150)):
        if j <= i: continue
        d = a.distance(L[j])
        if d > 150: continue
        if d == 0:
            x = a.intersection(L[j]); ps = [p for p in (x.geoms if hasattr(x, 'geoms') else [x]) if p.geom_type == 'Point']
        else:
            p1, p2 = nearest_points(a, L[j]); ps = [Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)]
        pts += [(p.x, p.y) for p in ps]
P = np.array(pts); cl = fcluster(linkage(P, 'single'), 150, 'distance'); N = np.array([P[cl == k].mean(0) for k in np.unique(cl)])
ok = (N[:, 0] > LEFT - 5000) & (N[:, 0] < RIGHT + 5000) & (N[:, 1] > BOT - 5000) & (N[:, 1] < TOP + 5000)
print('faults', len(Fw), 'nodes', int(ok.sum()))
D['nodes'] = dist([Point(*n) for n in N[ok]])

# 2. granites and host outcrops (Air / Tim Mersoi map)
geo = gpd.read_file(GEOL_TM); T = geo.TITRE.fillna('')
gran = shapely.union_all(geo[geo.GROUPE.isin(['03_Granites_Older', '04_Younger_Granites'])].geometry.values)
gran_km = dist([gran]) / 1000
carb = T.str.startswith(('Tarat', 'Guezouman', 'Madaouela', 'Tagora sup'))
hm = lambda sel: features.rasterize([(x, 1) for x in geo[sel.values].geometry], out_shape=shp, transform=tr, all_touched=False, fill=0).astype(bool)
d_carb = ndi.distance_transform_edt(~hm(carb)) * R

# 3. binary host rule on the regional map (computed on the large grid, then cropped to the window)
gr = gpd.read_file(GEOL_REG)
def cls(r):
    t_, c = str(r.TITRE), int(r.CLASSE)
    if c >= 7: return 0
    if c == 6: return 1 if (t_.startswith('Tarat') or t_.startswith('Tagora sup')) else 0
    if c == 1: return 9          # cover: takes the class of the nearest bedrock unit
    return 1
gr['F'] = gr.apply(cls, axis=1)
big_tr, big_shp = from_origin(240000, 2200000, R, R), (2800, 2100)
Lb = features.rasterize([(geom, int(f) + 1) for geom, f in zip(gr.geometry, gr.F)], out_shape=big_shp, transform=big_tr, fill=0, dtype='uint8')
cov = (Lb == 10) | (Lb == 0); _, (ii, jj) = ndi.distance_transform_edt(cov, return_indices=True)
Fb = (np.where(cov, Lb[ii, jj], Lb) == 2).astype(int)[1100:1100 + shp[0], 0:shp[1]]
print('host class 0 (% of window):', round((1 - Fb.mean()) * 100, 1))

# 4. reclassified scores and indices
Rc = {'N110': rc(D['N110_150'], (250, 500, 1000, 2000)), 'NS': rc(D['NS'], (250, 500, 1000, 2000)), 'N70': rc(D['N70'], (250, 500, 1000, 2000)),
      'N30': rc(D['N30'], (250, 500, 1000, 2000)), 'F3': rc(D['nodes'], (500, 1000, 2000, 4000)), 'F1': rc(gran_km, (20, 40, 60, 80))}
W_NEW = {'F4': 20.7089, 'N110': 32.1795, 'F3': 20.7089, 'N70': 11.2415, 'F1': 6.1131, 'NS': 6.1131, 'N30': 2.9349}
W_OLD = dict(F4=28.5, N110=28.5, F3=18.3, N70=10.4, F1=5.8, NS=5.8, N30=2.8)
full = (sum(W_NEW[k] * Rc[k] for k in W_NEW if k != 'F4') + W_NEW['F4'] * 4 * Fb) / sum(W_NEW.values())
rc_f4d = np.select([d_carb <= 1000, d_carb <= 3000, d_carb <= 6000], [4, 3, 2], 1)
prev = (sum(W_OLD[k] * Rc[k] for k in W_OLD if k != 'F4') + W_OLD['F4'] * rc_f4d) / sum(W_OLD.values())
Ws = {k: W_NEW[k] for k in ('N110', 'F3', 'N70', 'NS', 'N30')}; struct = sum(Ws[k] * Rc[k] for k in Ws) / sum(Ws.values())
prof = dict(driver='GTiff', dtype='float32', nodata=-9999.0, width=shp[1], height=shp[0], count=1, crs='EPSG:30732', transform=tr, compress='deflate')
for arr, fn in ((full, 'A4_full_index_binary_host_revised_weights_100m.tif'), (prev, 'A4_previous_full_index_100m.tif'), (struct, 'A4_structural_subindex_100m.tif')):
    with rasterio.open(os.path.join(OUT, fn), 'w', **prof) as w: w.write(arr.astype('float32'), 1)
print('written to', OUT)
