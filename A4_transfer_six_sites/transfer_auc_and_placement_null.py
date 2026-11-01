# -*- coding: utf-8 -*-
"""A4 · Transferability of a structural index to six sites: AUC, site bootstrap and placement null model.

Index: weighted sum of illustrative 1-5 classes derived from fault-family and node distances (weights as in the model:
N110-150 28.5, NS 5.8, N70 10.4, N30 2.8, nodes (F3) 18.3). For each site: AUC of the index (site cells vs background cells
of the host zone), mean index, and a placement null model: the site footprint is translated over a grid of offsets, excluding
offsets that overlap any site or leave the host zone, and the observed mean index is compared with the translated means.
Holm correction over all (index x site) tests.

Runs on the synthetic layers by default. Output: outputs/A4/transfer_results.json
Usage:  python A4_transfer_six_sites/transfer_auc_and_placement_null.py
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import numpy as np
from gis_tools import *

D = load_layers(); S = load_sites()
WTS = {'N110-150': 28.5, 'NS': 5.8, 'N70': 10.4, 'N30': 2.8, 'Nodes': 18.3}
Rc = {k: reclass(D[k]) for k in WTS}
structural = sum(WTS[k] * Rc[k] for k in WTS) / sum(WTS.values())
host_bonus = np.where(D['host'] == 0, 1.0, 0.0)                      # illustrative stand-in for the host-continuity factor
index = {'structural': structural, 'structural+host': 0.85 * structural + 0.15 * 5 * host_bonus}
zone = D['host'] <= 6000

# six "sites": pits of the main area, old workings, and four external circles (here fictitious)
sites = {'Pits': S[S.kind == 'pit'], 'Workings': S[S.kind == 'working']}
for n in ['E1', 'E2', 'E3', 'E4']: sites[n] = S[S.sector == n]
M = {k: mask_of(v) for k, v in sites.items()}
anyS = np.zeros((H, W), bool)
for m in M.values(): anyS |= m

res = {'sizes': {k: int(v.sum()) for k, v in M.items()}, 'auc': {}}
for nm, I in index.items():
    res['auc'][nm] = {s: round(auc(I, m, (~anyS) & zone), 2) for s, m in M.items()}
    res['auc'][nm]['pooled'] = round(auc(I, anyS, (~anyS) & zone), 2)
rng = np.random.default_rng(3); names = list(M); bs = []
for _ in range(500):                                                  # site bootstrap of the pooled AUC
    pick = rng.choice(names, len(names)); pos = np.zeros((H, W), bool)
    for s in pick: pos |= M[s]
    bs.append(auc(index['structural'], pos, (~pos) & zone))
res['pooled_auc_bootstrap_95'] = [float(np.percentile(bs, 2.5)), float(np.median(bs)), float(np.percentile(bs, 97.5))]

def placement_null(I, m, rng_km=15, step=10):
    yy, xx = np.nonzero(m); obs = I[m].mean(); vals = []; n = int(rng_km * 10)
    for dy in range(-n, n + 1, step):
        for dx in range(-n, n + 1, step):
            y2 = yy + dy; x2 = xx + dx
            if y2.min() < 0 or y2.max() >= H or x2.min() < 0 or x2.max() >= W: continue
            if anyS[y2, x2].any() or zone[y2, x2].mean() < 0.8: continue
            vals.append(I[y2, x2].mean())
    v = np.array(vals)
    if len(v) < 10: return dict(obs=float(obs), n=int(len(v)), note='fewer than 10 admissible offsets: test not performed', p=None)
    return dict(obs=float(obs), n=int(len(v)), null_mean=float(v.mean()), null_sd=float(v.std()), z=float((obs - v.mean()) / v.std()),
                p=float((np.sum(v >= obs) + 1) / (len(v) + 1)))
res['null'] = {nm: {s: placement_null(I, m) for s, m in M.items()} for nm, I in index.items()}
ps = {f'{nm}|{s}': d['p'] for nm in res['null'] for s, d in res['null'][nm].items() if d['p'] is not None}
res['holm'] = {k: dict(p=v[0], threshold=v[1], reject=v[2]) for k, v in holm(ps).items()}
o = OUT / 'A4'; o.mkdir(parents=True, exist_ok=True)
json.dump(res, open(o / 'transfer_results.json', 'w'), indent=1, default=float)
print('AUC', res['auc']['structural']); print('pooled bootstrap 95%', [round(x, 2) for x in res['pooled_auc_bootstrap_95']])
print({s: ((round(d['z'], 2), round(d['p'], 3)) if d['p'] is not None else 'not tested', d['n']) for s, d in res['null']['structural'].items()})
