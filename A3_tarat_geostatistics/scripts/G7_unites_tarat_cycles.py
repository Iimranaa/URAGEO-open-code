# -*- coding: utf-8 -*-
"""G7 · Unités du Tarat (U1 à U4/Madaouela) : épaisseurs, validation croisée RMSE et MAE, lecture séquentielle (cycles transgressifs / régressifs)
Figures produites : G7-01 (carte FR/EN/RU : épaisseur des quatre unités, proportion de l'unité sommitale, lecture séquentielle), G7-02 (RMSE/MAE par unité et par méthode)
Données : 09_CALCULS_STATISTIQUES_GEOSTATISTIQUES/01_Statistiques/Donnees/Tarat_731_donnees_calcul.csv (731 sondages)
          strati_units_synthetic.csv (equivalent of the STRATI table: intervalles des unités du Tarat : codes 310 = U1, 320 = U2, 330 = U3, 340/344 = U4 Madaouela)
Lecture séquentielle : à partir des milieux de dépôt de la colonne lithostratigraphique (AREVA NC Niger, 2009) ; ce sont des épaisseurs et une
interprétation proposée (à valider par l'encadrant), PAS une mesure de faciès sur sondages.
Hypothèse : une unité absente d'un sondage est exclue du calcul de cette unité (non remplacée par 0).
Lancer :  python G7_unites_tarat_cycles.py      (résultats dans 05_CARTES_FINALES/_recalcul_scripts/ ; voir commun_calculs.py)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from commun_calculs import *
from scipy import spatial
from scipy.optimize import curve_fit
from matplotlib.patches import FancyArrowPatch
d = load(); n0 = len(d)
st = pd.read_csv(STRATI_CSV)   # columns: hole_id, code, from_m, to_m (codes 310/320/330/340/344)
st['t'] = st.to_m.astype(float) - st.from_m.astype(float)
UN = {'U1': ['310'], 'U2': ['320'], 'U3': ['330'], 'U4': ['340', '344']}
for k, codes in UN.items():
    d[k] = d.hole_id.map(st[st.code.astype(str).isin(codes)].groupby('hole_id').t.sum())
d['P_U4'] = d.U4 / d.puiss_m
print('présence des unités :', {k: int(d[k].notna().sum()) for k in UN})
# ---- outils de validation croisée (mêmes méthodes que G5)
def sph(h, c0, c, a): h = np.asarray(h, float); r_ = np.minimum(h / a, 1); return np.where(h > 0, c0 + c * (1.5 * r_ - .5 * r_ ** 3), 0)
def expo(h, c0, c, a): h = np.asarray(h, float); return np.where(h > 0, c0 + c * (1 - np.exp(-3 * h / a)), 0)
def gaus(h, c0, c, a): h = np.asarray(h, float); return np.where(h > 0, c0 + c * (1 - np.exp(-3 * (h / a) ** 2)), 0)
MODELS = {'sphérique': sph, 'exponentiel': expo, 'gaussien': gaus}
def evario(z, XY, lag=100, maxd=1500):
    DM = spatial.distance.cdist(XY, XY); iu = np.triu_indices(len(z), 1); dd = DM[iu]; g = 0.5 * (z[iu[0]] - z[iu[1]]) ** 2
    b = np.floor(dd / lag).astype(int); ok = dd <= maxd; b = b[ok]; g = g[ok]; dd = dd[ok]; nb = b.max() + 1
    N = np.bincount(b, minlength=nb); G = np.bincount(b, weights=g, minlength=nb) / np.maximum(N, 1); H = np.bincount(b, weights=dd, minlength=nb) / np.maximum(N, 1); m = N >= 30
    return H[m], G[m], N[m]
def fit(H, G, N):
    best = None; v = G.max()
    for nm, f in MODELS.items():
        try: p, _ = curve_fit(f, H, G, p0=[G[0] * .5, max(v - G[0] * .5, 1e-9), 500], sigma=G / np.sqrt(N), bounds=([0, 1e-12, 40], [v * 2, v * 3, 4000]), maxfev=20000)
        except Exception: continue
        sse = np.sum(N * ((G - f(H, *p)) / np.maximum(G, 1e-12)) ** 2)
        if best is None or sse < best[3]: best = (nm, f, p, sse)
    return best
def ok_pred(XY, z, f, p, T, nn=24, excl=False):
    tree = spatial.cKDTree(XY); k = nn + (1 if excl else 0); dist, ix = tree.query(T, k=k)
    if excl: dist = dist[:, 1:]; ix = ix[:, 1:]
    m = T.shape[0]; P = XY[ix]; DD = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=3); Gm = f(np.maximum(DD, 1e-6), *p); Gm[:, np.arange(nn), np.arange(nn)] = 0
    A = np.ones((m, nn + 1, nn + 1)); A[:, :nn, :nn] = Gm; A[:, nn, nn] = 0; b = np.ones((m, nn + 1)); b[:, :nn] = f(np.maximum(dist, 1e-6), *p)
    s = np.linalg.solve(A, b[:, :, None])[:, :, 0]; return (s[:, :nn] * z[ix]).sum(1)
def idw(XY, z, T, pw=2, nn=12, excl=False):
    tree = spatial.cKDTree(XY); k = nn + (1 if excl else 0); dist, ix = tree.query(T, k=k)
    if excl: dist = dist[:, 1:]; ix = ix[:, 1:]
    w = 1 / np.maximum(dist, 1e-6) ** pw; return (w * z[ix]).sum(1) / w.sum(1)
LABS = {'U1': 'Unité U1', 'U2': 'Unité U2', 'U3': 'Unité U3', 'U4': 'Unité U4 (Madaouela)', 'P_U4': 'Part de U4 dans l’épaisseur du Tarat'}
rows = []; MOD = {}
for k, l in LABS.items():
    s_ = d.dropna(subset=[k]); XY = s_[['x', 'y']].values; z = s_[k].values.astype(float); n = len(z)
    H, G, N = evario(z, XY); nm, f, p, _ = fit(H, G, N); MOD[k] = (nm, f, p)
    DM = spatial.distance.cdist(XY, XY)
    pr = {'moyenne': np.array([(z.sum() - z[i]) / (n - 1) for i in range(n)]), 'plus proche voisin': z[np.argsort(DM + np.eye(n) * 1e9, axis=1)[:, 0]],
          'IDW (p=2, 12 pts)': idw(XY, z, XY, excl=True), 'krigeage ordinaire': ok_pred(XY, z, f, p, XY, excl=True)}
    for mt, v in pr.items():
        e = v - z; rm = np.sqrt((e ** 2).mean()); rows.append(dict(variable=l, cle=k, n=n, methode=mt, ME=e.mean(), MAE=np.abs(e).mean(), RMSE=rm, gain=1 - rm ** 2 / z.var(), modele=nm))
CV = pd.DataFrame(rows); CV.to_csv(OUT + 'res/G7_validation_croisee_unites_tarat.csv', index=False); print(CV.round(3).to_string())
# tableau RMSE/MAE commun (G5 + G7) pour l'annexe
try:
    g5 = pd.read_csv(OUT + 'res/G5_validation_croisee.csv'); g5['n'] = n0; g5['gain'] = g5['skill']; g5['modele'] = ''
    pd.concat([g5[['variable', 'cle', 'n', 'methode', 'ME', 'MAE', 'RMSE', 'gain']], CV[['variable', 'cle', 'n', 'methode', 'ME', 'MAE', 'RMSE', 'gain']]]).to_csv(OUT + 'res/G7_tableau_RMSE_MAE_complet.csv', index=False)
except FileNotFoundError:
    print('G5_validation_croisee.csv absent : lancer G5 avant G7 pour le tableau complet')
# ---- figure RMSE/MAE
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6)); keys = list(LABS); meths = ['moyenne', 'plus proche voisin', 'IDW (p=2, 12 pts)', 'krigeage ordinaire']; cc = [GREY, '#B8C2D6', CU, NAVY]
for a, met in zip(ax, ['RMSE', 'MAE']):
    for i, (m_, c_) in enumerate(zip(meths, cc)):
        a.bar(np.arange(len(keys)) + i * 0.2 - 0.3, [CV[(CV.cle == k) & (CV.methode == m_)][met].values[0] if k != 'P_U4' else CV[(CV.cle == k) & (CV.methode == m_)][met].values[0] * 100 for k in keys], 0.2, color=c_, label=m_)
    a.set_xticks(range(len(keys))); a.set_xticklabels(['U1', 'U2', 'U3', 'U4', 'Part U4 (× 100)']); a.set_ylabel(met + ' (m ; part en points de %)'); a.set_title(met + ' en validation croisée (un sondage retiré à la fois)')
ax[0].legend(frameon=False, fontsize=8); fig.suptitle('G7-02 · Erreur de prédiction de l’épaisseur des unités du Tarat', fontweight='bold'); fig.tight_layout(); save(fig, 'G7_Unites_Tarat', 'G7-02_RMSE_MAE_unites_Tarat')
# ---- cartes (FR/EN/RU) : IDW sur grille 50 m, limité à 1 km des sondages
T3 = {
 'fr': dict(t='G7-01 · Unités du Tarat : épaisseur, proportion de U4 et lecture séquentielle', e='Épaisseur (m)', p='Part de U4 (%)', x='Est (m)', y='Nord (m)', n='n = ', cap='Interpolation IDW (p = 2), limitée à 1 km des sondages. Une unité absente d’un sondage est exclue (non mise à 0).',
            col='Lecture séquentielle (colonne AREVA 2009) : proposition à valider', env=[('Arlit', 'éolien', 'continentalisation'), ('Madaouela (U4)', 'estuarien', 'TRANSGRESSION'), ('Tarat (U1-U3)', 'fluvio-deltaïque', 'RÉGRESSION (progradation)'), ('Tchinezogue', 'marin / lacustre', 'TRANSGRESSION'), ('Guezouman', 'fluvio-estuarien', '')], up='haut', dn='bas', note='Épaisseurs mesurées ; transgression/régression déduites des milieux de la colonne, non de faciès sur sondages.'),
 'en': dict(t='G7-01 · Tarat units: thickness, share of U4 and sequence reading', e='Thickness (m)', p='Share of U4 (%)', x='Easting (m)', y='Northing (m)', n='n = ', cap='IDW interpolation (p = 2), limited to 1 km from boreholes. A unit absent from a borehole is excluded (not set to 0).',
            col='Sequence reading (AREVA 2009 column): proposal to be validated', env=[('Arlit', 'aeolian', 'continentalisation'), ('Madaouela (U4)', 'estuarine', 'TRANSGRESSION'), ('Tarat (U1-U3)', 'fluvio-deltaic', 'REGRESSION (progradation)'), ('Tchinezogue', 'marine / lacustrine', 'TRANSGRESSION'), ('Guezouman', 'fluvio-estuarine', '')], up='top', dn='base', note='Thicknesses are measured; transgression/regression is inferred from the environments of the column, not from facies in boreholes.'),
 'ru': dict(t='G7-01 · Единицы тарата: мощность, доля U4 и секвенсная интерпретация', e='Мощность (м)', p='Доля U4 (%)', x='X, восток (м)', y='Y, север (м)', n='n = ', cap='Интерполяция IDW (p = 2), не далее 1 км от скважин. Единица, отсутствующая в скважине, исключается (не принимается за 0).',
            col='Секвенсная интерпретация (колонка AREVA 2009): предложение для проверки', env=[('Арлит', 'эоловые', 'континентализация'), ('Мадауэла (U4)', 'эстуарные', 'ТРАНСГРЕССИЯ'), ('Тарат (U1–U3)', 'флювиально-дельтовые', 'РЕГРЕССИЯ (проградация)'), ('Чинезогуэ', 'морские / озёрные', 'ТРАНСГРЕССИЯ'), ('Гуэзоумен', 'флювиально-эстуарные', '')], up='кровля', dn='подошва', note='Мощности измерены; трансгрессия/регрессия выведены из обстановок колонки, а не из фаций по скважинам.')}
xs = np.arange(d.x.min() - 500, d.x.max() + 500, 50); ys = np.arange(d.y.min() - 500, d.y.max() + 500, 50); GX, GY = np.meshgrid(xs, ys); TT = np.column_stack([GX.ravel(), GY.ravel()])
tree_all = spatial.cKDTree(d[['x', 'y']].values); near = tree_all.query(TT)[0].reshape(GX.shape) <= 1000
def surf(k, scale=1.0):
    s_ = d.dropna(subset=[k]); XY = s_[['x', 'y']].values; z = s_[k].values.astype(float) * scale
    return np.where(near, idw(XY, z, TT).reshape(GX.shape), np.nan)
S = {k: surf(k) for k in ['U1', 'U2', 'U3', 'U4']}; S['P_U4'] = surf('P_U4', 100)
for lg, L in T3.items():
    fig = plt.figure(figsize=(17, 10.5)); gs = fig.add_gridspec(2, 3, wspace=0.28, hspace=0.3)
    for i, (k, nm) in enumerate([('U1', 'U1'), ('U2', 'U2'), ('U3', 'U3'), ('U4', 'U4 (Madaouela)'), ('P_U4', None)]):
        a = fig.add_subplot(gs[i // 3, i % 3]); A = S[k]; lo, hi = np.nanpercentile(A, [2, 98])
        im = a.pcolormesh(GX, GY, A, cmap='YlGnBu' if k != 'P_U4' else 'PuOr_r', vmin=lo, vmax=hi, shading='auto')
        s_ = d.dropna(subset=[k]); a.scatter(s_.x, s_.y, s=1.5, color='k', alpha=0.35); a.set_aspect('equal')
        a.set_title((nm if nm else L['p']) + '   ' + L['n'] + str(len(s_)), fontsize=11); a.set_xlabel(L['x'], fontsize=8); a.set_ylabel(L['y'], fontsize=8); a.tick_params(labelsize=7)
        cb = fig.colorbar(im, ax=a, shrink=0.75); cb.set_label(L['e'] if k != 'P_U4' else L['p'], fontsize=8)
    a = fig.add_subplot(gs[1, 2]); a.axis('off'); a.set_xlim(0, 10); a.set_ylim(0, 10); a.set_title(L['col'], fontsize=10, fontweight='bold')
    cols = ['#F2D9A8', '#9CC3D5', '#E3B872', '#6FA0C0', '#C9BBA0']
    for j, (fm, env, ev) in enumerate(L['env']):
        y0 = 8.2 - j * 1.7; a.add_patch(plt.Rectangle((0.2, y0), 5.3, 1.4, color=cols[j], ec='k')); a.text(0.4, y0 + 0.7, fm + '\n' + env, va='center', fontsize=9)
        if ev: a.text(5.7, y0 + 0.7, ev, va='center', fontsize=9, color=RED if 'TRANS' in ev.upper() or 'ТРАНС' in ev.upper() else NAVY, fontweight='bold')
    a.annotate('', xy=(9.8, 9.5), xytext=(9.8, 0.8), arrowprops=dict(arrowstyle='->', color=GREY)); a.text(9.6, 9.6, L['up'], fontsize=8, ha='right'); a.text(9.6, 0.3, L['dn'], fontsize=8, ha='right')
    fig.suptitle(L['t'], fontweight='bold', fontsize=13); fig.text(0.01, 0.01, L['cap'] + '\n' + L['note'], fontsize=8.5, color=GREY)
    save(fig, 'G7_Unites_Tarat', 'G7-01_Unites_Tarat_cycles_' + lg.upper())
# ---- corrélations exploratoires unités du Tarat / indicateurs d'oxydoréduction (Spearman ; p non corrigées de l'autocorrélation spatiale)
from scipy import stats
cr = []
for v in ['acc_oxy', 'acc_sulf', 'acc_mo', 'PC1_calc']:
    for k in ['U1', 'U2', 'U3', 'U4', 'P_U4']:
        s_ = d[[v, k]].dropna(); rho, p_ = stats.spearmanr(s_[v], s_[k]); cr.append(dict(variable=v, unite=k, n=len(s_), rho_spearman=rho, p_brute=p_))
pd.DataFrame(cr).to_csv(OUT + 'res/G7_correlations_unites_redox.csv', index=False); print(pd.DataFrame(cr).round(3).to_string())
