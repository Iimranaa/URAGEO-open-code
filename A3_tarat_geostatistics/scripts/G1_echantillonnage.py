# -*- coding: utf-8 -*-
"""G1 · Géométrie de l’échantillonnage (maille, distances entre paires)
Figures produites : G1-01
Données : 09_CALCULS_STATISTIQUES_GEOSTATISTIQUES/01_Statistiques/Donnees/Tarat_731_donnees_calcul.csv (731 sondages).
Lancer :  python G1_echantillonnage.py      (résultats dans 05_CARTES_FINALES/_recalcul_scripts/ ; voir commun_calculs.py)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from commun_calculs import *
from scipy import stats, spatial
from scipy.optimize import curve_fit
d = load(); n = len(d)
d['PC1'] = d.PC1_calc; XY = d[['x','y']].values; rng = np.random.default_rng(11)
d['PC1']=d.PC1_calc; XY=d[['x','y']].values; rng=np.random.default_rng(11)
VARS={'puiss_m':'Épaisseur Tarat (m)','toit_m':'Profondeur du toit (m)','PC1':'Score PC1 (sulfures + MO)','acc_mo':'Matière organique','acc_sulf':'Sulfures','acc_oxy':'Oxy-hydroxydes'}
DM=spatial.distance.cdist(XY,XY); iu=np.triu_indices(n,1); print('dupes',(DM[iu]==0).sum())
nn1=np.sort(DM+np.eye(n)*1e9,axis=1)[:,0]; print('nn dist',np.percentile(nn1,[5,50,95]))
# G1
fig,ax=plt.subplots(1,3,figsize=(15,4.8)); a=ax[0]; a.scatter(d.x,d.y,s=5,color=NAVY); a.set_aspect('equal'); a.set_title(f'{n} sondages, emprise {np.ptp(d.x)/1000:.1f} × {np.ptp(d.y)/1000:.1f} km'); a.set_xlabel('X (m)'); a.set_ylabel('Y (m)')
a=ax[1]; a.hist(nn1,bins=40,color=CU,edgecolor='white'); a.axvline(np.median(nn1),color='k',ls='--'); a.set_xlabel('distance au sondage le plus proche (m)'); a.set_title(f'Maillage : médiane {np.median(nn1):.0f} m')
a=ax[2]; a.hist(DM[iu],bins=60,color=TEAL,edgecolor='white'); a.axvline(1500,color=RED,ls='--'); a.set_xlabel('distance entre paires de sondages (m)'); a.set_title('Distances entre paires (limite du variogramme : 1 500 m)')
fig.suptitle('G1-01 · Géométrie de l\'échantillonnage : ce qui borne les calculs géostatistiques',fontweight='bold'); fig.tight_layout(); save(fig,'G1_Echantillonnage','G1-01_Geometrie_echantillonnage')
RES['g1']=dict(n=n,nn_med=float(np.median(nn1)),nn_p95=float(np.percentile(nn1,95)),dupes=int((DM[iu]==0).sum()))
savejson('G1_echantillonnage.json')
