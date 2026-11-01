# -*- coding: utf-8 -*-
"""G2 · Autocorrélation spatiale : I de Moran global, diagramme de Moran, corrélogramme
Figures produites : G2-01 à G2-03
Données : 09_CALCULS_STATISTIQUES_GEOSTATISTIQUES/01_Statistiques/Donnees/Tarat_731_donnees_calcul.csv (731 sondages).
Lancer :  python G2_moran.py      (résultats dans 05_CARTES_FINALES/_recalcul_scripts/ ; voir commun_calculs.py)
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
# --- Moran
K=8; idxk=np.argsort(DM+np.eye(n)*1e9,axis=1)[:,:K]
W=np.zeros((n,n)); W[np.repeat(np.arange(n),K),idxk.ravel()]=1/K
def moran(z,W,perm=999):
    z=z-z.mean(); I=(z@W@z)/(z@z); sims=np.empty(perm)
    for i in range(perm): zp=rng.permutation(z); sims[i]=(zp@W@zp)/(zp@zp)
    p=(1+(np.abs(sims)>=abs(I)).sum())/(perm+1); return I,p,(I-sims.mean())/sims.std(),sims
def detrend(v):
    xs=(d.x-d.x.mean())/1000; ys=(d.y-d.y.mean())/1000; M=np.column_stack([np.ones(n),xs,ys,xs**2,xs*ys,ys**2]); b=np.linalg.lstsq(M,v,rcond=None)[0]; return v-M@b
rows=[]; 
for k,l in VARS.items():
    v=d[k].values.astype(float); I,p,z,_=moran(v,W); Ir,pr,zr,_=moran(detrend(v),W); rows.append(dict(variable=l,I=I,z=z,p=p,I_residus_tendance=Ir,z_res=zr,p_res=pr))
M=pd.DataFrame(rows); M.to_csv(OUT+'res/G2_Moran.csv',index=False); print(M.round(3).to_string()); RES['moran']=M.to_dict('records'); RES['moran_E']=-1/(n-1)
fig,a=plt.subplots(figsize=(9.5,4.8)); x=np.arange(len(M)); a.bar(x-.2,M.I,.4,color=NAVY,label='variable brute'); a.bar(x+.2,M.I_residus_tendance,.4,color=CU,label='après retrait d\'une tendance quadratique X-Y')
for i,r in M.iterrows(): a.text(i-.2,r.I+.01,f'{r.I:.2f}',ha='center',fontsize=8); a.text(i+.2,r.I_residus_tendance+.01,f'{r.I_residus_tendance:.2f}',ha='center',fontsize=8)
a.axhline(-1/(n-1),color='k',lw=.8,ls=':'); a.set_xticks(x); a.set_xticklabels([l.replace(' (','\n(') for l in M.variable],fontsize=8); a.set_ylabel("I de Moran (voisins = 8 plus proches)"); a.legend(frameon=False,fontsize=8)
a.set_title('G2-01 · Autocorrélation spatiale globale (999 permutations, tous p ≤ 0,005)',fontsize=10); fig.tight_layout(); save(fig,'G2_Moran','G2-01_Moran_global_par_variable')
fig,ax=plt.subplots(1,3,figsize=(15,4.8))
for a,k in zip(ax,['puiss_m','toit_m','PC1']):
    z=d[k].values-d[k].mean(); lag=W@z; sl,ic,r,_,_=stats.linregress(z,lag); a.scatter(z/z.std(),lag/z.std(),s=7,alpha=.5,color=NAVY); xs_=np.array([-3,3]); a.plot(xs_,sl*xs_,color=CU,lw=2)
    a.axhline(0,color='k',lw=.5); a.axvline(0,color='k',lw=.5); a.set_xlabel('valeur centrée-réduite'); a.set_ylabel('moyenne des 8 voisins'); a.set_title(f'{VARS[k]}\npente = I = {sl:.2f}')
fig.suptitle('G2-02 · Diagramme de Moran : la pente de la droite est l\'indice I',fontweight='bold'); fig.tight_layout(); save(fig,'G2_Moran','G2-02_Diagrammes_de_Moran')
# correlogram
edges=np.arange(0,1600,100); fig,ax=plt.subplots(1,4,figsize=(16,4),sharey=True); CG={}
for a,k in zip(ax,['puiss_m','toit_m','PC1','acc_mo']):
    z=d[k].values-d[k].mean(); Is=[];lo=[];hi=[];mid=[]
    for e0,e1 in zip(edges[:-1],edges[1:]):
        m=(DM>e0)&(DM<=e1); ii,jj=np.where(np.triu(m,1)); 
        if len(ii)<30: continue
        S0=len(ii)*2; I=(n/S0)*2*np.sum(z[ii]*z[jj])/(z@z); sims=[]
        for _ in range(199): zp=rng.permutation(z); sims.append((n/S0)*2*np.sum(zp[ii]*zp[jj])/(z@z))
        Is.append(I);lo.append(np.percentile(sims,2.5));hi.append(np.percentile(sims,97.5));mid.append((e0+e1)/2)
    a.fill_between(mid,lo,hi,color='#ddd',label='enveloppe du hasard (95 %)'); a.plot(mid,Is,'o-',color=NAVY,lw=2,label='I observé'); a.axhline(0,color='k',lw=.5); a.set_xlabel('distance (m)'); a.set_title(VARS[k],fontsize=9)
    CG[k]=dict(h=mid,I=Is,lo=lo,hi=hi)
ax[0].set_ylabel('I de Moran par classe de distance'); ax[0].legend(frameon=False,fontsize=7); fig.suptitle('G2-03 · Corrélogramme : jusqu\'à quelle distance les sondages se ressemblent-ils ?',fontweight='bold'); fig.tight_layout(); save(fig,'G2_Moran','G2-03_Correlogramme'); RES['correlogram']=CG
savejson('G2_moran.json')
