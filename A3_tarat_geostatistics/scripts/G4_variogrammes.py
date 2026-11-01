# -*- coding: utf-8 -*-
"""G4 · Variogrammes expérimentaux, modèles, directions, effet de la tendance
Figures produites : G4-01 à G4-04
Données : 09_CALCULS_STATISTIQUES_GEOSTATISTIQUES/01_Statistiques/Donnees/Tarat_731_donnees_calcul.csv (731 sondages).
Lancer :  python G4_variogrammes.py      (résultats dans 05_CARTES_FINALES/_recalcul_scripts/ ; voir commun_calculs.py)
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
# --- Variograms
def evario(z,lag=100,maxd=1500,mask=None):
    dd=DM[iu]; g=0.5*(z[iu[0]]-z[iu[1]])**2
    if mask is not None: dd=dd[mask]; g=g[mask]
    b=np.floor(dd/lag).astype(int); ok=dd<=maxd; b=b[ok]; g=g[ok]; dd=dd[ok]; nb=b.max()+1
    N=np.bincount(b,minlength=nb); G=np.bincount(b,weights=g,minlength=nb)/np.maximum(N,1); H=np.bincount(b,weights=dd,minlength=nb)/np.maximum(N,1); m=N>=30; return H[m],G[m],N[m]
def sph(h,c0,c,a): h=np.asarray(h,float); r=np.minimum(h/a,1); return np.where(h>0,c0+c*(1.5*r-.5*r**3),0)
def expo(h,c0,c,a): h=np.asarray(h,float); return np.where(h>0,c0+c*(1-np.exp(-3*h/a)),0)
def gaus(h,c0,c,a): h=np.asarray(h,float); return np.where(h>0,c0+c*(1-np.exp(-3*(h/a)**2)),0)
MODELS={'sphérique':sph,'exponentiel':expo,'gaussien':gaus}
def fit(H,G,N):
    best=None; v=G.max()
    for nm,f in MODELS.items():
        try:
            p,_=curve_fit(f,H,G,p0=[G[0]*.5,max(v-G[0]*.5,1e-9),500],sigma=G/np.sqrt(N),bounds=([0,1e-12,40],[v*2,v*3,4000]),maxfev=20000)
        except Exception as e: continue
        sse=np.sum(N*((G-f(H,*p))/np.maximum(G,1e-12))**2)
        if best is None or sse<best[3]: best=(nm,f,p,sse)
    return best
VM={}; fig,ax=plt.subplots(2,3,figsize=(15,8.5))
for a,(k,l) in zip(ax.ravel(),VARS.items()):
    z=d[k].values.astype(float); H,G,N=evario(z); nm,f,p,sse=fit(H,G,N); VM[k]=dict(model=nm,c0=float(p[0]),c=float(p[1]),a=float(p[2]),sill=float(p[0]+p[1]),var=float(z.var(ddof=1)),ratio=float(p[0]/(p[0]+p[1])))
    a.scatter(H,G,s=np.clip(N/40,8,70),color=NAVY,zorder=3); hh=np.linspace(1,1500,200); a.plot(hh,f(hh,*p),color=CU,lw=2); a.axhline(z.var(ddof=1),color=GREY,ls=':')
    a.set_title(f'{l}\n{nm} · pépite {p[0]:.3g} · palier {p[0]+p[1]:.3g} · portée {p[2]:.0f} m\npépite/palier = {100*p[0]/(p[0]+p[1]):.0f} %',fontsize=9); a.set_xlabel('distance h (m)'); a.set_ylabel('γ(h)'); a.set_ylim(0,None)
fig.suptitle('G4-02 · Variogrammes expérimentaux (points ; taille = nombre de paires) et modèles ajustés (ligne) ; pointillé = variance totale',fontweight='bold'); fig.tight_layout(); save(fig,'G4_Variogrammes','G4-02_Variogrammes_six_variables'); RES['vario']=VM; print(pd.DataFrame(VM).T.round(4))
# anatomy
z=d.puiss_m.values; H,G,N=evario(z); nm,f,p,sse=fit(H,G,N); fig,a=plt.subplots(figsize=(9,5.5)); a.scatter(H,G,color=NAVY,zorder=3,s=30); hh=np.linspace(1,1500,300); a.plot(hh,f(hh,*p),color=CU,lw=2.5)
a.axhline(p[0]+p[1],color=GREY,ls='--'); a.axvline(p[2],color=GREY,ls='--'); a.annotate('',xy=(0,p[0]),xytext=(0,0)); a.scatter([0],[p[0]],color=RED,zorder=5,s=60)
a.annotate(f'PÉPITE c₀ = {p[0]:.1f} m²\n(variabilité à très courte distance\n+ erreurs de mesure)',xy=(0,p[0]),xytext=(150,p[0]*.35),arrowprops=dict(arrowstyle='->'),fontsize=9)
a.annotate(f'PALIER c₀+c = {p[0]+p[1]:.1f} m²',xy=(1200,p[0]+p[1]),xytext=(850,(p[0]+p[1])*.8),arrowprops=dict(arrowstyle='->'),fontsize=9)
a.annotate(f'PORTÉE a = {p[2]:.0f} m\n(au-delà : sondages indépendants)',xy=(min(p[2],1400),(p[0]+p[1])*.55),xytext=(min(p[2],1400)+60,(p[0]+p[1])*.3),fontsize=9)
a.set_xlabel('distance h entre deux sondages (m)'); a.set_ylabel('γ(h) = ½ · moyenne de [z(x) − z(x+h)]²'); a.set_ylim(0,None); a.set_title(f'G4-01 · Anatomie d\'un variogramme (épaisseur du Tarat, modèle {nm})')
fig.text(.01,-.03,'Formule : γ(h) = 1 / (2 N(h)) · Σ [z(xᵢ) − z(xᵢ + h)]²   sur les N(h) paires séparées de la distance h.',fontsize=8,color=GREY); fig.tight_layout(); save(fig,'G4_Variogrammes','G4-01_Anatomie_du_variogramme')
# directional
def dirmask(az,tol=22.5):
    dx=XY[iu[1],0]-XY[iu[0],0]; dy=XY[iu[1],1]-XY[iu[0],1]; ang=(np.degrees(np.arctan2(dx,dy))%180); diff=np.abs((ang-az+90)%180-90); return diff<=tol
fig,ax=plt.subplots(1,3,figsize=(15,4.8)); DIR={}
for a,k in zip(ax,['puiss_m','toit_m','PC1']):
    z=d[k].values.astype(float)
    for az,cc in zip([0,45,90,135],[NAVY,CU,TEAL,RED]):
        H,G,N=evario(z,lag=150,maxd=1500,mask=dirmask(az)); a.plot(H,G,'o-',color=cc,label=f'{az}° (N={az}°)' if az==0 else f'{az}°',ms=4); DIR[f'{k}_{az}']=dict(H=H.tolist(),G=G.tolist())
    a.set_title(VARS[k]); a.set_xlabel('h (m)'); a.set_ylabel('γ(h)'); a.legend(frameon=False,fontsize=8,title='azimut',title_fontsize=8)
fig.suptitle('G4-03 · Variogrammes directionnels : même portée dans toutes les directions = isotrope ; sinon anisotrope (tolérance ±22,5°)',fontweight='bold',fontsize=10); fig.tight_layout(); save(fig,'G4_Variogrammes','G4-03_Variogrammes_directionnels'); RES['dir']=DIR
# detrended thickness variogram
fig,a=plt.subplots(figsize=(7,4.8))
for v_,lab_,cc in [(d.puiss_m.values,'épaisseur brute',NAVY),(detrend(d.puiss_m.values),'résidu de tendance quadratique',CU)]:
    H,G,N=evario(v_); a.plot(H,G,'o-',color=cc,label=lab_)
a.set_xlabel('h (m)'); a.set_ylabel('γ(h)'); a.legend(frameon=False); a.set_title('G4-04 · Effet de la tendance Est-Ouest sur le variogramme de l\'épaisseur'); fig.tight_layout(); save(fig,'G4_Variogrammes','G4-04_Effet_tendance_epaisseur')
savejson('G4_variogrammes.json')
