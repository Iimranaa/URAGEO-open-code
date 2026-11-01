# -*- coding: utf-8 -*-
"""G3 · Points chauds/froids Getis-Ord Gi* et Moran local (LISA)
Figures produites : G3-01, G3-02
Données : 09_CALCULS_STATISTIQUES_GEOSTATISTIQUES/01_Statistiques/Donnees/Tarat_731_donnees_calcul.csv (731 sondages).
Lancer :  python G3_gi_lisa.py      (résultats dans 05_CARTES_FINALES/_recalcul_scripts/ ; voir commun_calculs.py)
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
# --- Gi* and LISA
def gistar(v,band=250):
    Wb=(DM<=band).astype(float); S=Wb.sum(1); xb=v.mean(); s=np.sqrt((v**2).mean()-xb**2)
    z=(Wb@v-xb*S)/(s*np.sqrt((n*(Wb**2).sum(1)-S**2)/(n-1))); return z,S
def bh(p,q=.05):
    o=np.argsort(p); th=q*np.arange(1,len(p)+1)/len(p); ok=p[o]<=th; k=np.max(np.where(ok)[0]) if ok.any() else -1; rej=np.zeros(len(p),bool)
    if k>=0: rej[o[:k+1]]=True
    return rej
def gimap(a,v,title,band=250):
    z,S=gistar(v,band); p=2*(1-stats.norm.cdf(np.abs(z))); rej=bh(p); cls=np.where(rej&(z>0),2,np.where(rej&(z<0),0,1))
    col=np.array(['#2A5F9E','#D9D9D9','#B03A2E'])[cls]; a.scatter(d.x,d.y,c=col,s=9); a.set_aspect('equal'); a.set_title(title+f'\nchauds : {(cls==2).sum()}  froids : {(cls==0).sum()}',fontsize=9); a.set_xticks([]); a.set_yticks([]); return z,cls
fig,ax=plt.subplots(1,4,figsize=(17,4.8)); GI={}
for a,(k,v,t) in zip(ax,[('PC1',d.PC1.values,'Score PC1'),('acc_mo',d.acc_mo.values,'Matière organique'),('acc_sulf',d.acc_sulf.values,'Sulfures'),('puiss_m',detrend(d.puiss_m.values),'Épaisseur (résidu de tendance)')]):
    z,cls=gimap(a,v,t); GI[k]=dict(chauds=int((cls==2).sum()),froids=int((cls==0).sum()),zmax=float(z.max()),zmin=float(z.min())); d['Gi_'+k]=z
from matplotlib.lines import Line2D
ax[0].legend(handles=[Line2D([],[],marker='o',ls='',color=c,label=l) for c,l in [('#B03A2E','point chaud (FDR 5 %)'),('#D9D9D9','non significatif'),('#2A5F9E','point froid')]],frameon=False,fontsize=7,loc='lower left')
fig.suptitle('G3-01 · Points chauds et froids (Getis-Ord Gi*, rayon 250 m, correction du taux de fausses découvertes)',fontweight='bold'); fig.tight_layout(); save(fig,'G3_Points_chauds_Gi','G3-01_Gi_etoile'); RES['gi']=GI
# LISA PC1
def lisa(v,perm=499):
    z=(v-v.mean())/v.std(); lag=W@z; Ii=z*lag; p=np.empty(n)
    for i in range(n):
        oth=np.delete(np.arange(n),i); r=rng.choice(oth,(perm,K)); sim=z[i]*z[r].mean(1); p[i]=(1+(np.abs(sim)>=abs(Ii[i])).sum())/(perm+1)
    q=np.where((z>0)&(lag>0),'HH',np.where((z<0)&(lag<0),'LL',np.where((z>0)&(lag<0),'HL','LH'))); return z,lag,Ii,p,q
fig,ax=plt.subplots(1,3,figsize=(15,5)); LI={}
colq={'HH':'#B03A2E','LL':'#2A5F9E','HL':'#E3B77A','LH':'#8CB4D8','ns':'#D9D9D9'}
for a,(k,v,t) in zip(ax,[('PC1',d.PC1.values,'Score PC1'),('acc_mo',d.acc_mo.values,'Matière organique'),('puiss_m',detrend(d.puiss_m.values),'Épaisseur (résidu)')]):
    z,lag,Ii,p,q=lisa(v); sig=p<.05; qq=np.where(sig,q,'ns'); a.scatter(d.x,d.y,c=[colq[x] for x in qq],s=9); a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([])
    cnt=pd.Series(qq).value_counts().to_dict(); LI[k]=cnt; a.set_title(t+'\n'+'  '.join(f'{c} {cnt.get(c,0)}' for c in ['HH','LL','HL','LH']),fontsize=9)
ax[0].legend(handles=[Line2D([],[],marker='o',ls='',color=colq[c],label=l) for c,l in [('HH','forte entourée de fortes (HH)'),('LL','faible entourée de faibles (LL)'),('HL','forte isolée (HL)'),('LH','faible isolée (LH)'),('ns','non significatif')]],frameon=False,fontsize=7,loc='lower left')
fig.suptitle('G3-02 · Moran local (LISA) : regroupements et valeurs isolées (p < 0,05, sans correction : indicatif)',fontweight='bold'); fig.tight_layout(); save(fig,'G3_Points_chauds_Gi','G3-02_Moran_local_LISA'); RES['lisa']=LI
d[['hole_id','x','y','puiss_m','toit_m','PC1','Gi_PC1','Gi_acc_mo','Gi_acc_sulf','Gi_puiss_m']].to_csv(OUT+'res/G_Gi_scores.csv',index=False)
savejson('G3_gi_lisa.json')
