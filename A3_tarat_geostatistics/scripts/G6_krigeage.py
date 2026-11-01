# -*- coding: utf-8 -*-
"""G6 · Cartes de krigeage (épaisseur, toit, PC1) et rasters GeoTIFF
Figures produites : G6-01 à G6-03 + krigeage_*.tif
Données : 09_CALCULS_STATISTIQUES_GEOSTATISTIQUES/01_Statistiques/Donnees/Tarat_731_donnees_calcul.csv (731 sondages).
Lancer :  python G6_krigeage.py      (résultats dans 05_CARTES_FINALES/_recalcul_scripts/ ; voir commun_calculs.py)
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
# --- kriging machinery
def make_ok(z,model,nn=24):
    f,p=model; tree=spatial.cKDTree(XY)
    def pred(T,exclude_self=False):
        k=nn+(1 if exclude_self else 0); dist,ix=tree.query(T,k=k)
        if exclude_self: dist=dist[:,1:]; ix=ix[:,1:]
        m=T.shape[0]; P=XY[ix]; DD=np.linalg.norm(P[:,:,None,:]-P[:,None,:,:],axis=3); Gm=f(np.maximum(DD,1e-6),*p); Gm[:,np.arange(nn),np.arange(nn)]=0
        A=np.ones((m,nn+1,nn+1)); A[:,:nn,:nn]=Gm; A[:,nn,nn]=0; b=np.ones((m,nn+1)); b[:,:nn]=f(np.maximum(dist,1e-6),*p)
        s=np.linalg.solve(A,b[:,:,None])[:,:,0]; est=(s[:,:nn]*z[ix]).sum(1); var=(s[:,:nn]*b[:,:nn]).sum(1)+s[:,nn]; return est,np.maximum(var,0),dist[:,0]
    return pred
def idw(z,T,pw=2,nn=12,exclude_self=False):
    tree=spatial.cKDTree(XY); k=nn+(1 if exclude_self else 0); dist,ix=tree.query(T,k=k)
    if exclude_self: dist=dist[:,1:]; ix=ix[:,1:]
    w=1/np.maximum(dist,1e-6)**pw; return (w*z[ix]).sum(1)/w.sum(1)
CV=[]; PRED={}
for k,l in VARS.items():
    z=d[k].values.astype(float); vm=VM[k]; f=MODELS[vm['model']]; p=[vm['c0'],vm['c'],vm['a']]; ok=make_ok(z,(f,p)); est,var,_=ok(XY,True); ide=idw(z,XY,exclude_self=True)
    nnb=z[np.argsort(DM+np.eye(n)*1e9,axis=1)[:,0]]; mean_l=np.array([(z.sum()-z[i])/(n-1) for i in range(n)])
    PRED[k]=dict(ok=est,idw=ide,var=var,z=z)
    for nm,pr in [('moyenne',mean_l),('plus proche voisin',nnb),('IDW (p=2, 12 pts)',ide),('krigeage ordinaire',est)]:
        e=pr-z; rm=np.sqrt((e**2).mean()); CV.append(dict(variable=l,cle=k,methode=nm,ME=e.mean(),MAE=np.abs(e).mean(),RMSE=rm,skill=1-rm**2/z.var(),R=np.corrcoef(pr,z)[0,1]))
    se=(est-z)/np.sqrt(np.maximum(var,1e-12)); CV[-1]['zstd_moy']=se.mean(); CV[-1]['zstd_var']=se.var()
CVd=pd.DataFrame(CV); CVd.to_csv(OUT+'res/G5_validation_croisee.csv',index=False); print(CVd.round(3).to_string())
RES['cv']=CVd.to_dict('records')
fig,a=plt.subplots(figsize=(11,5)); meths=['moyenne','plus proche voisin','IDW (p=2, 12 pts)','krigeage ordinaire']; cc=[GREY,'#B8C2D6',CU,NAVY]; x=np.arange(len(VARS))
for i,(m_,c_) in enumerate(zip(meths,cc)):
    s_=[CVd[(CVd.cle==k)&(CVd.methode==m_)].skill.iloc[0] for k in VARS]; a.bar(x+(i-1.5)*.2,s_,.2,color=c_,label=m_)
a.axhline(0,color='k',lw=.8); a.set_xticks(x); a.set_xticklabels([l.replace(' (','\n(') for l in VARS.values()],fontsize=8); a.set_ylabel('gain par rapport à la moyenne  1 − RMSE²/variance'); a.legend(frameon=False,fontsize=8,ncol=2)
a.set_title('G5-01 · Validation croisée (un sondage retiré à la fois) : seul l\'épaisseur et le toit sont prévisibles',fontsize=10); fig.tight_layout(); save(fig,'G5_Validation_croisee','G5-01_Gain_par_methode_et_variable')
fig,ax=plt.subplots(2,3,figsize=(15,8.5))
for j,k in enumerate(['puiss_m','toit_m','PC1']):
    for i,(m_,c_) in enumerate([('ok','krigeage ordinaire'),('idw','IDW')]):
        a=ax[i,j]; z=PRED[k]['z']; pr=PRED[k][m_]; a.scatter(z,pr,s=6,alpha=.45,color=NAVY if i==0 else CU); lim=[min(z.min(),pr.min()),max(z.max(),pr.max())]; a.plot(lim,lim,color='k',lw=1); r=np.corrcoef(z,pr)[0,1]; rm=np.sqrt(((pr-z)**2).mean())
        a.set_title(f'{VARS[k]} · {c_}\nr = {r:.2f} · RMSE = {rm:.3g}',fontsize=9); a.set_xlabel('observé'); a.set_ylabel('prédit (sans le sondage)')
fig.suptitle('G5-02 · Observé contre prédit en validation croisée',fontweight='bold'); fig.tight_layout(); save(fig,'G5_Validation_croisee','G5-02_Observe_contre_predit')
fig,ax=plt.subplots(1,2,figsize=(11,4.4)); 
for a,k in zip(ax,['puiss_m','toit_m']):
    se=(PRED[k]['ok']-PRED[k]['z'])/np.sqrt(np.maximum(PRED[k]['var'],1e-12)); a.hist(se,bins=40,range=(-4,4),color=TEAL,edgecolor='white',density=True); xs_=np.linspace(-4,4,200); a.plot(xs_,stats.norm.pdf(xs_),color='k'); a.set_title(f'{VARS[k]}\nmoyenne {se.mean():.2f} · variance {se.var():.2f} (idéal : 0 et 1)',fontsize=9); a.set_xlabel('erreur standardisée = (prédit − observé) / écart-type de krigeage')
fig.suptitle('G5-03 · Les écarts-types de krigeage sont-ils crédibles ?',fontweight='bold'); fig.tight_layout(); save(fig,'G5_Validation_croisee','G5-03_Erreurs_standardisees')
# Moran des résidus de CV
rr=[]
for k in ['puiss_m','toit_m','PC1']:
    e=PRED[k]['ok']-PRED[k]['z']; I,p_,zz,_=moran(e,W); rr.append(dict(variable=VARS[k],I_residus_krigeage=I,p=p_))
RES['moran_resid']=rr; print(rr)
# --- Maps
res_=25; gx=np.arange(d.x.min()-100,d.x.max()+100,res_)+res_/2; gy=np.arange(d.y.min()-100,d.y.max()+100,res_)+res_/2; GX,GY=np.meshgrid(gx,gy); T=np.column_stack([GX.ravel(),GY.ravel()])
tree=spatial.cKDTree(XY); dmin=tree.query(T)[0]; mask=(dmin>200).reshape(GX.shape)
import rasterio; from rasterio.transform import from_origin
os.makedirs(OUT+'rasters',exist_ok=True)
def grid(v): A=v.reshape(GX.shape).copy(); A[mask]=np.nan; return A
def savetif(A,name):
    tr=from_origin(gx.min()-res_/2,gy.max()+res_/2,res_,res_); 
    with rasterio.open(OUT+'rasters/'+name+'.tif','w',driver='GTiff',height=A.shape[0],width=A.shape[1],count=1,dtype='float32',transform=tr,nodata=-9999) as ds: ds.write(np.where(np.isnan(A[::-1]),-9999,A[::-1]).astype('float32'),1)
ext=[gx.min()-res_/2,gx.max()+res_/2,gy.min()-res_/2,gy.max()+res_/2]
KR={}
for k,tag,unit in [('puiss_m','epaisseur','m'),('toit_m','toit','m'),('PC1','PC1','score')]:
    z=d[k].values.astype(float); vm=VM[k]; ok=make_ok(z,(MODELS[vm['model']],[vm['c0'],vm['c'],vm['a']])); est,var,_=ok(T); ide=idw(z,T)
    E=grid(est); S=grid(np.sqrt(var)); I_=grid(ide); df_=E-I_; savetif(E,f'krigeage_{tag}'); savetif(S,f'krigeage_{tag}_ecart_type'); savetif(I_,f'idw_{tag}')
    KR[k]=dict(est_min=float(np.nanmin(E)),est_max=float(np.nanmax(E)),se_med=float(np.nanmedian(S)),se_min=float(np.nanmin(S)),se_max=float(np.nanmax(S)),diff_rmse=float(np.sqrt(np.nanmean(df_**2))))
    fig,ax=plt.subplots(2,2,figsize=(13,11)); vmn,vmx=np.nanpercentile(E,[1,99])
    for a,(M_,t,cm,lims) in zip(ax.ravel(),[(E,f'Krigeage ordinaire ({unit})','viridis',(vmn,vmx)),(S,f'Écart-type de krigeage ({unit})','magma_r',(None,None)),(I_,f'IDW p=2 ({unit})','viridis',(vmn,vmx)),(df_,'Krigeage − IDW','RdBu_r',(-np.nanmax(np.abs(df_)),np.nanmax(np.abs(df_))))]):
        im=a.imshow(M_,extent=ext,origin='lower',cmap=cm,vmin=lims[0],vmax=lims[1]); a.scatter(d.x,d.y,s=1.2,color='k',alpha=.5); plt.colorbar(im,ax=a,shrink=.75); a.set_title(t); a.set_xlabel('X (m)'); a.set_ylabel('Y (m)')
    fig.suptitle(f'G6 · {VARS[k]} : {vm["model"]} (pépite/palier = {100*vm["ratio"]:.0f} %, portée {vm["a"]:.0f} m) ; cellule 25 m, masque à > 200 m d\'un sondage',fontweight='bold',fontsize=11); fig.tight_layout()
    save(fig,'G6_Krigeage','G6-0'+str(['puiss_m','toit_m','PC1'].index(k)+1)+'_Krigeage_'+tag)
RES['kr']=KR; print(KR)

savejson('G6_krigeage.json')
