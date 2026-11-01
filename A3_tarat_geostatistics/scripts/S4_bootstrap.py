# -*- coding: utf-8 -*-
"""S4 · Bootstrap simple et par blocs de 500 m : stabilité de l’ACP
Figures produites : S4-01, S4-02
Données : 09_CALCULS_STATISTIQUES_GEOSTATISTIQUES/01_Statistiques/Donnees/Tarat_731_donnees_calcul.csv (731 sondages).
Lancer :  python S4_bootstrap.py      (résultats dans 05_CARTES_FINALES/_recalcul_scripts/ ; voir commun_calculs.py)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from commun_calculs import *
from scipy import stats
from scipy.optimize import linear_sum_assignment
d = load(); V = ['acc_oxy','acc_sulf','acc_mo']; V4 = ['acc_oxy','acc_carb','acc_sulf','acc_mo']
# ---------- S3 PCA
def pca(X):
    mu=X.mean(0); sd=X.std(0,ddof=1); Z=(X-mu)/sd; R=np.corrcoef(X.T); w,v=np.linalg.eigh(R); o=w.argsort()[::-1]; return mu,sd,Z,R,w[o],v[:,o]
X=d[V].values; mu,sd,Z,R,w,v=pca(X)
# align sign with PC1_731 stored
for k in range(3):
    sc=Z@v[:,k]
    if np.corrcoef(sc,d[f'PC{k+1}_731'])[0,1]<0: v[:,k]*=-1
sc=Z@v; load=v*np.sqrt(w)
chk=[float(np.corrcoef(sc[:,k],d[f'PC{k+1}_731'])[0,1]) for k in range(3)]
RES['pca731']=dict(mu=mu.tolist(),sd=sd.tolist(),R=R.tolist(),eig=w.tolist(),var=(w/3).tolist(),vec=v.tolist(),load=load.tolist(),check_corr=chk)
print('eig',w,w/3,chk); print(v.round(3)); print(load.round(3))
# KMO, Bartlett
def kmo(R):
    Ri=np.linalg.inv(R); D=np.sqrt(np.outer(np.diag(Ri),np.diag(Ri))); Q=-Ri/D; np.fill_diagonal(Q,0); R0=R.copy(); np.fill_diagonal(R0,0)
    msa=(R0**2).sum(0)/((R0**2).sum(0)+(Q**2).sum(0)); return (R0**2).sum()/((R0**2).sum()+(Q**2).sum()),msa
def bartlett(R,n):
    p=R.shape[0]; chi=-(n-1-(2*p+5)/6)*np.log(np.linalg.det(R)); df=p*(p-1)/2; return chi,df,1-stats.chi2.cdf(chi,df)
k731,m731=kmo(R); b731=bartlett(R,len(d)); RES['kmo731']=dict(kmo=k731,msa=m731.tolist(),bart=list(b731)); print('KMO',k731,m731,b731)
# permutation parallel analysis
rng=np.random.default_rng(42); sim=np.zeros((1000,3))
for i in range(1000):
    Xp=np.column_stack([rng.permutation(X[:,j]) for j in range(3)]); sim[i]=np.sort(np.linalg.eigvalsh(np.corrcoef(Xp.T)))[::-1]
hor=np.percentile(sim,95,axis=0); RES['horn731']=hor.tolist()
fig,a=plt.subplots(figsize=(7.5,4.8)); a.plot([1,2,3],w,'o-',color=NAVY,lw=2,label='valeurs propres observées')
a.plot([1,2,3],hor,'s--',color=CU,label='seuil de Horn (95 %, permutations)'); a.axhline(1,color=GREY,ls=':',label='critère de Kaiser (λ = 1)')
for i in range(3): a.text(i+1.06,w[i],f'{w[i]:.3f}\n({100*w[i]/3:.0f} %)',fontsize=9,va='bottom')
a.set_xticks([1,2,3]); a.set_xticklabels(['PC1','PC2','PC3']); a.set_ylabel('valeur propre λ'); a.set_ylim(0,2.0); a.set_xlim(.8,3.4); a.legend(frameon=False,fontsize=8,loc='upper right'); a.set_title('S3-02 · Éboulis des valeurs propres : combien d\'axes garder ?')
fig.tight_layout(); save(fig,'S3_ACP','S3-02_Eboulis_Kaiser_Horn')
# loadings + circle
names=['Oxy-hydroxydes','Sulfures','Matière org.']
fig,ax=plt.subplots(1,2,figsize=(12,5)); a=ax[0]; x=np.arange(3)
for k,(cc,off) in enumerate(zip([NAVY,CU,TEAL],[-.27,0,.27])):
    a.bar(x+off,load[:,k],.27,color=cc,label=f'PC{k+1} ({100*w[k]/3:.0f} %)')
    for i in range(3): a.text(i+off,load[i,k]+(.03 if load[i,k]>=0 else -.08),f'{load[i,k]:.2f}',ha='center',fontsize=8)
a.axhline(0,color='k',lw=.8); a.set_xticks(x); a.set_xticklabels(names); a.set_ylabel('charge (corrélation variable–axe)'); a.legend(frameon=False); a.set_title('Charges (loadings)')
a=ax[1]; a.add_patch(plt.Circle((0,0),1,fill=False,color=GREY)); 
for i,n_ in enumerate(names): a.annotate('',xy=(load[i,0],load[i,1]),xytext=(0,0),arrowprops=dict(arrowstyle='->',color=NAVY,lw=2)); a.text(load[i,0]*1.1,load[i,1]*1.1+.03,n_,ha='center',fontsize=9)
a.axhline(0,color='k',lw=.5); a.axvline(0,color='k',lw=.5); a.set_xlim(-1.2,1.2); a.set_ylim(-1.2,1.2); a.set_aspect('equal'); a.set_xlabel(f'PC1 ({100*w[0]/3:.0f} %)'); a.set_ylabel(f'PC2 ({100*w[1]/3:.0f} %)'); a.set_title('Cercle des corrélations')
fig.suptitle('S3-03 · Ce que représente chaque axe',fontweight='bold'); fig.tight_layout(); save(fig,'S3_ACP','S3-03_Charges_et_cercle')
# biplot
fig,a=plt.subplots(figsize=(8,6.5)); colr={'Faible':'#9AA7B8','Modere':CU,'Eleve':RED}
for k,c in colr.items():
    m=d.cl_reduc==k; a.scatter(sc[m,0],sc[m,1],s=10,color=c,alpha=.6,label=f'réduction {k.lower().replace("modere","modérée").replace("eleve","élevée")} (n={m.sum()})')
for i,n_ in enumerate(names): a.annotate('',xy=(load[i,0]*4,load[i,1]*4),xytext=(0,0),arrowprops=dict(arrowstyle='->',color='k',lw=1.8)); a.text(load[i,0]*4.3,load[i,1]*4.3,n_,fontsize=9,weight='bold')
a.axhline(0,color='k',lw=.5); a.axvline(0,color='k',lw=.5); a.set_xlabel(f'PC1 ({100*w[0]/3:.0f} %) · sulfures + matière organique'); a.set_ylabel(f'PC2 ({100*w[1]/3:.0f} %) · oxy-hydroxydes'); a.legend(frameon=False,fontsize=8)
a.set_title('S3-04 · Biplot : les 731 sondages dans le plan PC1–PC2'); fig.tight_layout(); save(fig,'S3_ACP','S3-04_Biplot_731')
# step-by-step figure
fig=plt.figure(figsize=(14,8.5)); fig.suptitle('S3-01 · Les étapes de l\'ACP, avec les vrais chiffres (3 variables, 731 sondages)',fontweight='bold',fontsize=13)
def tbl(pos,title,data,rl,cl,fmt='{:.3f}'):
    a=fig.add_axes(pos); a.axis('off'); a.set_title(title,fontsize=10,loc='left',color=NAVY)
    t=a.table(cellText=[[fmt.format(x) if isinstance(x,(float,np.floating)) else str(x) for x in r] for r in data],rowLabels=rl,colLabels=cl,loc='center',cellLoc='center'); t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1,1.5)
    for (r,c),cell in t.get_celld().items():
        if r==0: cell.set_facecolor('#E8EDF6')
    return a
c3=['Oxy','Sulf','MO']
tbl([.03,.57,.43,.33],'① Données brutes (6 premiers sondages)',X[:6],list(d.hole_id[:6]),c3)
tbl([.53,.57,.43,.33],'② Standardisées  z = (x − moyenne) / écart-type',Z[:6],list(d.hole_id[:6]),c3)
tbl([.03,.25,.43,.30],'③ Moyennes et écarts-types utilisés',[mu,sd],['moyenne','écart-type'],c3,'{:.4f}')
tbl([.53,.25,.43,.30],'④ Matrice de corrélation R = Zᵀ Z / (n − 1)',R,c3,c3)
tbl([.03,-.06,.43,.29],'⑤ Valeurs propres λ (R·v = λ·v)',[[w[0],w[1],w[2]],[100*w[0]/3,100*w[1]/3,100*w[2]/3],[100*w[:1].sum()/3,100*w[:2].sum()/3,100]],['λ','% variance','% cumulé'],['PC1','PC2','PC3'],'{:.2f}')
tbl([.53,-.06,.43,.29],'⑥ Vecteurs propres v  →  scores = Z · v',v,c3,['PC1','PC2','PC3'])
save(fig,'S3_ACP','S3-01_Etapes_ACP_chiffrees')
# 31-hole PCA and compare
d31=d.dropna(subset=['acc_carb']); X31=d31[V4].values; mu31,sd31,Z31,R31,w31,v31=pca(X31); load31=v31*np.sqrt(w31)
k31,m31=kmo(R31); b31=bartlett(R31,len(d31)); print('31',w31,k31,b31)
RES['pca31']=dict(eig=w31.tolist(),load=load31.tolist(),vec=v31.tolist(),kmo=k31,bart=list(b31),msa=m31.tolist())
# horn 31
sim31=np.zeros((1000,4))
for i in range(1000):
    Xp=np.column_stack([rng.permutation(X31[:,j]) for j in range(4)]); sim31[i]=np.sort(np.linalg.eigvalsh(np.corrcoef(Xp.T)))[::-1]
hor31=np.percentile(sim31,95,axis=0); RES['horn31']=hor31.tolist()
fig,ax=plt.subplots(1,2,figsize=(12,4.8)); a=ax[0]
a.plot([1,2,3,4],w31,'o-',color=NAVY,label='observées (31 sondages)'); a.plot([1,2,3,4],hor31,'s--',color=CU,label='Horn 95 %'); a.axhline(1,color=GREY,ls=':')
a.set_xticks([1,2,3,4]); a.set_xticklabels(['PC1','PC2','PC3','PC4']); a.legend(frameon=False,fontsize=8); a.set_title(f'31 sondages (4 variables) : aucun axe au-dessus du hasard\nKMO = {k31:.2f} ; Bartlett p = {b31[2]:.3f}')
a=ax[1]; s31=np.sign(load31[2,0]); 
l1=load[:,0]; l31=load31[[0,2,3],0]*np.sign(load31[2,0]); x=np.arange(3)
a.bar(x-.2,l1,.4,color=NAVY,label=f'731 sondages (n=731, 3 var.) KMO={k731:.2f}'); a.bar(x+.2,l31,.4,color=CU,label=f'31 sondages (n=31, 4 var.)'); a.axhline(0,color='k',lw=.8)
a.set_xticks(x); a.set_xticklabels(names); a.set_ylabel('charge sur PC1'); a.legend(frameon=False,fontsize=8); a.set_title('PC1 : même signification (sulfures + matière organique)')
fig.suptitle('S3-05 · Pourquoi retenir l\'ACP à 731 sondages plutôt qu\'à 31',fontweight='bold'); fig.tight_layout(); save(fig,'S3_ACP','S3-05_Comparaison_731_vs_31')
# ---------- S4 bootstrap
def refit(Xs,vref):
    _,_,_,Rr,ww,vv=pca(Xs); M=np.abs(vv.T@vref); r,c=linear_sum_assignment(-M)  # r: replicate comp, c: ref comp
    order=np.zeros(3,int); 
    for ri,ci in zip(r,c): order[ci]=ri
    vv2=vv[:,order]; ww2=ww[order]
    for k in range(3):
        if vv2[:,k]@vref[:,k]<0: vv2[:,k]*=-1
    swapped=not np.array_equal(order,np.arange(3)); return ww2,vv2*np.sqrt(ww2),Rr[1,2],swapped
Bn=2000; bx=np.floor(d.x/500).astype(int).values; by=np.floor(d.y/500).astype(int).values; bid=np.array([f'{a}_{b}' for a,b in zip(bx,by)]); ub=list(pd.unique(bid)); idx={b:np.where(bid==b)[0] for b in ub}
Eb=[];Lb=[];Rb=[];Sb=0;Ei=[];Li=[];Ri_=[];Si=0
for i in range(Bn):
    chosen=rng.choice(len(ub),len(ub)); ii=np.concatenate([idx[ub[c]] for c in chosen]); 
    if X[ii].std(0).min()==0: continue
    e,l,r,s=refit(X[ii],v); Eb.append(e);Lb.append(l);Rb.append(r);Sb+=s
    jj=rng.integers(0,len(X),len(X)); e,l,r,s=refit(X[jj],v); Ei.append(e);Li.append(l);Ri_.append(r);Si+=s
Eb,Lb,Rb,Ei,Li,Ri_=map(np.array,(Eb,Lb,Rb,Ei,Li,Ri_))
ci=lambda a:np.percentile(a,[2.5,97.5],axis=0)
RES['boot']=dict(nblocks=len(ub),B=len(Eb),eig_block=ci(Eb).tolist(),eig_iid=ci(Ei).tolist(),load1_block=ci(Lb[:,:,0]).tolist(),load1_iid=ci(Li[:,:,0]).tolist(),load2_block=ci(Lb[:,:,1]).tolist(),load2_iid=ci(Li[:,:,1]).tolist(),
  r_block=np.percentile(Rb,[2.5,97.5]).tolist(),r_iid=np.percentile(Ri_,[2.5,97.5]).tolist(),swap_block=Sb/len(Eb),swap_iid=Si/len(Eb),est_r=float(R[1,2]))
print(RES['boot'])
fig,a=plt.subplots(figsize=(8.5,6.5)); cnt=pd.Series(bid).value_counts()
sc_=a.scatter(d.x,d.y,c=[cnt[b] for b in bid],s=8,cmap='viridis'); 
for u_ in ub:
    ux,uy=map(int,u_.split('_')); a.add_patch(plt.Rectangle((ux*500,uy*500),500,500,fill=False,ec='k',lw=.6))
a.set_aspect('equal'); a.set_xlabel('X (m)'); a.set_ylabel('Y (m)'); plt.colorbar(sc_,label='nombre de sondages du bloc'); a.set_title(f'S4-01 · Blocs de 500 m utilisés pour le bootstrap ({len(ub)} blocs)')
fig.tight_layout(); save(fig,'S4_Bootstrap','S4-01_Blocs_500m')
fig,ax=plt.subplots(1,3,figsize=(14,4.8)); 
for a,(arrB,arrI,est,lab,t) in zip(ax[:2],[(Eb,Ei,w,['λ1','λ2','λ3'],'Valeurs propres'),(Lb[:,:,0],Li[:,:,0],load[:,0],['Oxy','Sulf','MO'],'Charges sur PC1')]):
    for i in range(3):
        for off,A,cc,nm in [(-.12,arrI,GREY,'bootstrap simple'),(.12,arrB,NAVY,'bootstrap par blocs 500 m')]:
            lo,hi=np.percentile(A[:,i],[2.5,97.5]); a.plot([lo,hi],[i+off*1.2,i+off*1.2],color=cc,lw=3,label=nm if i==0 else None)
        a.scatter(est[i],i,color=CU,zorder=4,s=45,label='estimation' if i==0 else None)
    a.set_yticks(range(3)); a.set_yticklabels(lab); a.invert_yaxis(); a.set_title(t+' (IC 95 %)'); a.legend(frameon=False,fontsize=7)
a=ax[2]; a.hist(Ri_,bins=40,alpha=.6,color=GREY,label='simple'); a.hist(Rb,bins=40,alpha=.6,color=NAVY,label='par blocs'); a.axvline(R[1,2],color=CU,lw=2); a.set_xlabel('r (sulfures, matière organique)'); a.legend(frameon=False); a.set_title('Distribution bootstrap de r(Sulf, MO)')
fig.suptitle('S4-02 · Stabilité de l\'ACP à 731 sondages : bootstrap simple contre bootstrap par blocs',fontweight='bold'); fig.tight_layout(); save(fig,'S4_Bootstrap','S4-02_Intervalles_de_confiance')
# 31 iid bootstrap
E31=[];L31=[]
for i in range(3000):
    jj=rng.integers(0,31,31); Xs=X31[jj]
    if Xs.std(0).min()==0: continue
    _,_,_,Rr,ww,vv=pca(Xs); M=np.abs(vv.T@v31); r,c=linear_sum_assignment(-M); order=np.zeros(4,int)
    for ri_,ci_ in zip(r,c): order[ci_]=ri_
    vv=vv[:,order]; ww=ww[order]; 
    for k in range(4):
        if vv[:,k]@v31[:,k]<0: vv[:,k]*=-1
    E31.append(ww); L31.append(vv*np.sqrt(ww))
E31=np.array(E31); L31=np.array(L31); RES['boot31']=dict(eig=np.percentile(E31,[2.5,97.5],axis=0).tolist(),load_pc1=np.percentile(L31[:,:,0],[2.5,97.5],axis=0).tolist(),load31=load31.tolist(),eig31=w31.tolist())
print(RES['boot31'])
RES['sim_horn']=dict(h731=hor.tolist(),h31=hor31.tolist())

savejson('S4_bootstrap.json')
