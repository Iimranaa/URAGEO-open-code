# -*- coding: utf-8 -*-
"""S3 · ACP des trois indicateurs d’oxydoréduction (pas à pas, éboulis Kaiser/Horn, charges, biplot, 731 vs 31)
Figures produites : S3-01 à S3-05
Données : 09_CALCULS_STATISTIQUES_GEOSTATISTIQUES/01_Statistiques/Donnees/Tarat_731_donnees_calcul.csv (731 sondages).
Lancer :  python S3_acp.py      (résultats dans 05_CARTES_FINALES/_recalcul_scripts/ ; voir commun_calculs.py)
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
savejson('S3_acp.json')
