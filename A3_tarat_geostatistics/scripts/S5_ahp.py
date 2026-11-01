# -*- coding: utf-8 -*-
"""S5 · Méthode AHP : matrice de comparaison par paires, poids, cohérence (CR), stabilité Monte-Carlo
Figures produites : S5-01 à S5-03
Données : 09_CALCULS_STATISTIQUES_GEOSTATISTIQUES/01_Statistiques/Donnees/Tarat_731_donnees_calcul.csv (731 sondages).
Lancer :  python S5_ahp.py      (résultats dans 05_CARTES_FINALES/_recalcul_scripts/ ; voir commun_calculs.py)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from commun_calculs import *
from scipy import stats
import geopandas as gpd, shapely
d = load()
# ---------- S5 AHP
names=['F4','N110-150','F3','N70','F1','NS','N30']
A=np.array([[1,1,2,3,5,5,7],[1,1,2,3,5,5,7],[1/2,1/2,1,2,4,4,6],[1/3,1/3,1/2,1,2,2,4],[1/5,1/5,1/4,1/2,1,1,3],[1/5,1/5,1/4,1/2,1,1,3],[1/7,1/7,1/6,1/4,1/3,1/3,1]])
RI={3:.58,4:.90,5:1.12,6:1.24,7:1.32,8:1.41}
def ahp(A):
    n=len(A); gm=np.prod(A,axis=1)**(1/n); w=gm/gm.sum(); lam=np.linalg.eigvals(A).real.max(); ci=(lam-n)/(n-1); return gm,w,lam,ci,ci/RI[n]
gm,w,lam,ci,cr=ahp(A); ev=np.linalg.eigvals(A); lam_eig=ev.real.max(); we=np.linalg.eig(A)[1][:,ev.real.argmax()].real; we=we/we.sum()
print(w.round(4),lam,lam_eig,ci,cr,we.round(4))
RES['ahp']=dict(names=names,A=A.tolist(),gm=gm.tolist(),w=w.tolist(),lam=lam,lam_eig=lam_eig,ci=ci,cr=cr,w_eig=we.tolist(),colsum=A.sum(0).tolist())
fig,a=plt.subplots(figsize=(8,6.3)); im=a.imshow(np.log(A),cmap='RdBu_r',vmin=-np.log(9),vmax=np.log(9))
from fractions import Fraction
for i in range(7):
    for j in range(7):
        v=A[i,j]; t=str(int(v)) if v>=1 else '1/'+str(int(round(1/v))); a.text(j,i,t,ha='center',va='center',fontsize=11,color='white' if abs(np.log(v))>1.2 else 'k')
a.set_xticks(range(7)); a.set_yticks(range(7)); a.set_xticklabels(names); a.set_yticklabels(names); a.xaxis.tick_top(); a.spines[:].set_visible(False)
a.set_title('S5-01 · Matrice de comparaison par paires (lue : ligne est ... fois plus importante que colonne)',pad=34,fontsize=10); fig.tight_layout(); save(fig,'S5_AHP','S5-01_Matrice_AHP')
fig,ax=plt.subplots(1,2,figsize=(13,5)); a=ax[0]; a.axis('off'); a.set_title('Étapes du calcul des poids (méthode de la moyenne géométrique)',loc='left',color=NAVY)
cells=[[n_,f'{g:.3f}',f'{100*x:.1f} %'] for n_,g,x in zip(names,gm,w)]+[['Somme',f'{gm.sum():.3f}','100,0 %']]
t=a.table(cellText=cells,colLabels=['Facteur','① moyenne géométrique\ndes 7 valeurs de la ligne','② poids = ① / somme'],loc='center',cellLoc='center'); t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1,1.9)
for (r,c),cell in t.get_celld().items():
    if r==0: cell.set_facecolor('#E8EDF6')
a=ax[1]; cc=[NAVY,NAVY,CU,TEAL,GREY,GREY,GREY]; a.barh(range(7),100*w,color=cc)
for i,x in enumerate(w): a.text(100*x+.4,i,f'{100*x:.1f} %',va='center')
a.set_yticks(range(7)); a.set_yticklabels(names); a.invert_yaxis(); a.set_xlabel('poids (%)'); a.set_xlim(0,34); a.set_title('Poids obtenus')
fig.suptitle('S5-02 · De la matrice aux poids',fontweight='bold'); fig.tight_layout(); save(fig,'S5_AHP','S5-02_Calcul_des_poids')
# consistency illustration
fig,ax=plt.subplots(1,2,figsize=(12,4.6)); a=ax[0]
a.axis('off'); a.set_title('Contrôle de cohérence de Saaty',loc='left',color=NAVY)
txt=[f'③ Somme de chaque colonne de A : {np.round(A.sum(0),2).tolist()}',f'④ λmax = valeur propre principale de A = {lam:.3f}',f'      (équivalent : moyenne des (A·w)ᵢ/wᵢ, calculée dans Excel)',f'⑤ CI = (λmax − n) / (n − 1) = ({lam:.3f} − 7) / 6 = {ci:.4f}',f'⑥ CR = CI / RI = {ci:.4f} / 1,32 = {cr:.4f}','',f'Seuil d\'acceptation : CR < 0,10  →  {"jugements cohérents" if cr<.1 else "à revoir"}']
for i,s_ in enumerate(txt): a.text(0,.9-i*.12,s_,fontsize=10.5,transform=a.transAxes,family='DejaVu Sans')
# Monte-Carlo on judgments
ladder=np.array([1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2,1,2,3,4,5,6,7,8,9]); rng=np.random.default_rng(7)
def nearest(v): return int(np.abs(ladder-v).argmin())
iu=np.triu_indices(7,1); W=[];CR=[]
for _ in range(5000):
    B=np.ones((7,7))
    for i,j in zip(*iu):
        k=nearest(A[i,j])+rng.choice([-1,0,1]); k=min(max(k,0),16); B[i,j]=ladder[k]; B[j,i]=1/ladder[k]
    g_,w_,l_,c_,r_=ahp(B); W.append(w_); CR.append(r_)
W=np.array(W); CR=np.array(CR); top2=np.mean([set(np.argsort(-w_)[:2])=={0,1} for w_ in W]); okcr=(CR<.1).mean()
RES['ahp_mc']=dict(n=5000,cr_ok=okcr,top2=top2,pct=np.percentile(W,[2.5,50,97.5],axis=0).tolist(),cr_pct=np.percentile(CR,[5,50,95]).tolist())
print(RES['ahp_mc'])
a=ax[1]; bp=a.boxplot(100*W,vert=False,tick_labels=names,whis=(2.5,97.5),showfliers=False,patch_artist=True)
for p_ in bp['boxes']: p_.set_facecolor('#C9D3E6')
for i,x in enumerate(w): a.scatter(100*x,i+1,color=CU,zorder=4,label='poids retenu' if i==0 else None)
a.invert_yaxis(); a.set_xlabel('poids (%)'); a.legend(frameon=False); a.set_title(f'Sensibilité : chaque jugement déplacé d\'un cran (5 000 tirages)\nCR < 0,10 dans {100*okcr:.0f} % des tirages')
fig.suptitle('S5-03 · Cohérence de la matrice et stabilité des poids',fontweight='bold'); fig.tight_layout(); save(fig,'S5_AHP','S5-03_Coherence_et_stabilite_des_poids')
savejson('S5_ahp.json')
