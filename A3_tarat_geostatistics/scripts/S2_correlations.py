# -*- coding: utf-8 -*-
"""S2 · Corrélations de Pearson et Spearman, IC 95 %, nuages de points, sous-échantillon à 31 sondages
Figures produites : S2-01 à S2-04
Données : 09_CALCULS_STATISTIQUES_GEOSTATISTIQUES/01_Statistiques/Donnees/Tarat_731_donnees_calcul.csv (731 sondages).
Lancer :  python S2_correlations.py      (résultats dans 05_CARTES_FINALES/_recalcul_scripts/ ; voir commun_calculs.py)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from commun_calculs import *
from scipy import stats
from scipy.optimize import linear_sum_assignment
d = load(); V = ['acc_oxy','acc_sulf','acc_mo']; V4 = ['acc_oxy','acc_carb','acc_sulf','acc_mo']
# ---------- S2 correlations
C7=['acc_oxy','acc_sulf','acc_mo','puiss_m','toit_m','x','y']
P=d[C7].corr('pearson'); S=d[C7].corr('spearman'); P.to_csv(OUT+'res/S2_pearson.csv'); S.to_csv(OUT+'res/S2_spearman.csv')
fig,ax=plt.subplots(1,2,figsize=(15,6.5),gridspec_kw={'wspace':.75})
for a,M,t in zip(ax,[P,S],['Pearson (linéaire)','Spearman (rangs)']):
    im=a.imshow(M.values,cmap='RdBu_r',vmin=-1,vmax=1)
    a.set_xticks(range(7)); a.set_yticks(range(7)); a.set_xticklabels([LAB[c].replace(' (m)','') for c in C7],rotation=40,ha='right'); a.set_yticklabels([LAB[c].replace(' (m)','') for c in C7])
    for i in range(7):
        for j in range(7): a.text(j,i,f'{M.values[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(M.values[i,j])>.6 else 'k')
    a.set_title(t); a.spines[:].set_visible(False)
fig.colorbar(im,ax=ax,shrink=.7,label='coefficient de corrélation',pad=.02)
fig.suptitle('S2-01 · Matrices de corrélation, 731 sondages',fontweight='bold'); save(fig,'S2_Correlations','S2-01_Matrices_Pearson_Spearman')
def fisher(r,n):
    z=np.arctanh(r); se=1/np.sqrt(n-3); return np.tanh(z-1.96*se),np.tanh(z+1.96*se)
pairs=[('acc_sulf','acc_mo'),('acc_oxy','acc_sulf'),('acc_oxy','acc_mo'),('puiss_m','x'),('toit_m','x'),('puiss_m','toit_m'),('acc_mo','puiss_m'),('acc_sulf','puiss_m'),('acc_oxy','puiss_m')]
rows=[]
for a_,b_ in pairs:
    s=d[[a_,b_]].dropna(); r,p=stats.pearsonr(s[a_],s[b_]); rs,ps=stats.spearmanr(s[a_],s[b_]); lo,hi=fisher(r,len(s))
    rows.append(dict(paire=f'{LAB[a_]} – {LAB[b_]}',n=len(s),pearson=r,ic95_bas=lo,ic95_haut=hi,p_pearson=p,spearman=rs,p_spearman=ps))
D2=pd.DataFrame(rows); D2.to_csv(OUT+'res/S2_paires.csv',index=False); print(D2.round(3).to_string())
fig,a=plt.subplots(figsize=(9,5))
for i,r in D2.iterrows():
    a.plot([r.ic95_bas,r.ic95_haut],[i,i],color=NAVY,lw=2); a.scatter(r.pearson,i,color=CU,zorder=3,s=40)
    a.text(1.02,i,f"r = {r.pearson:.2f}  [{r.ic95_bas:.2f} ; {r.ic95_haut:.2f}]",va='center',fontsize=8)
a.axvline(0,color='k',lw=.8); a.set_yticks(range(len(D2))); a.set_yticklabels(D2.paire,fontsize=8); a.invert_yaxis(); a.set_xlim(-.4,1.0)
a.set_xlabel("corrélation de Pearson et intervalle de confiance à 95 % (transformation de Fisher)")
a.set_title('S2-02 · Corrélations clés avec leur incertitude (731 sondages)'); fig.tight_layout(); save(fig,'S2_Correlations','S2-02_Correlations_cles_IC95')
# scatter matrix
fig,ax=plt.subplots(3,3,figsize=(10,9.5)); S4=['acc_oxy','acc_sulf','acc_mo']
for i,u in enumerate(S4):
    for j,v in enumerate(S4):
        a=ax[i,j]
        if i==j: a.hist(d[u],bins=30,color=[TEAL,CU,GRN][i],edgecolor='white')
        else:
            a.scatter(d[v],d[u],s=6,alpha=.35,color=NAVY); r=d[[u,v]].corr().iloc[0,1]; rs=d[[u,v]].corr('spearman').iloc[0,1]
            a.text(.97,.97,f'r = {r:.2f}\nρ = {rs:.2f}',transform=a.transAxes,ha='right',va='top',fontsize=9,bbox=dict(fc='white',ec='#ccc'))
        if i==2: a.set_xlabel(LAB[v])
        if j==0: a.set_ylabel(LAB[u])
fig.suptitle('S2-03 · Nuages de points deux à deux : seule la paire sulfures – matière organique est liée',fontweight='bold'); fig.tight_layout(); save(fig,'S2_Correlations','S2-03_Nuages_de_points')
# 31 holes
d31=d.dropna(subset=['acc_carb']); V4=['acc_oxy','acc_carb','acc_sulf','acc_mo']
P31=d31[V4].corr(); n=len(d31)
fig,a=plt.subplots(figsize=(6.5,5.5)); im=a.imshow(P31.values,cmap='RdBu_r',vmin=-1,vmax=1)
a.set_xticks(range(4)); a.set_yticks(range(4)); a.set_xticklabels([LAB[c] for c in V4],rotation=30,ha='right'); a.set_yticklabels([LAB[c] for c in V4])
for i in range(4):
    for j in range(4):
        r=P31.values[i,j]; p=stats.pearsonr(d31[V4[i]],d31[V4[j]])[1] if i!=j else 0
        a.text(j,i,f'{r:.2f}'+('' if i==j else f'\np={p:.3f}'),ha='center',va='center',fontsize=8,color='white' if abs(r)>.6 else 'k')
a.set_title(f'S2-04 · Corrélations des 31 sondages avec carbonates'); a.spines[:].set_visible(False); fig.colorbar(im,shrink=.7); fig.tight_layout(); save(fig,'S2_Correlations','S2-04_Correlations_31_sondages')
savejson('S2_correlations.json')
