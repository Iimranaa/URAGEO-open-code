# -*- coding: utf-8 -*-
"""S6 (reduced) · Worked example of weights of evidence (WoE) and ROC/AUC on the borehole table.
Figures: S6-05 only. The original S6 script also re-plotted site-specific results taken from rasters of the project
(AUC per named site, success rates, scenarios 17-23); those values are NOT included in this public repository.
Runs on the synthetic table by default (URAGEO_DATA_CSV to change). Figure texts are in French.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from commun_calculs import *
from scipy import stats
d = load()
# ---------- WoE + AUC worked example (unité = sondage ; D = réduction élevée ; preuve = épaisseur/toit)
lab=(d.cl_reduc=='Eleve').values; N=len(d); Dn=lab.sum()
def woe(B):
    nbd=(B&lab).sum(); nbn=(B&~lab).sum(); nnd=(~B&lab).sum(); nnn=(~B&~lab).sum()
    Wp=np.log((nbd/Dn)/(nbn/(N-Dn))); Wm=np.log((nnd/Dn)/(nnn/(N-Dn)))
    sWp=np.sqrt(1/nbd+1/nbn); sWm=np.sqrt(1/nnd+1/nnn); C=Wp-Wm; sC=np.sqrt(sWp**2+sWm**2)
    return dict(B_et_D=int(nbd),B_et_nonD=int(nbn),nonB_et_D=int(nnd),nonB_et_nonD=int(nnn),Wp=Wp,Wm=Wm,C=C,sC=sC,Cs=C/sC)
th=d.puiss_m.values
cls=[('épaisseur ≥ 30 m',th>=30),('épaisseur ≥ 40 m',th>=40),('épaisseur ≥ 45 m',th>=45),('épaisseur ≥ 50 m',th>=50)]
WO=[]
for n_,m_ in cls:
    r=woe(m_); r['classe']=n_; r['unites_classe']=int(m_.sum()); WO.append(r)
WO=pd.DataFrame(WO); WO.to_csv(OUT+'res/S6_WoE_exemple.csv',index=False); print(WO.round(3).to_string())
score=th; rk=stats.rankdata(score); n1=lab.sum(); n0=(~lab).sum(); AUC=(rk[lab].sum()-n1*(n1+1)/2)/(n1*n0)
q=(AUC**2/(2-AUC)); q2=2*AUC**2/(1+AUC); se=np.sqrt((AUC*(1-AUC)+(n1-1)*(q-AUC**2)+(n0-1)*(q2-AUC**2))/(n1*n0))
order=np.argsort(-score); tp=np.cumsum(lab[order])/n1; fp=np.cumsum(~lab[order])/n0
print('AUC',AUC,se,n1,n0)
RES['auc_demo']=dict(auc=float(AUC),se_indep=float(se),n1=int(n1),n0=int(n0),N=int(N))
bi=2; r=WO.iloc[bi]
fig=plt.figure(figsize=(14,5.2)); a=fig.add_subplot(131)
a.scatter(d.x[~lab],d.y[~lab],s=5,color='#B8C2D6',label='réduction faible/modérée'); a.scatter(d.x[lab],d.y[lab],s=9,color=RED,label='réduction élevée (D)'); a.scatter(d.x[th>=45],d.y[th>=45],s=22,facecolors='none',edgecolors=NAVY,lw=.6,label='épaisseur ≥ 45 m (B)')
a.legend(frameon=False,fontsize=7,loc='lower right'); a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([]); a.set_title('Unités = 731 sondages')
a=fig.add_subplot(132); a.plot(fp,tp,color=NAVY,lw=2); a.plot([0,1],[0,1],color=GREY,ls=':'); a.fill_between(fp,tp,alpha=.15,color=NAVY); a.set_xlabel('faux positifs (sondages hors D classés favorables)'); a.set_ylabel('vrais positifs (sondages D captés)'); a.set_title(f'Courbe ROC, AUC = {AUC:.2f}'); a.set_aspect('equal')
a=fig.add_subplot(133); a.axis('off'); a.set_title('Poids de preuve, classe « épaisseur ≥ 45 m »',loc='left',color=NAVY)
tx=[f'Unités : N = {N}   ;   D (réduction élevée) = {Dn}',f'B (épaisseur ≥ 45 m) = {int(r.unites_classe)}',f'B∩D = {r.B_et_D}    B∩non-D = {r.B_et_nonD}',f'non-B∩D = {r.nonB_et_D}    non-B∩non-D = {r.nonB_et_nonD}','',f'W⁺ = ln[ P(B|D) / P(B|non-D) ] = {r.Wp:.2f}',f'W⁻ = ln[ P(non-B|D) / P(non-B|non-D) ] = {r.Wm:.2f}',f'C = W⁺ − W⁻ = {r.C:.2f}  (écart-type {r.sC:.2f})',f'C studentisé = {r.Cs:.1f}','','Exemple de mécanique de calcul sur les sondages :','les sondages voisins sont corrélés, le test est','donc trop optimiste. Ce n\'est pas la validation','du modèle (celle-ci : figures S6-01 à S6-04).']
for i,s_ in enumerate(tx): a.text(0,.93-i*.07,s_,fontsize=9.5,transform=a.transAxes)
fig.suptitle('S6-05 · Comment se calculent la courbe ROC/AUC et les poids de preuve (exemple pédagogique sur le jeu de sondages fourni)',fontweight='bold'); fig.tight_layout(); save(fig,'S6_Validation_et_sensibilite','S6-05_Exemple_ROC_AUC_et_poids_de_preuve')

savejson('S6_woe_roc_exemple.json')
