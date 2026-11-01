# -*- coding: utf-8 -*-
"""S1 · Statistiques descriptives des variables de base (histogrammes, QQ-plots, zéros et manquants)
Figures produites : S1-01, S1-02, S1-03
Données : 09_CALCULS_STATISTIQUES_GEOSTATISTIQUES/01_Statistiques/Donnees/Tarat_731_donnees_calcul.csv (731 sondages).
Lancer :  python S1_descriptif.py      (résultats dans 05_CARTES_FINALES/_recalcul_scripts/ ; voir commun_calculs.py)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from commun_calculs import *
from scipy import stats
from scipy.optimize import linear_sum_assignment
d = load(); V = ['acc_oxy','acc_sulf','acc_mo']; V4 = ['acc_oxy','acc_carb','acc_sulf','acc_mo']
# ---------- S1 descriptif
cols=['acc_oxy','acc_sulf','acc_mo','acc_carb','puiss_m','toit_m']
rows=[]
for c in cols:
    s=d[c].dropna(); sh=stats.shapiro(s) if len(s)<=5000 else None
    rows.append(dict(variable=LAB[c],n=len(s),moyenne=s.mean(),ecart_type=s.std(ddof=1),min=s.min(),q1=s.quantile(.25),mediane=s.median(),q3=s.quantile(.75),max=s.max(),
      asymetrie=stats.skew(s,bias=False),aplatissement=stats.kurtosis(s,bias=False),pct_zero=100*(s==0).mean(),shapiro_W=sh.statistic,shapiro_p=sh.pvalue))
D1=pd.DataFrame(rows); D1.to_csv(OUT+'res/S1_descriptif.csv',index=False); print(D1.round(3).to_string())
fig,ax=plt.subplots(2,3,figsize=(13,7))
cl=[TEAL,CU,GRN,GREY,NAVY,RED]
for a,c,k in zip(ax.ravel(),cols,cl):
    s=d[c].dropna(); a.hist(s,bins=35,color=k,alpha=.85,edgecolor='white')
    a.axvline(s.mean(),color='k',ls='--',lw=1.2); a.axvline(s.median(),color='k',ls=':',lw=1.2)
    r=D1[D1.variable==LAB[c]].iloc[0]
    a.set_title(f"{LAB[c]} (n = {len(s)})"); a.set_xlabel('accumulation (m·indice/m)' if c.startswith('acc') else 'm'); a.set_ylabel('nombre de sondages')
    a.text(.97,.95,f"moy. {s.mean():.3f}\nméd. {s.median():.3f}\nzéros {r.pct_zero:.0f} %\nasym. {r.asymetrie:.2f}",transform=a.transAxes,ha='right',va='top',fontsize=9,bbox=dict(fc='white',ec='#ccc'))
fig.suptitle('S1-01 · Distribution des six variables de base (tirets : moyenne ; pointillés : médiane)',fontweight='bold',y=1.0)
fig.text(.01,-.02,NOTE+' Carbonates : 31 sondages renseignés seulement.',fontsize=8,color=GREY)
fig.tight_layout(); save(fig,'S1_Descriptif','S1-01_Histogrammes_variables')
# QQ
fig,ax=plt.subplots(2,3,figsize=(13,7))
for a,c,k in zip(ax.ravel(),cols,cl):
    s=d[c].dropna(); (osm,osr),(sl,ic,r)=stats.probplot(s,dist='norm')
    a.scatter(osm,osr,s=8,color=k,alpha=.7); a.plot(osm,sl*osm+ic,color='k',lw=1)
    a.set_title(f"{LAB[c]}  (R² droite = {r**2:.2f})"); a.set_xlabel('quantiles théoriques (loi normale)'); a.set_ylabel('quantiles observés')
fig.suptitle('S1-02 · Diagramme quantile-quantile : un nuage qui suit la droite = distribution normale',fontweight='bold',y=1.0)
fig.tight_layout(); save(fig,'S1_Descriptif','S1-02_QQ_plots')
# zeros / missing
fig,a=plt.subplots(figsize=(8,4.2))
z=[100*(d[c].dropna()==0).mean() for c in cols]; m=[100*d[c].isna().mean() for c in cols]
x=np.arange(len(cols)); a.bar(x-.2,z,.4,color=CU,label='valeurs nulles (% des sondages renseignés)'); a.bar(x+.2,m,.4,color=GREY,label='valeur manquante (% des 731)')
for i,(u,v) in enumerate(zip(z,m)): a.text(i-.2,u+1,f'{u:.0f}',ha='center',fontsize=9); a.text(i+.2,v+1,f'{v:.0f}',ha='center',fontsize=9)
a.set_xticks(x); a.set_xticklabels([LAB[c].replace(' (m)','').replace(' ','\n',1) for c in cols],fontsize=9); a.set_ylim(0,110); a.set_ylabel('%'); a.legend(frameon=False,fontsize=8,loc='upper left')
a.set_title('S1-03 · Beaucoup de zéros, et carbonates quasi absents (95,8 % manquants)'); fig.tight_layout(); save(fig,'S1_Descriptif','S1-03_Zeros_et_manquants')
savejson('S1_descriptif.json')
