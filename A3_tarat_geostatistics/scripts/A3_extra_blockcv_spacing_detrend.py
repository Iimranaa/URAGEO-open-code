"""A3 additional analyses: spatial block cross-validation, spacing experiment, detrended variograms, thickness vs strong reduction (AUC).
Runs on the synthetic table by default (URAGEO_DATA_CSV to change). Output: outputs/A3/res/a3_extra_results.json"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, json
from scipy import spatial, stats
from scipy.optimize import curve_fit
rng=np.random.default_rng(2026)
REPO=Path(__file__).resolve().parents[2]
d=pd.read_csv(os.environ.get('URAGEO_DATA_CSV',REPO/'data_synthetic'/'tarat_boreholes_synthetic.csv')); d['PC1']=d.PC1_calc
XYall=d[['x','y']].values; n=len(d)
def sph(h,c0,c,a): h=np.asarray(h,float); r=np.minimum(h/a,1); return np.where(h>0,c0+c*(1.5*r-.5*r**3),0)
def expo(h,c0,c,a): h=np.asarray(h,float); return np.where(h>0,c0+c*(1-np.exp(-3*h/a)),0)
def gaus(h,c0,c,a): h=np.asarray(h,float); return np.where(h>0,c0+c*(1-np.exp(-3*(h/a)**2)),0)
MODELS={'spherical':sph,'exponential':expo,'gaussian':gaus}
def evario(XY,z,lag=100,maxd=1500):
    D=spatial.distance.pdist(XY); g=0.5*spatial.distance.pdist(z[:,None],'sqeuclidean')
    ok=D<=maxd; D=D[ok]; g=g[ok]; b=(D//lag).astype(int); nb=b.max()+1
    N=np.bincount(b,minlength=nb); G=np.bincount(b,weights=g,minlength=nb)/np.maximum(N,1); H=np.bincount(b,weights=D,minlength=nb)/np.maximum(N,1); m=N>=30
    return H[m],G[m],N[m]
def fit(H,G,N):
    best=None; v=G.max()
    for nm,f in MODELS.items():
        try: p,_=curve_fit(f,H,G,p0=[G[0]*.5,max(v-G[0]*.5,1e-9),500],sigma=G/np.sqrt(N),bounds=([0,1e-12,40],[v*2,v*3,4000]),maxfev=20000)
        except Exception: continue
        sse=np.sum(N*((G-f(H,*p))/np.maximum(G,1e-12))**2)
        if best is None or sse<best[3]: best=(nm,f,p,sse)
    return best
def krige(XYt,zt,f,p,T,nn=24):
    nn=min(nn,len(zt)); tree=spatial.cKDTree(XYt); dist,ix=tree.query(T,k=nn)
    P=XYt[ix]; DD=np.linalg.norm(P[:,:,None,:]-P[:,None,:,:],axis=3); Gm=f(np.maximum(DD,1e-6),*p); Gm[:,np.arange(nn),np.arange(nn)]=0
    A=np.ones((len(T),nn+1,nn+1)); A[:,:nn,:nn]=Gm; A[:,nn,nn]=0; b=np.ones((len(T),nn+1)); b[:,:nn]=f(np.maximum(dist,1e-6),*p)
    s=np.linalg.solve(A,b[:,:,None])[:,:,0]; return (s[:,:nn]*zt[ix]).sum(1)
def refit(XY,z,detr=False):
    H,G,N=evario(XY,z); nm,f,p,_=fit(H,G,N); return nm,f,p
def skill(pred,z,ref): return 1-np.mean((pred-z)**2)/np.mean((ref-z)**2)
R={}
VARS={'puiss_m':'thickness','toit_m':'roof depth','PC1':'PC1'}
# ---- 1. spatial block CV with per-fold variogram refit + buffer
def blockcv(k,B,buf=0,refit_each=True):
    z=d[k].values.astype(float); bx=((d.x-d.x.min())//B).astype(int).values; by=((d.y-d.y.min())//B).astype(int).values
    bid=bx*1000+by; ids=np.unique(bid); pred=np.full(n,np.nan); meanp=np.full(n,np.nan)
    nm0,f0,p0=refit(XYall,z)
    for b in ids:
        te=np.where(bid==b)[0]; 
        if buf>0:
            tr_mask=spatial.cKDTree(XYall[te]).query(XYall)[0]>buf
        else: tr_mask=bid!=b
        tr=np.where(tr_mask)[0]
        if len(tr)<100: continue
        if refit_each: nm,f,p=refit(XYall[tr],z[tr])
        else: f,p=f0,p0
        pred[te]=krige(XYall[tr],z[tr],f,p,XYall[te]); meanp[te]=z[tr].mean()
    m=~np.isnan(pred); return dict(B=B,buf=buf,n=int(m.sum()),rmse=float(np.sqrt(np.mean((pred[m]-z[m])**2))),rmse_mean=float(np.sqrt(np.mean((meanp[m]-z[m])**2))),skill=float(skill(pred[m],z[m],meanp[m])),r=float(np.corrcoef(pred[m],z[m])[0,1]))
R['blockcv']={}
for k in VARS:
    R['blockcv'][k]=[blockcv(k,B) for B in (500,1000,1500)]
    print(k,[(r['B'],round(r['skill'],3),round(r['rmse'],2)) for r in R['blockcv'][k]],flush=True)
# ---- 2. thinning experiment (LOO, variogram refit)
def loo(XY,z,f,p):
    tree=spatial.cKDTree(XY); nn=min(24,len(z)-1); dist,ix=tree.query(XY,k=nn+1); ix=ix[:,1:]; dist=dist[:,1:]
    P=XY[ix]; DD=np.linalg.norm(P[:,:,None,:]-P[:,None,:,:],axis=3); Gm=f(np.maximum(DD,1e-6),*p); Gm[:,np.arange(nn),np.arange(nn)]=0
    A=np.ones((len(XY),nn+1,nn+1)); A[:,:nn,:nn]=Gm; A[:,nn,nn]=0; b=np.ones((len(XY),nn+1)); b[:,:nn]=f(np.maximum(dist,1e-6),*p)
    s=np.linalg.solve(A,b[:,:,None])[:,:,0]; return (s[:,:nn]*z[ix]).sum(1)
def thin_mind(mind,seed):
    r=np.random.default_rng(seed); order=r.permutation(n); keep=[]; tree=None
    pts=[]
    for i in order:
        if not pts or np.min(np.linalg.norm(np.array(pts)-XYall[i],axis=1))>=mind: pts.append(XYall[i]); keep.append(i)
    return np.array(keep)
R['thin']={}
for k in ['puiss_m','toit_m']:
    z=d[k].values.astype(float); rows=[]
    for mind in (0,70,140,210,280):
        reps=[] ; ns=[]
        for s in range(4 if mind else 1):
            idx=np.arange(n) if mind==0 else thin_mind(mind,s)
            XY=XYall[idx]; zz=z[idx]; nm,f,p=refit(XY,zz); pr=loo(XY,zz,f,p)
            mloo=np.array([(zz.sum()-v)/(len(zz)-1) for v in zz]); reps.append((skill(pr,zz,mloo),np.sqrt(np.mean((pr-zz)**2)),p[0]/(p[0]+p[1]),p[2])); ns.append(len(idx))
        a=np.array(reps); rows.append(dict(mind=mind,n=float(np.mean(ns)),skill=float(a[:,0].mean()),skill_sd=float(a[:,0].std()),rmse=float(a[:,1].mean()),nug=float(a[:,2].mean()),rng=float(a[:,3].mean())))
        print(k,rows[-1],flush=True)
    R['thin'][k]=rows
# ---- 3. variograms: raw vs quadratic-detrended, nugget relative to sample variance
xs=(d.x-d.x.mean())/1000; ys=(d.y-d.y.mean())/1000; M=np.column_stack([np.ones(n),xs,ys,xs**2,xs*ys,ys**2])
R['vario']={}
for k in ['puiss_m','toit_m','acc_oxy','PC1']:
    z=d[k].values.astype(float); res=z-M@np.linalg.lstsq(M,z,rcond=None)[0]
    for lab,v in (('raw',z),('detrended',res)):
        H,G,N=evario(XYall,v); nm,f,p,_=fit(H,G,N); R['vario'][f'{k}_{lab}']=dict(model=nm,c0=float(p[0]),c=float(p[1]),a=float(p[2]),sill=float(p[0]+p[1]),var=float(v.var(ddof=1)),nug_over_sill=float(p[0]/(p[0]+p[1])),nug_over_var=float(p[0]/v.var(ddof=1)))
        print(k,lab,{a:round(b,3) if not isinstance(b,str) else b for a,b in R['vario'][f'{k}_{lab}'].items()},flush=True)
# ---- 4. Gi* raw thickness
DM=spatial.distance.cdist(XYall,XYall)
def gistar(v,band=250):
    Wb=(DM<=band).astype(float); S=Wb.sum(1); xb=v.mean(); s=np.sqrt((v**2).mean()-xb**2)
    return (Wb@v-xb*S)/(s*np.sqrt((n*(Wb**2).sum(1)-S**2)/(n-1)))
def bh(p,q=.05):
    o=np.argsort(p); th=q*np.arange(1,len(p)+1)/len(p); ok=p[o]<=th; k=np.max(np.where(ok)[0]) if ok.any() else -1; rej=np.zeros(len(p),bool)
    if k>=0: rej[o[:k+1]]=True
    return rej
z=gistar(d.puiss_m.values); p=2*(1-stats.norm.cdf(np.abs(z))); rej=bh(p)
R['gi_raw_thickness']=dict(hot=int((rej&(z>0)).sum()),cold=int((rej&(z<0)).sum()))
print(R['gi_raw_thickness'])
# ---- 5. thickness vs strong reduction: Easting control, block bootstrap AUC
Dv=(d.cl_reduc=='Eleve').values.astype(int); th=d.puiss_m.values
def auc(s,y):
    r=stats.rankdata(s); n1=y.sum(); n0=len(y)-n1; return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
a0=auc(th,Dv); R['auc_thickness']=dict(auc=float(a0),D=int(Dv.sum()))
bx=((d.x-d.x.min())//500).astype(int).values; by=((d.y-d.y.min())//500).astype(int).values; bid=bx*1000+by; ids=np.unique(bid)
bs=[]
for _ in range(1000):
    pick=rng.choice(ids,len(ids)); idx=np.concatenate([np.where(bid==b)[0] for b in pick])
    if Dv[idx].sum()>5: bs.append(auc(th[idx],Dv[idx]))
R['auc_thickness']['ci_block500']=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
# Easting quintile strata
q=pd.qcut(d.x,5,labels=False).values; ws=[];aucs=[]
for s in range(5):
    m=q==s; 
    if Dv[m].sum()>=3: aucs.append((int(s),int(m.sum()),int(Dv[m].sum()),float(auc(th[m],Dv[m]))))
R['auc_thickness']['by_easting_quintile']=aucs
# Mantel-Haenszel-like pooled: weighted mean of stratum AUC
w=np.array([a[2]*(a[1]-a[2]) for a in aucs]); R['auc_thickness']['pooled_within_easting']=float(np.sum([a[3]*wi for a,wi in zip(aucs,w)])/w.sum())
R['auc_thickness']['rho_thickness_easting']=float(stats.spearmanr(d.x,th)[0]); R['auc_thickness']['rho_D_easting']=float(stats.spearmanr(d.x,Dv)[0])
print(R['auc_thickness'])
# logistic reg thickness controlling x,y (simple Newton)
def logit(X,y):
    b=np.zeros(X.shape[1])
    for _ in range(50):
        pz=1/(1+np.exp(-X@b)); W=pz*(1-pz); H=X.T@(X*W[:,None]); g=X.T@(y-pz); b=b+np.linalg.solve(H,g)
    cov=np.linalg.inv(X.T@(X*W[:,None])); return b,np.sqrt(np.diag(cov))
Xs=np.column_stack([np.ones(n),(th-th.mean())/th.std(),(d.x-d.x.mean())/1000,(d.y-d.y.mean())/1000])
b,se=logit(Xs,Dv); R['logit_thickness_ctrl_xy']=dict(beta=float(b[1]),se_iid=float(se[1]),z=float(b[1]/se[1]))
# block bootstrap for beta
bb=[]
for _ in range(500):
    pick=rng.choice(ids,len(ids)); idx=np.concatenate([np.where(bid==b_)[0] for b_ in pick])
    try: bb.append(logit(Xs[idx],Dv[idx])[0][1])
    except Exception: pass
R['logit_thickness_ctrl_xy']['ci_block']=[float(np.percentile(bb,2.5)),float(np.percentile(bb,97.5))]
print(R['logit_thickness_ctrl_xy'])
# thickness by Easting bin
R['thick_by_easting']=[(float(a),float(b_),int(m.sum()),float(th[m].mean())) for a,b_ in [(0,323e3),(323e3,324e3),(324e3,324.5e3),(324.5e3,325e3),(325e3,1e9)] for m in [((d.x>=a)&(d.x<b_)).values]]
o=Path(os.environ.get('URAGEO_OUTPUT',REPO/'outputs'/'A3'))/'res'; o.mkdir(parents=True,exist_ok=True)
json.dump(R,open(o/'a3_extra_results.json','w'),indent=1,default=float)
