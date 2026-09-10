#!/usr/bin/env python3
"""Parameter holdout: vary one primitive at a time around the paper baseline;
measure the SRN sc-cubic (fast-alpha) RMSE vs its own RHJB. Tests whether the
elementary closed form fits OUT of the base calibration (not just because it was
tuned to the base RHJB). Incremental log + CSV + plot."""
import os, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG = open(ROOT/"rerun_campaign"/"logs"/"param_holdout.log","w",buffering=1)
CSV = open(ROOT/"rerun_campaign"/"csv"/"param_holdout.csv","w",buffering=1)
def log(m): print(m); LOG.write(m+"\n")
CSV.write("param,value,rel_to_base,sc_vs_srn_unif,sc_vs_srn_stat\n")
QS=(-5,0,10,15); yy=np.linspace(-6,6,241)
BASE={"ASOU5_ALPHA_BETA":0.0125,"ASOU5_ALPHA_ETA":0.32,"ASOU5_ALPHA_ZETA":0.25,
      "ASOU5_ALPHA_EPSILON":0.005,"ASOU5_ALPHA_K":2.0}
SWEEP={"ASOU5_ALPHA_BETA":[0.00625,0.0125,0.025],"ASOU5_ALPHA_ETA":[0.24,0.32,0.42],
       "ASOU5_ALPHA_ZETA":[0.15,0.25,0.40],"ASOU5_ALPHA_EPSILON":[0.0025,0.005,0.01],
       "ASOU5_ALPHA_K":[1.0,2.0,3.0]}
def load():
    for m in list(sys.modules):
        if "alpha_reduced" in m: del sys.modules[m]
    sys.path.insert(0,str(ROOT/"alpha_studies")); import alpha_reduced_cole_hopf as A; return A
def sc(A,q,y):
    P=A.P;k=P.k;rG=2*P.p_glft;D0=P.delta0; qf=np.asarray(q,float);y=np.asarray(y,float)
    tf,ap,g=A.toy_h_flow_alpha_components(y); th=y-k*tf; r=rG*np.sqrt(np.cosh(y)/np.cosh(th)); m=r/rG
    sh=m*tf+(1+g*(m-1))*ap; thf=y-k*sh; s=(r**2/6)*np.tanh(thf)
    a=D0+sh-(r/(2*k))*(2*qf-1)+(s/k)*(qf**2-qf+1/3); b=D0-sh+(r/(2*k))*(2*qf+1)-(s/k)*(qf**2+qf+1/3)
    return np.where(q>-P.qmax,a,np.inf),np.where(q<P.qmax,b,np.inf)
def rmse(A,u,w):
    rh=lambda q,y:A.y_aware_depths(u,np.asarray(q,int),np.asarray(y,float)); E,W=[],[]
    for qv in QS:
        qa=np.full_like(yy,qv);x,y2=sc(A,qa,yy);ar,br=rh(qa,yy)
        for a,r in ((x,ar),(y2,br)):
            g=np.isfinite(a)&np.isfinite(r);E.append((a-r)[g]);W.append((w if w is not None else np.ones_like(yy))[g])
    e=np.concatenate(E);ww=np.concatenate(W);return float(np.sqrt(np.mean(e**2))),float(np.sqrt(np.sum(ww*e**2)/np.sum(ww)))
log("param holdout: sc-cubic RMSE vs RHJB, one primitive varied at a time (base gamma=0.01)")
res={}
for pname,vals in SWEEP.items():
    res[pname]=[]
    for v in vals:
        for kk,bv in BASE.items(): os.environ[kk]=str(bv)
        os.environ[pname]=str(v); os.environ["ASOU5_ALPHA_GAMMA"]="0.01"
        A=load();P=A.P; sig=P.eta/np.sqrt(2*P.beta); w=np.exp(-yy**2/(2*sig**2));w/=w.sum()
        u,_=A.solve_ou_ergodic(A.ALPHA_BAR); ru,rs=rmse(A,u,w)
        rel=v/BASE[pname]; res[pname].append((rel,rs))
        log(f"  {pname.split('_')[-1]:8s}={v:<8g} (x{rel:.2f} base): unif {ru:.4f}  stat {rs:.4f}")
        CSV.write(f"{pname.split('_')[-1]},{v},{rel:.3f},{ru:.5f},{rs:.5f}\n")
fig,ax=plt.subplots(figsize=(8,5))
for pname,pts in res.items():
    x=[p[0] for p in pts]; y=[p[1] for p in pts]
    ax.plot(x,y,"o-",label=pname.split("_")[-1].lower())
ax.axhline(0.049,color="k",ls="--",lw=.8,label="baseline 0.049")
ax.set_xlabel("parameter / baseline"); ax.set_ylabel("stationary-weighted sc-cubic RMSE vs RHJB")
ax.legend(fontsize=9); ax.grid(alpha=.3); ax.set_title("Closed-form fit is stable across the calibration neighbourhood")
fig.tight_layout(); fig.savefig(ROOT/"rerun_campaign"/"plots"/"param_holdout.png",dpi=150)
log("wrote plots/param_holdout.png and csv/param_holdout.csv")
