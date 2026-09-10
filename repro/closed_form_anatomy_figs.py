"""Closed-form anatomy and closure diagnostics (Figs. toy_closed_form_anatomy, alpha_hybrid_anatomy,
alpha_depth_error_vs_i, alpha_frozen_vs_exact_mean).  Uses the Table-1 baseline constants, the committed
fast-alpha depth CSV, and a resolvent BVP for the exact conditional mean m_alpha(i).  Writes to paper/images_final/."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper" / "images_final"; DATA = REPO / "imagens_tex"
plt.rcParams.update({"font.size":14,"axes.titlesize":15,"axes.labelsize":14,"xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":12,"axes.grid":True,"grid.alpha":.25})
# ---- baseline parameters (Table 1)
g,sig,L,k,beta,eta,zeta,eps=0.01,0.3,0.9,2.0,0.0125,0.32,0.25,0.005; tauR=120.0
chi=(1+g/k)**(-(1+k/g)); ac=0.5*k*g*sig**2; cG=L*chi; rG=np.sqrt(ac/cG); kap=np.sqrt(ac*cG); dg=np.log(1+g/k)/g
A1=kap/(k*(kap+beta))*eta/np.sqrt(beta); A2=np.sqrt(beta)/eta
Aflow=(2/k)*kap/(beta+2*kap); Aal=(2*L*eps/zeta)/(beta+2*kap); AR=beta/(1/tauR+2*kap)*(Aflow+Aal)
print(f"chi={chi:.4f} a_c={ac:.5f} c_GLFT={cG:.4f} r_G={rG:.5f} kappa={kap:.5f} delta_g={dg:.4f}")
print(f"A1={A1:.4f} A2={A2:.4f} A_I={A1*A2:.4f} | A_flow={Aflow:.4f} A_alpha={Aal:.4f} A_R={AR:.4f}")
i=np.linspace(-6,6,1201); q0=10
# ---- toy closed form
hI=A1*np.tanh(A2*i); th=i-k*hI; M=np.sqrt(np.cosh(i)/np.cosh(th)); r=rG*M; s=r**2/6*np.tanh(th)
cub=lambda s,q:(s/k)*(q*q-q+1/3); curv=lambda r,q:-(r/(2*k))*(2*q-1)
print(f"toy: h_I(6)={hI[-1]:.3f}, r/rG(6)={M[-1]:.3f}, max|s|/rG^2={np.max(np.abs(s))/rG**2:.3f}, cubic contrib at q=10,i=6: {cub(s[-1],q0):+.4f}, curvature contrib at q=10,i=0: {curv(rG,q0):+.4f}, at i=6: {curv(r[-1],q0):+.4f}")
fig,ax=plt.subplots(1,3,figsize=(15,4.4))
ax[0].plot(i,hI,color="#8c564b",lw=2.5,label=r"$h_I(i)=A_1\tanh(A_2 i)$"); ax[0].plot(i,A1*A2*i,color="k",lw=1,ls=":",label=r"origin slope $A_I\,i$"); ax[0].set_ylim(-1.1,1.1); ax[0].set_title("loading"); ax[0].legend(loc="upper left")
ax[1].plot(i,M,color="#1f77b4",lw=2.5,label=r"$r(i)/r_{\rm G}$"); ax[1].plot(i,np.tanh(th),color="#d62728",lw=2,ls="--",label=r"$\tanh\vartheta(i)=6s/r^2$"); ax[1].set_title("curvature ratio and cubic sign"); ax[1].legend(loc="upper center")
ax[2].plot(i,hI,color="#8c564b",lw=2,label=r"loading $h_I$"); ax[2].plot(i,curv(r,q0),color="#1f77b4",lw=2,label=r"curvature $-\frac{r(i)}{2k}(2q-1)$"); ax[2].plot(i,cub(s,q0),color="#d62728",lw=2,label=r"cubic $\frac{s(i)}{k}(q^2-q+\frac{1}{3})$")
ax[2].plot(i,hI+curv(r,q0)+cub(s,q0),color="k",lw=2.5,ls="--",label=r"$\delta^a_{\rm toy}-\delta_\gamma$"); ax[2].set_title(r"ask-depth terms at $q=10$"); ax[2].legend(loc="upper left",fontsize=10.5)
for a in ax: a.set_xlabel("imbalance $i$"); a.axhline(0,color="k",lw=.7,alpha=.4)
fig.tight_layout(); fig.savefig(f"{OUT}/toy_closed_form_anatomy.png",dpi=170); plt.close(fig); print("wrote toy anatomy")
# ---- fast-alpha hybrid
hal=Aal*np.tanh(i); thI=i-k*hI; MI=np.sqrt(np.cosh(i)/np.cosh(thI)); gI=np.tanh(A2*i)**2
gate=1+gI*(MI-1); hshift=MI*hI+gate*hal; thfull=i-k*hshift; rh=rG*MI; sh=rh**2/6*np.tanh(thfull)
band=np.abs(i)<=3; print(f"hybrid: h_shift(6)={hshift[-1]:.3f}, M_I(6)={MI[-1]:.3f}, sign(theta_full*i) on |i|<=3: {np.unique(np.sign(thfull[band]*i[band]))}, theta_full(6)={thfull[-1]:.3f}, theta_I(6)={thI[-1]:.3f}, k(A_I+A_alpha)={k*(A1*A2+Aal):.3f}")
fig,ax=plt.subplots(2,2,figsize=(13,8.6))
a=ax[0,0]; a.plot(i,MI*hI,color="#8c564b",lw=2,label=r"flow $\mathcal{M}_I h_I$"); a.plot(i,gate*hal,color="#2ca02c",lw=2,label=r"gated alpha $[1+g(\mathcal{M}_I-1)]h_\alpha$"); a.plot(i,hshift,color="k",lw=2.5,ls="--",label=r"$h_{\rm shift}$"); a.set_title("skew components"); a.legend(loc="upper left")
a=ax[0,1]; a.plot(i,thI,color="#8c564b",lw=2,label=r"flow shift $\vartheta_I=i-kh_I$"); a.plot(i,thfull,color="k",lw=2.5,label=r"full shift $\vartheta_{\rm full}=i-kh_{\rm shift}$"); a.plot(i,i,color="gray",lw=1,ls=":",label="$i$"); a.set_title("effective imbalances"); a.legend(loc="upper left")
a=ax[1,0]; a.plot(i,MI,color="#1f77b4",lw=2.5,label=r"$\mathcal{M}_I(i)=r(i)/r_{\rm G}$"); a.plot(i,gI,color="#9467bd",lw=2,ls="--",label=r"gate $g(i)=\tanh^2(A_2 i)$"); a.plot(i,np.tanh(thfull),color="#d62728",lw=2,ls="-.",label=r"$\tanh\vartheta_{\rm full}=6s/r^2$"); a.set_title("curvature multiplier, gate, cubic sign"); a.legend(loc="upper center",fontsize=11)
a=ax[1,1]; a.plot(i,hshift,color="k",lw=2,label=r"skew $h_{\rm shift}$"); a.plot(i,curv(rh,q0),color="#1f77b4",lw=2,label=r"curvature $-\frac{r(i)}{2k}(2q-1)$"); a.plot(i,cub(sh,q0),color="#d62728",lw=2,label=r"cubic $\frac{s(i)}{k}(q^2-q+\frac{1}{3})$"); a.plot(i,hshift+curv(rh,q0)+cub(sh,q0),color="#E69F00",lw=2.5,ls="--",label=r"$\delta^a_{\rm hybrid}-\delta_\gamma$"); a.set_title(r"ask-depth terms at $q=10$"); a.legend(loc="upper left",fontsize=10.5)
for a in ax.ravel(): a.set_xlabel("imbalance $i$"); a.axhline(0,color="k",lw=.7,alpha=.4)
fig.tight_layout(); fig.savefig(f"{OUT}/alpha_hybrid_anatomy.png",dpi=170); plt.close(fig); print("wrote hybrid anatomy")
# ---- error vs i with stationary density (fast-alpha CSV)
d=np.genfromtxt(f"{DATA}/alpha_studies_T10000_qmax60/alpha_reduced_galerkin_depths_vs_imbalance.csv",delimiter=",",names=True)
sd=np.sqrt(eta**2/(2*beta)); ii=np.linspace(-6,6,601); pi=np.exp(-ii**2/(2*sd**2))/(sd*np.sqrt(2*np.pi))
fig,ax=plt.subplots(1,2,figsize=(13,4.6),sharey=True)
for a,(ca,cb),name in zip(ax,[("ask_sc","bid_sc"),("ask_gal","bid_gal")],["Alpha/$I$ hybrid cubic","Alpha/$I$ Galerkin-cubic"]):
    for qv,col in zip([0,10,15],["#1f77b4","#E69F00","#d62728"]):
        m=np.isclose(d["q"],qv); x=d["y"][m]; e=np.sqrt(0.5*((d[ca][m]-d["ask_hjb"][m])**2+(d[cb][m]-d["bid_hjb"][m])**2))
        a.plot(x,e,lw=2.2,color=col,label=f"$q={qv}$")
        w=np.exp(-x**2/(2*sd**2)); w/=w.sum(); print(f"  {name} q={qv}: uniform RMS={np.sqrt(np.mean(e**2)):.3f} weighted RMS={np.sqrt(np.sum(w*e**2)):.3f} max={e.max():.3f} at i={x[np.argmax(e)]:+.1f}")
    a.set_yscale("log"); a.set_ylim(3e-3,6); a.set_xlabel("imbalance $i$"); a.set_title(name); a.legend(loc="upper center",ncol=3)
    a2=a.twinx(); a2.fill_between(ii,0,pi,color="gray",alpha=.18,lw=0); a2.set_ylim(0,pi.max()*3.2); a2.set_yticks([]); a2.grid(False)
    if a is ax[1]: a2.set_ylabel(r"stationary law $\pi_I$ (shaded)",color="gray")
ax[0].set_ylabel(r"depth error vs RHJB (RMS over ask, bid)")
fig.tight_layout(); fig.savefig(f"{OUT}/alpha_depth_error_vs_i.png",dpi=170); plt.close(fig); print("wrote error vs i")
# ---- frozen-I closure vs exact conditional mean (resolvent BVP)
N=6001; x=np.linspace(-20,20,N); h=x[1]-x[0]
rhs=2*L*eps*np.tanh(x)
main=np.full(N, zeta+eta**2/h**2); lower=np.zeros(N-1); upper=np.zeros(N-1)
# (zeta - L_I) m = rhs, L_I m = -beta x m' + eta^2/2 m'' ; centered differences
lower[:]=-(eta**2/(2*h**2))-beta*x[1:]/(2*h)   # coefficient of m_{j-1} in row j:  -(eta^2/2h^2) - beta x_j/(2h)
upper[:]=-(eta**2/(2*h**2))+beta*x[:-1]/(2*h)  # coefficient of m_{j+1} in row j
A=np.zeros((N,N)); idx=np.arange(N); A[idx,idx]=main; A[idx[1:],idx[:-1]]=lower; A[idx[:-1],idx[1:]]=upper
A[0,:]=0; A[0,0]=1; A[0,1]=-1; rhs[0]=0; A[-1,:]=0; A[-1,-1]=1; A[-1,-2]=-1; rhs[-1]=0   # Neumann ends
m=np.linalg.solve(A,rhs); abar=2*L*eps/zeta*np.tanh(x); j0=N//2
print(f"m_alpha'(0)={(m[j0+1]-m[j0-1])/(2*h):.4f}  abar'(0)={2*L*eps/zeta:.4f}  ratio={(m[j0+1]-m[j0-1])/(2*h)/(2*L*eps/zeta):.3f}; m(6)={m[np.argmin(np.abs(x-6))]:.4f} abar(6)={abar[np.argmin(np.abs(x-6))]:.4f}")
sel=np.abs(x)<=6
fig,ax=plt.subplots(1,1,figsize=(7.5,4.2))
ax.plot(x[sel],abar[sel],color="#E69F00",lw=2.5,label=r"frozen-$I$ closure $\bar{\alpha}(i)=\frac{2\bar{\Lambda}\epsilon}{\zeta}\tanh i$")
ax.plot(x[sel],m[sel],color="k",lw=2.5,ls="--",label=r"exact stationary mean $m_\alpha(i)$")
ax.axhline(0,color="k",lw=.7,alpha=.4); ax.set_xlabel("imbalance $i$"); ax.set_ylabel(r"conditional alpha"); ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/alpha_frozen_vs_exact_mean.png",dpi=170); plt.close(fig); print("wrote frozen vs exact")
