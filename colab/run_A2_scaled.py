"""
run_A2_scaled.py: the configuration that produces the Section 6.1 results.

run_A2.py alone does not reproduce them. The opponent-shaping term is second
order in the payoffs while the naive gradient is first order, and the calibrated
stage payoffs are small (R=0, T=0.218, P=-0.093, S=-0.30), so at unit scale the
shaping term is negligible and LOLA sits on top of naive learning.

This wrapper rescales the payoff matrix by 12, a monotone transformation that
leaves the preference ordering and every equilibrium unchanged, and divides
reported payoffs back by 12 so results stay in original units.

Configuration: SCALE=12.0, eta=40, alpha=0.08, steps=700, 6 seeds.
Reproduces: naive -0.0904 (sd 0.0001), LOLA -0.0329 (sd 0.0048),
LOLA vs fixed defector -0.0965 against P=-0.093 and S=-0.30.
"""

import numpy as np, json
import run_A2 as A2
SCALE=12.0
A2.U1=A2.U1*SCALE; A2.U2=A2.U2*SCALE
from run_A2 import values, grad, mixed_hess_V2, mixed_hess_V1, sig, R,T,P,S,DELTA
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def train(mode,steps=700,alpha=0.08,eta=40,seed=0,th2fix=None,log=False):
    rng=np.random.default_rng(seed); th1=rng.normal(0,0.5,5)
    th2=(th2fix.copy() if th2fix is not None else rng.normal(0,0.5,5))
    traj=[]
    for t in range(steps):
        g1=grad(0,th1,th2,True); g2=grad(1,th2,th1,False)
        if mode=="lola":
            d1=g1+eta*(mixed_hess_V2(th1,th2)@grad(0,th2,th1,self_is_1=False))
            d2=g2+eta*(mixed_hess_V1(th1,th2)@grad(1,th1,th2,self_is_1=True))
        else: d1,d2=g1,g2
        th1=th1+alpha*d1
        if th2fix is None: th2=th2+alpha*d2
        if log and (t%10==0 or t==steps-1):
            v1,v2=values(th1,th2); traj.append((t,(v1+v2)/2/SCALE))
    v1,v2=values(th1,th2)
    return (v1+v2)/2/SCALE, sig(th1), sig(th2), traj

# 1) distribution over seeds
res={"delta":DELTA,"scale":SCALE,"stage":{"R":R,"T":T,"P":P,"S":S}}
for mode in ["naive","lola"]:
    js=[train(mode,seed=s)[0] for s in range(6)]
    res[mode+"_joint"]=[float(x) for x in js]
    print(f"{mode:5s}: joint/period mean={np.mean(js):+.4f}  sd={np.std(js):.4f}")

# 2) representative trajectories + emergent policy
_,_,_,tj_naive=train("naive",seed=1,log=True)
jl,p1,p2,tj_lola=train("lola",seed=1,log=True)
res["traj_naive"]=tj_naive; res["traj_lola"]=tj_lola
res["policy_p1"]=p1.tolist(); res["policy_p2"]=p2.tolist()
print("emergent LOLA P1:",[f'{x:.2f}' for x in p1])
print("emergent LOLA P2:",[f'{x:.2f}' for x in p2])

# 3) exploitability: LOLA vs fixed ALLD
alld=np.full(5,-8.0)
_,pe,_,_=train("lola",seed=0,th2fix=alld)
th_e=np.log(np.clip(pe,1e-6,1-1e-6)/(1-np.clip(pe,1e-6,1-1e-6)))
ve1,ve2=values(th_e,alld)
res["lola_vs_alld"]={"policy":pe.tolist(),"V_lola":ve1/SCALE,"V_alld":ve2/SCALE}
print(f"LOLA vs ALLD: V_lola={ve1/SCALE:+.4f} (P={P}, S={S}); policy after being defected (CD)={pe[2]:.2f}")
json.dump(res,open("results_A2.json","w"),indent=2)

# ---- figure ----
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9.6,4.0),gridspec_kw={"width_ratios":[1.3,1]})
ax1.axhspan(-0.02,0.01,color="#2e7d32",alpha=0.10); ax1.text(20,-0.008,"cooperation (R)",color="#2e7d32",fontsize=8)
ax1.axhline(P,ls="--",lw=1,color="#b71c1c"); ax1.text(20,P-0.006,"defection (P)",color="#b71c1c",fontsize=8)
tn=np.array(tj_naive); tl=np.array(tj_lola)
ax1.plot(tn[:,0],tn[:,1],color="#b71c1c",lw=1.8,label="naive learners")
ax1.plot(tl[:,0],tl[:,1],color="#1565c0",lw=1.8,label="transparent (LOLA) learners")
ax1.set_xlabel("learning step"); ax1.set_ylabel("joint payoff per period")
ax1.set_title("A2  Mutual transparency crosses the gap"); ax1.legend(fontsize=8,loc="center right"); ax1.set_ylim(-0.11,0.02)
# right: emergent reciprocal policy
states=["after\nCC","after\nCD","after\nDC","after\nDD"]; x=np.arange(4); w=0.38
ax2.bar(x-w/2,p1[1:],w,color="#1565c0",label="learner 1")
ax2.bar(x+w/2,p2[1:][A2.SWAP],w,color="#66aaff",label="learner 2")
ax2.set_xticks(x); ax2.set_xticklabels(states,fontsize=8); ax2.set_ylabel("P(cooperate)")
ax2.set_title("Emergent policy is reciprocal\n(cooperate after C, punish D)"); ax2.legend(fontsize=8); ax2.set_ylim(0,1.05)
fig.tight_layout(); fig.savefig("figA2_transparency.png",dpi=150)
print("wrote figA2_transparency.png")