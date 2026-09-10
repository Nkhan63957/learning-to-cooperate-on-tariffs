"""
A2 -- Partner transparency as a route across the gap (exact LOLA).

Memory-1 repeated PD with the calibrated tariff stage game (C = free trade, D = Nash):
    R=0.0 (CC), T=0.218 (defect vs cooperator), P=-0.093 (DD), S=-0.30 (cooperator vs defector).
Each agent's policy is a 5-logit memory-1 strategy: theta = [start, CC, CD, DC, DD] giving
P(cooperate) = sigmoid(theta). Discounted value of two memory-1 strategies is closed form
via the stationary/resolvent construction (Press & Dyson 2012).

NAIVE learning: each agent ascends its own value assuming the opponent is fixed.
LOLA learning: each agent ascends its value while accounting for the opponent's one
naive learning step (opponent-shaping). Update for agent 1:
    dtheta1 = grad_th1 V1 + eta * (d/dth1 d/dth2 V2) @ (grad_th2 V1)
The shaping term is what lets mutually transparent learners reach reciprocal cooperation
with NO exogenous committed reciprocator. Question: does transparency substitute for
enforcement, and is the cooperation it reaches self-enforcing (reciprocal, not naive)?
"""
import numpy as np, json

R,T,P,S = 0.0, 0.218, -0.093, -0.30
DELTA = 0.9
SWAP = [0,2,1,3]                 # P1-perspective joint state -> P2 own-perspective index
U1 = np.array([R,S,T,P])         # P1 payoff by joint state [CC,CD,DC,DD]
U2 = np.array([R,T,S,P])         # P2 payoff

def sig(x): return 1/(1+np.exp(-x))

def values(th1, th2, delta=DELTA):
    p1 = sig(th1); q = sig(th2)          # coop probs; index0 = start, 1..4 = states
    p0, p = p1[0], p1[1:]
    q0, qc = q[0], q[1:]
    qs = qc[SWAP]                         # P2 coop prob aligned to P1-perspective states
    x0 = np.array([p0*q0, p0*(1-q0), (1-p0)*q0, (1-p0)*(1-q0)])
    M = np.empty((4,4))
    M[:,0] = p*qs; M[:,1] = p*(1-qs); M[:,2] = (1-p)*qs; M[:,3] = (1-p)*(1-qs)
    d = (1-delta) * x0 @ np.linalg.inv(np.eye(4) - delta*M)
    return float(d@U1), float(d@U2)

def grad(fun_idx, th_self, th_other, self_is_1=True, h=1e-5):
    """gradient of V_{fun_idx} wrt th_self."""
    g = np.zeros(5)
    for i in range(5):
        e = np.zeros(5); e[i]=h
        if self_is_1:
            vp = values(th_self+e, th_other); vm = values(th_self-e, th_other)
        else:
            vp = values(th_other, th_self+e); vm = values(th_other, th_self-e)
        g[i] = (vp[fun_idx]-vm[fun_idx])/(2*h)
    return g

def grad_th2_V2(th1, th2, h=1e-5):
    g=np.zeros(5)
    for i in range(5):
        e=np.zeros(5); e[i]=h
        g[i]=(values(th1,th2+e)[1]-values(th1,th2-e)[1])/(2*h)
    return g

def mixed_hess_V2(th1, th2, h=1e-4):
    """H[a,b] = d^2 V2 / d th1[a] d th2[b]."""
    H=np.zeros((5,5))
    for a in range(5):
        e=np.zeros(5); e[a]=h
        gp=grad_th2_V2(th1+e, th2); gm=grad_th2_V2(th1-e, th2)
        H[a,:]=(gp-gm)/(2*h)
    return H

def train(mode, steps=600, alpha=1.0, eta=1.0, seed=0, th2_fixed=None):
    rng=np.random.default_rng(seed)
    th1=rng.normal(0,0.5,5); th2=(th2_fixed.copy() if th2_fixed is not None else rng.normal(0,0.5,5))
    traj=[]
    for t in range(steps):
        g1=grad(0, th1, th2, True); g2=grad(1, th2, th1, False)
        if mode=="lola":
            H2=mixed_hess_V2(th1,th2); d1=g1+eta*(H2@grad(0,th2,th1,False)*0+ H2@grad_th2_V2(th1,th2)*0)
            # correction term: (d/dth1 d/dth2 V2) @ (grad_th2 V1)
            gth2V1=grad(0, th2, th1, self_is_1=False)  # grad of V1 wrt th2
            d1=g1+eta*(H2@gth2V1)
            # symmetric for agent 2
            H1=mixed_hess_V1(th1,th2); gth1V2=grad(1, th1, th2, self_is_1=True)
            d2=g2+eta*(H1@gth1V2)
        else:
            d1,d2=g1,g2
        th1=th1+alpha*d1
        if th2_fixed is None: th2=th2+alpha*d2
        if t%20==0 or t==steps-1:
            v1,v2=values(th1,th2); traj.append((t,v1,v2,sig(th1).tolist(),sig(th2).tolist()))
    return th1,th2,traj

def grad_th1_V1(th1,th2,h=1e-5):
    g=np.zeros(5)
    for i in range(5):
        e=np.zeros(5); e[i]=h
        g[i]=(values(th1+e,th2)[0]-values(th1-e,th2)[0])/(2*h)
    return g
def mixed_hess_V1(th1,th2,h=1e-4):
    """H[a,b] = d^2 V1 / d th2[a] d th1[b]."""
    H=np.zeros((5,5))
    for a in range(5):
        e=np.zeros(5); e[a]=h
        gp=grad_th1_V1(th1,th2+e); gm=grad_th1_V1(th1,th2-e)
        H[a,:]=(gp-gm)/(2*h)
    return H

if __name__=="__main__":
    # 1) naive vs LOLA self-play, several seeds
    out={"delta":DELTA,"payoffs":{"R":R,"T":T,"P":P,"S":S}}
    for mode in ["naive","lola"]:
        finals=[]; sample=None
        for s in range(6):
            th1,th2,traj=train(mode,seed=s)
            v1,v2=values(th1,th2); finals.append((v1+v2)/2)
            if s==0: sample={"traj":traj,"p1":sig(th1).tolist(),"p2":sig(th2).tolist()}
        out[mode]={"mean_joint_payoff":float(np.mean(finals)),
                   "all_joint":[float(x) for x in finals],
                   "sample":sample}
        print(f"{mode:5s}: mean joint payoff={np.mean(finals):+.3f}  (R={R} coop, P={P} defect)")
        print(f"        emergent P1 policy [start,CC,CD,DC,DD]=",[f'{x:.2f}' for x in sample['p1']])
    # 2) exploitability: LOLA agent vs fixed ALLD
    alld=np.full(5,-6.0)
    th1,_,_=train("lola",seed=0,th2_fixed=alld)
    v1,v2=values(th1,alld)
    out["lola_vs_alld"]={"p1":sig(th1).tolist(),"V_lola":v1,"V_alld":v2}
    print(f"LOLA vs ALLD: LOLA policy=",[f'{x:.2f}' for x in sig(th1)],f" V_lola={v1:+.3f} V_alld={v2:+.3f}")
    json.dump(out,open("results_A2.json","w"),indent=2)
    print("saved results_A2.json")