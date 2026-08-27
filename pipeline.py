"""
pipeline.py — all the science in one self-contained file.

Config, the N-country game, BACI calibration, and every experiment live here.
Run it to produce data/processed/results.json. dashboard.py and deployment.py
read ONLY that JSON (no shared module), so these three files are standalone.

    python pipeline.py

Recomputed here (numpy, from the constants below + the BACI files):
    two_country, multilateral (N=30), sensitivity (16-combo robustness).
Ingested if present in data/processed/ (computed in Colab; torch / World Bank API):
    qre, reciprocity, mechanism, ppo, phase3 (natural experiment).
"""
from __future__ import annotations
from pathlib import Path
import json, numpy as np, pandas as pd

# ===================== CONFIG (single source of truth) =====================
ROOT = Path(__file__).parent
DATA_RAW, DATA_PROC, FIGURES = ROOT/"data"/"raw", ROOT/"data"/"processed", ROOT/"figures"
RESULTS_JSON = DATA_PROC/"results.json"

KAPPA, DELTA = 1.0, 0.10                       # DWL curvature; trade-contraction coupling
TWO_COUNTRY = {"iso": ["USA","CHN"], "b": [0.66, 0.56],
               "expect": {"nash": [0.610, 0.499], "binding_delta_star": 0.700}}
COUNTRIES_30 = ["USA","CHN","JPN","DEU","IND","GBR","FRA","ITA","BRA","CAN","RUS",
    "KOR","AUS","MEX","IDN","ESP","NLD","SAU","TUR","CHE","POL","SWE","BEL","THA",
    "ARG","ZAF","ARE","EGY","VNM","MYS"]
BACI_CSV, BACI_CODES_CSV = DATA_RAW/"BACI_HS22_Y2024_V202601.csv", DATA_RAW/"country_codes_V202601.csv"
B_LO, B_HI, B_MAPPING = 0.50, 0.68, "linear"
SWEEP_DELTAS, SWEEP_MAPPINGS = [0.05,0.10,0.20,0.30], ["linear","rank","sqrt","log"]
EXTERNAL = {"qre":"qre_results.json","reciprocity":"reciprocity_results.json",
            "mechanism":"mechanism_results.json","ppo":"ppo_results.json",
            "natural_expt":"phase3_results.json"}

# ===================== THE N-COUNTRY GAME =====================
class NCountryTradeGame:
    """W_i = b_i*t_i - sum_j w_ij*b_j*t_j - 0.5*k_i*t_i^2 - delta*t_i*(sum_j w_ij*t_j).
       N=2 with W=[[0,1],[1,0]] reproduces the calibrated US-China game."""
    def __init__(self, b, W, kappa=1.0, delta=0.10):
        self.b=np.asarray(b,float); self.N=len(self.b); self.W=np.asarray(W,float)
        self.k=np.full(self.N,kappa) if np.isscalar(kappa) else np.asarray(kappa,float)
        self.d=float(delta)
    def welfare(self,t):
        t=np.asarray(t,float)
        return self.b*t - self.W@(self.b*t) - 0.5*self.k*t**2 - self.d*t*(self.W@t)
    def nash(self,iters=4000,tol=1e-12):
        t=np.zeros(self.N)
        for _ in range(iters):
            nt=np.clip((self.b-self.d*(self.W@t))/self.k,0,1)
            if np.max(np.abs(nt-t))<tol: t=nt; break
            t=nt
        return t
    def folk_delta_star(self):
        nash=self.nash(); Wn=self.welfare(nash); ds=np.full(self.N,np.nan)
        for i in range(self.N):
            t=np.zeros(self.N); t[i]=np.clip(self.b[i]/self.k[i],0,1); Wdef=self.welfare(t)[i]
            ds[i]=np.inf if Wn[i]>=0 else (Wdef/(-Wn[i]))/(1+Wdef/(-Wn[i]))
        return ds, nash, Wn

# ===================== CALIBRATION FROM BACI =====================
def load_baci_matrix():
    iso=COUNTRIES_30; N=len(iso); cc=pd.read_csv(BACI_CODES_CSV)
    cand={}
    for _,r in cc.iterrows(): cand.setdefault(r["country_iso3"],[]).append(int(r["country_code"]))
    assert not [c for c in iso if c not in cand], "ISO3 missing from codes file"
    tot={}                                            # resolve codes by DATA PRESENCE
    for ch in pd.read_csv(BACI_CSV,usecols=["i","v"],chunksize=3_000_000):
        for k,v in ch.groupby("i")["v"].sum().items(): tot[k]=tot.get(k,0)+v
    codes={c:max(cand[c],key=lambda k:tot.get(k,0)) for c in iso}
    idx={codes[c]:k for k,c in enumerate(iso)}; keep=set(codes.values()); M=np.zeros((N,N))
    for ch in pd.read_csv(BACI_CSV,usecols=["i","j","v"],chunksize=2_000_000):
        ch=ch[ch["i"].isin(keep)&ch["j"].isin(keep)]
        for (j,i),v in ch.groupby(["j","i"])["v"].sum().items(): M[idx[j],idx[i]]+=v
    np.fill_diagonal(M,0.0)
    assert (M.sum(1)>0).all(), "a country has zero trade -> code resolution failed"
    return M, codes

def market_power(M, mapping=B_MAPPING):
    s=M.sum(1)/M.sum(); N=len(s)
    if   mapping=="linear": x=s/s.max()
    elif mapping=="rank":   x=np.argsort(np.argsort(s))/(N-1)
    elif mapping=="sqrt":   x=np.sqrt(s)/np.sqrt(s.max())
    elif mapping=="log":    x=(np.log(s)-np.log(s.min()))/(np.log(s.max())-np.log(s.min()))
    else: raise ValueError(mapping)
    return B_LO+(B_HI-B_LO)*x

def trade_matrix(M):
    W=M/M.sum(1,keepdims=True); np.fill_diagonal(W,0.0); return W/W.sum(1,keepdims=True)

# ===================== EXPERIMENTS =====================
def exp_two_country():
    g=NCountryTradeGame(TWO_COUNTRY["b"],[[0,1],[1,0]],KAPPA,DELTA)
    ds,nash,Wn=g.folk_delta_star(); e=TWO_COUNTRY["expect"]
    assert abs(nash[0]-e["nash"][0])<1e-2 and abs(np.nanmax(ds)-e["binding_delta_star"])<1e-2, \
        "two-country regression guard FAILED — calibration drifted"
    return {"iso":TWO_COUNTRY["iso"],"b":TWO_COUNTRY["b"],"nash":nash.tolist(),
            "delta_star":ds.tolist(),"binding":float(np.nanmax(ds)),"guard_passed":True}

def exp_multilateral():
    M,codes=load_baci_matrix(); iso=COUNTRIES_30
    b=market_power(M); W=trade_matrix(M)
    ds,nash,Wn=NCountryTradeGame(b,W,KAPPA,DELTA).folk_delta_star()
    exposure=W@b; imp=M.sum(1)/M.sum(); fin=np.isfinite(ds)
    def resid(y,x): A=np.vstack([x,np.ones_like(x)]).T; c,*_=np.linalg.lstsq(A,y,rcond=None); return y-A@c
    pr=float(np.corrcoef(resid(ds[fin],b[fin]),resid(exposure[fin],b[fin]))[0,1])
    heg=[iso[i] for i in range(len(iso)) if not np.isfinite(ds[i])]
    return {"iso":iso,"codes":codes,"b":b.tolist(),"nash":nash.tolist(),
            "delta_star":[None if not np.isfinite(x) else x for x in ds],
            "W_nash":Wn.tolist(),"exposure":exposure.tolist(),"import_share":imp.tolist(),
            "concentration_hhi":((W**2).sum(1)).tolist(),
            "binding_country":iso[int(np.nanargmax(np.where(fin,ds,-1)))],
            "binding_delta_star":float(np.nanmax(ds[fin])),"hegemons":heg,
            "corr_b":float(np.corrcoef(b[fin],ds[fin])[0,1]),
            "partial_corr_exposure_given_b":pr}

def exp_sensitivity():
    M,_=load_baci_matrix(); W=trade_matrix(M); iso=COUNTRIES_30
    iMEX,iCAN=iso.index("MEX"),iso.index("CAN"); out={}; allbelow=True
    for mp in SWEEP_MAPPINGS:
        b=market_power(M,mp)
        for dl in SWEEP_DELTAS:
            ds,*_=NCountryTradeGame(b,W,KAPPA,dl).folk_delta_star(); m=np.isfinite(ds)
            A=np.vstack([b[m],np.ones(m.sum())]).T; c,*_=np.linalg.lstsq(A,ds[m],rcond=None)
            rM=float(ds[iMEX]-(c[0]*b[iMEX]+c[1])); rC=float(ds[iCAN]-(c[0]*b[iCAN]+c[1]))
            out[f"{mp}_d{dl}"]={"MEX_resid":rM,"CAN_resid":rC}; allbelow&=(rM<0 and rC<0)
    return {"combinations":out,"all_below_size_trend":bool(allbelow)}

def ingest_external():
    out={}
    for name,fn in EXTERNAL.items():
        p=DATA_PROC/fn
        out[name]={"status":"ingested","data":json.load(open(p))} if p.exists() \
                  else {"status":"external","note":f"compute in Colab -> {fn}"}
    return out

def main():
    DATA_PROC.mkdir(parents=True,exist_ok=True)
    R={"two_country":exp_two_country(),"multilateral":exp_multilateral(),
       "sensitivity":exp_sensitivity(),"external":ingest_external(),
       "config":{"kappa":KAPPA,"delta":DELTA,"B_LO":B_LO,"B_HI":B_HI,
                 "B_MAPPING":B_MAPPING,"N":len(COUNTRIES_30)}}
    json.dump(R,open(RESULTS_JSON,"w"),indent=1)
    m=R["multilateral"]
    print("pipeline OK ->",RESULTS_JSON)
    print(f"  two-country guard: {R['two_country']['guard_passed']}")
    print(f"  N=30 binding: {m['binding_country']} (delta*={m['binding_delta_star']:.2f}); hegemons: {m['hegemons'] or 'none'}")
    print(f"  partial-corr(delta*,exposure|b): {m['partial_corr_exposure_given_b']:.2f}")
    print(f"  sensitivity all-below-trend: {R['sensitivity']['all_below_size_trend']}")

if __name__=="__main__": main()
