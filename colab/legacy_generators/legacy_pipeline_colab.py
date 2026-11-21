# ARCHIVED (added by the provenance audit, 2026-11-20). Recovered verbatim from
# archived session logs, UNMODIFIED. It ran in Colab (working dir /content/makoto)
# and is NOT part of the runnable repo pipeline. See docs/provenance.md (the
# "Archival Extraction Note") for the session-log locator and how this relates to
# the committed files.
#
# Role:   Monolithic Colab precursor of pipeline.py. RECOMPUTES two_country / multilateral / sensitivity from the BACI files; INGESTS qre/mechanism/reciprocity/ppo/phase3 _results.json as uploaded inputs. Superseded by the committed pipeline.py.
# ============================================================================

# ===== CELL 1: pipeline.py =====
import os, json, numpy as np, pandas as pd
os.makedirs("data/raw", exist_ok=True); os.makedirs("data/processed", exist_ok=True); os.makedirs("figures", exist_ok=True)
# >>> upload BACI_HS22_Y2024_V202601.csv + country_codes_V202601.csv into data/raw/
# >>> upload qre/mechanism/reciprocity/ppo/phase3 _results.json into data/processed/
BACI_CSV="data/raw/BACI_HS22_Y2024_V202601.csv"; BACI_CODES_CSV="data/raw/country_codes_V202601.csv"
KAPPA,DELTA=1.0,0.10
TWO={"iso":["USA","CHN"],"b":[0.66,0.56],"expect":{"nash":[0.610,0.499],"binding":0.700}}
ISO=["USA","CHN","JPN","DEU","IND","GBR","FRA","ITA","BRA","CAN","RUS","KOR","AUS","MEX","IDN","ESP","NLD","SAU","TUR","CHE","POL","SWE","BEL","THA","ARG","ZAF","ARE","EGY","VNM","MYS"]
B_LO,B_HI,B_MAPPING=0.50,0.68,"linear"
SWEEP_D=[0.05,0.10,0.20,0.30]; SWEEP_M=["linear","rank","sqrt","log"]
EXT={"qre":"qre_results.json","reciprocity":"reciprocity_results.json","mechanism":"mechanism_results.json","ppo":"ppo_results.json","natural_expt":"phase3_results.json"}

class Game:
    def __init__(s,b,W,k=1.0,d=0.10): s.b=np.asarray(b,float);s.N=len(s.b);s.W=np.asarray(W,float);s.k=np.full(s.N,k) if np.isscalar(k) else np.asarray(k,float);s.d=float(d)
    def welf(s,t): t=np.asarray(t,float); return s.b*t-s.W@(s.b*t)-0.5*s.k*t**2-s.d*t*(s.W@t)
    def nash(s,it=4000,tol=1e-12):
        t=np.zeros(s.N)
        for _ in range(it):
            nt=np.clip((s.b-s.d*(s.W@t))/s.k,0,1)
            if np.max(np.abs(nt-t))<tol: t=nt;break
            t=nt
        return t
    def folk(s):
        nz=s.nash(); Wn=s.welf(nz); ds=np.full(s.N,np.nan)
        for i in range(s.N):
            t=np.zeros(s.N); t[i]=np.clip(s.b[i]/s.k[i],0,1); Wd=s.welf(t)[i]
            ds[i]=np.inf if Wn[i]>=0 else (Wd/(-Wn[i]))/(1+Wd/(-Wn[i]))
        return ds,nz,Wn

def load_baci():
    cc=pd.read_csv(BACI_CODES_CSV); cand={}
    for _,r in cc.iterrows(): cand.setdefault(r["country_iso3"],[]).append(int(r["country_code"]))
    tot={}
    for ch in pd.read_csv(BACI_CSV,usecols=["i","v"],chunksize=3_000_000):
        for k,v in ch.groupby("i")["v"].sum().items(): tot[k]=tot.get(k,0)+v
    codes={c:max(cand[c],key=lambda k:tot.get(k,0)) for c in ISO}
    idx={codes[c]:k for k,c in enumerate(ISO)}; keep=set(codes.values()); N=len(ISO); M=np.zeros((N,N))
    for ch in pd.read_csv(BACI_CSV,usecols=["i","j","v"],chunksize=2_000_000):
        ch=ch[ch["i"].isin(keep)&ch["j"].isin(keep)]
        for (j,i),v in ch.groupby(["j","i"])["v"].sum().items(): M[idx[j],idx[i]]+=v
    np.fill_diagonal(M,0.0); assert (M.sum(1)>0).all(),"zero-trade country: code resolution failed"
    return M,codes

def bmap(M,mp):
    s=M.sum(1)/M.sum(); N=len(s)
    if mp=="linear": x=s/s.max()
    elif mp=="rank": x=np.argsort(np.argsort(s))/(N-1)
    elif mp=="sqrt": x=np.sqrt(s)/np.sqrt(s.max())
    elif mp=="log": x=(np.log(s)-np.log(s.min()))/(np.log(s.max())-np.log(s.min()))
    return B_LO+(B_HI-B_LO)*x
def tmat(M): W=M/M.sum(1,keepdims=True); np.fill_diagonal(W,0.0); return W/W.sum(1,keepdims=True)

def two_country():
    ds,nz,_=Game(TWO["b"],[[0,1],[1,0]],KAPPA,DELTA).folk(); e=TWO["expect"]
    assert abs(nz[0]-e["nash"][0])<1e-2 and abs(np.nanmax(ds)-e["binding"])<1e-2,"GUARD FAILED: calibration drifted"
    return {"iso":TWO["iso"],"b":TWO["b"],"nash":nz.tolist(),"delta_star":ds.tolist(),"binding":float(np.nanmax(ds)),"guard_passed":True}

def multilateral():
    M,codes=load_baci(); b=bmap(M,B_MAPPING); W=tmat(M); ds,nz,Wn=Game(b,W,KAPPA,DELTA).folk()
    exp=W@b; imp=M.sum(1)/M.sum(); fin=np.isfinite(ds)
    def res(y,x): A=np.vstack([x,np.ones_like(x)]).T; c,*_=np.linalg.lstsq(A,y,rcond=None); return y-A@c
    pr=float(np.corrcoef(res(ds[fin],b[fin]),res(exp[fin],b[fin]))[0,1])
    heg=[ISO[i] for i in range(len(ISO)) if not np.isfinite(ds[i])]
    return {"iso":ISO,"codes":codes,"b":b.tolist(),"nash":nz.tolist(),
            "delta_star":[None if not np.isfinite(x) else x for x in ds],"W_nash":Wn.tolist(),
            "exposure":exp.tolist(),"import_share":imp.tolist(),"concentration_hhi":((W**2).sum(1)).tolist(),
            "binding_country":ISO[int(np.nanargmax(np.where(fin,ds,-1)))],"binding_delta_star":float(np.nanmax(ds[fin])),
            "hegemons":heg,"corr_b":float(np.corrcoef(b[fin],ds[fin])[0,1]),"partial_corr_exposure_given_b":pr}

def sensitivity():
    M,_=load_baci(); W=tmat(M); iM,iC=ISO.index("MEX"),ISO.index("CAN"); out={}; allb=True
    for mp in SWEEP_M:
        b=bmap(M,mp)
        for dl in SWEEP_D:
            ds,*_=Game(b,W,KAPPA,dl).folk(); m=np.isfinite(ds)
            A=np.vstack([b[m],np.ones(m.sum())]).T; c,*_=np.linalg.lstsq(A,ds[m],rcond=None)
            rM=float(ds[iM]-(c[0]*b[iM]+c[1])); rC=float(ds[iC]-(c[0]*b[iC]+c[1]))
            out[f"{mp}_d{dl}"]={"MEX_resid":rM,"CAN_resid":rC}; allb&=(rM<0 and rC<0)
    return {"combinations":out,"all_below_size_trend":bool(allb)}

def ingest():
    out={}
    for n,f in EXT.items():
        p=f"data/processed/{f}"
        out[n]={"status":"ingested","data":json.load(open(p))} if os.path.exists(p) else {"status":"external","note":f"missing {f}"}
    return out

R={"two_country":two_country(),"multilateral":multilateral(),"sensitivity":sensitivity(),"external":ingest(),
   "config":{"kappa":KAPPA,"delta":DELTA,"B_LO":B_LO,"B_HI":B_HI,"B_MAPPING":B_MAPPING,"N":len(ISO)}}
json.dump(R,open("data/processed/results.json","w"),indent=1)
m=R["multilateral"]; print("pipeline OK")
print(f"  guard: {R['two_country']['guard_passed']}")
print(f"  N=30 binding: {m['binding_country']} d*={m['binding_delta_star']:.2f}; hegemons: {m['hegemons'] or 'none'}")
print(f"  partial-corr: {m['partial_corr_exposure_given_b']:.2f}; sensitivity all-below: {R['sensitivity']['all_below_size_trend']}")
print("  ingested:",{k:v['status'] for k,v in R['external'].items()})
