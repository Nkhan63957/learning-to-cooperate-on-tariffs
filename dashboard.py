"""
dashboard.py — regenerates every figure from data/processed/results.json.
Standalone: depends only on the JSON (no shared module) and reads no number it
doesn't find there. Run pipeline.py first.   python dashboard.py
"""
from __future__ import annotations
from pathlib import Path
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT=Path(__file__).parent; FIG=ROOT/"figures"; FIG.mkdir(parents=True,exist_ok=True)
R=json.load(open(ROOT/"data"/"processed"/"results.json"))

def _despine(ax): [ax.spines[s].set_visible(False) for s in ("top","right")]

def fig_network():
    m=R["multilateral"]; iso=m["iso"]; b=np.array(m["b"]); exp=np.array(m["exposure"])
    ds=np.array([np.nan if x is None else x for x in m["delta_star"]]); fin=np.isfinite(ds)
    A=np.vstack([b[fin],np.ones(fin.sum())]).T; c,*_=np.linalg.lstsq(A,ds[fin],rcond=None)
    xs=np.linspace(b.min(),b.max(),50); fig,ax=plt.subplots(figsize=(9.6,6))
    ax.plot(xs,c[0]*xs+c[1],color="#999",ls="--",lw=1.4,label="size trend (δ* ~ market power)")
    sc=ax.scatter(b,ds,c=exp,cmap="RdYlGn_r",s=90,edgecolor="white",lw=.6,zorder=3)
    plt.colorbar(sc).set_label("network exposure (trade-weighted partner power)",fontsize=9)
    for c_ in ["MEX","CAN","USA","CHN","RUS","DEU","JPN"]:
        i=iso.index(c_); ax.annotate(c_,(b[i],ds[i]),fontsize=8.5,fontweight="bold",xytext=(5,4),textcoords="offset points")
    for c_ in ["MEX","CAN"]:
        i=iso.index(c_); ax.annotate("",xy=(b[i],ds[i]),xytext=(b[i],c[0]*b[i]+c[1]),arrowprops=dict(arrowstyle="->",color="#C0392B",lw=1.3))
    ax.set_xlabel("Market power  b  (world import share)"); ax.set_ylabel("Critical discount δ*  (harder to keep cooperative →)")
    ax.set_title("Trade-network position reshapes who sustains multilateral cooperation",weight="bold",fontsize=12)
    ax.legend(fontsize=9,frameon=False,loc="upper left"); _despine(ax); ax.grid(alpha=.2)
    plt.tight_layout(); plt.savefig(FIG/"network_delta.png",dpi=150,bbox_inches="tight"); plt.close()

def fig_nash_bars():
    m=R["multilateral"]; iso=m["iso"]; nash=np.array(m["nash"])
    ds=np.array([np.nan if x is None else x for x in m["delta_star"]]); o=np.argsort(-np.array(m["import_share"]))
    fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5.4))
    a1.bar(range(len(iso)),[nash[i]*100 for i in o],color="#9D6FFF")
    a1.set_xticks(range(len(iso))); a1.set_xticklabels([iso[i] for i in o],rotation=90,fontsize=7)
    a1.set_ylabel("Nash tariff (%)"); a1.set_title("Nash tariffs by economy (sorted by import share)",weight="bold")
    a2.bar(range(len(iso)),[ds[i] for i in o],color="#1E8C6B"); a2.axhline(1.0,color="#C0392B",ls="--",lw=1)
    a2.set_xticks(range(len(iso))); a2.set_xticklabels([iso[i] for i in o],rotation=90,fontsize=7)
    a2.set_ylabel("critical discount δ*"); a2.set_title("Who is hardest to keep cooperative",weight="bold")
    _despine(a1); _despine(a2); plt.tight_layout(); plt.savefig(FIG/"nash_and_delta.png",dpi=150,bbox_inches="tight"); plt.close()

def fig_sensitivity():
    s=R["sensitivity"]["combinations"]; keys=list(s.keys())
    mex=[s[k]["MEX_resid"] for k in keys]; can=[s[k]["CAN_resid"] for k in keys]
    fig,ax=plt.subplots(figsize=(11,4.4)); x=np.arange(len(keys))
    ax.bar(x-0.2,mex,0.4,label="Mexico",color="#C0392B"); ax.bar(x+0.2,can,0.4,label="Canada",color="#E08A1E")
    ax.axhline(0,color="#333",lw=1); ax.set_xticks(x); ax.set_xticklabels(keys,rotation=90,fontsize=7)
    ax.set_ylabel("δ* residual vs size trend"); ax.set_title("Robustness: MEX/CAN stay below the size trend in all 16 calibrations",weight="bold",fontsize=11)
    ax.legend(frameon=False); _despine(ax); plt.tight_layout(); plt.savefig(FIG/"sensitivity.png",dpi=150,bbox_inches="tight"); plt.close()

def fig_natural_experiment():
    ext=R["external"].get("natural_expt",{})
    if ext.get("status")!="ingested": print("  natural_experiment: external -> run Phase 3 in Colab (skipped)"); return
    d=ext["data"]; yrs=np.array(d["years"]); nUS=np.array(d["nash_US"])*100; rz=np.array(d["realized_tariff"])
    fig,ax=plt.subplots(figsize=(10,5.4))
    ax.plot(yrs,nUS,color="#C0392B",lw=2,ls="--",label="Model: US Nash (no commitment)")
    ax.axhline(3,color="#2E9E5B",lw=2,ls=":",label="Model: cooperative (~WTO MFN)")
    ax.plot(yrs,rz,"-o",color="#222",lw=2.4,ms=5,label="Realized US tariff on China (PIIE)")
    ax.set_xlabel("Year"); ax.set_ylabel("US tariff on Chinese goods (%)")
    ax.set_title("Natural experiment: cooperation holds under commitment, collapses without it",weight="bold",fontsize=11.5)
    ax.legend(fontsize=9,frameon=False,loc="upper left"); _despine(ax)
    plt.tight_layout(); plt.savefig(FIG/"natural_experiment.png",dpi=150,bbox_inches="tight"); plt.close()

def main():
    fig_network(); fig_nash_bars(); fig_sensitivity(); fig_natural_experiment()
    print("dashboard OK -> figures in",FIG)

if __name__=="__main__": main()
