#!/usr/bin/env python3
"""Test whether PhyTopo skill appears specifically under directional marine-exposure contrast."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/cma_validation_prd_all_events/terrain_glo30_v2_only/directional_physics_audit"
SRC = BASE / "wrf_directional_physical_site_pairs.csv.gz"
RNG = np.random.default_rng(20260820)


def event_boot(q, n=10000):
    e = (q.groupby("event_id", as_index=False)
         .agg(era_sum=("era_err", "sum"), v2_sum=("v2_err", "sum"), n=("era_err", "size")))
    draw = RNG.integers(0, len(e), (n, len(e)))
    era = e.era_sum.to_numpy()[draw].sum(axis=1) / e.n.to_numpy()[draw].sum(axis=1)
    v2 = e.v2_sum.to_numpy()[draw].sum(axis=1) / e.n.to_numpy()[draw].sum(axis=1)
    red = 100*(era-v2)/era
    return [float(v) for v in np.percentile(red, [2.5, 50, 97.5])]


def summarize(d):
    d = d.copy()
    d["era_err"] = abs(d.era_contrast-d.obs_contrast)
    d["v2_err"] = abs(d.v2_contrast-d.obs_contrast)
    d["exposure_gap"] = abs(d.ocean_contrast)
    cuts = d.exposure_gap.quantile([1/3, 2/3]).to_numpy()
    d["gap_group"] = pd.cut(d.exposure_gap, [-np.inf, cuts[0], cuts[1], np.inf],
                            labels=["low", "mid", "high"], include_lowest=True)
    groups = {}
    for label, q in d.groupby("gap_group", observed=True):
        ev = (q.groupby("event_id", as_index=False)
              .agg(era=("era_err", "mean"), v2=("v2_err", "mean")))
        groups[str(label)] = {"pair_hours": len(q), "pairs": int(q.pair_id.nunique()),
                              "events": int(q.event_id.nunique()),
                              "exposure_gap_median": float(q.exposure_gap.median()),
                              "era_mae": float(q.era_err.mean()), "v2_mae": float(q.v2_err.mean()),
                              "reduction_percent": float(100*(q.era_err.mean()-q.v2_err.mean())/q.era_err.mean()),
                              "event_bootstrap_reduction_95ci": event_boot(q),
                              "events_improved": int((ev.v2 < ev.era).sum()),
                              "event_sign_p": float(stats.binomtest(int((ev.v2 < ev.era).sum()), len(ev), .5,
                                                                    alternative="greater").pvalue)}
    # Within each physical pair, compare high- and low-exposure states.
    rows=[]
    for pid,g in d.groupby("pair_id"):
        if len(g)<12 or g.exposure_gap.nunique()<3: continue
        lo,hi=g.exposure_gap.quantile([1/3,2/3]); a=g[g.exposure_gap<=lo];b=g[g.exposure_gap>=hi]
        if min(len(a),len(b))<3: continue
        rows.append({"pair_id":pid,"events":g.event_id.nunique(),
                     "low_gain":float(a.era_err.mean()-a.v2_err.mean()),
                     "high_gain":float(b.era_err.mean()-b.v2_err.mean()),
                     "gain_increase":float((b.era_err.mean()-b.v2_err.mean())-(a.era_err.mean()-a.v2_err.mean()))})
    pr=pd.DataFrame(rows)
    within={"pairs":len(pr),"median_low_gain_ms":float(pr.low_gain.median()),
            "median_high_gain_ms":float(pr.high_gain.median()),
            "median_gain_increase_ms":float(pr.gain_increase.median()),
            "pairs_gain_increases":int((pr.gain_increase>0).sum()),
            "sign_test_p":float(stats.binomtest(int((pr.gain_increase>0).sum()),len(pr),.5,
                                                alternative="greater").pvalue)}
    return {"rows":len(d),"pairs":int(d.pair_id.nunique()),"events":int(d.event_id.nunique()),
            "tertile_cutoffs":cuts.tolist(),"groups":groups,"within_pair":within},d,pr


def main():
    d=pd.read_csv(SRC)
    out={}
    tables=[]
    for name,q in [("all_strong_background",d),("observed_contrast_ge5",d[abs(d.obs_contrast)>=5])]:
        out[name],aug,pr=summarize(q)
        aug.assign(sample=name).to_csv(BASE/f"exposure_skill_{name}.csv.gz",index=False,compression="gzip")
        pr.assign(sample=name).to_csv(BASE/f"exposure_within_pair_{name}.csv",index=False)
    (BASE/"exposure_conditioned_skill_summary.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))


if __name__=="__main__": main()
