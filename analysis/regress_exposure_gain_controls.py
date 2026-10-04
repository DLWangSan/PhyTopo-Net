#!/usr/bin/env python3
"""Control-test directional exposure skill against error magnitude and fixed effects."""

import json
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import patsy

ROOT=Path(__file__).resolve().parent
BASE=ROOT/"outputs/cma_validation_prd_all_events/terrain_glo30_v2_only/directional_physics_audit"
SRC=BASE/"wrf_directional_physical_site_pairs.csv.gz"


def extract(fit, term):
    return {"coef":float(fit.params[term]),"se":float(fit.bse[term]),"p":float(fit.pvalues[term]),
            "ci95":[float(v) for v in fit.conf_int().loc[term]],"n":int(fit.nobs),"r2":float(fit.rsquared)}


def main():
    d=pd.read_csv(SRC);d=d[abs(d.obs_contrast)>=5].copy()
    d["era_err"]=abs(d.era_contrast-d.obs_contrast);d["v2_err"]=abs(d.v2_contrast-d.obs_contrast)
    d["gain"]=d.era_err-d.v2_err;d["exposure_gap"]=abs(d.ocean_contrast);d["obs_abs"]=abs(d.obs_contrast)
    forms={
      "gain_basic":"gain ~ exposure_gap + obs_abs + era_speed_mean + distance_km",
      "gain_pair_event_fe":"gain ~ exposure_gap + obs_abs + era_speed_mean + distance_km + C(pair_id) + C(event_id)",
      "v2_error_conditional":"v2_err ~ era_err + exposure_gap + era_speed_mean + distance_km + C(pair_id) + C(event_id)",
    }
    out={}
    for name,form in forms.items():
        fit=smf.ols(form,d).fit(cov_type="cluster",cov_kwds={"groups":d.pair_id})
        out[name]=extract(fit,"exposure_gap")
    # Each coefficient is per unit fraction; report a physically readable 0.4 shift.
    for q in out.values():q["effect_for_0p4_exposure_shift_ms"]=0.4*q["coef"]
    # Small-cluster robustness: residualized within-pair/event permutation and leave-one-pair-out.
    controls="obs_abs + era_speed_mean + distance_km + C(pair_id) + C(event_id)"
    X=np.asarray(patsy.dmatrix(controls,d,return_type="dataframe"),float)
    y=d.gain.to_numpy(float);x=d.exposure_gap.to_numpy(float)
    def resid(v,mat=X):return v-mat@np.linalg.lstsq(mat,v,rcond=None)[0]
    yr=resid(y);xr=resid(x);b=float(xr@yr/(xr@xr))
    rng=np.random.default_rng(20260820);perms=[]
    groups=[g.index.to_numpy() for _,g in d.reset_index(drop=True).groupby(["pair_id","event_id"])]
    for _ in range(5000):
        xp=x.copy()
        for ix in groups:xp[ix]=rng.permutation(xp[ix])
        xpr=resid(xp);perms.append(float(xpr@yr/(xpr@xpr)))
    loo=[]
    for pid in d.pair_id.unique():
        q=d[d.pair_id!=pid].copy();Xm=np.asarray(patsy.dmatrix(controls,q,return_type="dataframe"),float)
        yy=q.gain.to_numpy(float);xx=q.exposure_gap.to_numpy(float)
        yy=yy-Xm@np.linalg.lstsq(Xm,yy,rcond=None)[0];xx=xx-Xm@np.linalg.lstsq(Xm,xx,rcond=None)[0]
        loo.append(float(xx@yy/(xx@xx)))
    out["fixed_effect_robustness"]={"residualized_coef":b,
        "within_pair_event_permutation_p":float((1+np.sum(np.asarray(perms)>=b))/(1+len(perms))),
        "permutation_95pct":[float(v) for v in np.percentile(perms,[2.5,97.5])],
        "leave_one_physical_pair_out_coef_range":[float(min(loo)),float(max(loo))],
        "leave_one_out_all_positive":bool(np.all(np.asarray(loo)>0))}
    (BASE/"exposure_gain_controlled_regression.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))


if __name__=="__main__":main()
