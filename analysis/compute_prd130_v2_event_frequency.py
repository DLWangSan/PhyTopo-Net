#!/usr/bin/env python3
import argparse, glob, os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--era-root', required=True)
    p.add_argument('--recon-root', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--thresholds', default='20,25,30,35,40')
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    thresholds = np.array([float(x) for x in a.thresholds.split(',')])
    era_files = sorted(glob.glob(os.path.join(a.era_root, '*_ERA5_sfc.nc')))
    rows, count_e, count_r = [], None, None
    lat2 = lon2 = None
    for k, ef in enumerate(era_files, 1):
        tag = os.path.basename(ef).replace('_ERA5_sfc.nc','')
        rf = os.path.join(a.recon_root, tag + '_phytopo_recon.nc')
        if not os.path.isfile(rf):
            continue
        with xr.open_dataset(rf) as r:
            ur = r.u10.values.astype('float32')
            vr = r.v10.values.astype('float32')
            rmax = np.nanmax(np.hypot(ur, vr), axis=0)
            if lat2 is None:
                lat2 = r.XLAT.values.astype('float32')
                lon2 = r.XLONG.values.astype('float32')
                shp = rmax.shape
                count_e = np.zeros((len(thresholds),)+shp, np.uint16)
                count_r = np.zeros_like(count_e)
        with xr.open_dataset(ef) as e:
            ue = e.u10.values.astype('float32')
            ve = e.v10.values.astype('float32')
            emax_native = np.nanmax(np.hypot(ue, ve), axis=0)
            lat = e.latitude.values.astype(float)
            lon = e.longitude.values.astype(float)
        if lat[0] > lat[-1]:
            lat = lat[::-1]; emax_native = emax_native[::-1, :]
        interp = RegularGridInterpolator((lat, lon), emax_native,
                                         bounds_error=False, fill_value=np.nan)
        emax = interp(np.column_stack([lat2.ravel(),lon2.ravel()])).reshape(lat2.shape)
        for i,t in enumerate(thresholds):
            count_e[i] += np.isfinite(emax) & (emax >= t)
            count_r[i] += np.isfinite(rmax) & (rmax >= t)
        rows.append({'storm_id':tag,
                     'era5_event_max_ms':float(np.nanmax(emax_native)),
                     'phytopo_event_max_ms':float(np.nanmax(rmax)),
                     'increment_ms':float(np.nanmax(rmax)-np.nanmax(emax_native))})
        print(f'{k}/{len(era_files)} {tag}', flush=True)
    df = pd.DataFrame(rows)
    n = len(df)
    freq_e = count_e.astype(float)/n
    freq_r = count_r.astype(float)/n
    np.savez_compressed(os.path.join(a.out_dir,'prd130_v2_event_frequency.npz'),
                        lat=lat2, lon=lon2, thresholds=thresholds,
                        count_era5=count_e, count_phytopo=count_r,
                        freq_era5=freq_e, freq_phytopo=freq_r,
                        delta_frequency_pp=(freq_r-freq_e)*100.0,
                        n_events=np.array([n]))
    df.to_csv(os.path.join(a.out_dir,'prd130_v2_event_stats.csv'), index=False)
    summary=[]
    for t in thresholds:
        ce=int((df.era5_event_max_ms>=t).sum())
        cr=int((df.phytopo_event_max_ms>=t).sum())
        newly=int(((df.era5_event_max_ms<t)&(df.phytopo_event_max_ms>=t)).sum())
        summary.append({'threshold_ms':t,'era5_events':ce,'phytopo_events':cr,
                        'newly_classified':newly,'n_events':n})
    pd.DataFrame(summary).to_csv(os.path.join(a.out_dir,'prd130_v2_threshold_summary.csv'),index=False)
    print(df.describe().to_string())
    print(pd.DataFrame(summary).to_string(index=False))

if __name__ == '__main__':
    main()
