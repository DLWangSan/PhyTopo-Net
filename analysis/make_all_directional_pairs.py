from itertools import combinations
from pathlib import Path
import pandas as pd
import numpy as np

root=Path('outputs/cma_validation_prd_all_events/directional_mechanism')
h=pd.read_csv(root/'hourly_with_uv_exposure.csv.gz')
h=h[(h.era_bilinear_speed_uv>=8)&h.obs_speed.notna()].copy()
rows=[]
for (event,time,cell),g in h.groupby(['event_id','time_utc','era_cell'],sort=False):
    for a,b in combinations(g.itertuples(index=False),2):
        rows.append({'event_id':event,'time_utc':time,'era_cell':cell,
          'station_a':a.station_key,'station_b':b.station_key,
          'obs_contrast':a.obs_speed-b.obs_speed,'phy_contrast':a.phy_speed_uv-b.phy_speed_uv,
          'ocean_contrast':a.ocean_fraction_upwind-b.ocean_fraction_upwind,
          'blocking_contrast':a.blocking_angle_deg-b.blocking_angle_deg})
d=pd.DataFrame(rows)
d.to_csv(root/'same_cell_directional_pairs_all.csv.gz',index=False,compression='gzip')
print({'pairs':len(d),'events':d.event_id.nunique(),'abs_obs_ge2':int((d.obs_contrast.abs()>=2).sum()),'abs_obs_ge5':int((d.obs_contrast.abs()>=5).sum())})
