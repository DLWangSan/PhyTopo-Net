# Data format

## ERA5 driver files

The inference pipeline reads hourly NetCDF files with either `time` or `valid_time` as the temporal coordinate. The required atmospheric fields are:

| Canonical field | Expected physical quantity | Unit before normalization |
|---|---|---|
| `U10` | eastward 10-m wind | m/s |
| `V10` | northward 10-m wind | m/s |
| `SLP` | mean sea-level pressure | hPa or Pa, detected by the loader |
| `T2` | 2-m air temperature | K |
| `SST` | sea-surface temperature | K |
| `BLH` | boundary-layer height | m |
| `FSR` | forecast surface roughness | m |

Common ERA5 lower-case variable names are mapped internally. The seven fields are interpolated to the target 3-km grid before inference.

## Static geographical fields

The WRF `geo_em` file supplies five static channels:

1. terrain height (`HGT_M`);
2. terrain slope derived from terrain height;
3. latitude (`XLAT_M` or `XLAT`);
4. longitude (`XLONG_M` or `XLONG`);
5. land mask (`LANDMASK`).

The released model was trained for the supplied Greater Bay Area grid. Applying it to another geographical domain requires a compatible `geo_em` file and should be treated as spatial extrapolation unless the model is retrained.

## Training patches

Each serialized training sample stores:

- `input`: 12 channels ordered as `[U10, V10, SLP, T2, SST, BLH, FSR, HGT, SLOPE, LAT, LON, LANDMASK]`;
- `label`: four WRF channels ordered as `[U10, V10, SLP, T2]`.

Patches are 64 × 64 grid points in the released training setup.

## Direct reconstruction output

`batch_prd130_era5_to_recon_nc.py` writes:

| Variable | Description | Unit |
|---|---|---|
| `u10` | reconstructed eastward 10-m wind | m/s |
| `v10` | reconstructed northward 10-m wind | m/s |
| `msl` | reconstructed mean sea-level pressure | hPa |
| `XLAT` | target-grid latitude | degrees north |
| `XLONG` | target-grid longitude | degrees east |

## Observation-constrained output

The optional engineering analysis writes `u10_analysis`, `v10_analysis`, `wspd10_analysis`, `analysis_increment`, `cma_influence_weight`, `ib_center_pressure`, and `assimilated_station_count`. Its input CMA observations cannot be redistributed with this repository. The derived files retain method and provenance attributes in each NetCDF header.

## Data licenses

ERA5, IBTrACS, WRF inputs, HKO observations, CMA observations, and derived products remain subject to their providers' licenses. The MIT license in this repository applies to source code only.
