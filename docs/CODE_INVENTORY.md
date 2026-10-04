# Code inventory

This repository contains the final, publication-facing workflow. Scripts that produced the released checkpoint, the final overlap-blended reconstruction, the 117-event diagnostics, and the observation-constrained archive were selected from the project history. Superseded experiments and one-off debugging copies were not included.

## Core model

- `model.py`: PhyTopo-Net and baseline architectures.
- `model_discriminator.py`: conditional discriminator used by optional adversarial training.
- `dataset.py`: patch loading, channel separation, and normalization.
- `loss.py`, `loss_physics.py`: supervised, spectral, cycle-consistency, and teacher-guided physical losses.
- `train.py`, `opts.py`: training and checkpoint management.

## Production inference

- `inference/predict_full_field.py`: ERA5 loading, regridding, normalization, and base patch inference.
- `inference/predict_full_field_blend_adv.py`: Hann-window overlap blending and shifted-grid ensemble.
- `inference/batch_prd130_era5_to_recon_nc.py`: final catalogue driver.
- `inference/infer_date_range_to_netcdf.py`: arbitrary date-range export.

The `blend_adv` predictor is the release default. The legacy predictor is retained only for backward compatibility.

## Post-processing

- `postprocessing/apply_cma_observation_analysis.py`: central-pressure anchor and CMA residual analysis.
- `postprocessing/extract_cma_analysis_event_peaks.py`: event- and station-level summaries.
- `postprocessing/validate_cma_analysis_archive.py`: archive integrity checks.

## Manuscript analysis

The scripts in `analysis/`, `evaluation/`, and `figures/` reproduce selected diagnostics when the required licensed or large input datasets are supplied. Set the paths in `configs/paths.example.env` rather than editing personal absolute paths in source files.

## Online application

`webapp/` is a self-contained mirror of the Hugging Face Spaces deployment. It intentionally contains local copies of the checkpoint and geographical file so that the directory can be deployed directly as a Space.
