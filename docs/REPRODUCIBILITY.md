# Reproducibility notes

## Authoritative release

The released network is the Stage-3 semi-supervised checkpoint in `models/best_model.pth`. It has 838,404 trainable parameters and was fine-tuned from the physics-guided model using 763,776 supervised training patches, 54,912 validation patches, and 8,526 unlabeled ERA5 samples. Hyperparameters are recorded in `configs/stage3_semisup.yaml`.

The final 117-event archive was generated with the advanced overlap-blending workflow in `inference/predict_full_field_blend_adv.py`. Production settings are stored in `configs/inference_v2.yaml` and are also the defaults of `inference/batch_prd130_era5_to_recon_nc.py`.

## Recommended reproduction sequence

1. Prepare paired ERA5–WRF patches with `preprocessing/make_dataset.py`.
2. Prepare additional ERA5-only patches with `preprocessing/make_unlabeled_dataset.py`.
3. Train the supervised model and physics-guided model.
4. Fine-tune Stage 3 with the command in the main README.
5. Reconstruct each historical event with `inference/batch_prd130_era5_to_recon_nc.py`.
6. Run catalogue diagnostics in `analysis/`.
7. If licensed CMA observations are available, create the optional observation-constrained archive with `postprocessing/apply_cma_observation_analysis.py`.

## Dataset partitioning

Training, validation, and independent-event evaluation serve different purposes. The WRF-supervised training and validation patches are used for model optimization and checkpoint selection. Mangkhut and the station-based analyses provide independent event-level assessment. The 117-event catalogue is used after validation to quantify how reconstruction changes historical strong-wind identification; it is not treated as 117 independent WRF validation simulations.

## Determinism

GPU kernels, interpolation libraries, and NetCDF encoders may cause small platform-dependent numerical differences. Preserve checkpoint identity, normalization ranges, static geography, patch size, stride, blend window, shifted-grid settings, and smoothing parameter when reproducing the released catalogue.

## Provenance checks

The batch script writes a manifest and method metadata. The observation-constrained archive can be checked with `postprocessing/validate_cma_analysis_archive.py`. Keep the event-pair table in `metadata/event_peak_pairs.csv` with every archive release.
