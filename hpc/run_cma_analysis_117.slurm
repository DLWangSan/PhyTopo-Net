#!/bin/bash
#SBATCH -J cma_analysis
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 01:00:00
#SBATCH --array=0-116%8
#SBATCH -o cma_analysis_%A_%a.out
#SBATCH -e cma_analysis_%A_%a.err

set -euo pipefail

# Required environment variables:
# ERA5_ROOT, RECON_ROOT, CMA_HOURLY and IBTRACS_FILE.
# REPO, OUTPUT_ROOT, PYTHON_BIN, PAIR_FILE and MODEL_FILE are optional.
REPO="${REPO:-$HOME/PhyTopo-Net}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO/outputs/cma_analysis_117}"
PAIR_FILE="${PAIR_FILE:-$REPO/metadata/event_peak_pairs.csv}"
MODEL_FILE="${MODEL_FILE:-$REPO/models/unified_hierarchical_model_v2.npz}"

: "${ERA5_ROOT:?Set ERA5_ROOT to the ERA5 event archive}"
: "${RECON_ROOT:?Set RECON_ROOT to the direct reconstruction archive}"
: "${CMA_HOURLY:?Set CMA_HOURLY to the CMA hourly observation table}"
: "${IBTRACS_FILE:?Set IBTRACS_FILE to the IBTrACS NetCDF file}"

mkdir -p "$REPO/logs" "$OUTPUT_ROOT"
mapfile -t EVENTS < <(tail -n +2 "$PAIR_FILE" | cut -d, -f1)
STORM_ID="${EVENTS[$SLURM_ARRAY_TASK_ID]}"

"$PYTHON_BIN" "$REPO/postprocessing/apply_cma_observation_analysis.py" "$STORM_ID" \
  --era-root "$ERA5_ROOT" \
  --recon-root "$RECON_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --cma-hourly "$CMA_HOURLY" \
  --global-model "$MODEL_FILE" \
  --pairs "$PAIR_FILE" \
  --ibtracs "$IBTRACS_FILE"
