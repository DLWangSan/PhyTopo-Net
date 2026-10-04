#!/bin/bash
#SBATCH -J prd130_v2
#SBATCH -p A40
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 120:00:00
#SBATCH -o experiments/prd130_recon_v2_%j.out
#SBATCH -e experiments/prd130_recon_v2_%j.err

# prd130 全量 *_ERA5_sfc.nc → blend_adv 重构（与 infer_2017_08 同款配置）
# 输出至新目录 era5_ibtracs_prd130_recon_v2，不覆盖旧版 era5_ibtracs_prd130_recon

set -eo pipefail

PYTORCH_PY="${PYTORCH_PY:-/public/software/apps/anaconda3/2024.06/envs/mytorch/bin/python}"
REPO="${REPO:-$HOME/PhyTopo-Net}"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

ERA5_ROOT="${ERA5_ROOT:-$REPO/era5_ibtracs_prd130}"
OUT_DIR="${OUT_DIR:-$REPO/era5_ibtracs_prd130_recon_v2}"
GEO_FILE="${GEO_FILE:-$REPO/assets/geo_em.d03.nc}"
MODEL_PATH="${MODEL_PATH:-$REPO/models/best_model.pth}"

mkdir -p "$REPO/experiments" "$OUT_DIR"

echo "=========================================="
echo "prd130 batch recon v2  $(date)  $(hostname)"
echo "ERA5_ROOT=$ERA5_ROOT"
echo "OUT_DIR=$OUT_DIR"
echo "MODEL_PATH=$MODEL_PATH"
echo "predictor=blend_adv stride=16 dual_grid=1 triple_grid=1 blend=hann"
echo "=========================================="

[[ -d "$ERA5_ROOT" ]] || { echo "缺少 ERA5_ROOT"; exit 1; }
[[ -f "$GEO_FILE" ]] || { echo "缺少 GEO_FILE"; exit 1; }
[[ -f "$MODEL_PATH" ]] || { echo "缺少 MODEL_PATH"; exit 1; }

N_IN=$(ls -1 "$ERA5_ROOT"/*_ERA5_sfc.nc 2>/dev/null | wc -l)
echo "输入 ERA5 文件数: $N_IN"

"$PYTORCH_PY" -u "$REPO/inference/batch_prd130_era5_to_recon_nc.py" \
  --era5-root "$ERA5_ROOT" \
  --out-dir "$OUT_DIR" \
  --geo-file "$GEO_FILE" \
  --model-path "$MODEL_PATH" \
  --device cuda \
  --predictor blend_adv \
  --stride 16 \
  --blend-window hann \
  --norm-uv-smooth-sigma 0.35 \
  --overwrite

echo "完成 $(date)"
N_OUT=$(ls -1 "$OUT_DIR"/*_phytopo_recon.nc 2>/dev/null | wc -l)
du -sh "$OUT_DIR"
echo "输出 recon 文件数: $N_OUT"
ls -lh "$OUT_DIR"/batch_recon_manifest.json
