#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda:0}
DATA_ROOT=${DATA_ROOT:-/path/to/RRSIS-D}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints/rrsisd}
COARSE_DIR=${COARSE_DIR:-$OUTPUT_ROOT/coarse}
COARSE_CKPT=${COARSE_CKPT:-$COARSE_DIR/coarse_best.pth}
OFFLINE_BANK_DIR=${OFFLINE_BANK_DIR:-$OUTPUT_ROOT/offline_bank}
LOCALIZATION_DIR=${LOCALIZATION_DIR:-$OUTPUT_ROOT/localization}
REFINER_DIR=${REFINER_DIR:-$OUTPUT_ROOT/refiner}

COMMON_ARGS=(
  --device "$DEVICE"
  --dataset rrsisd
  --refer-data-root "$DATA_ROOT"
  --img-size 480
  --swin-type base
  --num-vmsf-blocks 3
  --num-heads-fusion 1
  --visual-fusion lvmsf
)

"$PYTHON" build_offline_bank.py \
  "${COMMON_ARGS[@]}" \
  --bank-type all \
  --coarse-dir "$COARSE_DIR" \
  --coarse-ckpt "$COARSE_CKPT" \
  --output-dir "$OFFLINE_BANK_DIR" \
  --batch-size 16 \
  --snapshot-epochs "10,15,20,30,39" \
  --lcr-iou-range 0.5 0.99

"$PYTHON" train_localization_guide.py \
  "${COMMON_ARGS[@]}" \
  --seed 42 \
  --coarse-ckpt "$COARSE_CKPT" \
  --offline-bank-dir "$OFFLINE_BANK_DIR" \
  --output-dir "$LOCALIZATION_DIR" \
  --batch-size 16 \
  --epochs 20 \
  --evidence-lr 8e-4 \
  --evidence-weight-decay 1e-4 \
  --winner-lr 5e-4 \
  --winner-weight-decay 1e-4 \
  --alpha 0.5 \
  --lambda-geo 0.5

"$PYTHON" train_refiner.py \
  "${COMMON_ARGS[@]}" \
  --seed 42 \
  --coarse-ckpt "$COARSE_CKPT" \
  --offline-bank-dir "$OFFLINE_BANK_DIR" \
  --output-dir "$REFINER_DIR" \
  --batch-size 8 \
  --epochs 40 \
  --lr 5e-5 \
  --weight-decay 1e-2 \
  --aug-morph-prob 0.6 \
  --morph-max-radius 4
