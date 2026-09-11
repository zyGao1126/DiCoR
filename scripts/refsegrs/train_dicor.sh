#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda:0}
DATA_ROOT=${DATA_ROOT:-/path/to/RefSegRS}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints/refsegrs}
COARSE_DIR=${COARSE_DIR:-$OUTPUT_ROOT/coarse}
COARSE_CKPT=${COARSE_CKPT:-$COARSE_DIR/coarse_best.pth}
OFFLINE_BANK_DIR=${OFFLINE_BANK_DIR:-$OUTPUT_ROOT/offline_bank}
REFINER_DIR=${REFINER_DIR:-$OUTPUT_ROOT/refiner}

COMMON_ARGS=(
  --device "$DEVICE"
  --dataset refsegrs
  --refer-data-root "$DATA_ROOT"
  --img-size 480
  --swin-type base
  --num-vmsf-blocks 1
  --num-heads-fusion 8
  --visual-fusion lvmsf
)

# RefSegRS does not use DLG, so only construct the LCR probability bank.
"$PYTHON" build_offline_bank.py \
  "${COMMON_ARGS[@]}" \
  --bank-type lcr \
  --coarse-dir "$COARSE_DIR" \
  --output-dir "$OFFLINE_BANK_DIR" \
  --batch-size 16 \
  --snapshot-epochs "20,30,40,50,59" \
  --lcr-iou-range 0.5 0.99

"$PYTHON" train_refiner.py \
  "${COMMON_ARGS[@]}" \
  --seed 1126 \
  --coarse-ckpt "$COARSE_CKPT" \
  --offline-bank-dir "$OFFLINE_BANK_DIR" \
  --output-dir "$REFINER_DIR" \
  --batch-size 8 \
  --epochs 40 \
  --lr 5e-5 \
  --weight-decay 1e-2 \
  --aug-morph-prob 0.4 \
  --morph-max-radius 2
