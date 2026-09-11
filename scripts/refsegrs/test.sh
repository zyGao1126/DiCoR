#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda:0}
DATA_ROOT=${DATA_ROOT:-/path/to/RefSegRS}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints/refsegrs}
COARSE_CKPT=${COARSE_CKPT:-$OUTPUT_ROOT/coarse/coarse_best.pth}
REFINER_CKPT=${REFINER_CKPT:-$OUTPUT_ROOT/refiner/refiner_best.pth}
SPLIT=${SPLIT:-test}

"$PYTHON" test.py \
  --device "$DEVICE" \
  --dataset refsegrs \
  --split "$SPLIT" \
  --refer-data-root "$DATA_ROOT" \
  --img-size 480 \
  --swin-type base \
  --num-vmsf-blocks 1 \
  --num-heads-fusion 8 \
  --visual-fusion lvmsf \
  --batch-size 16 \
  --coarse-ckpt "$COARSE_CKPT" \
  --refiner-ckpt "$REFINER_CKPT"
