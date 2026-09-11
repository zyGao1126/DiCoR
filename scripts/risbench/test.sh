#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda:0}
DATA_ROOT=${DATA_ROOT:-/path/to/RISBench}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints/risbench}
COARSE_CKPT=${COARSE_CKPT:-$OUTPUT_ROOT/coarse/coarse_best.pth}
REFINER_CKPT=${REFINER_CKPT:-$OUTPUT_ROOT/refiner/refiner_best.pth}
LOCATE_CKPT=${LOCATE_CKPT:-$OUTPUT_ROOT/localization/localization_guidance_best.pth}
SPLIT=${SPLIT:-test}

"$PYTHON" test.py \
  --device "$DEVICE" \
  --dataset risbench \
  --split "$SPLIT" \
  --refer-data-root "$DATA_ROOT" \
  --img-size 480 \
  --swin-type base \
  --num-vmsf-blocks 3 \
  --num-heads-fusion 1 \
  --visual-fusion lvmsf \
  --batch-size 16 \
  --alpha 0.5 \
  --lambda-geo 0.5 \
  --coarse-ckpt "$COARSE_CKPT" \
  --refiner-ckpt "$REFINER_CKPT" \
  --locate-ckpt "$LOCATE_CKPT"
