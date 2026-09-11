#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda:0}
DATA_ROOT=${DATA_ROOT:-/path/to/RISBench}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints/risbench}
COARSE_DIR=${COARSE_DIR:-$OUTPUT_ROOT/coarse}

"$PYTHON" train_baseline.py \
  --device "$DEVICE" \
  --dataset risbench \
  --refer-data-root "$DATA_ROOT" \
  --img-size 480 \
  --swin-type base \
  --num-vmsf-blocks 3 \
  --num-heads-fusion 1 \
  --visual-fusion lvmsf \
  --batch-size 8 \
  --epochs 40 \
  --lr 3e-5 \
  --weight-decay 1e-2 \
  --snapshot-epochs "10,15,20,30,39" \
  --output-dir "$COARSE_DIR"
