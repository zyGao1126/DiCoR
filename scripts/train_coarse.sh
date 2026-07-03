#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA=${DATA:-/path/to/RefSegRS}
BANK=${BANK:-checkpoints/refsegrs}
DATASET=${DATASET:-refsegrs}
DEVICE=${DEVICE:-cuda:0}
IMG_SIZE=${IMG_SIZE:-480}
NUM_TMEM=${NUM_TMEM:-3}
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-40}

mkdir -p "$BANK/coarse"

python train_baseline.py \
  --device "$DEVICE" \
  --dataset "$DATASET" \
  --refer-data-root "$DATA" \
  --img-size "$IMG_SIZE" \
  --window12 \
  --num-tmem "$NUM_TMEM" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --output-dir "$BANK/coarse"
