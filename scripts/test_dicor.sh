#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA=${DATA:-/path/to/RefSegRS}
BANK=${BANK:-checkpoints/refsegrs}
COARSE_CKPT=${COARSE_CKPT:-$BANK/localization/joint_best.pth}
REFINER_CKPT=${REFINER_CKPT:-$BANK/refiner/refiner.pth}
LOCATE_CKPT=${LOCATE_CKPT:-$BANK/localization/localization_guidance_best.pth}
DATASET=${DATASET:-refsegrs}
SPLIT=${SPLIT:-test}
DEVICE=${DEVICE:-cuda:0}
IMG_SIZE=${IMG_SIZE:-480}
NUM_TMEM=${NUM_TMEM:-3}
BATCH_SIZE=${BATCH_SIZE:-16}

python test.py \
  --device "$DEVICE" \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --refer-data-root "$DATA" \
  --img-size "$IMG_SIZE" \
  --window12 \
  --num-tmem "$NUM_TMEM" \
  --batch-size "$BATCH_SIZE" \
  --coarse-ckpt "$COARSE_CKPT" \
  --refiner-ckpt "$REFINER_CKPT" \
  --locate-ckpt "$LOCATE_CKPT"
