#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA=${DATA:-/path/to/RefSegRS}
BANK=${BANK:-checkpoints/refsegrs}
PROMPT_BANK_DIR=${PROMPT_BANK_DIR:-$BANK/prompt_bank}
COARSE_CKPT=${COARSE_CKPT:-$BANK/coarse/coarse_best.pth}
DATASET=${DATASET:-refsegrs}
DEVICE=${DEVICE:-cuda:0}
IMG_SIZE=${IMG_SIZE:-480}
NUM_TMEM=${NUM_TMEM:-3}
BATCH_SIZE=${BATCH_SIZE:-8}
GUIDE_PRETRAIN_EPOCHS=${GUIDE_PRETRAIN_EPOCHS:-40}
EPOCHS=${EPOCHS:-10}

mkdir -p "$BANK/localization"

python train_localization_guide.py \
  --device "$DEVICE" \
  --dataset "$DATASET" \
  --refer-data-root "$DATA" \
  --img-size "$IMG_SIZE" \
  --window12 \
  --num-tmem "$NUM_TMEM" \
  --batch-size "$BATCH_SIZE" \
  --guide-pretrain-epochs "$GUIDE_PRETRAIN_EPOCHS" \
  --epochs "$EPOCHS" \
  --coarse-ckpt "$COARSE_CKPT" \
  --prompt-bank-dir "$PROMPT_BANK_DIR" \
  --output-dir "$BANK/localization"
