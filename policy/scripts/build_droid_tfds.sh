#!/bin/bash
# Build the filtered DROID TFDS dataset used by OpenVLA / MiniVLA launchers.

set -euo pipefail

ROOT_DIR="/home/yashagar/multimodal/multimodal-project"
OUTPUT_DIR="${OUTPUT_DIR:-/data/user_data/wenjiel2/datasets/droid_rlds_cache}"
RAW_DIR="${RAW_DIR:-/data/user_data/wenjiel2/datasets/droid_rlds}"
METADATA_CACHE="${METADATA_CACHE:-${ROOT_DIR}/data/droid_tokenizer_metadata.csv}"
VAL_FRAC="${VAL_FRAC:-0.1}"
SEED="${SEED:-42}"

cd "$ROOT_DIR"
source venv/bin/activate

python -m datasets.tfds_builders.droid_dataset \
    --output_dir "$OUTPUT_DIR" \
    --droid_rlds_dir "$RAW_DIR" \
    --metadata_cache "$METADATA_CACHE" \
    --val_fraction "$VAL_FRAC" \
    --seed "$SEED"