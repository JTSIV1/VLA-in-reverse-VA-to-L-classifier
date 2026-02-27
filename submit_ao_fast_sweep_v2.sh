#!/bin/bash
# Action-only FAST sweep v2 — same scale×vocab grid as round 3b, two fixes:
#   1. 21 sparse classes (--min_class_count 30), no weighted loss
#   2. max_seq_len=512  (round 3b used 64, truncating scale≥5 sequences)
#      avg tokens/traj by config:
#        scale=1:  23–27   scale=5:  70–95   scale=10: 138
#        scale=20: 230–264  scale=50: 255–303
# All tokenizers already exist — no fitting step needed.

set -euo pipefail

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
cd "$PROJECT_DIR"
mkdir -p logs

CONFIGS=(
    "1,256,./checkpoints/fast_tokenizer_s1_v256"
    "1,512,./checkpoints/fast_tokenizer_s1_v512"
    "1,1024,./checkpoints/fast_tokenizer_s1_v1024"
    "5,256,./checkpoints/fast_tokenizer_s5_v256"
    "5,512,./checkpoints/fast_tokenizer_s5_v512"
    "5,1024,./checkpoints/fast_tokenizer_s5_v1024"
    "10,512,./checkpoints/fast_tokenizer_s10_v512"
    "20,512,./checkpoints/fast_tokenizer_s20_v512"
    "20,768,./checkpoints/fast_tokenizer_s20_v768"
    "20,1024,./checkpoints/fast_tokenizer_s20_v1024"
    "50,1024,./checkpoints/fast_tokenizer_s50_v1024"
    "50,1536,./checkpoints/fast_tokenizer_s50_v1536"
)

N=0
for CFG in "${CONFIGS[@]}"; do
    SCALE=$(echo "$CFG" | cut -d, -f1)
    VOCAB=$(echo "$CFG" | cut -d, -f2)
    TOK=$(echo "$CFG"   | cut -d, -f3)
    TAG="ao_fast21_s${SCALE}_v${VOCAB}"

    sbatch --job-name="$TAG" \
        -o "logs/${TAG}-%j.out" \
        -e "logs/${TAG}-%j.err" \
        run_experiment.sh "$TAG" action_only fast \
        --fast_tokenizer_path "$TOK" \
        --min_class_count 30 \
        --max_seq_len 512

    echo "  Submitted: $TAG"
    (( N++ )) || true
done

echo "Submitted $N jobs (21 classes, max_seq_len=512, same grid as round 3b)"
