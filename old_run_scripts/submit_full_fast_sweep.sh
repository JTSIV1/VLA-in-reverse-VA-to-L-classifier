#!/bin/bash
# Full FAST scale × vocab sweep: 5 scales × 3 vocab sizes = 15 combinations
#
# Grid:
#   Scales: 1, 5, 10, 20, 50
#   Vocab:  256, 512, 1024
#
# Already have results for:
#   (1,256), (5,256), (10,512) — from scale sweep
#   (10,256), (10,1024) — from round 1 vocab sweep
# Need to fit+train: 10 new combinations

set -euo pipefail

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints

# All (scale, vocab) pairs that need NEW tokenizers
PAIRS=(
    "1,512"
    "1,1024"
    "5,512"
    "5,1024"
    "20,256"
    "20,512"
    "20,1024"
    "50,256"
    "50,512"
    "50,1024"
)

if [[ "${1:-fit}" == "fit" ]]; then
    # Step 1: Fit all missing tokenizers (CPU job)
    sbatch --job-name=fit_fast_full_sweep \
        -o logs/fit_fast_full_sweep-%j.out -e logs/fit_fast_full_sweep-%j.err \
        --partition=cpu --cpus-per-task=4 --mem=16G --time=04:00:00 \
        --wrap "
source \$(conda info --base)/etc/profile.d/conda.sh && conda activate mmml && cd $PROJECT_DIR && \
for PAIR in ${PAIRS[*]}; do \
    SCALE=\${PAIR%%,*} && \
    VOCAB=\${PAIR##*,} && \
    SAVE=./checkpoints/fast_tokenizer_s\${SCALE}_v\${VOCAB} && \
    if [ -d \"\$SAVE\" ]; then echo \"=== Skipping s\${SCALE}_v\${VOCAB} (exists) ===\"; continue; fi && \
    echo \"=== Fitting scale=\$SCALE vocab=\$VOCAB ===\" && \
    python fast_tokenizer.py --save_path \$SAVE --vocab_size \$VOCAB --scale \$SCALE; \
done && echo 'All tokenizers fitted'
"
    echo "Submitted tokenizer fitting job. Once complete, run: bash submit_full_fast_sweep.sh train"

elif [[ "$1" == "train" ]]; then
    # Step 2: Train classifiers for NEW pairs only
    for PAIR in "${PAIRS[@]}"; do
        SCALE=${PAIR%%,*}
        VOCAB=${PAIR##*,}
        TOK_PATH="./checkpoints/fast_tokenizer_s${SCALE}_v${VOCAB}"
        TAG="ao_fast_s${SCALE}_v${VOCAB}"

        if [[ ! -d "$TOK_PATH" ]]; then
            echo "WARNING: $TOK_PATH not found, skipping $TAG"
            continue
        fi

        sbatch --job-name="$TAG" \
            -o "logs/${TAG}-%j.out" -e "logs/${TAG}-%j.err" \
            run_experiment.sh "$TAG" action_only fast \
            --fast_tokenizer_path "$TOK_PATH"
    done

    echo "Submitted training jobs for 10 new scale×vocab combinations"
fi
