#!/bin/bash
# Round 1 addendum: FAST scale sweep with matched vocab sizes
#
# Each scale produces a different alphabet size. We match vocab_size to ~2x
# the alphabet so BPE has comparable merge headroom across scales:
#
#   scale=1:  alphabet=13,  vocab=256   (243 merge slots)
#   scale=5:  alphabet=64,  vocab=256   (192 merge slots)
#   scale=10: alphabet=126, vocab=512   (386 merge slots) — matches best round 1 config
#   scale=20: alphabet=252, vocab=768   (516 merge slots)
#   scale=50: alphabet=628, vocab=1536  (908 merge slots)
#
# Also re-run pretrained FAST+ (scale=10, vocab=2048, trained on 1M trajs)
# with best-val saving for comparison.

set -euo pipefail

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints

# Scale → vocab mapping
declare -A SCALE_VOCAB
SCALE_VOCAB[1]=256
SCALE_VOCAB[5]=256
SCALE_VOCAB[10]=512
SCALE_VOCAB[20]=768
SCALE_VOCAB[50]=1536

if [[ "${1:-fit}" == "fit" ]]; then
    # Step 1: Fit tokenizers (CPU job, ~30 min)
    # Skip scale=10/v512 if it already exists
    sbatch --job-name=fit_fast_scales \
        -o logs/fit_fast_scales-%j.out -e logs/fit_fast_scales-%j.err \
        --partition=cpu --cpus-per-task=4 --mem=16G --time=02:00:00 \
        --wrap "
source \$(conda info --base)/etc/profile.d/conda.sh && conda activate mmml && cd $PROJECT_DIR && \
for SCALE in 1 5 10 20 50; do \
    VOCAB=\$(python -c \"d={1:256,5:256,10:512,20:768,50:1536}; print(d[\$SCALE])\") && \
    SAVE=./checkpoints/fast_tokenizer_s\${SCALE}_v\${VOCAB} && \
    if [ -d \"\$SAVE\" ]; then echo \"=== Skipping scale=\$SCALE (already exists at \$SAVE) ===\"; continue; fi && \
    echo \"=== Fitting scale=\$SCALE vocab=\$VOCAB ===\" && \
    python fast_tokenizer.py --save_path \$SAVE --vocab_size \$VOCAB --scale \$SCALE; \
done && echo 'All tokenizers fitted'
"
    echo "Submitted tokenizer fitting job. Once complete, run: bash submit_scale_sweep.sh train"

elif [[ "$1" == "train" ]]; then
    # Step 2: Train classifiers
    for SCALE in 1 5 10 20 50; do
        VOCAB=${SCALE_VOCAB[$SCALE]}
        TOK_PATH="./checkpoints/fast_tokenizer_s${SCALE}_v${VOCAB}"

        if [[ ! -d "$TOK_PATH" ]]; then
            echo "WARNING: $TOK_PATH not found, skipping scale=$SCALE"
            continue
        fi

        sbatch --job-name=ao_fast_s${SCALE}_v${VOCAB} \
            -o logs/ao_fast_s${SCALE}_v${VOCAB}-%j.out -e logs/ao_fast_s${SCALE}_v${VOCAB}-%j.err \
            run_experiment.sh ao_fast_s${SCALE}_v${VOCAB} action_only fast \
            --fast_tokenizer_path "$TOK_PATH"
    done

    # Pretrained FAST+ for comparison (with best-val saving)
    sbatch --job-name=ao_fast_pretrained_v2 \
        -o logs/ao_fast_pretrained_v2-%j.out -e logs/ao_fast_pretrained_v2-%j.err \
        run_experiment.sh ao_fast_pretrained_v2 action_only fast \
        --fast_tokenizer_path ./checkpoints/fast_pretrained \
        --max_seq_len 192

    echo "Submitted scale sweep training jobs"
fi
