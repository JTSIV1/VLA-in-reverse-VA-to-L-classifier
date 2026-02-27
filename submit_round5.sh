#!/bin/bash
# Round 5: Multimodal capacity and robustness improvements
# Baseline: full_vc1_d16_late2_sp_wt = 42.4% acc / 40.7% MacF1
# Improvements: (a) d=256, (b) K=49 patches, (c) modal dropout, (d) aux losses
# All: VC-1, native actions, cross_layers=2, sp+wt (21 classes), --time=12:00:00
set -euo pipefail
cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
mkdir -p logs

# ── (a) Scale up: d_model=256, K=16 ─────────────────────────────────────────
JID1=$(sbatch --parsable --job-name=full_vc1_d256_k16_late2 \
    --time=12:00:00 \
    -o logs/full_vc1_d256_k16_late2-%j.out \
    -e logs/full_vc1_d256_k16_late2-%j.err \
    run_experiment.sh full_vc1_d256_k16_late2 full native \
    --vision_encoder vc1 --delta_patches 16 --cross_layers 2 --d_model 256 \
    --min_class_count 30 --weighted_loss)
echo "  $JID1  full_vc1_d256_k16_late2     (a: d=256, K=16)"

# ── (b) Token balance: K=49 delta patches (max with pool_size=7) ─────────────
JID2=$(sbatch --parsable --job-name=full_vc1_k49_late2 \
    --time=12:00:00 \
    -o logs/full_vc1_k49_late2-%j.out \
    -e logs/full_vc1_k49_late2-%j.err \
    run_experiment.sh full_vc1_k49_late2 full native \
    --vision_encoder vc1 --delta_patches 49 --cross_layers 2 \
    --min_class_count 30 --weighted_loss)
echo "  $JID2  full_vc1_k49_late2          (b: d=128, K=49)"

# ── (c) Modality dropout: p=0.3 (30% vision-only, 30% action-only, 40% both) ─
JID3=$(sbatch --parsable --job-name=full_vc1_mdrop_late2 \
    --time=12:00:00 \
    -o logs/full_vc1_mdrop_late2-%j.out \
    -e logs/full_vc1_mdrop_late2-%j.err \
    run_experiment.sh full_vc1_mdrop_late2 full native \
    --vision_encoder vc1 --delta_patches 16 --cross_layers 2 \
    --modal_dropout 0.3 \
    --min_class_count 30 --weighted_loss)
echo "  $JID3  full_vc1_mdrop_late2        (c: d=128, K=16, mdrop=0.3)"

# ── (d) Auxiliary unimodal losses: lambda=0.3 ────────────────────────────────
JID4=$(sbatch --parsable --job-name=full_vc1_aux_late2 \
    --time=12:00:00 \
    -o logs/full_vc1_aux_late2-%j.out \
    -e logs/full_vc1_aux_late2-%j.err \
    run_experiment.sh full_vc1_aux_late2 full native \
    --vision_encoder vc1 --delta_patches 16 --cross_layers 2 \
    --aux_loss_weight 0.3 \
    --min_class_count 30 --weighted_loss)
echo "  $JID4  full_vc1_aux_late2          (d: d=128, K=16, aux=0.3)"

# ── (a+b+c+d) Combined: d=256 + K=49 + modal dropout + aux losses ────────────
JID5=$(sbatch --parsable --job-name=full_vc1_d256_k49_mdrop_aux_late2 \
    --time=12:00:00 \
    -o logs/full_vc1_d256_k49_mdrop_aux_late2-%j.out \
    -e logs/full_vc1_d256_k49_mdrop_aux_late2-%j.err \
    run_experiment.sh full_vc1_d256_k49_mdrop_aux_late2 full native \
    --vision_encoder vc1 --delta_patches 49 --cross_layers 2 --d_model 256 \
    --modal_dropout 0.3 --aux_loss_weight 0.3 \
    --min_class_count 30 --weighted_loss)
echo "  $JID5  full_vc1_d256_k49_mdrop_aux_late2  (a+b+c+d combined)"

echo ""
echo "Round 5 jobs submitted. Monitor with: squeue -u wenjiel2"
