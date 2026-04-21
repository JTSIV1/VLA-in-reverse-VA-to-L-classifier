# `analyze_vla_failures.py` Guide

This script is a specialized evaluation tool designed specifically to probe **VLA policies (both full OpenVLA adapters and scratch-trained MiniVLAs)** and automatically extract both their worst-performing (failures) and best-performing (successes) scenarios.

## What it Analyzes
1. **Model Loading:** It loads your VLA model and its corresponding discrete action tokenizer (`vq_bet`, `quest`, `oat`, or standard `bin`).
2. **Evaluation:** It iterates over the CALVIN validation dataset (up to `--max_batches`), generating action predictions for each frame.
3. **Error Calculation:** It decodes the predicted discrete tokens back into continuous 7D actions and computes the **L1 Element-wise Error** against the ground-truth trajectory.
4. **Failure and Success Extraction:** It sorts all evaluated samples to find the `--top_k` samples with the absolute highest L1 errors (failures) and the `--bottom_k` samples with the lowest L1 errors (successes).
5. **Organized Export:** Results are saved into two subdirectories: `failures/` and `successes/`. For each sample, it generates a data folder containing:
   - `frame.png`: The exact RGB observation the model failed on.
   - `trajectory.png`: A plotted bar chart comparing the predicted action vs the ground-truth action side-by-side.
   - `meta.json`: A readable JSON tracking the instruction, raw error, and the exact discrete tokens predicted vs the ground truth tokens.
   - `data.npz`: All raw arrays saved for offline probing and custom plotting.

## How to Run It

Because Hugging Face uses file locks that conflict with shared read-only caches, you should export your `HF_HOME` locally to bypass permission lock errors when reading the base foundation models.

**Example 1: Evaluating a MiniVLA (`family=scratch`)**
```bash
export HF_HOME=/home/istepka/.cache/huggingface

python policy/scripts/analyze_vla_failures.py \
    --family scratch \
    --condition minivla_vq_bet_5_16_4 \
    --checkpoint_dir /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier/checkpoints/calvin_sweep/policy/minivla_vq_bet_5_16_4 \
    --sweep_tokenizer_type vq_bet \
    --sweep_checkpoint_path /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier/checkpoints/calvin_sweep/tokenizers/vq_bet_5_16_4/full.pth \
    --out_dir results/vla_failure_analysis/minivla_vq_bet_5_16_4 \
    --top_k 10 \
    --bottom_k 10 \
    --max_batches 50
```

**Example 2: Evaluating a Full OpenVLA LoRA Adapter (`family=openvla`)**
```bash
export HF_HOME=/home/istepka/.cache/huggingface

python policy/scripts/analyze_vla_failures.py \
    --family openvla \
    --condition bin \
    --checkpoint_dir /home/istepka/11777/runs/calvind_policy_adapter_tmp/openvla-7b+calvin_dataset+b16+lr-0.0005+lora-r32+dropout-0.0--bin_baseline--image_aug \
    --out_dir results/vla_failure_analysis/bin_baseline \
    --top_k 10 \
    --bottom_k 10 \
    --max_batches 50
```
