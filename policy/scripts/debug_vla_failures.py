"""
analyze_vla_failures.py

Specialized failure analysis for VLA policies (OpenVLA and MiniVLA).
Identifies frames with the highest L1 error between predicted and ground-truth actions,
and exports detailed samples (images, trajectories, raw data) for offline inspection.

Usage:
    python policy/scripts/analyze_vla_failures.py \
        --family scratch \
        --condition vb_c5e16g4 \
        --checkpoint_dir /home/istepka/11777/runs/calvind_scratch/... \
        --out_dir results/vla_failure_analysis/vb_c5e16g4 \
        --top_k 10
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# --- Path Setup ---
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
OPENVLA_DIR  = str(Path("/data/user_data/wenjiel2/Code/openvla-mini"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TOKENIZATION_DIR = os.path.join(PROJECT_ROOT, "tokenization")
if TOKENIZATION_DIR not in sys.path:
    sys.path.append(TOKENIZATION_DIR)

if OPENVLA_DIR not in sys.path:
    sys.path.append(OPENVLA_DIR)

# --- Monkey-Patch Prismatic Tokenizer ---
os.environ.setdefault("PRISMATIC_DATA_ROOT", "/data/user_data/wenjiel2/datasets/calvin_rlds")
os.environ.setdefault("HF_HUB_DISABLE_FILE_LOCKS", "1")
try:
    import tokenization.local_sweep_action_tokenizer
    sys.modules["prismatic.vla.calvin_sweep_action_tokenizer"] = tokenization.local_sweep_action_tokenizer
except Exception as e:
    print(f"Warning: Failed to monkey-patch local sweep action tokenizer: {e}")

# --- Constants ---
RLDS_DATA_ROOT = "/data/user_data/wenjiel2/datasets/calvin_rlds"

# --- Model Loading Helpers (HF) ---
def load_hf_model(checkpoint_dir, device):
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    print(f"Loading HF processor and model from {checkpoint_dir} ...")
    processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    vla.eval()
    return vla, processor

# --- Model Loading Helpers (FSDP) ---
def load_fsdp_model(checkpoint_path, device):
    from prismatic.models import load_vla
    from prismatic.overwatch import initialize_overwatch
    path = Path(checkpoint_path)
    if path.is_dir():
        # Look for .pt files in checkpoints/
        ckpt_dir = path / "checkpoints"
        if ckpt_dir.exists():
            ckpts = sorted(list(ckpt_dir.glob("*.pt")))
            if ckpts:
                checkpoint_path = str(ckpts[-1])
                print(f"Found latest checkpoint: {checkpoint_path}")
            else:
                raise ValueError(f"No .pt files found in {ckpt_dir}")
        else:
            raise ValueError(f"No checkpoints directory found in {path}")
            
    # HACK: Prismatic's load_vla fails if base_vlm in config.json is an absolute path that doesn't exist.
    # We monkey-patch the config loading by temporarily overriding open if needed, 
    # but a cleaner way is just to fix the config in memory. 
    # Since load_vla is a bit rigid, we'll do a surgical strike.
    
    print(f"Loading native VLA from FSDP checkpoint {checkpoint_path} ...")
    
    # We'll use a local import to avoid issues
    import builtins
    import json
    import io
    original_open = builtins.open
    
    def hooked_open(file, *args, **kwargs):
        # We only want to hook the config.json inside the run directory or checkpoint directory
        if str(file).endswith("config.json"):
            with original_open(file, *args, **kwargs) as f:
                content = f.read()
            data = json.loads(content)
            if "vla" in data and "base_vlm" in data["vla"]:
                base = data["vla"]["base_vlm"]
                if isinstance(base, str) and base.startswith("/") and not os.path.exists(base):
                    # Try to infer model ID from path
                    if "prism-qwen25" in base:
                        new_base = "prism-qwen25-dinosiglip-224px+0_5b"
                        if "extra" in base:
                            new_base = "prism-qwen25-extra-dinosiglip-224px+0_5b"
                        print(f"HACK: Redirecting base_vlm {base} -> {new_base}")
                        data["vla"]["base_vlm"] = new_base
            return io.StringIO(json.dumps(data))
        return original_open(file, *args, **kwargs)

    # Apply hook
    builtins.open = hooked_open
    try:
        vla = load_vla(checkpoint_path, load_for_training=False)
    finally:
        # Restore open
        builtins.open = original_open
        
    vla = vla.to(device)
    vla.eval()
    return vla

# --- Visualization Helpers ---
def save_failure_sample(out_dir, idx, sample_data):
    """
    Saves a directory for a single failure instance.
    sample_data keys: image_np, instruction, gt_action, pred_action, gt_tokens, pred_tokens, l1_error
    """
    sample_dir = os.path.join(out_dir, f"failure_{idx:03d}_err{sample_data['l1_error']:.4f}")
    os.makedirs(sample_dir, exist_ok=True)

    # 1. Save Image
    img = Image.fromarray(sample_data["image_np"])
    img.save(os.path.join(sample_dir, "frame.png"))

    # 2. Save Trajectory Plot (1st step of predicted vs GT)
    dims = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
    gt = sample_data["gt_action"]
    pred = sample_data["pred_action"]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(dims))
    plt.bar(x - 0.2, gt, 0.4, label="Ground Truth", color="tab:blue")
    plt.bar(x + 0.2, pred, 0.4, label="Predicted", color="tab:orange")
    plt.xticks(x, dims)
    plt.ylabel("Action Value")
    plt.title(f"Action Comparison (L1={sample_data['l1_error']:.4f})\n{sample_data['instruction']}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(sample_dir, "trajectory.png"))
    plt.close()

    # 3. Save Raw Data (NPZ for offline access)
    np.savez(
        os.path.join(sample_dir, "data.npz"),
        image=sample_data["image_np"],
        instruction=sample_data["instruction"],
        gt_action=gt,
        pred_action=pred,
        gt_tokens=sample_data["gt_tokens"],
        pred_tokens=sample_data["pred_tokens"],
        l1_error=sample_data["l1_error"]
    )

    # 4. Save Meta JSON (Readable)
    meta = {
        "instruction": sample_data["instruction"],
        "l1_error": float(sample_data["l1_error"]),
        "gt_action": gt.tolist(),
        "pred_action": pred.tolist(),
        "gt_tokens": sample_data["gt_tokens"].tolist(),
        "pred_tokens": sample_data["pred_tokens"].tolist()
    }
    with open(os.path.join(sample_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["scratch", "openvla"], required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--vqvla_checkpoint_dir", default="")
    parser.add_argument("--sweep_tokenizer_type", default="")
    parser.add_argument("--sweep_checkpoint_path", default="")
    parser.add_argument("--data_root_dir", default=RLDS_DATA_ROOT)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.environ.setdefault("PRISMATIC_DATA_ROOT", args.data_root_dir)

    device = args.device

    # --- Load Model and Tokenizer ---
    if args.family == "openvla":
        vla, processor = load_hf_model(args.checkpoint_dir, device)
        tokenizer = processor.tokenizer
        image_processor = processor.image_processor
        from prismatic.vla.action_tokenizer import ActionTokenizer
        if args.condition == "bin":
            action_tokenizer = ActionTokenizer(tokenizer)
        elif args.sweep_tokenizer_type:
            from prismatic.vla.calvin_sweep_action_tokenizer import CalvinSweepActionTokenizer
            action_tokenizer = CalvinSweepActionTokenizer(tokenizer, args.sweep_tokenizer_type, args.sweep_checkpoint_path, use_extra=True)
        else:
            from prismatic.vla.calvin_vq_action_tokenizer import CalvinVQActionTokenizer
            action_tokenizer = CalvinVQActionTokenizer(tokenizer, vqvla_checkpoint_dir=args.vqvla_checkpoint_dir)
    else:  # scratch (FSDP)
        vla = load_fsdp_model(args.checkpoint_dir, device)
        tokenizer = vla.llm_backbone.get_tokenizer()
        image_processor = vla.vision_backbone.get_image_transform()
        
        # Determine action tokenizer string for native loader
        if not (args.sweep_tokenizer_type and args.sweep_checkpoint_path):
             action_tok_str = "action_tokenizer"  # bin
        else:
             action_tok_str = "sweep:{}:{}".format(args.sweep_tokenizer_type, args.sweep_checkpoint_path)
             
        # Reload dataset/collator/tokenizer via native materials
        from prismatic.vla.materialize import get_vla_dataset_and_collator
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder
        _, action_tokenizer, _ = get_vla_dataset_and_collator(
            data_root_dir=Path(args.data_root_dir),
            data_mix="calvin_dataset",
            image_transform=lambda x: x, # dummy
            tokenizer=tokenizer,
            prompt_builder_fn=PurePromptBuilder,
            default_image_resolution=(3, 224, 224),
            padding_side="right",
            train=False,
            image_aug=False,
            action_tokenizer=action_tok_str,
        )

    # --- Setup Dataset ---
    from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    import tensorflow_datasets as tfds

    # For bin tokenizer, load normalization stats if available
    is_bin = not (args.sweep_tokenizer_type and args.sweep_checkpoint_path)
    unnorm_q01, unnorm_q99, unnorm_mask = None, None, None
    if is_bin:
        # Try to find dataset_statistics.json in the run directory (for FSDP)
        # or in the checkpoint directory (for HF)
        search_dirs = [Path(args.checkpoint_dir).parents[1], Path(args.checkpoint_dir)]
        for d in search_dirs:
            stats_path = d / "dataset_statistics.json"
            if stats_path.exists():
                with open(stats_path) as f:
                    stats = json.load(f)
                if "calvin_dataset" in stats:
                    action_stats = stats["calvin_dataset"]["action"]
                    unnorm_q01 = np.array(action_stats["q01"], dtype=np.float32)
                    unnorm_q99 = np.array(action_stats["q99"], dtype=np.float32)
                    unnorm_mask = np.array(action_stats.get("mask", [True]*7), dtype=bool)
                    print(f"Loaded unnorm stats from {stats_path}")
                    break

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        tokenizer,
        image_transform=image_processor.apply_transform if hasattr(image_processor, "apply_transform") else image_processor,
        prompt_builder_fn=PurePromptBuilder,
    )
    # Use RLDS with raw actions iter for comparison
    future_horizon = action_tokenizer.required_future_horizon if hasattr(action_tokenizer, "required_future_horizon") else 0
    
    val_dataset = RLDSDataset(
        data_root_dir=Path(args.data_root_dir),
        data_mix="calvin_dataset",
        batch_transform=batch_transform,
        resize_resolution=(224, 224),
        shuffle_buffer_size=1,        # Crucial for alignment: disable shuffling
        train=False,
        image_aug=False,
        future_action_window_size=future_horizon,
    )
    collator = PaddedCollatorForActionPrediction(tokenizer.model_max_length, tokenizer.pad_token_id, padding_side="right")
    dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collator)

    # Raw iterator for GT comparison and original images
    # We must ensure this iterates in the EXACT SAME order as RLDSDataset(train=False)
    rlds_path = os.path.join(args.data_root_dir, "calvin_dataset", "1.0.0")
    raw_ds = tfds.builder_from_directory(rlds_path).as_dataset(split="val")
    
    def raw_iterator():
        for ep in raw_ds:
            steps = ep["steps"]
            for step in steps:
                obs = step["observation"]
                img = obs["rgb_static"] if "rgb_static" in obs else obs["image"]
                yield {
                    "image": img.numpy(),
                    "action": step["action"].numpy(),
                    "instruction": step["language_instruction"].numpy().decode("utf-8")
                }
    
    raw_iter = raw_iterator()

    failures = [] # List of (l1_error, sample_data)

    print(f"Starting analysis on {args.max_batches} samples...")
    n_codes = getattr(action_tokenizer, 'n_codes_per_chunk', 7)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= args.max_batches:
                break
            
            try:
                raw_data = next(raw_iter)
            except StopIteration:
                break

            pv = batch["pixel_values"]
            if isinstance(pv, dict):
                pixel_values = {k: v.to(torch.bfloat16).to(device) for k, v in pv.items()}
            else:
                pixel_values = pv.to(torch.bfloat16).to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = vla(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            
            try:
                if hasattr(vla.vision_backbone, "num_patches"):
                    num_patches = vla.vision_backbone.num_patches
                else:
                    num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches
            except Exception:
                num_patches = 256

            action_logits = output.logits[:, num_patches:-1]
            action_preds = action_logits.argmax(dim=2)
            
            action_gt = labels[:, 1:].to(device)
            mask = (action_tokenizer.action_token_end_idx > action_gt) & (action_gt > action_tokenizer.action_token_begin_idx)
            
            gt_tokens = action_gt[mask].cpu().numpy()
            pred_tokens = action_preds[mask].cpu().numpy()

            print(f"\n--- DEBUG ---")
            print(f"vla.vision_backbone.num_patches: {getattr(vla.vision_backbone, 'num_patches', 'N/A')}")
            print(f"input_ids shape: {input_ids.shape}")
            print(f"labels shape: {labels.shape}")
            print(f"output.logits shape: {output.logits.shape}")
            print(f"action_logits shape: {action_logits.shape}")
            print(f"action_preds shape: {action_preds.shape}")
            print(f"action_gt shape: {action_gt.shape}")
            print(f"mask shape: {mask.shape}")
            print(f"gt_tokens length: {len(gt_tokens)} pred_tokens length: {len(pred_tokens)} n_codes: {n_codes}")
            print(f"action_gt max: {action_gt.max().item()} min: {action_gt.min().item()}")
            print(f"mask matches: {mask.sum().item()}")
            import sys; sys.exit(0)

            if len(gt_tokens) < n_codes or len(pred_tokens) < n_codes:
                continue
            
            pred_cont = action_tokenizer.decode_token_ids_to_actions(pred_tokens[:n_codes])
            if isinstance(pred_cont, torch.Tensor):
                pred_cont = pred_cont.cpu().numpy()
            
            if pred_cont.ndim > 1:
                pred_cont = pred_cont[0]
            
            # Unnormalize if needed (for bin)
            if is_bin and unnorm_q01 is not None:
                pred_cont = np.where(
                    unnorm_mask,
                    0.5 * (pred_cont + 1) * (unnorm_q99 - unnorm_q01) + unnorm_q01,
                    pred_cont,
                )
            
            gt_action = raw_data["action"]
            l1_error = np.abs(pred_cont - gt_action).mean()

            
            sample_data = {
                "image_np": raw_data["image"],
                "instruction": raw_data["instruction"],
                "gt_action": gt_action,
                "pred_action": pred_cont,
                "gt_tokens": gt_tokens[:n_codes],
                "pred_tokens": pred_tokens[:n_codes],
                "l1_error": l1_error
            }
            
            failures.append((l1_error, sample_data))
            
            if i % 10 == 0:
                print(f"  Processed {i} steps...")

    # Sort failures by L1 error descending
    failures.sort(key=lambda x: x[0], reverse=True)
    
    # Save top K
    print(f"Saving top {args.top_k} failures to {args.out_dir} ...")
    for idx, (err, data) in enumerate(failures[:args.top_k]):
        save_failure_sample(args.out_dir, idx + 1, data)

    print("Analysis complete.")

if __name__ == "__main__":
    main()
