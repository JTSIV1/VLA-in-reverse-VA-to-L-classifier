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
from typing import Any, Dict
from contextlib import nullcontext

# Silence HF tokenizer warnings and bypass permission issues in shared cache dirs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ["HF_HUB_DISABLE_FILE_LOCKS"] = "1"

_SHARED_HF_HUB = "/data/user_data/wenjiel2/.cache/huggingface/hub:/home/wenjiel2/.cache/huggingface/hub"
if os.path.isdir(_SHARED_HF_HUB):
    os.environ["HF_HUB_CACHE"] = _SHARED_HF_HUB
    os.environ["HUGGINGFACE_HUB_CACHE"] = _SHARED_HF_HUB
    os.environ["TRANSFORMERS_CACHE"] = _SHARED_HF_HUB
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# --- Path Setup ---
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
OPENVLA_DIR = str(Path("/data/user_data/wenjiel2/Code/openvla-mini"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TOKENIZATION_DIR = os.path.join(PROJECT_ROOT, "tokenization")
if TOKENIZATION_DIR not in sys.path:
    sys.path.append(TOKENIZATION_DIR)

if OPENVLA_DIR not in sys.path:
    sys.path.append(OPENVLA_DIR)

# --- Monkey-Patch Prismatic Tokenizer ---
os.environ.setdefault(
    "PRISMATIC_DATA_ROOT", "/data/user_data/wenjiel2/datasets/calvin_rlds"
)
os.environ.setdefault("HF_HUB_DISABLE_FILE_LOCKS", "1")
try:
    import tokenization.local_sweep_action_tokenizer

    sys.modules["prismatic.vla.calvin_sweep_action_tokenizer"] = (
        tokenization.local_sweep_action_tokenizer
    )
except Exception as e:
    print(f"Warning: Failed to monkey-patch local sweep action tokenizer: {e}")

# --- Constants ---
RLDS_DATA_ROOT = "/data/user_data/wenjiel2/datasets/calvin_rlds"
DEFAULT_DATASET_NAME = "calvin_dataset"


# --- Model Loading Helpers (HF) ---
def load_hf_model(checkpoint_dir, device):
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForVision2Seq,
        AutoProcessor,
    )
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import (
        PrismaticImageProcessor,
        PrismaticProcessor,
    )

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
    from safetensors import safe_open
    from transformers import Qwen2Config

    original_open = builtins.open

    def hooked_open(file, *args, **kwargs):
        # We only want to hook the config.json inside the run directory or checkpoint directory
        if str(file).endswith("config.json"):
            with original_open(file, *args, **kwargs) as f:
                content = f.read()
            data = json.loads(content)
            if "vla" in data and "base_vlm" in data["vla"]:
                base = data["vla"]["base_vlm"]
                if (
                    isinstance(base, str)
                    and base.startswith("/")
                    and not os.path.exists(base)
                ):
                    # Try to infer model ID from path
                    if "prism-qwen25" in base:
                        new_base = "prism-qwen25-dinosiglip-224px+0_5b"
                        if "extra" in base:
                            new_base = "prism-qwen25-extra-dinosiglip-224px+0_5b"
                        print(f"HACK: Redirecting base_vlm {base} -> {new_base}")
                        data["vla"]["base_vlm"] = new_base
            return io.StringIO(json.dumps(data))
        return original_open(file, *args, **kwargs)

    import timm
    original_create_model = timm.create_model

    def offline_create_model(*args, **kwargs):
        # Loading from the FSDP checkpoint will restore the final backbone weights,
        # so we can safely skip TIMM/HF pretrained downloads during model construction.
        kwargs["pretrained"] = False
        return original_create_model(*args, **kwargs)

    qwen_snapshot = (
        "/data/user_data/wenjiel2/.cache/huggingface/hub:/home/wenjiel2/.cache/huggingface/hub/"
        "models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
    )
    if os.path.isdir(qwen_snapshot):
        offline_qwen_dir = "/tmp/qwen25_0_5b_offline"
        os.makedirs(offline_qwen_dir, exist_ok=True)

        for filename in ("merges.txt", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json"):
            src = os.path.join(qwen_snapshot, filename)
            dst = os.path.join(offline_qwen_dir, filename)
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(src, dst)

        config_path = os.path.join(offline_qwen_dir, "config.json")
        if not os.path.exists(config_path):
            model_path = os.path.join(qwen_snapshot, "model.safetensors")
            with safe_open(model_path, framework="pt") as f:
                hidden_size = int(f.get_tensor("model.embed_tokens.weight").shape[1])
                vocab_size = int(f.get_tensor("model.embed_tokens.weight").shape[0])
                intermediate_size = int(f.get_tensor("model.layers.0.mlp.gate_proj.weight").shape[0])
                q_out = int(f.get_tensor("model.layers.0.self_attn.q_proj.weight").shape[0])
                kv_out = int(f.get_tensor("model.layers.0.self_attn.k_proj.weight").shape[0])
                layer_ids = {
                    int(k.split(".")[2]) for k in f.keys() if k.startswith("model.layers.")
                }
            head_dim = 64
            num_attention_heads = q_out // head_dim
            num_key_value_heads = max(1, kv_out // head_dim)
            cfg = Qwen2Config(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_hidden_layers=max(layer_ids) + 1,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                max_position_embeddings=32768,
                rms_norm_eps=1e-6,
                rope_theta=1_000_000.0,
                hidden_act="silu",
                tie_word_embeddings=False,
                eos_token_id=151643,
                pad_token_id=151643,
            )
            with open(config_path, "w") as f:
                json.dump(cfg.to_dict(), f, indent=2)

        from prismatic.models.backbones.llm import qwen25 as qwen25_mod
        for key in ("qwen25-0_5b-extra", "qwen25-0_5b-pure"):
            if key in qwen25_mod.QWEN25_MODELS:
                qwen25_mod.QWEN25_MODELS[key]["hf_hub_path"] = offline_qwen_dir

    # Apply hook
    builtins.open = hooked_open
    timm.create_model = offline_create_model
    try:
        vla = load_vla(checkpoint_path, load_for_training=False)
    finally:
        # Restore open
        builtins.open = original_open
        timm.create_model = original_create_model

    vla = vla.to(device)
    vla.eval()
    return vla


# --- Visualization Helpers ---
def save_failure_sample(out_dir, idx, sample_data):
    """
    Saves a directory for a single failure instance.
    sample_data keys: image_np, instruction, gt_action, pred_action, gt_tokens, pred_tokens, l1_error
    """
    # Handle episodic vs single-frame error
    err_raw = sample_data["l1_error"]
    is_episodic = isinstance(err_raw, (list, np.ndarray)) and np.array(err_raw).ndim > 0
    mean_err = np.mean(err_raw) if is_episodic else float(err_raw)

    sample_dir = os.path.join(out_dir, f"failure_{idx:03d}_err{mean_err:.4f}")
    os.makedirs(sample_dir, exist_ok=True)

    # 1. Save Image (First frame for episodic)
    if is_episodic:
        img = Image.fromarray(sample_data["image_np"][0])
    else:
        img = Image.fromarray(sample_data["image_np"])
    img.save(os.path.join(sample_dir, "frame.png"))

    # 2. Save Trajectory Plot (First step comparison for episodic, or the single step)
    dims = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
    if is_episodic:
        gt = sample_data["gt_action"][0]
        pred = sample_data["pred_action"][0]
        title_suffix = " (First Step)"
    else:
        gt = sample_data["gt_action"]
        pred = sample_data["pred_action"]
        title_suffix = ""

    plt.figure(figsize=(10, 6))
    x = np.arange(len(dims))
    plt.bar(x - 0.2, gt, 0.4, label="Ground Truth", color="tab:blue")
    plt.bar(x + 0.2, pred, 0.4, label="Predicted", color="tab:orange")
    plt.xticks(x, dims)
    plt.ylabel("Action Value")
    plt.title(
        f"Action Comparison{title_suffix} (Mean L1={mean_err:.4f})\n{sample_data['instruction']}"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(sample_dir, "trajectory.png"))
    plt.close()

    # 3. Save Raw Data and Plots
    if is_episodic:
        # Save full data
        np.savez(
            os.path.join(sample_dir, "data.npz"),
            images=sample_data["image_np"],  # (T, H, W, 3)
            instruction=sample_data["instruction"],
            gt_actions=sample_data["gt_action"],  # (T, 7)
            pred_actions=sample_data["pred_action"],  # (T, 7)
            gt_tokens=sample_data["gt_tokens"],  # (T, n_codes)
            pred_tokens=sample_data["pred_tokens"],  # (T, n_codes)
            l1_errors=sample_data["l1_error"],  # (T,)
            mean_l1_error=mean_err,
        )

        # Save an error-over-time plot
        plt.figure(figsize=(10, 4))
        plt.plot(sample_data["l1_error"])
        plt.xlabel("Step")
        plt.ylabel("L1 Error")
        plt.title(
            f"Episode Error over Time (Mean={mean_err:.4f})\n{sample_data['instruction']}"
        )
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(sample_dir, "error_over_time.png"))
        plt.close()
    else:
        np.savez(
            os.path.join(sample_dir, "data.npz"),
            image=sample_data["image_np"],
            instruction=sample_data["instruction"],
            gt_action=gt,
            pred_action=pred,
            gt_tokens=sample_data["gt_tokens"],
            pred_tokens=sample_data["pred_tokens"],
            l1_error=sample_data["l1_error"],
        )

    # 4. Save Meta JSON (Readable)
    meta = {
        "instruction": sample_data["instruction"],
        "l1_error": float(mean_err),  # Convert numpy.float32 to Python float
        "gt_action": gt.tolist(),  # First step for episodic, or the single step
        "pred_action": pred.tolist(),
        "gt_tokens": sample_data["gt_tokens"].tolist(),
        "pred_tokens": sample_data["pred_tokens"].tolist(),
    }
    if is_episodic:
        # Add full errors (as list of floats) to meta for episodic mode
        meta["all_l1_errors"] = [float(e) for e in sample_data["l1_error"]]

    with open(os.path.join(sample_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["scratch", "openvla"], required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--vqvla_checkpoint_dir", default="")
    parser.add_argument("--sweep_tokenizer_type", default="")
    parser.add_argument("--sweep_checkpoint_path", default="")
    parser.add_argument("--data_root_dir", default=RLDS_DATA_ROOT)
    parser.add_argument(
        "--dataset_name",
        default=DEFAULT_DATASET_NAME,
        help="RLDS data mix / dataset name (e.g. calvin_dataset, bridge_dataset)",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument(
        "--episodic",
        action="store_true",
        help="Analyze by episode instead of by frame.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--debug", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    os.environ.setdefault("PRISMATIC_DATA_ROOT", args.data_root_dir)

    device = args.device
    dataset_name = args.dataset_name
    print(f"Dataset: {dataset_name}")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to cpu")
        device = "cpu"

    # --- Load Model and Tokenizer ---
    if args.family == "openvla":
        vla, processor = load_hf_model(args.checkpoint_dir, device)
        tokenizer = processor.tokenizer
        image_processor = processor.image_processor
        from prismatic.vla.action_tokenizer import ActionTokenizer

        if args.condition == "bin":
            action_tokenizer = ActionTokenizer(tokenizer)
        elif args.sweep_tokenizer_type:
            from prismatic.vla.calvin_sweep_action_tokenizer import (
                CalvinSweepActionTokenizer,
            )

            action_tokenizer = CalvinSweepActionTokenizer(
                tokenizer,
                args.sweep_tokenizer_type,
                args.sweep_checkpoint_path,
                use_extra=True,
            )
        else:
            from prismatic.vla.calvin_vq_action_tokenizer import CalvinVQActionTokenizer

            action_tokenizer = CalvinVQActionTokenizer(
                tokenizer, vqvla_checkpoint_dir=args.vqvla_checkpoint_dir
            )
    else:  # scratch (FSDP)
        vla = load_fsdp_model(args.checkpoint_dir, device)
        tokenizer = vla.llm_backbone.get_tokenizer()
        image_processor = vla.vision_backbone.get_image_transform()

        # Determine action tokenizer string for native loader
        if not (args.sweep_tokenizer_type and args.sweep_checkpoint_path):
            action_tok_str = "action_tokenizer"  # bin
        else:
            action_tok_str = "sweep:{}:{}".format(
                args.sweep_tokenizer_type, args.sweep_checkpoint_path
            )

        # Monkey-patch dlimp to prevent validation dataset truncation when shuffle_buffer_size=1
        try:
            import dlimp as dl

            if not hasattr(dl.DLataset, "_original_take"):
                dl.DLataset._original_take = dl.DLataset.take
                dl.DLataset._original_shuffle = dl.DLataset.shuffle

                def _hooked_take(self, count, *args, **kwargs):
                    if count <= 1:
                        return self
                    return self._original_take(count, *args, **kwargs)

                def _hooked_shuffle(self, buffer_size, *args, **kwargs):
                    if buffer_size <= 1:
                        return self
                    return self._original_shuffle(buffer_size, *args, **kwargs)

                dl.DLataset.take = _hooked_take
                dl.DLataset.shuffle = _hooked_shuffle
        except Exception as e:
            print(f"Warning: Failed to monkey-patch dlimp: {e}")

        # Reload dataset/collator/tokenizer via native materials
        from prismatic.vla.materialize import get_vla_dataset_and_collator
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder

        _, action_tokenizer, _ = get_vla_dataset_and_collator(
            data_root_dir=Path(args.data_root_dir),
            data_mix=dataset_name,
            image_transform=lambda x: x,  # dummy
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
                if dataset_name in stats:
                    action_stats = stats[dataset_name]["action"]
                    unnorm_q01 = np.array(action_stats["q01"], dtype=np.float32)
                    unnorm_q99 = np.array(action_stats["q99"], dtype=np.float32)
                    unnorm_mask = np.array(
                        action_stats.get("mask", [True] * 7), dtype=bool
                    )
                    print(f"Loaded unnorm stats from {stats_path}")
                    break

    # --- Monkey-patch RLDSBatchTransform to include raw data for visualization ---
    from prismatic.vla.datasets import RLDSBatchTransform

    original_transform_call = RLDSBatchTransform.__call__

    def wrapped_transform(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        result = original_transform_call(self, rlds_batch)
        # Store original data for qualitative analysis (no-sync-required)
        # Handle both single-frame and windowed/chunked actions
        if rlds_batch["observation"]["image_primary"].ndim == 4:  # [T, H, W, 3]
            result["raw_image"] = rlds_batch["observation"]["image_primary"][0]
        else:  # [H, W, 3]
            result["raw_image"] = rlds_batch["observation"]["image_primary"]

        # The 'current' action is the first one in the chunk if we're chunking
        if rlds_batch["action"].ndim == 2:  # [T, 7]
            result["raw_action"] = torch.from_numpy(np.array(rlds_batch["action"][0], copy=True)).float()
        else:  # [7]
            result["raw_action"] = torch.from_numpy(np.array(rlds_batch["action"], copy=True)).float()

        raw_instruction = rlds_batch["task"]["language_instruction"]
        if isinstance(raw_instruction, bytes):
            raw_instruction = raw_instruction.decode()
        result["raw_instruction"] = raw_instruction
        return result

    RLDSBatchTransform.__call__ = wrapped_transform

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        tokenizer,
        image_transform=image_processor.apply_transform
        if hasattr(image_processor, "apply_transform")
        else image_processor,
        prompt_builder_fn=PurePromptBuilder,
    )
    # Use future horizon if chunking
    future_horizon = (
        action_tokenizer.required_future_horizon
        if hasattr(action_tokenizer, "required_future_horizon")
        else 0
    )

    # Use EpisodicRLDSDataset if requested
    if args.episodic:
        print("=" * 50)
        print("EPISODIC")
        print("=" * 50)
        from prismatic.vla.datasets import EpisodicRLDSDataset

        val_dataset = EpisodicRLDSDataset(
            data_root_dir=Path(args.data_root_dir),
            data_mix=dataset_name,
            batch_transform=batch_transform,
            resize_resolution=(224, 224),
            shuffle_buffer_size=1,
            train=False,
            image_aug=False,
            future_action_window_size=future_horizon,
        )
    else:
        val_dataset = RLDSDataset(
            data_root_dir=Path(args.data_root_dir),
            data_mix=dataset_name,
            batch_transform=batch_transform,
            resize_resolution=(224, 224),
            shuffle_buffer_size=1,  # Crucial for alignment: disable shuffling
            train=False,
            image_aug=False,
            future_action_window_size=future_horizon,
        )
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side="right"
    )
    # Use identity collation so we can access raw_* keys before collating for the model
    dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x
    )

    # No separate raw_iter needed as it's now integrated!

    failures = []  # List of (l1_error, sample_data)

    print(f"Starting analysis on {args.max_batches} samples...")
    n_codes = getattr(action_tokenizer, "n_codes_per_chunk", 7)
    use_autocast = str(device).startswith("cuda")
    vision_dtype = torch.bfloat16 if use_autocast else torch.float32

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= args.max_batches:
                break

            # Handle Episodic vs Single-Frame
            if args.episodic:
                episode = batch[0]  # List of steps (each contains raw_image, etc.)

                ep_l1_errors = []
                ep_images = []
                ep_gt_actions = []
                ep_pred_actions = []
                ep_gt_tokens_list = []
                ep_pred_tokens_list = []

                print(f"  Processing Episode {i} ({len(episode)} steps)...")

                for step_idx, step in enumerate(episode):
                    # Use collator to get attention_mask and handle tensor formatting
                    step_batch = collator([step])

                    pv = step_batch["pixel_values"]
                    if isinstance(pv, dict):
                        pixel_values = {
                            k: v.to(device=device, dtype=vision_dtype)
                            for k, v in pv.items()
                        }
                    else:
                        pixel_values = pv.to(device=device, dtype=vision_dtype)

                    input_ids = step_batch["input_ids"].to(device)
                    attention_mask = step_batch["attention_mask"].to(device)

                    autocast_ctx = (
                        torch.autocast("cuda", dtype=torch.bfloat16)
                        if use_autocast
                        else nullcontext()
                    )
                    with autocast_ctx:
                        output = vla(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                        )

                    try:
                        if hasattr(vla.vision_backbone, "num_patches"):
                            num_patches = vla.vision_backbone.num_patches
                        else:
                            num_patches = (
                                vla.vision_backbone.featurizer.patch_embed.num_patches
                            )
                    except Exception:
                        num_patches = 256

                    action_logits = output.logits[:, num_patches:-1]
                    action_preds = action_logits.argmax(dim=2)

                    # Use input_ids instead of labels to avoid the -100 masking bug from datasets.py
                    action_gt = input_ids[:, 1:].to(device)
                    mask = (action_tokenizer.action_token_end_idx > action_gt) & (
                        action_gt > action_tokenizer.action_token_begin_idx
                    )

                    gt_tokens = action_gt[mask].cpu().numpy()
                    pred_tokens = action_preds[mask].cpu().numpy()

                    if len(gt_tokens) < n_codes or len(pred_tokens) < n_codes:
                        continue

                    pred_cont = action_tokenizer.decode_token_ids_to_actions(
                        pred_tokens[:n_codes]
                    )
                    if isinstance(pred_cont, torch.Tensor):
                        pred_cont = pred_cont.cpu().numpy()

                    if pred_cont.ndim > 1:
                        pred_cont = pred_cont[0]

                    # Unnormalize if needed (for bin)
                    if is_bin and unnorm_q01 is not None:
                        pred_cont = np.where(
                            unnorm_mask,
                            0.5 * (pred_cont + 1) * (unnorm_q99 - unnorm_q01)
                            + unnorm_q01,
                            pred_cont,
                        ).astype(np.float32)

                    current_raw_action = np.array(step["raw_action"], copy=True)
                    if current_raw_action.shape[-1] == 7:
                        # Map RLDS gripper [0, 1] to training range [-1, 1]
                        current_raw_action[6] = current_raw_action[6] * 2.0 - 1.0
                    l1_err = np.abs(pred_cont - current_raw_action).mean()

                    ep_l1_errors.append(l1_err)
                    ep_images.append(step["raw_image"])
                    ep_gt_actions.append(current_raw_action)
                    ep_pred_actions.append(pred_cont)
                    ep_gt_tokens_list.append(gt_tokens[:n_codes])
                    ep_pred_tokens_list.append(pred_tokens[:n_codes])

                if len(ep_l1_errors) > 0:
                    mean_ep_error = np.mean(ep_l1_errors)
                    sample_data = {
                        "image_np": np.array(ep_images),  # (T, H, W, 3)
                        "instruction": episode[0]["raw_instruction"],
                        "gt_action": np.array(ep_gt_actions),
                        "pred_action": np.array(ep_pred_actions),
                        "gt_tokens": np.array(ep_gt_tokens_list),
                        "pred_tokens": np.array(ep_pred_tokens_list),
                        "l1_error": ep_l1_errors,  # list of errors
                    }
                    failures.append((mean_ep_error, sample_data))

            else:  # Single-Frame Mode
                # batch is a list of 1 raw sample; collate it for the model
                sample = batch[0]
                step_batch = collator([sample])

                pv = step_batch["pixel_values"]
                if isinstance(pv, dict):
                    pixel_values = {
                        k: v.to(device=device, dtype=vision_dtype)
                        for k, v in pv.items()
                    }
                else:
                    pixel_values = pv.to(device=device, dtype=vision_dtype)

                input_ids = step_batch["input_ids"].to(device)
                attention_mask = step_batch["attention_mask"].to(device)

                autocast_ctx = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if use_autocast
                    else nullcontext()
                )
                with autocast_ctx:
                    output = vla(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                    )

                try:
                    if hasattr(vla.vision_backbone, "num_patches"):
                        num_patches = vla.vision_backbone.num_patches
                    else:
                        num_patches = (
                            vla.vision_backbone.featurizer.patch_embed.num_patches
                        )
                except Exception:
                    num_patches = 256

                action_logits = output.logits[:, num_patches:-1]
                action_preds = action_logits.argmax(dim=2)

                # Use input_ids instead of labels to avoid the -100 masking bug from datasets.py
                action_gt = input_ids[:, 1:].to(device)
                mask = (action_tokenizer.action_token_end_idx > action_gt) & (
                    action_gt > action_tokenizer.action_token_begin_idx
                )

                gt_tokens = action_gt[mask].cpu().numpy()
                pred_tokens = action_preds[mask].cpu().numpy()

                if len(gt_tokens) < n_codes or len(pred_tokens) < n_codes:
                    continue

                pred_cont = action_tokenizer.decode_token_ids_to_actions(
                    pred_tokens[:n_codes]
                )
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
                    ).astype(np.float32)

                # Use the raw data from our monkey-patched transform
                gt_action = np.array(sample["raw_action"], copy=True)
                if gt_action.shape[-1] == 7:
                    # Map RLDS gripper [0, 1] to training range [-1, 1]
                    gt_action[6] = gt_action[6] * 2.0 - 1.0
                l1_error = np.abs(pred_cont - gt_action).mean()

                sample_data = {
                    "image_np": sample["raw_image"],
                    "instruction": sample["raw_instruction"],
                    "gt_action": gt_action,
                    "pred_action": pred_cont,
                    "gt_tokens": gt_tokens[:n_codes],
                    "pred_tokens": pred_tokens[:n_codes],
                    "l1_error": l1_error,
                }

                failures.append((l1_error, sample_data))

                if i % 10 == 0:
                    print(f"  Processed {i} samples...")

    # Sort failures by L1 error descending
    failures.sort(key=lambda x: x[0], reverse=True)

    # Save top K worst (highest error)
    worst_dir = os.path.join(args.out_dir, "worst")
    os.makedirs(worst_dir, exist_ok=True)
    print(f"Saving top {args.top_k} worst (highest error) to {worst_dir} ...")
    for idx, (err, data) in enumerate(failures[: args.top_k]):
        save_failure_sample(worst_dir, idx + 1, data)

    # Save top K best (lowest error)
    best_dir = os.path.join(args.out_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    print(f"Saving top {args.top_k} best (lowest error) to {best_dir} ...")
    for idx, (err, data) in enumerate(reversed(failures[-args.top_k :])):
        save_failure_sample(best_dir, idx + 1, data)

    print("Analysis complete.")


if __name__ == "__main__":
    main()
