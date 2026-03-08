"""
evaluate_openvla.py

Stage 3: Evaluate fine-tuned OpenVLA models on CALVIN val split.

Two evaluation modes:
  A. Teacher-forcing NLL + action accuracy  (requires merged OpenVLA checkpoint on GPU)
  B. Verb decodability probe via tokenizer round-trip (lightweight, CPU-friendly)
     - Encodes CALVIN val trajectories through the action tokenizer and decodes back
     - Runs reconstructed trajectories through our pre-trained verb classifier
     - Measures how much verb information survives the encode-decode round-trip

Usage:
  # NLL only (Stage 2a baseline):
  python -m openvla_experiment.scripts.evaluate_openvla \\
      --condition bin \\
      --checkpoint_dir /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier/runs/openvla/... \\
      --data_root_dir /data/user_data/wenjiel2/datasets/calvin_rlds \\
      --output_dir results/stage3/bin --eval_nll --max_nll_batches 500

  # Verb probe only (no OpenVLA model needed):
  python -m openvla_experiment.scripts.evaluate_openvla \\
      --condition vq_verb \\
      --vqvla_checkpoint_dir checkpoints/vqvla_ft_verb_l0.5 \\
      --verb_classifier_ckpt checkpoints/ao_native_sparse_weighted_j6457852_best.pth \\
      --output_dir results/stage3/vq_verb --eval_verb_probe
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
OPENVLA_DIR  = str(Path("/data/user_data/wenjiel2/Code/openvla-mini"))
for p in [PROJECT_ROOT, OPENVLA_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Constants ─────────────────────────────────────────────────────────────────
CALVIN_VAL_DIR  = "/data/user_data/yashagar/task_D_D/validation"
CALVIN_TRAIN_DIR = "/data/user_data/yashagar/task_D_D/training"
DEFAULT_VERB_CLF = os.path.join(
    PROJECT_ROOT, "checkpoints", "ao_native_sparse_weighted_j6457852_best.pth"
)
RLDS_DATA_ROOT  = "/data/user_data/wenjiel2/datasets/calvin_rlds"
DEFAULT_OUT     = os.path.join(PROJECT_ROOT, "results", "stage3")

# ── Bin tokenizer round-trip ───────────────────────────────────────────────────

def bin_roundtrip(actions: np.ndarray, n_bins: int = 256,
                  min_action: float = -1.0, max_action: float = 1.0) -> np.ndarray:
    """Quantize continuous actions through the bin tokenizer and back.

    actions: (T, A) float32 in [-1, 1]
    Returns: (T, A) float32, each value snapped to the nearest bin center.
    """
    bins        = np.linspace(min_action, max_action, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    clipped     = np.clip(actions, min_action, max_action)
    indices     = np.digitize(clipped, bins) - 1       # in [0, n_bins]
    indices     = np.clip(indices, 0, len(bin_centers) - 1)
    return bin_centers[indices]


# ── VQ tokenizer round-trip ───────────────────────────────────────────────────

# VQ-VLA config JSONs live in tokenization/vqvla/config/
VQVLA_CONFIG_DIR = os.path.join(PROJECT_ROOT, "tokenization", "vqvla", "config")


def load_vq_wrapper(vqvla_checkpoint_dir: str, device: str = "cpu"):
    """Load the VQ-VLA model for action tokenization.

    Works around the diffusers `from_config(file_path)` API change by
    monkeypatching ActionVQVAEPE.from_config to accept JSON file paths.
    """
    import json
    from tokenization.vqvla.modeling_causal_vae import ActionVQVAEPE
    from tokenization.vqvla.action_vqvae_wrapper import ActionVQVAELossWrapper

    # Patch from_config so it accepts ".json" file paths (pre-0.30 diffusers behavior)
    _orig_from_config = ActionVQVAEPE.from_config.__func__

    @classmethod  # type: ignore[misc]
    def _patched_from_config(cls, config, **kwargs):
        if isinstance(config, str) and config.endswith(".json"):
            with open(config, "r") as f:
                config = json.load(f)
        return _orig_from_config(cls, config, **kwargs)

    ActionVQVAEPE.from_config = _patched_from_config

    try:
        wrapper = ActionVQVAELossWrapper(
            model_path=VQVLA_CONFIG_DIR,   # config JSONs live here
            use_action_type_pe=True,
            use_time_pe=True,
            freeze=True,
            is_eval=True,
        )
        weights_path = os.path.join(vqvla_checkpoint_dir, "vqvla_weights.pth")
        inner_weights = torch.load(weights_path, map_location="cpu", weights_only=False)
        load_result = wrapper.vqvae.load_state_dict(inner_weights, strict=True)
        print(f"[load_vq_wrapper] Loaded weights from {weights_path}: {load_result}")
        wrapper.to(device)
        wrapper.eval()
    finally:
        ActionVQVAEPE.from_config = classmethod(_orig_from_config)

    # Expose attributes expected by vq_roundtrip_trajectory
    wrapper.vqvae_n_embed = wrapper.vqvae.vqvae_n_embed
    wrapper.vqvae_groups  = wrapper.vqvae.vqvae_groups
    wrapper.input_dim_h   = wrapper.vqvae.temporal_compression_ratio
    wrapper.input_dim_w   = 7
    return wrapper


def vq_roundtrip_trajectory(actions: np.ndarray, wrapper, device: str = "cpu") -> np.ndarray:
    """Round-trip a full trajectory (T, 7) through the VQ tokenizer in 5-step chunks.

    Returns: (n_chunks * 5, 7) reconstructed actions.
    If T < 5, returns an empty array.
    """
    T       = len(actions)
    n_chunks = T // 5
    if n_chunks == 0:
        return np.zeros((0, 7), dtype=np.float32)

    recon_list = []
    with torch.no_grad():
        for i in range(n_chunks):
            chunk = actions[i * 5 : (i + 1) * 5]              # (5, 7)
            x     = torch.from_numpy(chunk).float().to(device).unsqueeze(0)  # (1, 5, 7)
            # get_code returns either codes directly or (None, codes)
            codes_result = wrapper.get_code(x)
            codes = codes_result[1] if isinstance(codes_result, tuple) else codes_result  # (1, 4)
            z_embed  = wrapper.draw_code_forward(codes)        # (1, latent_dim)
            decoded  = wrapper.get_action_from_latent(z_embed) # (1, 5, 7) or tensor-like
            if hasattr(decoded, "sample"):
                decoded = decoded.sample
            if isinstance(decoded, torch.Tensor):
                decoded = decoded.squeeze(0).cpu().numpy()     # (5, 7)
            else:
                decoded = np.array(decoded).squeeze(0)         # fallback
            recon_list.append(decoded)
    return np.concatenate(recon_list, axis=0)                  # (n_chunks*5, 7)


# ── Verb probe ────────────────────────────────────────────────────────────────

def run_verb_probe(condition: str, vqvla_checkpoint_dir: str,
                   verb_classifier_ckpt: str, device: str,
                   min_class_count: int = 30) -> dict:
    """
    Evaluate verb decodability by round-tripping trajectories through the
    action tokenizer then running through the pre-trained verb classifier.

    Returns a dict with keys: verb_acc, macro_f1, n_samples, id_to_verb, per_class.
    """
    from sklearn.metrics import classification_report
    from utils import load_calvin_to_dataframe
    from train_transformer import ActionToVerbTransformer, CalvinVerbDataset
    from config import VAL_DIR, TRAIN_DIR, MAX_SEQ_LEN, D_MODEL, NHEAD, NUM_LAYERS

    print("\n=== Verb Probe (tokenizer round-trip) ===")

    # ── Load verb classifier ──────────────────────────────────────────────────
    raw = torch.load(verb_classifier_ckpt, map_location=device, weights_only=False)
    state_dict  = raw["state_dict"]
    num_verbs   = raw["num_verbs"]
    verb_to_id  = raw["verb_to_id"]
    id_to_verb  = raw["id_to_verb"]
    d_model     = raw.get("d_model", D_MODEL)
    nhead       = raw.get("nhead", NHEAD)
    num_layers  = raw.get("num_layers", NUM_LAYERS)
    action_dim  = raw.get("action_dim", 7)
    max_action_len = raw.get("max_action_len", MAX_SEQ_LEN)

    clf = ActionToVerbTransformer(
        num_verbs=num_verbs,
        action_vocab_size=None,
        d_model=d_model, nhead=nhead, num_layers=num_layers,
        action_dim=action_dim,
        max_action_len=max_action_len,
        modality="action_only",
        action_rep="native",
    ).to(device)
    clf.load_state_dict(state_dict, strict=True)
    clf.eval()
    print(f"Loaded verb classifier: {num_verbs} classes, d_model={d_model}")

    # ── Load VQ wrapper (for VQ conditions) ───────────────────────────────────
    vq_wrapper = None
    if condition.startswith("vq_"):
        assert vqvla_checkpoint_dir, f"--vqvla_checkpoint_dir required for condition={condition}"
        vq_wrapper = load_vq_wrapper(vqvla_checkpoint_dir, device=device)
        print(f"Loaded VQ wrapper from {vqvla_checkpoint_dir}")

    # ── Load val DataFrame ────────────────────────────────────────────────────
    val_df  = load_calvin_to_dataframe(VAL_DIR)
    # Filter to verbs seen in training (sparse classes)
    if min_class_count > 0:
        train_df    = load_calvin_to_dataframe(TRAIN_DIR)
        verb_counts = train_df["primary_verb"].value_counts()
        keep_verbs  = set(verb_counts[verb_counts >= min_class_count].index)
        val_df      = val_df[val_df["primary_verb"].isin(keep_verbs)].copy()
        print(f"Kept {len(keep_verbs)} sparse verbs; val samples: {len(val_df)}")

    # ── Build val dataset (native action rep, action_only) ────────────────────
    val_ds = CalvinVerbDataset(
        val_df, VAL_DIR,
        max_seq_len=max_action_len,
        modality="action_only",
        action_tokenizer=None,
    )
    # Share verb vocab with classifier's verb_to_id
    val_ds.verb_to_id = verb_to_id
    val_ds.id_to_verb = id_to_verb
    # Filter val samples to known verbs
    valid_mask = val_df["primary_verb"].isin(verb_to_id.keys())
    val_df     = val_df[valid_mask].reset_index(drop=True)
    val_ds.df  = val_df
    print(f"Val samples after vocab filter: {len(val_df)}")

    dataloader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4,
                            collate_fn=lambda batch: _collate_action_only(batch, max_action_len))

    all_preds, all_labels = [], []

    for frames, actions, scene_vecs, labels, seq_lengths in dataloader:
        # actions: (B, max_seq_len, 7) continuous, float32
        # seq_lengths: (B,) int — actual trajectory length before padding

        actions_np = actions.numpy()           # (B, T, 7)
        recon_list = []

        for b in range(len(actions_np)):
            L    = int(seq_lengths[b].item()) - 1  # seq_len includes CLS token; subtract
            traj = actions_np[b, :L]               # (L, 7) actual actions

            if condition == "bin":
                recon = bin_roundtrip(traj)        # (L, 7)
            elif condition.startswith("vq_"):
                recon = vq_roundtrip_trajectory(traj, vq_wrapper, device=device)
                if len(recon) == 0:
                    recon = traj  # fallback: too short, keep original
            else:
                raise ValueError(f"Unknown condition: {condition}")

            # Pad/truncate to max_action_len
            L_recon = len(recon)
            if L_recon < max_action_len:
                pad = np.zeros((max_action_len - L_recon, 7), dtype=np.float32)
                recon = np.concatenate([recon, pad], axis=0)
            else:
                recon = recon[:max_action_len]
            recon_list.append(recon)

        recon_batch = torch.tensor(np.stack(recon_list), dtype=torch.float32).to(device)

        # Run verb classifier
        with torch.no_grad():
            logits = clf(frames=None, trajectories=recon_batch,
                         seq_lengths=seq_lengths.to(device))
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    # ── Compute metrics ───────────────────────────────────────────────────────
    correct   = sum(p == l for p, l in zip(all_preds, all_labels))
    verb_acc  = correct / len(all_labels) if all_labels else 0.0
    verb_names = [id_to_verb[i] for i in range(num_verbs)]
    report     = classification_report(all_labels, all_preds,
                                       target_names=verb_names,
                                       output_dict=True, zero_division=0)
    macro_f1   = report.get("macro avg", {}).get("f1-score", 0.0)

    print(f"Verb probe results ({condition}):")
    print(f"  Accuracy : {verb_acc:.4f}")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print(f"  Samples  : {len(all_labels)}")

    return {
        "verb_acc":   verb_acc,
        "macro_f1":   macro_f1,
        "n_samples":  len(all_labels),
        "id_to_verb": {str(k): v for k, v in id_to_verb.items()},
        "per_class":  {k: v for k, v in report.items()
                       if k not in ("accuracy", "macro avg", "weighted avg")},
        "macro_avg":  report.get("macro avg", {}),
        "weighted_avg": report.get("weighted avg", {}),
    }


def _collate_action_only(batch, max_seq_len):
    """Collate CalvinVerbDataset samples for action_only modality."""
    frames_list, actions_list, scene_list, labels_list, seqlens_list = zip(*batch)
    # frames: dummy zeros in action_only mode
    actions = torch.stack([a.float() if a.dtype != torch.float32 else a
                           for a in actions_list])
    labels  = torch.stack(labels_list)
    seqlens = torch.tensor(seqlens_list, dtype=torch.long)
    return None, actions, None, labels, seqlens


# ── NLL evaluation ────────────────────────────────────────────────────────────

def run_nll_eval(condition: str, checkpoint_dir: str, vqvla_checkpoint_dir: str,
                 data_root_dir: str, max_batches: int, device: str) -> dict:
    """
    Continuous L1 loss evaluation on the RLDS val split (teacher-forcing).

    Loads the merged fine-tuned OpenVLA model, runs teacher-forcing on the
    CALVIN validation set, decodes predicted action tokens to continuous
    actions, and computes mean L1 vs ground-truth. Returns avg_l1, n_batches.
    """
    print("\n=== L1 Evaluation ===")

    os.environ.setdefault("PRISMATIC_DATA_ROOT", data_root_dir)

    import torch
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder

    # Register HF custom model classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    print(f"Loading processor and model from {checkpoint_dir} ...")
    processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    vla.eval()
    print("Model loaded.")

    # Create action tokenizer matching the condition
    if condition == "bin":
        action_tokenizer = ActionTokenizer(processor.tokenizer)
        future_horizon   = 0
    elif condition.startswith("vq_"):
        from prismatic.vla.calvin_vq_action_tokenizer import CalvinVQActionTokenizer
        action_tokenizer = CalvinVQActionTokenizer(
            processor.tokenizer,
            vqvla_checkpoint_dir=vqvla_checkpoint_dir,
        )
        future_horizon = action_tokenizer.required_future_horizon
        print(f"VQ tokenizer: groups={action_tokenizer.vq_vae.vqvae_groups}, "
              f"n_embed={action_tokenizer.vq_vae.vqvae_n_embed}, "
              f"future_horizon={future_horizon}")
    else:
        raise ValueError(f"Unknown condition: {condition}")

    # Build val dataloader
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    val_dataset = RLDSDataset(
        data_root_dir=Path(data_root_dir),
        data_mix="calvin_dataset",
        batch_transform=batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=1000,        # small buffer for val (order doesn't matter much)
        train=False,                     # use val split
        image_aug=False,
        future_action_window_size=future_horizon,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    dataloader = DataLoader(
        val_dataset, batch_size=8, sampler=None,
        collate_fn=collator, num_workers=0,
    )

    total_l1, l1_batches = 0.0, 0
    n_batches = 0

    # For VQ: tokens per chunk = n_groups; for bin: tokens per step = action_dim (7)
    if condition.startswith("vq_"):
        _tokens_per_unit = action_tokenizer.vq_vae.vqvae_groups   # 4
    else:
        _tokens_per_unit = 7  # bin: 7 dims per timestep

    with torch.no_grad():
        for batch in dataloader:
            if max_batches > 0 and n_batches >= max_batches:
                break

            pixel_values   = batch["pixel_values"].to(torch.bfloat16).to(device)
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = vla(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                )

            num_patches   = vla.vision_backbone.featurizer.patch_embed.num_patches
            action_logits = output.logits[:, num_patches:-1]
            action_preds  = action_logits.argmax(dim=2)
            action_gt     = labels[:, 1:].to(device)
            mask          = action_gt > action_tokenizer.action_token_begin_idx

            # Continuous L1: decode tokens → continuous actions → L1 vs GT
            try:
                pred_flat = action_preds[mask].cpu().numpy()  # (N,)
                gt_flat   = action_gt[mask].cpu().numpy()     # (N,)
                tpu = _tokens_per_unit
                n = (len(pred_flat) // tpu) * tpu
                if n > 0:
                    if condition == "bin":
                        pred_cont = action_tokenizer.decode_token_ids_to_actions(pred_flat[:n]).reshape(-1, 7)
                        gt_cont   = action_tokenizer.decode_token_ids_to_actions(gt_flat[:n]).reshape(-1, 7)
                    else:  # vq_*
                        pred_cont = action_tokenizer.decode_token_ids_to_actions(pred_flat[:n].reshape(-1, tpu))
                        gt_cont   = action_tokenizer.decode_token_ids_to_actions(gt_flat[:n].reshape(-1, tpu))
                    if isinstance(pred_cont, torch.Tensor):
                        pred_cont = pred_cont.cpu().numpy()
                    if isinstance(gt_cont, torch.Tensor):
                        gt_cont = gt_cont.cpu().numpy()
                    total_l1 += float(np.abs(pred_cont - gt_cont).mean())
                    l1_batches += 1
            except Exception:
                pass  # L1 not available (e.g., decode error)

            n_batches += 1
            if n_batches % 50 == 0:
                avg_l1_so_far = total_l1 / l1_batches if l1_batches > 0 else float("nan")
                print(f"  [{n_batches} batches] avg_l1={avg_l1_so_far:.4f}")

    avg_l1 = total_l1 / l1_batches if l1_batches > 0 else None
    print(f"\nL1 eval done ({n_batches} batches):")
    print(f"  Avg L1  : {avg_l1:.4f}" if avg_l1 is not None else "  Avg L1  : N/A")

    return {
        "avg_l1":    avg_l1,
        "n_batches": n_batches,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Stage 3 OpenVLA evaluation")
    p.add_argument("--condition", required=True,
                   help="Which fine-tuned condition to evaluate (e.g. bin, vq_vanilla, vq_verb, vq_verb01)")

    # NLL eval args
    p.add_argument("--eval_nll", action="store_true",
                   help="Run teacher-forcing NLL evaluation (requires checkpoint_dir)")
    p.add_argument("--checkpoint_dir", type=str, default="",
                   help="Path to merged fine-tuned OpenVLA checkpoint")
    p.add_argument("--data_root_dir", type=str, default=RLDS_DATA_ROOT)
    p.add_argument("--max_nll_batches", type=int, default=500,
                   help="Max val batches for NLL eval (0 = full val set)")

    # Verb probe args
    p.add_argument("--eval_verb_probe", action="store_true",
                   help="Run verb decodability probe (tokenizer round-trip)")
    p.add_argument("--vqvla_checkpoint_dir", type=str, default="",
                   help="Path to fine-tuned VQ-VLA checkpoint dir (for vq_* conditions)")
    p.add_argument("--verb_classifier_ckpt", type=str, default=DEFAULT_VERB_CLF,
                   help="Path to action-only verb classifier checkpoint")
    p.add_argument("--min_class_count", type=int, default=30,
                   help="Min train samples for a verb class to be included")

    # Output
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUT)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    results = {"condition": args.condition}

    if args.eval_nll:
        assert args.checkpoint_dir, "--checkpoint_dir required for --eval_nll"
        nll_metrics = run_nll_eval(
            condition=args.condition,
            checkpoint_dir=args.checkpoint_dir,
            vqvla_checkpoint_dir=args.vqvla_checkpoint_dir,
            data_root_dir=args.data_root_dir,
            max_batches=args.max_nll_batches,
            device=args.device,
        )
        results["nll_eval"] = nll_metrics

    if args.eval_verb_probe:
        verb_metrics = run_verb_probe(
            condition=args.condition,
            vqvla_checkpoint_dir=args.vqvla_checkpoint_dir,
            verb_classifier_ckpt=args.verb_classifier_ckpt,
            device=args.device,
            min_class_count=args.min_class_count,
        )
        results["verb_probe"] = verb_metrics

    # Save results (merge with existing to avoid overwriting prior eval modes)
    out_path = os.path.join(args.output_dir, f"eval_{args.condition}.json")
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            existing = json.load(f)
        existing.update(results)
        results = existing
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(json.dumps({k: v for k, v in results.items() if k != "verb_probe" or True},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
