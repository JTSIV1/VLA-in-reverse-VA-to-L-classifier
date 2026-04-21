"""Failure analysis for verb classifiers (probe models).

Loads a trained verb probe checkpoint (MotionVerbClassifier), runs inference
on the CALVIN validation set, and produces:
  - per_class_metrics.json   — precision/recall/F1 per verb
  - confusion_matrix.json    — raw NxN confusion counts
  - confusion_matrix.png     — annotated heatmap
  - top_failures.json        — ranked failures per confusion pair
  - samples/                 — per-failure frames + trajectory plots

Supports the same tokenizer variants as the VLA failure analysis:
  - native (raw 7-DoF actions)
  - vq_bet / quest / oat (discrete token IDs from frozen tokenizer)
  - latent (continuous encoder latents from frozen tokenizer)

Usage:
    # Native actions
    python analyze_failures.py \\
        --probe_path checkpoints/probe_native.pth \\
        --out_dir results/failure_analysis/native

    # VQ-BeT tokid probe
    python analyze_failures.py \\
        --probe_path checkpoints/calvin_sweep/tokenizers/vq_bet_5_16_4/probe_tokid.pth \\
        --tokenizer_ckpt checkpoints/calvin_sweep/tokenizers/vq_bet_5_16_4/full.pth \\
        --out_dir results/failure_analysis/vqbet_5_16_4_tokid

    # Smoke-test (32 samples)
    python analyze_failures.py \\
        --probe_path checkpoints/probe_native.pth \\
        --out_dir results/failure_analysis/smoke \\
        --top_k 2 --debug 32
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

try:
    from torchvision import transforms
    from PIL import Image as PILImage
except (ImportError, RuntimeError):
    transforms = None
    PILImage = None

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    VAL_DIR, DATA_DIR, ACTION_DIM, D_MODEL, NHEAD, NUM_LAYERS, DROPOUT_RATE,
    BATCH_SIZE, MAX_SEQ_LEN, NUM_WORKERS, PATCH_SIZE, IMAGE_SIZE,
    IMG_MEAN, IMG_STD, IMAGE_ENCODER, EPISODE_TEMPLATE, IMAGE_KEY, ACTION_KEY,
)
from verb_probe.models import MotionVerbClassifier, GoalVerbClassifier


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_checkpoint(args, device):
    """Load probe checkpoint and return (model, ckpt_meta, id_to_verb, verb_to_id)."""
    raw = torch.load(args.probe_path, map_location=device, weights_only=False)

    if not isinstance(raw, dict) or "state_dict" not in raw:
        raise ValueError(
            "Unsupported checkpoint format. Expected dict with 'state_dict' key "
            "(saved by verb_probe/train_verb_probe.py)."
        )

    state_dict = raw["state_dict"]
    num_verbs = raw["num_verbs"]
    verb_to_id = raw["verb_to_id"]
    id_to_verb = raw["id_to_verb"]
    d_model = raw.get("d_model", D_MODEL)
    nhead = raw.get("nhead", NHEAD)
    num_layers = raw.get("num_layers", NUM_LAYERS)
    action_dim = raw.get("action_dim", ACTION_DIM)
    modality = raw.get("modality", "action_only")
    action_rep = raw.get("action_rep", "native")
    image_encoder = raw.get("image_encoder", "scratch")
    delta_patches = raw.get("delta_patches", 0)
    num_frames = raw.get("num_frames", 2)
    max_action_len = raw.get("max_action_len", MAX_SEQ_LEN)
    tokenizer_type = raw.get("tokenizer_type", None)
    tokenizer_ckpt = raw.get("tokenizer_ckpt", None)
    action_vocab_size = raw.get("action_vocab_size", None)
    latent_dim = raw.get("latent_dim", None)

    print(f"[ckpt] modality={modality}  action_rep={action_rep}  "
          f"tokenizer_type={tokenizer_type}  num_verbs={num_verbs}")

    # Build model based on modality + action_rep
    if modality == "goal_only":
        img_size = 224 if image_encoder in ("r3m", "dinov2_s", "dinov2_b", "vc1", "dinov2") else IMAGE_SIZE[0]
        model = GoalVerbClassifier(
            num_verbs=num_verbs,
            image_encoder=image_encoder,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=DROPOUT_RATE,
            img_size=img_size,
            patch_size=PATCH_SIZE,
            num_frames=num_frames,
            delta_patches=delta_patches,
        )
    elif action_rep == "latent":
        model = MotionVerbClassifier(
            num_verbs=num_verbs,
            action_rep="latent",
            latent_dim=latent_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=DROPOUT_RATE,
        )
    elif action_rep in ("vq_bet", "oat", "quest", "fast") and action_vocab_size:
        model = MotionVerbClassifier(
            num_verbs=num_verbs,
            action_rep="token_id",
            action_vocab_size=action_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=DROPOUT_RATE,
        )
    else:
        model = MotionVerbClassifier(
            num_verbs=num_verbs,
            action_rep="native",
            action_dim=action_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=DROPOUT_RATE,
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    ckpt_meta = dict(
        modality=modality, action_rep=action_rep,
        image_encoder=image_encoder, max_action_len=max_action_len,
        num_frames=num_frames, delta_patches=delta_patches,
        tokenizer_type=tokenizer_type, tokenizer_ckpt=tokenizer_ckpt,
        action_vocab_size=action_vocab_size, latent_dim=latent_dim,
    )
    return model, ckpt_meta, id_to_verb, verb_to_id


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _build_dataset(args, ckpt_meta, verb_to_id):
    """Build CalvinTokenizerDataset or CalvinVerbProbeDataset from ckpt_meta."""
    from utils import load_calvin_to_dataframe

    action_rep = ckpt_meta["action_rep"]
    tokenizer_reps = {"vq_bet", "oat", "quest", "latent"}

    if action_rep in tokenizer_reps:
        return _build_tokenizer_dataset(args, ckpt_meta, verb_to_id)
    else:
        return _build_standard_dataset(args, ckpt_meta, verb_to_id)


def _build_tokenizer_dataset(args, ckpt_meta, verb_to_id):
    """Build CalvinTokenizerDataset + frozen tokenizer for on-the-fly encoding."""
    from utils import load_calvin_to_dataframe
    from datasets.calvin_dataset import CalvinTokenizerDataset
    from verb_probe.train_verb_probe import _load_frozen_tokenizer, _make_tokenizer_batch_transform

    tok_type = ckpt_meta["tokenizer_type"]
    tok_ckpt = args.tokenizer_ckpt or ckpt_meta["tokenizer_ckpt"]

    if not tok_ckpt:
        raise ValueError(
            "Tokenizer checkpoint required for action_rep="
            f"{ckpt_meta['action_rep']}. Use --tokenizer_ckpt."
        )

    # Build a namespace that _load_frozen_tokenizer expects
    tok_args = argparse.Namespace(
        tokenizer_type=tok_type,
        tokenizer_ckpt=tok_ckpt,
    )

    print(f"Loading frozen {tok_type} from {tok_ckpt}")
    tok_model = _load_frozen_tokenizer(tok_args)

    # Get chunk_size from tokenizer checkpoint
    ckpt = torch.load(tok_ckpt, map_location="cpu", weights_only=False)
    ckpt_args = ckpt["args"]
    if not isinstance(ckpt_args, dict):
        ckpt_args = vars(ckpt_args)
    chunk_size = ckpt_args.get("chunk_size", 16)
    sampling = ckpt_args.get("sampling", "random")
    max_chunks = ckpt_args.get("max_chunks", 8)

    # Load validation DataFrame
    val_df = load_calvin_to_dataframe(args.data_dir)
    if args.debug:
        val_df = val_df.head(min(args.debug, len(val_df))).copy()
        print(f"[debug] Using {len(val_df)} samples")

    val_ds = CalvinTokenizerDataset(
        args.data_dir, val_df, chunk_size=chunk_size,
        max_chunks=max_chunks, sampling=sampling,
        verb_to_id=verb_to_id, cache_actions=True,
    )

    # Drop samples with unseen verbs
    verb_col = val_ds._verb_col
    valid_mask = val_df[verb_col].isin(verb_to_id.keys())
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        print(f"[dataset] Dropping {n_dropped} samples with OOV verbs")
        val_ds.df = val_df[valid_mask].reset_index(drop=True)

    # Determine mode
    mode = "latent" if ckpt_meta["action_rep"] == "latent" else "token_id"

    # Move tokenizer to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok_model = tok_model.to(device)

    batch_transform = _make_tokenizer_batch_transform(tok_model, tok_type, mode)

    return val_ds, batch_transform


def _build_standard_dataset(args, ckpt_meta, verb_to_id):
    """Build CalvinVerbProbeDataset for native/fast/goal_only modes."""
    from utils import load_calvin_to_dataframe
    from datasets.calvin_dataset import CalvinVerbProbeDataset

    modality = ckpt_meta["modality"]
    image_encoder = ckpt_meta["image_encoder"]
    img_size = 224 if image_encoder in ("r3m", "dinov2_s", "dinov2_b", "vc1", "dinov2") else IMAGE_SIZE[0]

    transform = None
    if modality == "goal_only" and transforms is not None:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ])

    internal_modality = "action_only" if modality == "action_only" else "vision_only"

    val_df = load_calvin_to_dataframe(args.data_dir)
    if args.debug:
        val_df = val_df.head(min(args.debug, len(val_df))).copy()
        print(f"[debug] Using {len(val_df)} samples")

    val_ds = CalvinVerbProbeDataset(
        args.data_dir, val_df, modality=internal_modality,
        verb_to_id=verb_to_id,
        max_seq_len=ckpt_meta["max_action_len"],
        num_frames=ckpt_meta["num_frames"],
        delta_patches=ckpt_meta["delta_patches"],
        image_encoder=image_encoder,
        transform=transform, img_size=img_size, cache_actions=True,
    )

    # Drop val samples with unseen verbs
    verb_col = val_ds._verb_col
    valid_mask = val_df[verb_col].isin(verb_to_id.keys())
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        print(f"[dataset] Dropping {n_dropped} samples with OOV verbs")
        val_ds.df = val_df[valid_mask].reset_index(drop=True)

    return val_ds, None  # no batch transform needed


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, dataset, device, batch_size, num_workers,
                  batch_transform_fn=None):
    """Run inference over entire dataset, collecting per-sample results."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    all_labels, all_preds, all_confs, all_df_indices = [], [], [], []
    sample_idx = 0

    with torch.no_grad():
        for batch in loader:
            if batch_transform_fn is not None:
                frames, actions, scene_vecs, labels, seq_lengths = \
                    batch_transform_fn(batch, device)
            else:
                frames, actions, scene_vecs, labels, seq_lengths = batch
                frames = frames.to(device)
                actions = actions.to(device)
                scene_vecs = scene_vecs.to(device)
                seq_lengths = seq_lengths.to(device)
                labels = labels.to(device)

            logits = model(frames, actions, seq_lengths=seq_lengths,
                           scene_vec=scene_vecs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            confs = probs[range(len(preds)), preds]

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_confs.extend(confs.cpu().tolist())
            all_df_indices.extend(range(sample_idx, sample_idx + len(labels)))
            sample_idx += len(labels)

    return all_labels, all_preds, all_confs, all_df_indices


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _save_frames(data_dir, start_idx, end_idx, out_dir):
    """Save start and end RGB frames as PNGs."""
    paths = {}
    for name, frame_idx in [("frame_start", start_idx), ("frame_end", end_idx)]:
        ep_path = os.path.join(data_dir, EPISODE_TEMPLATE.format(frame_idx))
        try:
            ep = np.load(ep_path, mmap_mode="r")
            img_arr = np.array(ep[IMAGE_KEY])
            img = PILImage.fromarray(img_arr)
            png_path = os.path.join(out_dir, f"{name}.png")
            img.save(png_path)
            paths[name] = png_path
        except Exception as e:
            print(f"  [warn] Could not save {name}: {e}")
            paths[name] = None
    return paths


def _save_trajectory(data_dir, start_idx, end_idx, out_dir, gt_verb, pred_verb):
    """Load action steps and save a 7-dim time-series plot."""
    actions = []
    for i in range(start_idx, end_idx + 1):
        ep_path = os.path.join(data_dir, EPISODE_TEMPLATE.format(i))
        try:
            ep = np.load(ep_path, mmap_mode="r")
            actions.append(np.array(ep[ACTION_KEY]))
        except Exception:
            break
    if not actions:
        return None
    actions = np.array(actions)
    T = actions.shape[0]
    dims = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
    colors = plt.cm.tab10(np.linspace(0, 1, 7))

    fig, axes = plt.subplots(7, 1, figsize=(8, 7), sharex=True)
    for d in range(7):
        axes[d].plot(range(T), actions[:, d], color=colors[d], linewidth=1.5)
        axes[d].set_ylabel(dims[d], fontsize=8, rotation=0, labelpad=25)
        axes[d].axhline(0, color="grey", linewidth=0.5, linestyle="--")
        axes[d].grid(alpha=0.3)
    axes[-1].set_xlabel("Timestep")
    fig.suptitle(f"Action trajectory\nGT: {gt_verb}  →  Predicted: {pred_verb}",
                 fontsize=9, fontweight="bold")
    plt.tight_layout()
    traj_path = os.path.join(out_dir, "trajectory.png")
    plt.savefig(traj_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return traj_path


def save_failure_sample(sample_dir, data_dir, df_row, gt_verb, pred_verb, conf,
                        df_idx, label_id, pred_id):
    """Save frames + trajectory for one failure case. Returns metadata dict."""
    os.makedirs(sample_dir, exist_ok=True)
    start_idx = int(df_row["start_idx"])
    end_idx = int(df_row["end_idx"])
    instruction = str(df_row["instruction"])

    frame_paths = _save_frames(data_dir, start_idx, end_idx, sample_dir)
    traj_path = _save_trajectory(data_dir, start_idx, end_idx, sample_dir,
                                 gt_verb, pred_verb)
    meta = {
        "df_idx": df_idx,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "instruction": instruction,
        "gt_verb": gt_verb,
        "pred_verb": pred_verb,
        "confidence": round(float(conf), 4),
        "frame_start": frame_paths.get("frame_start"),
        "frame_end": frame_paths.get("frame_end"),
        "trajectory": traj_path,
        "sample_dir": sample_dir,
    }
    with open(os.path.join(sample_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)
    samples_dir = os.path.join(args.out_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # --- Load model ---
    model, ckpt_meta, id_to_verb, verb_to_id = _load_checkpoint(args, device)
    modality = ckpt_meta["modality"]
    action_rep = ckpt_meta["action_rep"]

    # --- Load dataset ---
    print(f"Loading validation data from {args.data_dir} ...")
    dataset, batch_transform_fn = _build_dataset(args, ckpt_meta, verb_to_id)
    df = dataset.df

    print(f"Dataset: {len(dataset)} samples, {len(verb_to_id)} verbs")

    # --- Inference ---
    print("Running inference ...")
    all_labels, all_preds, all_confs, all_df_indices = run_inference(
        model, dataset, device, args.batch_size, args.num_workers,
        batch_transform_fn=batch_transform_fn)

    accuracy = 100.0 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
    print(f"Overall accuracy: {accuracy:.2f}%  ({len(all_preds)} samples)")

    # --- Per-class metrics ---
    present_labels = sorted(set(all_labels + all_preds))
    target_names = [id_to_verb[i] for i in present_labels]

    report_dict = classification_report(
        all_labels, all_preds, labels=present_labels,
        target_names=target_names, digits=4, output_dict=True)
    per_class_path = os.path.join(args.out_dir, "per_class_metrics.json")
    with open(per_class_path, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "modality": modality,
            "action_rep": action_rep,
            "report": report_dict,
        }, f, indent=2)
    print(f"Per-class metrics → {per_class_path}")

    # --- Confusion matrix ---
    cm = confusion_matrix(all_labels, all_preds, labels=present_labels)
    cm_dict = {"labels": target_names, "matrix": cm.tolist()}
    cm_json_path = os.path.join(args.out_dir, "confusion_matrix.json")
    with open(cm_json_path, "w") as f:
        json.dump(cm_dict, f, indent=2)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(target_names)))
    ax.set_yticks(range(len(target_names)))
    ax.set_xticklabels(target_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(target_names, fontsize=8)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=7)
    ax.set_xlabel("Predicted verb")
    ax.set_ylabel("True verb")
    ax.set_title(
        f"Confusion Matrix — {modality} / {action_rep}\n"
        f"Accuracy: {accuracy:.1f}%"
    )
    plt.tight_layout()
    cm_png_path = os.path.join(args.out_dir, "confusion_matrix.png")
    plt.savefig(cm_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix → {cm_png_path}")

    # --- Find top-K failures per (gt, pred) confusion pair ---
    confusion_buckets = defaultdict(list)
    for i, (label, pred, conf, df_idx) in enumerate(
            zip(all_labels, all_preds, all_confs, all_df_indices)):
        if label != pred:
            gt_verb = id_to_verb[label]
            pred_verb = id_to_verb[pred]
            confusion_buckets[(gt_verb, pred_verb)].append(
                (conf, df_idx, label, pred))

    # Sort each bucket by confidence descending (most egregious first)
    for key in confusion_buckets:
        confusion_buckets[key].sort(key=lambda x: -x[0])

    # Sort pairs by total count (most frequent confusions first)
    sorted_pairs = sorted(confusion_buckets.items(), key=lambda x: -len(x[1]))

    all_failures = []
    print(f"\nTop confusion pairs (total pairs: {len(sorted_pairs)}):")
    for rank, ((gt_verb, pred_verb), samples) in enumerate(sorted_pairs):
        count = len(samples)
        print(f"  [{rank+1:2d}] {gt_verb:15s} → {pred_verb:15s}  ({count} samples)")

        for k, (conf, df_idx, label_id, pred_id) in enumerate(samples[:args.top_k]):
            row = df.iloc[df_idx]
            safe_gt = gt_verb.replace(" ", "_")
            safe_pred = pred_verb.replace(" ", "_")
            sample_dir = os.path.join(
                samples_dir,
                f"pair{rank+1:02d}_k{k+1}_gt{safe_gt}_pred{safe_pred}"
                f"_conf{int(conf*100):02d}")
            meta = save_failure_sample(
                sample_dir, args.data_dir, row,
                gt_verb, pred_verb, conf, int(df_idx), label_id, pred_id)
            meta["pair_rank"] = rank + 1
            meta["pair_count"] = count
            all_failures.append(meta)

    # Save master failures list
    failures_path = os.path.join(args.out_dir, "top_failures.json")
    with open(failures_path, "w") as f:
        json.dump(all_failures, f, indent=2)
    print(f"\nTop failures → {failures_path}  ({len(all_failures)} cases saved)")

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"ANALYSIS COMPLETE  [{modality} / {action_rep}]")
    print(f"  Accuracy:   {accuracy:.2f}%")
    print(f"  Failures:   {sum(p != l for p, l in zip(all_preds, all_labels))}")
    print(f"  Output dir: {args.out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Failure analysis for verb probes")
    parser.add_argument("--probe_path", type=str, required=True,
                        help="Path to verb probe checkpoint (.pth)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to write all outputs")
    parser.add_argument("--data_dir", type=str, default=VAL_DIR,
                        help="CALVIN validation directory")
    parser.add_argument("--tokenizer_ckpt", type=str, default=None,
                        help="Frozen tokenizer checkpoint (overrides ckpt path)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Top-K failures per (gt, pred) confusion pair")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--debug", type=int, default=0, metavar="N",
                        help="Use only N validation samples (0 = full)")
    args = parser.parse_args()
    main(args)
