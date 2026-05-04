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
import pandas as pd
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
    dataset = raw.get("dataset", "calvin")

    print(f"[ckpt] dataset={dataset}  modality={modality}  action_rep={action_rep}  "
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
    elif action_rep in ("latent", "vla_embed"):
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
        dataset=dataset,
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
    """Build the dataset described by the checkpoint metadata."""
    dataset_name = args.dataset or ckpt_meta.get("dataset", "calvin")

    action_rep = ckpt_meta["action_rep"]
    tokenizer_reps = {"vq_bet", "oat", "quest", "latent", "vla_embed"}

    if dataset_name == "bridge":
        if action_rep in tokenizer_reps:
            return _build_bridge_tokenizer_dataset(args, ckpt_meta, verb_to_id)
        return _build_bridge_standard_dataset(args, ckpt_meta, verb_to_id)

    if action_rep in tokenizer_reps:
        return _build_tokenizer_dataset(args, ckpt_meta, verb_to_id)
    return _build_standard_dataset(args, ckpt_meta, verb_to_id)


def _build_tokenizer_dataset(args, ckpt_meta, verb_to_id):
    """Build CalvinTokenizerDataset + frozen tokenizer for on-the-fly encoding."""
    from utils import load_calvin_to_dataframe
    from datasets.calvin_dataset import CalvinTokenizerDataset
    from verb_probe.load_tokenizer import (
        load_frozen_tokenizer, encode_tokenizer_batch, load_vla_embedding,
    )

    tok_type = ckpt_meta["tokenizer_type"]
    tok_ckpt = args.tokenizer_ckpt or ckpt_meta["tokenizer_ckpt"]

    if not tok_ckpt:
        raise ValueError(
            "Tokenizer checkpoint required for action_rep="
            f"{ckpt_meta['action_rep']}. Use --tokenizer_ckpt."
        )

    print(f"Loading frozen {tok_type} from {tok_ckpt}")
    tok_model = load_frozen_tokenizer(tok_type, tok_ckpt)

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
    mode = ckpt_meta["action_rep"] if ckpt_meta["action_rep"] in ("latent", "vla_embed") else "token_id"

    # Move tokenizer to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok_model = tok_model.to(device)
    vla_embed_info = None
    if mode == "vla_embed":
        policy_dir = args.policy_dir or getattr(args, "policy_dir", None)
        if not policy_dir:
            raise ValueError("--policy_dir is required for action_rep=vla_embed")
        vla_embed_info = load_vla_embedding(policy_dir, device=device)

    def batch_transform(batch, batch_device):
        actions, labels, seq_lengths = encode_tokenizer_batch(
            batch, batch_device, tok_model, tok_type, mode,
            vla_embed_info=vla_embed_info)
        bsz = labels.shape[0]
        frames = torch.zeros((bsz, 2, 3, 224, 224), device=batch_device)
        scene_vecs = torch.zeros((bsz, 48), device=batch_device)
        return frames, actions, scene_vecs, labels, seq_lengths

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


def _load_and_split_bridge(args):
    """Load bridge actions, filter to CSV episodes, and reproduce the val split."""
    from datasets.bridge_dataset import load_bridge_actions

    if not args.shard_dir or not args.bridge_csv:
        raise ValueError(
            "Bridge failure analysis requires --shard_dir and --bridge_csv."
        )

    all_actions, all_keys = load_bridge_actions(args.shard_dir)
    csv_df = pd.read_csv(args.bridge_csv)
    csv_key_set = set(csv_df["episode_key"])

    keep_idx = [i for i, key in enumerate(all_keys) if key in csv_key_set]
    all_actions = [all_actions[i] for i in keep_idx]
    all_keys = [all_keys[i] for i in keep_idx]
    print(f"Filtered to {len(all_actions)} Bridge episodes using {args.bridge_csv}")

    np.random.seed(42)
    perm = np.random.permutation(len(all_actions))
    n_val = max(1, int(len(all_actions) * args.val_fraction))
    val_idx = perm[:n_val]

    return csv_df, all_actions, all_keys, val_idx


def _build_bridge_standard_dataset(args, ckpt_meta, verb_to_id):
    """Build BridgeVerbDataset for native-action Bridge checkpoints."""
    from datasets.bridge_dataset import BridgeVerbDataset, load_bridge_verb_labels

    if ckpt_meta["modality"] != "action_only":
        raise ValueError("Bridge failure analysis currently supports action_only probes only.")
    if ckpt_meta["action_rep"] != "native":
        raise ValueError(
            f"Unsupported Bridge standard action_rep={ckpt_meta['action_rep']}; "
            "expected native."
        )

    csv_df, all_actions, all_keys, val_idx = _load_and_split_bridge(args)
    all_verb_ids, csv_verb_to_id = load_bridge_verb_labels(
        args.bridge_csv, all_keys, min_class_count=args.min_class_count)
    key_to_instruction = dict(zip(csv_df["episode_key"], csv_df["instruction"]))
    key_to_csv_verb = dict(zip(csv_df["episode_key"], csv_df["verb"]))
    id_to_verb = {idx: verb for verb, idx in verb_to_id.items()}

    rows = []
    actions_cache = {}
    for seg_idx, global_idx in enumerate(val_idx):
        vid = all_verb_ids[global_idx]
        if vid < 0:
            continue
        episode_key = all_keys[global_idx]
        csv_verb = key_to_csv_verb.get(episode_key)
        if csv_verb not in verb_to_id:
            continue
        rows.append({
            "seg_idx": seg_idx,
            "episode_key": episode_key,
            "instruction": key_to_instruction.get(episode_key, ""),
            "verb": csv_verb,
            "traj_len": int(len(all_actions[global_idx])),
        })
        actions_cache[f"actions_{seg_idx}"] = all_actions[global_idx]

    val_df = pd.DataFrame(rows)
    dropped = len([i for i in val_idx if all_verb_ids[i] >= 0]) - len(val_df)
    if dropped > 0:
        print(f"[dataset] Dropping {dropped} Bridge val samples with OOV verbs")
    if args.debug:
        val_df = val_df.head(min(args.debug, len(val_df))).reset_index(drop=True)
        keep_keys = {f"actions_{int(seg_idx)}" for seg_idx in val_df["seg_idx"].tolist()}
        actions_cache = {k: v for k, v in actions_cache.items() if k in keep_keys}
        print(f"[debug] Using {len(val_df)} Bridge native samples")
    val_ds = BridgeVerbDataset(
        val_df, actions_cache,
        max_seq_len=ckpt_meta["max_action_len"],
        verb_to_id=verb_to_id,
    )
    val_ds.df = val_df.reset_index(drop=True)
    return val_ds, None


def _build_bridge_tokenizer_dataset(args, ckpt_meta, verb_to_id):
    """Build BridgeTokenizerDataset + frozen tokenizer for on-the-fly encoding."""
    from datasets.bridge_dataset import BridgeTokenizerDataset, load_bridge_verb_labels
    from verb_probe.load_tokenizer import (
        load_frozen_tokenizer, encode_tokenizer_batch, load_vla_embedding,
    )

    csv_df, all_actions, all_keys, val_idx = _load_and_split_bridge(args)
    all_verb_ids, csv_verb_to_id = load_bridge_verb_labels(
        args.bridge_csv, all_keys, min_class_count=args.min_class_count)
    key_to_instruction = dict(zip(csv_df["episode_key"], csv_df["instruction"]))
    key_to_csv_verb = dict(zip(csv_df["episode_key"], csv_df["verb"]))
    id_to_verb = {idx: verb for verb, idx in verb_to_id.items()}

    val_actions = [all_actions[i] for i in val_idx]
    val_keys = [all_keys[i] for i in val_idx]
    val_verb_ids = []
    kept_actions = []
    kept_keys = []
    kept_episode_index = []
    dropped = 0
    for local_idx, global_idx in enumerate(val_idx):
        raw_vid = all_verb_ids[global_idx]
        if raw_vid < 0:
            dropped += 1
            continue
        episode_key = all_keys[global_idx]
        csv_verb = key_to_csv_verb.get(episode_key)
        if csv_verb not in verb_to_id:
            dropped += 1
            continue
        kept_actions.append(all_actions[global_idx])
        kept_keys.append(episode_key)
        kept_episode_index.append(local_idx)
        val_verb_ids.append(verb_to_id[csv_verb])

    val_actions = kept_actions
    val_keys = kept_keys
    if dropped > 0:
        print(f"[dataset] Dropping {dropped} Bridge val samples with OOV verbs")

    tok_type = ckpt_meta["tokenizer_type"]
    tok_ckpt = args.tokenizer_ckpt or ckpt_meta["tokenizer_ckpt"]
    if not tok_ckpt:
        raise ValueError(
            "Tokenizer checkpoint required for Bridge tokenizer failure analysis. "
            "Use --tokenizer_ckpt."
        )

    print(f"Loading frozen {tok_type} from {tok_ckpt}")
    tok_model = load_frozen_tokenizer(tok_type, tok_ckpt)
    ckpt = torch.load(tok_ckpt, map_location="cpu", weights_only=False)
    ckpt_args = ckpt["args"]
    if not isinstance(ckpt_args, dict):
        ckpt_args = vars(ckpt_args)

    val_ds = BridgeTokenizerDataset(
        val_actions,
        chunk_size=ckpt_args.get("chunk_size", 16),
        max_chunks=ckpt_args.get("max_chunks", 8),
        sampling=ckpt_args.get("sampling", "random"),
        verb_ids=val_verb_ids,
        verb_to_id=verb_to_id,
        instructions=[key_to_instruction.get(k, "") for k in val_keys],
    )

    rows = []
    for ep_idx in val_ds.ep_indices:
        episode_key = val_keys[ep_idx]
        vid = val_verb_ids[ep_idx]
        rows.append({
            "dataset_index": int(ep_idx),
            "episode_index": int(kept_episode_index[ep_idx]),
            "episode_key": episode_key,
            "instruction": key_to_instruction.get(episode_key, ""),
            "verb": id_to_verb[vid],
            "traj_len": int(len(val_actions[ep_idx])),
        })
    val_ds.df = pd.DataFrame(rows).reset_index(drop=True)
    if args.debug:
        n = min(args.debug, len(val_ds.ep_indices))
        val_ds.ep_indices = val_ds.ep_indices[:n]
        val_ds.df = val_ds.df.head(n).reset_index(drop=True)
        print(f"[debug] Using {n} Bridge tokenizer samples")

    mode = ckpt_meta["action_rep"] if ckpt_meta["action_rep"] in ("latent", "vla_embed") else "token_id"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok_model = tok_model.to(device)
    vla_embed_info = None
    if mode == "vla_embed":
        policy_dir = args.policy_dir or getattr(args, "policy_dir", None)
        if not policy_dir:
            raise ValueError("--policy_dir is required for action_rep=vla_embed")
        vla_embed_info = load_vla_embedding(policy_dir, device=device)

    def batch_transform(batch, batch_device):
        actions, labels, seq_lengths = encode_tokenizer_batch(
            batch, batch_device, tok_model, tok_type, mode,
            vla_embed_info=vla_embed_info)
        bsz = labels.shape[0]
        frames = torch.zeros((bsz, 2, 3, 224, 224), device=batch_device)
        scene_vecs = torch.zeros((bsz, 48), device=batch_device)
        return frames, actions, scene_vecs, labels, seq_lengths

    return val_ds, batch_transform


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


def _save_bridge_trajectory(actions, out_dir, gt_verb, pred_verb):
    """Save a Bridge action trajectory plot from an in-memory action array."""
    if actions is None or len(actions) == 0:
        return None
    actions = np.asarray(actions)
    T = actions.shape[0]
    dims = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
    colors = plt.cm.tab10(np.linspace(0, 1, min(actions.shape[1], 7)))

    fig, axes = plt.subplots(7, 1, figsize=(8, 7), sharex=True)
    for d in range(min(actions.shape[1], 7)):
        axes[d].plot(range(T), actions[:, d], color=colors[d], linewidth=1.5)
        axes[d].set_ylabel(dims[d], fontsize=8, rotation=0, labelpad=25)
        axes[d].axhline(0, color="grey", linewidth=0.5, linestyle="--")
        axes[d].grid(alpha=0.3)
    axes[-1].set_xlabel("Timestep")
    fig.suptitle(f"Action trajectory\nGT: {gt_verb}  ->  Predicted: {pred_verb}",
                 fontsize=9, fontweight="bold")
    plt.tight_layout()
    traj_path = os.path.join(out_dir, "trajectory.png")
    plt.savefig(traj_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return traj_path


def save_failure_sample(sample_dir, data_dir, dataset_name, dataset, df_row, gt_verb,
                        pred_verb, conf, df_idx, label_id, pred_id):
    """Save dataset-specific assets for one failure case. Returns metadata dict."""
    os.makedirs(sample_dir, exist_ok=True)
    instruction = str(df_row.get("instruction", ""))

    if dataset_name == "bridge":
        seg_idx = int(df_row.get("seg_idx", df_row.get("dataset_index", df_row.get("episode_index", df_idx))))
        actions = None
        if hasattr(dataset, "actions_cache"):
            actions = dataset.actions_cache.get(f"actions_{seg_idx}")
        elif hasattr(dataset, "actions"):
            ep_idx = int(df_row.get("dataset_index", seg_idx))
            actions = dataset.actions[ep_idx]
        traj_path = _save_bridge_trajectory(actions, sample_dir, gt_verb, pred_verb)
        meta = {
            "df_idx": df_idx,
            "episode_key": str(df_row.get("episode_key", "")),
            "dataset_index": int(df_row.get("dataset_index", seg_idx)),
            "episode_index": int(df_row.get("episode_index", seg_idx)),
            "seg_idx": seg_idx,
            "traj_len": int(df_row.get("traj_len", len(actions) if actions is not None else 0)),
            "instruction": instruction,
            "gt_verb": gt_verb,
            "pred_verb": pred_verb,
            "confidence": round(float(conf), 4),
            "trajectory": traj_path,
            "sample_dir": sample_dir,
        }
    else:
        start_idx = int(df_row["start_idx"])
        end_idx = int(df_row["end_idx"])
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
    dataset_name = args.dataset or ckpt_meta.get("dataset", "calvin")
    modality = ckpt_meta["modality"]
    action_rep = ckpt_meta["action_rep"]

    # --- Load dataset ---
    print(f"Loading validation data from {args.data_dir} ...")
    dataset, batch_transform_fn = _build_dataset(args, ckpt_meta, verb_to_id)
    df = dataset.df

    print(f"Dataset: {dataset_name}  |  {len(dataset)} samples, {len(verb_to_id)} verbs")

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
                sample_dir, args.data_dir, dataset_name, dataset, row,
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
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["calvin", "bridge"],
                        help="Override dataset type stored in checkpoint")
    parser.add_argument("--data_dir", type=str, default=VAL_DIR,
                        help="CALVIN validation directory")
    parser.add_argument("--tokenizer_ckpt", type=str, default=None,
                        help="Frozen tokenizer checkpoint (overrides ckpt path)")
    parser.add_argument("--policy_dir", type=str, default=None,
                        help="Policy directory for action_rep=vla_embed")
    parser.add_argument("--shard_dir", type=str, default=None,
                        help="Bridge action shards directory")
    parser.add_argument("--bridge_csv", type=str, default=None,
                        help="Bridge episode CSV used for filtering + verb labels")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Bridge validation fraction; keep at 0.1 to match training")
    parser.add_argument("--min_class_count", type=int, default=0,
                        help="Bridge min class count; should match probe training")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Top-K failures per (gt, pred) confusion pair")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--debug", type=int, default=0, metavar="N",
                        help="Use only N validation samples (0 = full)")
    args = parser.parse_args()
    main(args)
