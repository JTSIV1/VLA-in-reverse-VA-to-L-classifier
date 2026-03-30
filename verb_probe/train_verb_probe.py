"""Verb classification probe for CALVIN / BridgeV2 datasets.

Trains MotionVerbClassifier (action_only) or GoalVerbClassifier (goal_only)
to classify verbs from action trajectories or vision.

Action representations:
  - native:      raw 7-DoF actions
  - fast:        FAST (DCT+BPE) discrete token IDs
  - vq_bet/oat/quest: discrete codebook IDs from frozen tokenizer
  - latent:      continuous encoder latents from frozen tokenizer
  - vla_embed:   VLA's learned token embeddings (codes → LLM embedding lookup)

Usage:
    # Action-only with native actions
    python verb_probe/train_verb_probe.py --modality action_only

    # Action-only with VQ-BeT discrete codes
    python verb_probe/train_verb_probe.py --modality action_only \
        --action_rep vq_bet --tokenizer_ckpt checkpoints/vq_bet/full.pth

    # Action-only with continuous latents from frozen tokenizer
    python verb_probe/train_verb_probe.py --modality action_only \
        --action_rep latent --tokenizer_type oat \
        --tokenizer_ckpt checkpoints/oat/full.pth

    # Bridge dataset with tokenizer probe
    python verb_probe/train_verb_probe.py --dataset bridge \
        --shard_dir /path/to/bridge_actions --bridge_csv data/bridge_episodes_filtered.csv \
        --action_rep latent --tokenizer_type quest \
        --tokenizer_ckpt checkpoints/bridge_sweep/tokenizers/quest_16_855_4/full.pth

    # Goal-only (uses DINOv2-S image patches)
    python verb_probe/train_verb_probe.py --modality goal_only \
        --image_encoder dinov2_s --delta_patches 16
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
try:
    from torchvision import transforms
except (ImportError, RuntimeError):
    transforms = None

from config import (
    DATA_DIR, VAL_DIR, ACTION_DIM,
    D_MODEL, NHEAD, NUM_LAYERS, DROPOUT_RATE,
    PATCH_SIZE, IMAGE_SIZE, IMG_MEAN, IMG_STD,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, MAX_SEQ_LEN, NUM_WORKERS,
    WARMUP_EPOCHS, GRAD_CLIP_NORM, IMAGE_ENCODER,
    FAST_TOKENIZER_PATH,
)
from verb_probe.models import MotionVerbClassifier, GoalVerbClassifier
from verb_probe.train_utils import (
    build_criterion, build_optimizer_scheduler, run_training_loop,
)
from datasets.calvin_dataset import build_calvin_tokenizer_data
from verb_probe.load_tokenizer import (
    load_frozen_tokenizer, get_tokenizer_chunk_params, get_vocab_size, load_fast_tokenizer
)

_TOKENIZER_REPS = {"vq_bet", "oat", "quest", "latent", "vla_embed"}


# ======================================================================
# Bridge data helpers
# ======================================================================

def _load_and_split_bridge(args):
    """Load bridge actions, filter to CSV episodes, split train/val.

    Uses the same seed=42 and val_fraction=0.1 as tokenizer training
    so the verb probe evaluates on the same val set.

    Returns: (train_actions, val_actions, train_keys, val_keys, perm, n_val)
    """
    import pandas as pd
    from datasets.bridge_dataset import load_bridge_actions

    all_actions, all_keys = load_bridge_actions(args.shard_dir)
    csv_df = pd.read_csv(args.bridge_csv)
    csv_key_set = set(csv_df["episode_key"])

    n_total = len(all_actions)
    keep_idx = [i for i, k in enumerate(all_keys) if k in csv_key_set]
    all_actions = [all_actions[i] for i in keep_idx]
    all_keys = [all_keys[i] for i in keep_idx]
    print(f"Filtered to {len(all_actions)}/{n_total} episodes "
          f"using {args.bridge_csv}")

    val_fraction = getattr(args, 'val_fraction', 0.1)
    np.random.seed(42)
    perm = np.random.permutation(len(all_actions))
    n_val = max(1, int(len(all_actions) * val_fraction))

    train_actions = [all_actions[i] for i in perm[n_val:]]
    val_actions = [all_actions[i] for i in perm[:n_val]]
    train_keys = [all_keys[i] for i in perm[n_val:]]
    val_keys = [all_keys[i] for i in perm[:n_val]]
    print(f"Train: {len(train_actions)} episodes, Val: {len(val_actions)} episodes")

    return (all_actions, all_keys, train_actions, val_actions,
            train_keys, val_keys, perm, n_val)


def _build_bridge_tokenizer_data(args, chunk_params):
    """Build BridgeTokenizerDataset for verb probe (tokenizer reps)."""
    from datasets.bridge_dataset import (
        BridgeTokenizerDataset, load_bridge_verb_labels,
    )
    (all_actions, all_keys, train_actions, val_actions,
     _, _, perm, n_val) = _load_and_split_bridge(args)

    all_verb_ids, verb_to_id = load_bridge_verb_labels(
        args.bridge_csv, all_keys, min_class_count=args.min_class_count)
    train_verb_ids = [all_verb_ids[i] for i in perm[n_val:]]
    val_verb_ids = [all_verb_ids[i] for i in perm[:n_val]]

    train_ds = BridgeTokenizerDataset(
        train_actions, verb_ids=train_verb_ids, verb_to_id=verb_to_id,
        **chunk_params)
    val_ds = BridgeTokenizerDataset(
        val_actions, verb_ids=val_verb_ids, verb_to_id=verb_to_id,
        **chunk_params)

    id_to_verb = {v: k for k, v in verb_to_id.items()}
    num_verbs = len(verb_to_id)

    # verb_counts from train split
    from collections import Counter
    cnt = Counter(v for v in train_verb_ids if v >= 0)
    verb_counts = {id_to_verb[vid]: c for vid, c in cnt.items()}

    return train_ds, val_ds, num_verbs, id_to_verb, verb_to_id, verb_counts


def _build_bridge_native_data(args):
    """Build BridgeVerbDataset for native action probe."""
    import pandas as pd
    from datasets.bridge_dataset import (
        BridgeVerbDataset, load_bridge_verb_labels,
    )
    (all_actions, all_keys, train_actions, val_actions,
     train_keys, val_keys, perm, n_val) = _load_and_split_bridge(args)

    all_verb_ids, verb_to_id = load_bridge_verb_labels(
        args.bridge_csv, all_keys, min_class_count=args.min_class_count)
    train_verb_ids = [all_verb_ids[i] for i in perm[n_val:]]
    val_verb_ids = [all_verb_ids[i] for i in perm[:n_val]]

    id_to_verb = {v: k for k, v in verb_to_id.items()}

    # Build DataFrames and action caches for BridgeVerbDataset
    def _make_df_and_cache(actions, verb_ids):
        rows = []
        cache = {}
        for i, (act, vid) in enumerate(zip(actions, verb_ids)):
            if vid < 0:
                continue
            rows.append({"seg_idx": i, "verb": id_to_verb[vid]})
            cache[f"actions_{i}"] = act
        return pd.DataFrame(rows), cache

    train_df, train_cache = _make_df_and_cache(train_actions, train_verb_ids)
    val_df, val_cache = _make_df_and_cache(val_actions, val_verb_ids)
    print(f"Native probe: {len(train_df)} train / {len(val_df)} val episodes")

    train_ds = BridgeVerbDataset(train_df, train_cache,
                                 max_seq_len=args.max_seq_len,
                                 verb_to_id=verb_to_id)
    val_ds = BridgeVerbDataset(val_df, val_cache,
                               max_seq_len=args.max_seq_len,
                               verb_to_id=verb_to_id)

    num_verbs = len(verb_to_id)
    from collections import Counter
    cnt = Counter(train_df["verb"])
    verb_counts = dict(cnt)

    return train_ds, val_ds, num_verbs, id_to_verb, verb_to_id, verb_counts


# ======================================================================
# Dataset construction
# ======================================================================

def build_datasets(args):
    """Build train/val datasets.

    Returns: (train_ds, val_ds, num_verbs, id_to_verb, verb_to_id, verb_counts)
    """
    if args.action_rep in _TOKENIZER_REPS:
        return _build_tokenizer_datasets(args)
    else:
        return _build_standard_datasets(args)


def _build_tokenizer_datasets(args):
    """Build tokenizer dataset + frozen tokenizer for on-the-fly encoding."""

    # Load frozen tokenizer and read chunking params from its checkpoint
    print(f"Loading frozen {args.tokenizer_type} from {args.tokenizer_ckpt}")
    tok_model = load_frozen_tokenizer(args.tokenizer_type, args.tokenizer_ckpt)
    chunk_params = get_tokenizer_chunk_params(args.tokenizer_ckpt)
    print(f"  chunk_size={chunk_params['chunk_size']}, "
          f"sampling={chunk_params['sampling']}, "
          f"max_chunks={chunk_params['max_chunks']}")

    # Build datasets
    if args.dataset == "bridge":
        train_ds, val_ds, num_verbs, id_to_verb, verb_to_id, verb_counts = \
            _build_bridge_tokenizer_data(args, chunk_params)
    else:
        train_ds, val_ds, num_verbs, id_to_verb, verb_to_id, verb_counts = \
            build_calvin_tokenizer_data(
                args.data_dir, args.val_dir,
                min_class_count=args.min_class_count, cache_actions=True,
                **chunk_params)

    # Determine mode: latent (continuous), vla_embed (VLA embeddings), or token_id (discrete)
    if args.action_rep == "latent":
        mode = "latent"
    elif args.action_rep == "vla_embed":
        mode = "vla_embed"
    else:
        mode = "token_id"

    # Probe shapes from one sample
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_model = tok_model.to(device)
    from tokenization.train_utils import extract_episode_batch
    sample_batch = next(iter(DataLoader(train_ds, batch_size=1, shuffle=False)))
    with torch.no_grad():
        result = extract_episode_batch(tok_model, sample_batch, device, args.tokenizer_type)

    if mode == "vla_embed":
        from verb_probe.load_tokenizer import load_vla_embedding
        vla_info = load_vla_embedding(args.policy_dir, device=device)
        args._vla_embed_info = vla_info
        args._latent_dim = vla_info['embed_dim']
        print(f"  vla_embed_dim={args._latent_dim}")
    elif mode == "latent":
        args._latent_dim = result['latents'].shape[-1]
        print(f"  latent_dim={args._latent_dim}")
    else:
        args._action_vocab_size = get_vocab_size(tok_model, args.tokenizer_type)
        print(f"  vocab_size={args._action_vocab_size}")

    # Store frozen tokenizer for on-the-fly encoding in training loop
    args._tok_model = tok_model
    args._tok_mode = mode

    return train_ds, val_ds, num_verbs, id_to_verb, verb_to_id, verb_counts


def _build_standard_datasets(args):
    """Build standard datasets for native/fast/goal_only modes."""

    if args.dataset == "bridge":
        return _build_bridge_native_data(args)

    from datasets.calvin_dataset import build_calvin_verb_probe_data

    # Vision transform (goal_only)
    img_size = 224 if args.image_encoder in ("r3m", "dinov2_s", "dinov2_b", "vc1", "dinov2") else IMAGE_SIZE[0]
    if args.modality == "goal_only" and transforms is not None:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
        ])
    else:
        transform = None

    # FAST tokenizer (action_only with discrete tokens)
    tok = None
    if args.modality == "action_only":
        tok, _ = load_fast_tokenizer(args)

    internal_modality = "action_only" if args.modality == "action_only" else "vision_only"

    train_ds, val_ds, num_verbs, id_to_verb, verb_to_id, verb_counts = \
        build_calvin_verb_probe_data(
            args.data_dir, args.val_dir,
            min_class_count=args.min_class_count,
            cache_actions=True,
            modality=internal_modality,
            action_tokenizer=tok,
            max_seq_len=args.max_seq_len, num_frames=args.num_frames,
            delta_patches=args.delta_patches, image_encoder=args.image_encoder,
            transform=transform, img_size=img_size)

    if args.debug:
        n = min(args.debug, len(train_ds))
        train_ds.df = train_ds.df.head(n).copy()
        val_ds.df = val_ds.df.head(n).copy()
        args.epochs = min(args.epochs, 2)
        print(f"[DEBUG] {n} train / {len(val_ds)} val, {args.epochs} epochs")

    return train_ds, val_ds, num_verbs, id_to_verb, verb_to_id, verb_counts


# ======================================================================
# Main
# ======================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Modality: {args.modality} | Action rep: {args.action_rep}")

    # Build datasets
    train_ds, val_ds, num_verbs, id_to_verb, verb_to_id, verb_counts = \
        build_datasets(args)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    # Build model
    img_size = 224 if args.image_encoder in ("r3m", "dinov2_s", "dinov2_b", "vc1", "dinov2") else IMAGE_SIZE[0]

    if args.modality == "goal_only":
        model = GoalVerbClassifier(
            num_verbs=num_verbs,
            image_encoder=args.image_encoder,
            d_model=args.d_model,
            num_layers=args.num_layers,
            dropout=DROPOUT_RATE,
            img_size=img_size,
            patch_size=PATCH_SIZE,
            num_frames=args.num_frames,
            delta_patches=args.delta_patches,
        ).to(device)

        if hasattr(train_ds, 'num_patches') and hasattr(model, 'num_patches'):
            train_ds.num_patches = model.num_patches
            val_ds.num_patches = model.num_patches

    elif args.action_rep in ("latent", "vla_embed"):
        model = MotionVerbClassifier(
            num_verbs=num_verbs,
            action_rep="latent",
            latent_dim=args._latent_dim,
            d_model=args.d_model,
            num_layers=args.num_layers,
            dropout=DROPOUT_RATE,
        ).to(device)

    elif args.action_rep == "native":
        model = MotionVerbClassifier(
            num_verbs=num_verbs,
            action_rep="native",
            action_dim=ACTION_DIM,
            d_model=args.d_model,
            num_layers=args.num_layers,
            dropout=DROPOUT_RATE,
        ).to(device)

    else:
        # Discrete token IDs (fast, vq_bet, oat, quest)
        if hasattr(args, '_action_vocab_size'):
            action_vocab_size = args._action_vocab_size
        else:
            _, action_vocab_size = load_fast_tokenizer(args)
        model = MotionVerbClassifier(
            num_verbs=num_verbs,
            action_rep="token_id",
            action_vocab_size=action_vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            dropout=DROPOUT_RATE,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} trainable params")

    # Loss, optimizer, scheduler
    criterion = build_criterion(
        verb_counts, verb_to_id, num_verbs, device,
        weighted=args.weighted_loss, label_smoothing=args.label_smoothing)

    total_steps = len(train_loader) * args.epochs
    optimizer, scheduler = build_optimizer_scheduler(
        model, args.lr, args.weight_decay, total_steps,
        args.warmup_epochs, args.epochs)

    args.grad_clip = GRAD_CLIP_NORM

    # Checkpoint metadata
    def ckpt_fn(model, args, best_val_acc, best_epoch):
        meta = {
            "num_verbs": num_verbs,
            "verb_to_id": verb_to_id,
            "id_to_verb": id_to_verb,
            "d_model": args.d_model,
            "action_dim": ACTION_DIM,
            "nhead": NHEAD,
            "num_layers": args.num_layers,
            "max_action_len": args.max_seq_len,
            "modality": args.modality,
            "action_rep": args.action_rep,
            "image_encoder": args.image_encoder,
            "delta_patches": args.delta_patches,
            "min_class_count": args.min_class_count,
            "dataset": args.dataset,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
        }
        if args.action_rep in _TOKENIZER_REPS:
            meta["tokenizer_type"] = args.tokenizer_type
            meta["tokenizer_ckpt"] = args.tokenizer_ckpt
        if args.action_rep in ("latent", "vla_embed"):
            meta["latent_dim"] = args._latent_dim
        if hasattr(args, '_action_vocab_size'):
            meta["action_vocab_size"] = args._action_vocab_size
        return meta

    pass_seq = (args.modality != "goal_only")

    print(f"\nTraining: {num_verbs} verbs, {args.epochs} epochs, "
          f"d_model={args.d_model}")

    run_training_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, args, num_verbs, id_to_verb,
        checkpoint_metadata_fn=ckpt_fn,
        track_per_class_loss=True,
        pass_seq_lengths=pass_seq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verb classification probe")

    # Dataset
    parser.add_argument("--dataset", type=str, default="calvin",
                        choices=["calvin", "bridge"])
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--val_dir", type=str, default=VAL_DIR)
    parser.add_argument("--shard_dir", type=str, default=None,
                        help="Bridge action shards directory")
    parser.add_argument("--bridge_csv", type=str, default=None,
                        help="Bridge episode CSV (for filtering + verb labels)")

    # Modality
    parser.add_argument("--modality", type=str, default="action_only",
                        choices=["action_only", "goal_only"])

    # Action representation
    parser.add_argument("--action_rep", type=str, default="native",
                        choices=["native", "fast", "vq_bet", "quest",
                                 "oat", "latent", "vla_embed"])
    parser.add_argument("--fast_tokenizer_path", type=str,
                        default=FAST_TOKENIZER_PATH)

    # Tokenizer checkpoint (for vq_bet/oat/quest/latent)
    parser.add_argument("--tokenizer_type", type=str, default=None,
                        choices=["vq_bet", "oat", "quest"],
                        help="Tokenizer type (required for vq_bet/oat/quest/latent)")
    parser.add_argument("--tokenizer_ckpt", type=str, default=None,
                        help="Frozen tokenizer checkpoint (required for vq_bet/oat/quest/latent)")
    parser.add_argument("--policy_dir", type=str, default=None,
                        help="Policy run directory for vla_embed mode")

    # Vision (goal_only)
    parser.add_argument("--image_encoder", type=str, default=IMAGE_ENCODER,
                        choices=["dinov2_s", "dinov2_b", "vc1"])
    parser.add_argument("--num_frames", type=int, default=2)
    parser.add_argument("--delta_patches", type=int, default=0)

    # Model
    parser.add_argument("--d_model", type=int, default=D_MODEL)
    parser.add_argument("--num_layers", type=int, default=NUM_LAYERS)

    # Training
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--weighted_loss", action="store_true")
    parser.add_argument("--min_class_count", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0)

    # Output
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--debug", type=int, default=0, metavar="N")

    args = parser.parse_args()

    # Validate tokenizer args
    if args.action_rep in _TOKENIZER_REPS:
        # For vq_bet/oat/quest, default tokenizer_type from action_rep
        if args.tokenizer_type is None and args.action_rep not in ("latent", "vla_embed"):
            args.tokenizer_type = args.action_rep
        if not args.tokenizer_type or not args.tokenizer_ckpt:
            parser.error(
                f"--action_rep {args.action_rep} requires --tokenizer_type and --tokenizer_ckpt")
        if args.action_rep == "vla_embed" and not args.policy_dir:
            parser.error("--action_rep vla_embed requires --policy_dir")

    # Validate bridge args
    if args.dataset == "bridge":
        if not args.shard_dir or not args.bridge_csv:
            parser.error("--dataset bridge requires --shard_dir and --bridge_csv")

    main(args)
