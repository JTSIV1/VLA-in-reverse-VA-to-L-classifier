"""Unified verb classification probe for CALVIN, Bridge, and DROID datasets.

Trains ActionToVerbTransformer to classify verbs from:
  - action trajectories (raw or tokenized)
  - goal signal (scene_obs for CALVIN, image frames for Bridge/DROID)
  - both (fusion)

Usage:
    # CALVIN action-only with native actions
    python verb_probe/train_verb_probe.py --dataset calvin --modality action_only

    # CALVIN action-only with VQ-BeT tokens
    python verb_probe/train_verb_probe.py --dataset calvin --modality action_only \
        --action_rep vq_bet --tokenizer_ckpt checkpoints/vq_bet/best.pth

    # CALVIN goal-only (uses scene_obs)
    python verb_probe/train_verb_probe.py --dataset calvin --modality goal_only

    # Bridge goal-only (uses image frames)
    python verb_probe/train_verb_probe.py --dataset bridge --modality goal_only \
        --image_encoder dinov2_s

    # CALVIN fusion (action + scene_obs)
    python verb_probe/train_verb_probe.py --dataset calvin --modality fusion
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import pandas as pd
import torch
try:
    from torchvision import transforms
except (ImportError, RuntimeError):
    transforms = None
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from config import (
    DATA_DIR, VAL_DIR,
    SCENE_OBS_DIM, SCENE_REP_DIM, ACTION_DIM,
    D_MODEL, NHEAD, NUM_LAYERS, CROSS_LAYERS, DROPOUT_RATE,
    PATCH_SIZE, IMAGE_SIZE, IMG_MEAN, IMG_STD,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, MAX_SEQ_LEN, NUM_WORKERS,
    WARMUP_EPOCHS, GRAD_CLIP_NORM, IMAGE_ENCODER,
    FAST_TOKENIZER_PATH, QUEST_TOKENIZER_CKPT, OAT_TOKENIZER_CKPT,
    TOKENIZER_HORIZON, TOKENIZER_FIT_NORM_MAX_TRAJS,
)
from verb_probe.models import (
    ActionToVerbTransformer, SCENE_FUSION_MODALITIES,
)
from verb_probe.training import (
    build_criterion, build_optimizer_scheduler, run_training_loop,
)


# ======================================================================
# Dataset-specific loading
# ======================================================================

def _resolve_internal_modality(args):
    """Map clean CLI modality to internal model modality strings.

    Returns: (internal_modality, goal_source)
        internal_modality: string the model understands
        goal_source: "scene_obs" | "image" | None
    """
    if args.modality == "action_only":
        return "action_only", None
    elif args.modality == "goal_only":
        if args.dataset == "calvin":
            return "scene_obs", "scene_obs"
        else:
            return "vision_only", "image"
    elif args.modality == "fusion":
        if args.dataset == "calvin":
            # Use scene_obs fusion via scene_token modality
            return "full", "scene_obs"
        else:
            return "full", "image"
    raise ValueError(f"Unknown modality: {args.modality}")


def _load_calvin(args):
    """Load CALVIN train/val DataFrames and build datasets."""
    from utils import load_calvin_to_dataframe
    from datasets.calvin_dataset import CalvinVerbProbeDataset

    print(f"Loading CALVIN training data from {args.data_dir}...")
    df = load_calvin_to_dataframe(args.data_dir)
    print(f"Loading CALVIN validation data from {args.val_dir}...")
    val_df = load_calvin_to_dataframe(args.val_dir)

    return df, val_df, 'primary_verb'


def _load_bridge(args):
    """Load Bridge train/val DataFrames."""
    from datasets.bridge_dataset import BRIDGE_CSV, BRIDGE_ACTIONS_NPZ

    csv_path = getattr(args, 'csv_path', None) or BRIDGE_CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} Bridge segments, {df['verb'].nunique()} verbs")

    train_df, val_df = train_test_split(
        df, test_size=args.val_fraction, random_state=42, stratify=df["verb"])
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), 'verb'


def _load_droid(args):
    """Load DROID train/val DataFrames."""
    from datasets.droid_dataset import DROID_CSV

    csv_path = getattr(args, 'csv_path', None) or DROID_CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} DROID episodes, {df['verb'].nunique()} verbs")

    train_df, val_df = train_test_split(
        df, test_size=args.val_fraction, random_state=42, stratify=df["verb"])
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), 'verb'


def _filter_sparse_classes(df, val_df, verb_col, min_count):
    """Filter classes with fewer than min_count training examples."""
    if min_count <= 0:
        return df, val_df
    verb_counts = df[verb_col].value_counts()
    keep_verbs = set(verb_counts[verb_counts >= min_count].index)
    n_before = len(df)
    df = df[df[verb_col].isin(keep_verbs)].reset_index(drop=True)
    val_df = val_df[val_df[verb_col].isin(keep_verbs)].reset_index(drop=True)
    dropped = verb_counts.index.difference(keep_verbs)
    print(f"Filtered: {len(verb_counts)}->{len(keep_verbs)} classes, "
          f"train {n_before}->{len(df)}, val->{len(val_df)}")
    if len(dropped) > 0:
        print(f"  Dropped: {sorted(dropped.tolist())}")
    return df, val_df


# ======================================================================
# Action tokenizer loading
# ======================================================================

def _load_action_tokenizer(args):
    """Load action tokenizer. Returns (tok, vocab_size)."""
    from tokenization.action_tokenizers import load_action_tokenizer

    if args.action_rep == "native":
        return None, None

    if args.action_rep == "fast":
        from tokenization.fast_tokenizer import load_fast_tokenizer, tokenize_trajectory
        _fast_tok = load_fast_tokenizer(args.fast_tokenizer_path)
        def _fast_wrapper(actions_batch):
            return [tokenize_trajectory(_fast_tok, actions_batch[0])]
        _fast_wrapper.vocab_size = _fast_tok.vocab_size
        print(f"Loaded FAST tokenizer (vocab_size={_fast_tok.vocab_size})")
        return _fast_wrapper, _fast_tok.vocab_size

    if args.action_rep in ("bin", "quest", "oat"):
        tok = load_action_tokenizer(
            args.action_rep,
            train_dir=args.data_dir,
            horizon=TOKENIZER_HORIZON,
            max_tokens=args.max_seq_len,
            quest_ckpt=args.quest_ckpt,
            oat_ckpt=args.oat_ckpt,
            fit_norm_max_trajs=TOKENIZER_FIT_NORM_MAX_TRAJS,
        )
        print(f"Loaded {args.action_rep} tokenizer (vocab_size={tok.vocab_size})")
        return tok, tok.vocab_size

    if args.action_rep == "vq_vae":
        from tokenization.vqvae_tokenizer import load_vqvae_tokenizer, tokenize_trajectory_vqvae
        from functools import partial
        _vq = load_vqvae_tokenizer(args.tokenizer_ckpt)
        tok = partial(tokenize_trajectory_vqvae, _vq)
        tok.vocab_size = _vq.num_codes
        print(f"Loaded VQ-VAE tokenizer (num_codes={_vq.num_codes})")
        return tok, _vq.num_codes

    if args.action_rep == "vq_bet":
        from tokenization.vqbet_tokenizer import load_vqbet_tokenizer, tokenize_trajectory_vqbet
        from functools import partial
        _vqb = load_vqbet_tokenizer(args.tokenizer_ckpt)
        tok = partial(tokenize_trajectory_vqbet, _vqb)
        tok.vocab_size = _vqb.total_vocab_size
        print(f"Loaded VQ-BeT tokenizer (vocab_size={_vqb.total_vocab_size})")
        return tok, _vqb.total_vocab_size

    if args.action_rep == "vqvla":
        from tokenization.vqvae_tokenizer import load_vqvla_tokenizer, VQVLA_VOCAB_SIZE
        _vqvla = load_vqvla_tokenizer(
            config_dir=args.vqvla_config_dir,
            checkpoint_path=args.vqvla_checkpoint_path)
        from tokenization.vqvae_tokenizer import tokenize_trajectory_vqvla
        from functools import partial
        tok = partial(tokenize_trajectory_vqvla, _vqvla)
        tok.vocab_size = VQVLA_VOCAB_SIZE
        print(f"Loaded VQ-VLA tokenizer (vocab_size={VQVLA_VOCAB_SIZE})")
        return tok, VQVLA_VOCAB_SIZE

    raise ValueError(f"Unknown action_rep: {args.action_rep}")


# ======================================================================
# Dataset construction
# ======================================================================

def _build_calvin_datasets(args, df, val_df, internal_modality, tok, img_size, transform):
    """Build CalvinVerbProbeDataset for train and val."""
    from datasets.calvin_dataset import CalvinVerbProbeDataset

    delta_patches = args.delta_patches

    train_ds = CalvinVerbProbeDataset(
        args.data_dir, df, modality=internal_modality,
        action_tokenizer=tok,
        max_seq_len=args.max_seq_len, num_frames=args.num_frames,
        delta_patches=delta_patches, image_encoder=args.image_encoder,
        transform=transform, img_size=img_size, cache_actions=True)

    val_ds = CalvinVerbProbeDataset(
        args.val_dir, val_df, modality=internal_modality,
        action_tokenizer=tok, verb_to_id=train_ds.verb_to_id,
        max_seq_len=args.max_seq_len, num_frames=args.num_frames,
        delta_patches=delta_patches, image_encoder=args.image_encoder,
        transform=transform, img_size=img_size, cache_actions=True)

    # Drop val samples with unseen verbs
    valid_mask = val_df[train_ds._verb_col].isin(train_ds.verb_to_id.keys())
    if (~valid_mask).sum() > 0:
        print(f"Dropping {(~valid_mask).sum()} val samples with unseen verbs")
        val_ds.df = val_df[valid_mask].reset_index(drop=True)

    return train_ds, val_ds


def _build_bridge_datasets(args, df, val_df, internal_modality, tok, img_size, transform):
    """Build Bridge datasets for train and val."""
    from datasets.bridge_dataset import BridgeVerbDataset, BRIDGE_ACTIONS_NPZ

    actions_npz_path = getattr(args, 'actions_npz', None) or BRIDGE_ACTIONS_NPZ
    print(f"Loading Bridge actions from {actions_npz_path}...")
    npz = np.load(actions_npz_path, allow_pickle=True)
    actions_cache = {k: npz[k] for k in npz.files if k.startswith("actions_")}
    npz.close()

    train_ds = BridgeVerbDataset(df, actions_cache, max_seq_len=args.max_seq_len)
    val_ds = BridgeVerbDataset(val_df, actions_cache, max_seq_len=args.max_seq_len,
                                verb_to_id=train_ds.verb_to_id)
    return train_ds, val_ds


def _build_droid_datasets(args, df, val_df, internal_modality, tok, img_size, transform):
    """Build DROID datasets for train and val."""
    from datasets.droid_dataset import (
        DroidVerbDataset, DroidGoalDataset, load_actions_cache,
        build_frames_index, DROID_ACTIONS_DIR, DROID_FRAMES_DIR,
    )

    if internal_modality == "vision_only":
        frames_dir = getattr(args, 'frames_dir', None) or DROID_FRAMES_DIR
        frames_index = build_frames_index(frames_dir)
        has_frames = df["episode_idx"].isin(frames_index)
        if not has_frames.all():
            print(f"Warning: {(~has_frames).sum()} episodes missing frames, dropping")
            df = df[has_frames].reset_index(drop=True)
            val_df = val_df[val_df["episode_idx"].isin(frames_index)].reset_index(drop=True)
        train_ds = DroidGoalDataset(df, frames_index, img_size=img_size)
        val_ds = DroidGoalDataset(val_df, frames_index, img_size=img_size,
                                   verb_to_id=train_ds.verb_to_id)
    else:
        actions_dir = getattr(args, 'actions_dir', None) or DROID_ACTIONS_DIR
        actions_cache, _ = load_actions_cache(actions_dir)
        train_ds = DroidVerbDataset(df, actions_cache, max_seq_len=args.max_seq_len)
        val_ds = DroidVerbDataset(val_df, actions_cache, max_seq_len=args.max_seq_len,
                                   verb_to_id=train_ds.verb_to_id)
    return train_ds, val_ds


# ======================================================================
# Main
# ======================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve modality
    internal_modality, goal_source = _resolve_internal_modality(args)
    print(f"Dataset: {args.dataset} | Modality: {args.modality} "
          f"(internal={internal_modality}) | Action rep: {args.action_rep}")
    if goal_source:
        print(f"  Goal source: {goal_source}")

    # Image size
    img_size = 224 if args.image_encoder in ("r3m", "dinov2_s", "dinov2_b", "vc1", "dinov2") else IMAGE_SIZE[0]

    # Vision transform (only for image-based goals)
    needs_vision = internal_modality in ("full", "vision_only")
    if needs_vision and transforms is not None:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
        ])
    else:
        transform = None

    # Action tokenizer
    tok, action_vocab_size = _load_action_tokenizer(args)

    # Load data
    loaders = {
        "calvin": _load_calvin,
        "bridge": _load_bridge,
        "droid": _load_droid,
    }
    df, val_df, verb_col = loaders[args.dataset](args)

    if args.debug:
        n = min(args.debug, len(df))
        df = df.head(n).copy()
        val_df = val_df.head(n).copy()
        args.epochs = min(args.epochs, 2)
        print(f"[DEBUG] {n} train / {len(val_df)} val, {args.epochs} epochs")

    # Filter sparse classes
    df, val_df = _filter_sparse_classes(df, val_df, verb_col, args.min_class_count)

    # Build datasets
    builders = {
        "calvin": _build_calvin_datasets,
        "bridge": _build_bridge_datasets,
        "droid": _build_droid_datasets,
    }
    train_ds, val_ds = builders[args.dataset](
        args, df, val_df, internal_modality, tok, img_size, transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers,
                               pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    # Model
    num_verbs = len(train_ds.verb_to_id)
    if internal_modality == "scene_obs":
        action_dim = SCENE_OBS_DIM
    elif args.dataset == "bridge":
        from datasets.bridge_dataset import BRIDGE_ACTION_DIM
        action_dim = BRIDGE_ACTION_DIM
    elif args.dataset == "droid":
        from datasets.droid_dataset import DROID_ACTION_DIM
        action_dim = DROID_ACTION_DIM
    else:
        action_dim = ACTION_DIM

    scene_dim = SCENE_REP_DIM if internal_modality in SCENE_FUSION_MODALITIES else 0

    model = ActionToVerbTransformer(
        num_verbs=num_verbs, d_model=args.d_model,
        num_layers=args.num_layers, action_dim=action_dim,
        max_action_len=args.max_seq_len, img_size=img_size,
        modality=internal_modality, action_rep=args.action_rep,
        cross_layers=args.cross_layers, image_encoder=args.image_encoder,
        action_vocab_size=action_vocab_size,
        freeze_vision=args.freeze_vision, num_frames=args.num_frames,
        delta_patches=args.delta_patches,
        modal_dropout=args.modal_dropout,
        aux_loss_weight=args.aux_loss_weight,
        scene_dim=scene_dim,
    ).to(device)

    # Sync num_patches for vision modalities
    if hasattr(train_ds, 'num_patches') and hasattr(model, 'num_patches'):
        train_ds.num_patches = model.num_patches
        val_ds.num_patches = model.num_patches

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} trainable params")

    # Loss, optimizer, scheduler
    criterion = build_criterion(
        df[verb_col].value_counts(), train_ds.verb_to_id,
        num_verbs, device, weighted=args.weighted_loss,
        label_smoothing=args.label_smoothing)

    total_steps = len(train_loader) * args.epochs
    optimizer, scheduler = build_optimizer_scheduler(
        model, args.lr, args.weight_decay, total_steps,
        args.warmup_epochs, args.epochs)

    args.grad_clip = GRAD_CLIP_NORM

    # Checkpoint metadata
    def ckpt_fn(model, args, best_val_acc, best_epoch):
        return {
            "num_verbs": num_verbs,
            "verb_to_id": train_ds.verb_to_id,
            "id_to_verb": train_ds.id_to_verb,
            "d_model": args.d_model,
            "action_dim": action_dim,
            "nhead": NHEAD,
            "num_layers": args.num_layers,
            "max_action_len": args.max_seq_len,
            "modality": internal_modality,
            "action_rep": args.action_rep,
            "action_vocab_size": action_vocab_size,
            "cross_layers": args.cross_layers,
            "image_encoder": args.image_encoder,
            "delta_patches": args.delta_patches,
            "min_class_count": args.min_class_count,
            "dataset": args.dataset,
            "scene_dim": scene_dim,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
        }

    # Whether to pass seq_lengths to training loop
    pass_seq = internal_modality != "vision_only"

    print(f"\nTraining: {num_verbs} verbs, {args.epochs} epochs, "
          f"d_model={args.d_model}")

    run_training_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, args, num_verbs, train_ds.id_to_verb,
        checkpoint_metadata_fn=ckpt_fn,
        use_aux=(args.aux_loss_weight > 0.0),
        track_per_class_loss=True,
        pass_seq_lengths=pass_seq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified verb classification probe")

    # Dataset
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["calvin", "bridge", "droid"])
    parser.add_argument("--data_dir", type=str, default=DATA_DIR,
                        help="CALVIN training dir (ignored for bridge/droid)")
    parser.add_argument("--val_dir", type=str, default=VAL_DIR,
                        help="CALVIN validation dir (ignored for bridge/droid)")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="CSV path override for bridge/droid")
    parser.add_argument("--actions_npz", type=str, default=None,
                        help="Bridge actions npz path override")
    parser.add_argument("--actions_dir", type=str, default=None,
                        help="DROID actions dir override")
    parser.add_argument("--frames_dir", type=str, default=None,
                        help="DROID frames dir override")
    parser.add_argument("--val_fraction", type=float, default=0.15,
                        help="Val split fraction for bridge/droid")

    # Modality
    parser.add_argument("--modality", type=str, default="action_only",
                        choices=["action_only", "goal_only", "fusion"])

    # Action representation
    parser.add_argument("--action_rep", type=str, default="native",
                        choices=["native", "fast", "bin", "vq_bet", "vq_vae",
                                 "vqvla", "quest", "oat"])
    parser.add_argument("--tokenizer_ckpt", type=str, default=None,
                        help="Checkpoint path for vq_bet/vq_vae tokenizer")
    parser.add_argument("--fast_tokenizer_path", type=str,
                        default=FAST_TOKENIZER_PATH)
    parser.add_argument("--quest_ckpt", type=str, default=QUEST_TOKENIZER_CKPT)
    parser.add_argument("--oat_ckpt", type=str, default=OAT_TOKENIZER_CKPT)
    parser.add_argument("--vqvla_config_dir", type=str,
                        default="./tokenization/vqvla/config")
    parser.add_argument("--vqvla_checkpoint_path", type=str,
                        default="./checkpoints/vqvla_pretrained/action_tokenizer_weight/all_data_vq.pth")

    # Vision (for goal_only/fusion with image source)
    parser.add_argument("--image_encoder", type=str, default=IMAGE_ENCODER,
                        choices=["scratch", "patch", "resnet18", "dinov2",
                                 "dinov2_s", "dinov2_b", "vc1", "r3m"])
    parser.add_argument("--freeze_vision", action="store_true", default=True)
    parser.add_argument("--no_freeze_vision", dest="freeze_vision",
                        action="store_false")
    parser.add_argument("--num_frames", type=int, default=2)
    parser.add_argument("--delta_patches", type=int, default=0)

    # Model
    parser.add_argument("--d_model", type=int, default=D_MODEL)
    parser.add_argument("--num_layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--cross_layers", type=int, default=CROSS_LAYERS)
    parser.add_argument("--modal_dropout", type=float, default=0.0)
    parser.add_argument("--aux_loss_weight", type=float, default=0.0)

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
    main(args)
