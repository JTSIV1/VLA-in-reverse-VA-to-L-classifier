"""Verb classification probe for CALVIN dataset.

Trains MotionVerbClassifier (action_only) or GoalVerbClassifier (goal_only)
to classify verbs from action trajectories or vision.

Action representations:
  - native:      raw 7-DoF actions
  - fast:        FAST (DCT+BPE) discrete token IDs
  - vq_bet/oat/quest: discrete codebook IDs from frozen tokenizer
  - latent:      continuous encoder latents from frozen tokenizer

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

    # Goal-only (uses DINOv2-S image patches)
    python verb_probe/train_verb_probe.py --modality goal_only \
        --image_encoder dinov2_s --delta_patches 16
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

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
from verb_probe.training import (
    build_criterion, build_optimizer_scheduler, run_training_loop,
)


# ======================================================================
# Frozen tokenizer loading and on-the-fly encoding
# ======================================================================

def _load_frozen_tokenizer(args):
    """Load frozen tokenizer using the same builders as train_tokenizer.py."""
    from tokenization.train_tokenizer import build_vqbet, build_oat, build_quest

    ckpt = torch.load(args.tokenizer_ckpt, map_location="cpu")
    ckpt_args = ckpt["args"]
    if not isinstance(ckpt_args, dict):
        ckpt_args = vars(ckpt_args)

    import argparse
    build_args = argparse.Namespace(**ckpt_args)

    if args.tokenizer_type == "vq_bet":
        model = build_vqbet(build_args)
    elif args.tokenizer_type == "oat":
        model = build_oat(build_args)
    elif args.tokenizer_type == "quest":
        model = build_quest(build_args)
    else:
        raise ValueError(f"Unknown tokenizer_type: {args.tokenizer_type}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if "normalizer" in ckpt:
        model.set_normalizer(ckpt["normalizer"])

    return model


def _make_tokenizer_batch_transform(tok_model, tok_type, mode):
    """Build a batch transform that encodes chunks on-the-fly.

    Args:
        tok_model: frozen tokenizer model.
        tok_type: 'vq_bet', 'oat', or 'quest'.
        mode: 'latent' — return continuous latent vectors as actions.
              'token_id' — return discrete code IDs as actions.
                  VQ-BeT codes get group-offset encoding so each group's
                  codes occupy a separate range of the embedding table.

    Returns a callable(batch, device) -> (frames, actions, scene_vecs, labels, seq_lengths)
    """
    from tokenization.train_utils import extract_episode_batch

    def transform(batch, device):
        with torch.no_grad():
            result = extract_episode_batch(tok_model, batch, device, tok_type)

        n_valid = result['n_valid']
        labels = result['verb_ids']

        if mode == 'latent':
            actions = result['latents']
        else:
            codes = result['codes']
            if tok_type == 'vq_bet' and codes.ndim == 3:
                # (B, K, groups) → group-offset encoding → (B, K*groups)
                B, K, G = codes.shape
                offsets = torch.arange(G, device=device) * tok_model.n_embed
                codes = (codes + offsets.view(1, 1, G)).reshape(B, K * G)
                n_valid = n_valid * G
            actions = codes.long()

        B = actions.size(0)
        dummy = torch.zeros(B, 1, device=device)
        return dummy, actions, dummy, labels, n_valid

    return transform


def _load_fast_tokenizer(args):
    """Load FAST tokenizer for on-the-fly tokenization in CalvinVerbProbeDataset.

    Returns: (tok_wrapper, vocab_size) or (None, None).
    """
    if args.action_rep != "fast":
        return None, None
    from tokenization.fast.fast_tokenizer import load_fast_tokenizer, tokenize_trajectory
    _fast_tok = load_fast_tokenizer(args.fast_tokenizer_path)
    def _fast_wrapper(actions_batch):
        return [tokenize_trajectory(_fast_tok, actions_batch[0])]
    _fast_wrapper.vocab_size = _fast_tok.vocab_size
    print(f"Loaded FAST tokenizer (vocab_size={_fast_tok.vocab_size})")
    return _fast_wrapper, _fast_tok.vocab_size


# ======================================================================
# Dataset construction
# ======================================================================

_TOKENIZER_REPS = {"vq_bet", "oat", "quest", "latent"}


def build_datasets(args):
    """Build train/val datasets.

    Returns: (train_ds, val_ds, num_verbs, id_to_verb, verb_to_id, verb_counts)
    """
    if args.action_rep in _TOKENIZER_REPS:
        return _build_tokenizer_datasets(args)
    else:
        return _build_standard_datasets(args)


def _build_tokenizer_datasets(args):
    """Build CalvinTokenizerDataset + frozen tokenizer for on-the-fly encoding.

    Handles both latent (continuous) and token_id (discrete codes) modes.
    """
    from utils import load_calvin_to_dataframe
    from datasets.calvin_dataset import CalvinTokenizerDataset

    # Load frozen tokenizer
    print(f"Loading frozen {args.tokenizer_type} from {args.tokenizer_ckpt}")
    tok_model = _load_frozen_tokenizer(args)

    # Get chunk_size and sampling from the tokenizer checkpoint's saved args
    ckpt = torch.load(args.tokenizer_ckpt, map_location="cpu")
    ckpt_args = ckpt["args"]
    if not isinstance(ckpt_args, dict):
        ckpt_args = vars(ckpt_args)
    chunk_size = ckpt_args.get("chunk_size", 16)
    sampling = ckpt_args.get("sampling", "random")
    max_chunks = ckpt_args.get("max_chunks", 8)
    print(f"  chunk_size={chunk_size}, sampling={sampling}, max_chunks={max_chunks}")

    # Load DataFrames and filter sparse classes
    train_df = load_calvin_to_dataframe(args.data_dir)
    val_df = load_calvin_to_dataframe(args.val_dir)

    if args.min_class_count > 0:
        verb_col = 'primary_verb' if 'primary_verb' in train_df.columns else 'verb'
        vc = train_df[verb_col].value_counts()
        keep_verbs = set(vc[vc >= args.min_class_count].index)
        train_df = train_df[train_df[verb_col].isin(keep_verbs)].reset_index(drop=True)
        val_df = val_df[val_df[verb_col].isin(keep_verbs)].reset_index(drop=True)
        print(f"  Filtered to {len(keep_verbs)} classes")

    # Build CalvinTokenizerDataset (same format as tokenizer training)
    train_ds = CalvinTokenizerDataset(
        args.data_dir, train_df, chunk_size=chunk_size,
        max_chunks=max_chunks, sampling=sampling, cache_actions=True)
    val_ds = CalvinTokenizerDataset(
        args.val_dir, val_df, chunk_size=chunk_size,
        max_chunks=max_chunks, sampling=sampling,
        verb_to_id=train_ds.verb_to_id, cache_actions=True)

    verb_to_id = train_ds.verb_to_id
    id_to_verb = train_ds.id_to_verb

    # Drop val samples with unseen verbs
    verb_col = train_ds._verb_col
    valid_mask = val_df[verb_col].isin(verb_to_id.keys())
    if (~valid_mask).sum() > 0:
        print(f"  Dropping {(~valid_mask).sum()} val samples with unseen verbs")
        val_ds.df = val_df[valid_mask].reset_index(drop=True)

    # Determine mode: latent (continuous) vs token_id (discrete codes)
    mode = "latent" if args.action_rep == "latent" else "token_id"

    # Probe shapes from one sample
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_model = tok_model.to(device)
    from tokenization.train_utils import extract_episode_batch
    sample_batch = next(iter(DataLoader(train_ds, batch_size=1, shuffle=False)))
    with torch.no_grad():
        result = extract_episode_batch(tok_model, sample_batch, device, args.tokenizer_type)

    if mode == "latent":
        args._latent_dim = result['latents'].shape[-1]
        print(f"  latent_dim={args._latent_dim}")
    else:
        # vocab_size for nn.Embedding
        if args.tokenizer_type == "vq_bet":
            # group-offset encoding: each group occupies [g*n_embed, (g+1)*n_embed)
            args._action_vocab_size = tok_model.n_embed * tok_model.groups
        else:
            args._action_vocab_size = tok_model.vocab_size
        print(f"  vocab_size={args._action_vocab_size}")

    # Store batch transform for on-the-fly encoding in training loop
    args._batch_transform_fn = _make_tokenizer_batch_transform(
        tok_model, args.tokenizer_type, mode)

    # Verb counts for weighted loss
    verb_counts = train_ds.df[verb_col].value_counts().to_dict()

    num_verbs = len(verb_to_id)
    return train_ds, val_ds, num_verbs, id_to_verb, verb_to_id, verb_counts


def _build_standard_datasets(args):
    """Build CalvinVerbProbeDataset for native/fast/goal_only modes."""
    from utils import load_calvin_to_dataframe
    from datasets.calvin_dataset import CalvinVerbProbeDataset

    print(f"Loading CALVIN data from {args.data_dir} / {args.val_dir}...")
    train_df = load_calvin_to_dataframe(args.data_dir)
    val_df = load_calvin_to_dataframe(args.val_dir)

    # Filter sparse verb classes
    if args.min_class_count > 0:
        verb_col = 'primary_verb' if 'primary_verb' in train_df.columns else 'verb'
        vc = train_df[verb_col].value_counts()
        keep_verbs = set(vc[vc >= args.min_class_count].index)
        n_before = len(train_df)
        train_df = train_df[train_df[verb_col].isin(keep_verbs)].reset_index(drop=True)
        val_df = val_df[val_df[verb_col].isin(keep_verbs)].reset_index(drop=True)
        print(f"Filtered: {len(vc)}->{len(keep_verbs)} classes, "
              f"train {n_before}->{len(train_df)}, val->{len(val_df)}")

    if args.debug:
        n = min(args.debug, len(train_df))
        train_df = train_df.head(n).copy()
        val_df = val_df.head(n).copy()
        args.epochs = min(args.epochs, 2)
        print(f"[DEBUG] {n} train / {len(val_df)} val, {args.epochs} epochs")

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
        tok, _ = _load_fast_tokenizer(args)

    # Internal modality string
    internal_modality = "action_only" if args.modality == "action_only" else "vision_only"

    train_ds = CalvinVerbProbeDataset(
        args.data_dir, train_df, modality=internal_modality,
        action_tokenizer=tok,
        max_seq_len=args.max_seq_len, num_frames=args.num_frames,
        delta_patches=args.delta_patches, image_encoder=args.image_encoder,
        transform=transform, img_size=img_size, cache_actions=True)

    val_ds = CalvinVerbProbeDataset(
        args.val_dir, val_df, modality=internal_modality,
        action_tokenizer=tok, verb_to_id=train_ds.verb_to_id,
        max_seq_len=args.max_seq_len, num_frames=args.num_frames,
        delta_patches=args.delta_patches, image_encoder=args.image_encoder,
        transform=transform, img_size=img_size, cache_actions=True)

    # Drop val samples with unseen verbs
    verb_col = train_ds._verb_col
    valid_mask = val_df[verb_col].isin(train_ds.verb_to_id.keys())
    if (~valid_mask).sum() > 0:
        print(f"Dropping {(~valid_mask).sum()} val samples with unseen verbs")
        val_ds.df = val_df[valid_mask].reset_index(drop=True)

    num_verbs = len(train_ds.verb_to_id)
    id_to_verb = train_ds.id_to_verb
    verb_to_id = train_ds.verb_to_id
    verb_counts = train_ds.df[verb_col].value_counts().to_dict()

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

    elif args.action_rep == "latent":
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
        # Discrete token IDs (fast from standard path, vq_bet/oat/quest from tokenizer path)
        if hasattr(args, '_action_vocab_size'):
            action_vocab_size = args._action_vocab_size
        else:
            # FAST: vocab_size stored by _load_fast_tokenizer via CalvinVerbProbeDataset
            _, action_vocab_size = _load_fast_tokenizer(args)
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
            "dataset": "calvin",
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
        }
        if args.action_rep in _TOKENIZER_REPS:
            meta["tokenizer_type"] = args.tokenizer_type
            meta["tokenizer_ckpt"] = args.tokenizer_ckpt
        if args.action_rep == "latent":
            meta["latent_dim"] = args._latent_dim
        if hasattr(args, '_action_vocab_size'):
            meta["action_vocab_size"] = args._action_vocab_size
        return meta

    pass_seq = (args.modality != "goal_only")

    print(f"\nTraining: {num_verbs} verbs, {args.epochs} epochs, "
          f"d_model={args.d_model}")

    batch_transform_fn = getattr(args, '_batch_transform_fn', None)

    run_training_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, args, num_verbs, id_to_verb,
        checkpoint_metadata_fn=ckpt_fn,
        track_per_class_loss=True,
        pass_seq_lengths=pass_seq,
        batch_transform_fn=batch_transform_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CALVIN verb classification probe")

    # Dataset
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--val_dir", type=str, default=VAL_DIR)

    # Modality
    parser.add_argument("--modality", type=str, default="action_only",
                        choices=["action_only", "goal_only"])

    # Action representation
    parser.add_argument("--action_rep", type=str, default="native",
                        choices=["native", "fast", "vq_bet", "quest",
                                 "oat", "latent"])
    parser.add_argument("--fast_tokenizer_path", type=str,
                        default=FAST_TOKENIZER_PATH)

    # Tokenizer checkpoint (for vq_bet/oat/quest/latent)
    parser.add_argument("--tokenizer_type", type=str, default=None,
                        choices=["vq_bet", "oat", "quest"],
                        help="Tokenizer type (required for vq_bet/oat/quest/latent)")
    parser.add_argument("--tokenizer_ckpt", type=str, default=None,
                        help="Frozen tokenizer checkpoint (required for vq_bet/oat/quest/latent)")

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
        if args.tokenizer_type is None and args.action_rep != "latent":
            args.tokenizer_type = args.action_rep
        if not args.tokenizer_type or not args.tokenizer_ckpt:
            parser.error(
                f"--action_rep {args.action_rep} requires --tokenizer_type and --tokenizer_ckpt")

    main(args)
