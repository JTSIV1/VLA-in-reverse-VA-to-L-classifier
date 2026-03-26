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
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
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


# ======================================================================
# Loss, optimizer, scheduler
# ======================================================================

def build_criterion(verb_counts, verb_to_id, num_verbs, device,
                    weighted=False, label_smoothing=0.0):
    """Build CE loss, optionally weighted by inverse class frequency."""
    if weighted:
        weights = torch.zeros(num_verbs)
        for verb, cid in verb_to_id.items():
            count = verb_counts.get(verb, 1)
            weights[cid] = 1.0 / count
        weights = weights / weights.sum() * num_verbs
        criterion = nn.CrossEntropyLoss(weight=weights.to(device),
                                        label_smoothing=label_smoothing)
        print(f"Weighted CE (min={weights.min():.3f}, max={weights.max():.3f}), "
              f"label_smoothing={label_smoothing}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    return criterion


def build_optimizer_scheduler(model, lr, weight_decay, total_steps,
                              warmup_epochs, epochs):
    """Build AdamW optimizer with OneCycleLR scheduler."""
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay)
    warmup_pct = min(warmup_epochs / epochs, 0.3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=warmup_pct, anneal_strategy="cos")
    return optimizer, scheduler


# ======================================================================
# Training and validation epochs
# ======================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler,
                    device, grad_clip, epoch, total_epochs,
                    use_aux=False, aux_weight=0.0, track_per_class_loss=False,
                    batch_transform_fn=None):
    """Run one training epoch with the standard batch format.

    Batch format: (frames, actions, scene_vecs, labels, seq_lengths)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_loss_sum = defaultdict(float) if track_per_class_loss else None

    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f"Epoch {epoch}/{total_epochs}")
    for batch_idx, batch in pbar:
        if batch_transform_fn is not None:
            frames, actions, scene_vecs, labels, seq_lengths = batch_transform_fn(batch, device)
        else:
            frames, actions, scene_vecs, labels, seq_lengths = batch
            frames = frames.to(device)
            actions = actions.to(device)
            labels = labels.to(device)
            scene_vecs = scene_vecs.to(device)
            seq_lengths = seq_lengths.to(device)

        optimizer.zero_grad()

        if use_aux and aux_weight > 0.0:
            main_logits, aux_v_logits, aux_a_logits = model.forward_with_aux(
                frames, actions, seq_lengths=seq_lengths, scene_vec=scene_vecs)
            loss = criterion(main_logits, labels)
            if aux_v_logits is not None:
                loss = loss + aux_weight * criterion(aux_v_logits, labels)
            if aux_a_logits is not None:
                loss = loss + aux_weight * criterion(aux_a_logits, labels)
            logits = main_logits
        else:
            logits = model(frames, actions, seq_lengths=seq_lengths,
                           scene_vec=scene_vecs)
            loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Per-class stats
        with torch.no_grad():
            if track_per_class_loss:
                per_sample_loss = nn.functional.cross_entropy(
                    logits, labels, reduction='none')
                for lbl, pred, sl in zip(labels.cpu().tolist(),
                                         preds.cpu().tolist(),
                                         per_sample_loss.cpu().tolist()):
                    class_total[lbl] += 1
                    class_correct[lbl] += int(pred == lbl)
                    class_loss_sum[lbl] += sl
            else:
                for lbl, pred in zip(labels.cpu().tolist(),
                                     preds.cpu().tolist()):
                    class_total[lbl] += 1
                    class_correct[lbl] += int(pred == lbl)

        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         acc=f"{100*correct/total:.1f}%")

    avg_loss = total_loss / len(dataloader)
    train_acc = 100 * correct / total
    current_lr = scheduler.get_last_lr()[0]

    result = {
        "loss": avg_loss,
        "acc": train_acc,
        "lr": current_lr,
        "class_correct": dict(class_correct),
        "class_total": dict(class_total),
    }
    if track_per_class_loss:
        result["class_loss_sum"] = dict(class_loss_sum)
    return result


def validate(model, dataloader, criterion, device, num_verbs, id_to_verb,
             pass_seq_lengths=True, batch_transform_fn=None):
    """Run validation with the standard batch format.

    Returns:
        dict with keys: loss, acc, macro_recall, per_class_val,
                        class_correct, class_total
    """
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    val_class_correct = defaultdict(int)
    val_class_total = defaultdict(int)
    val_class_loss_sum = defaultdict(float)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Validating"):
            if batch_transform_fn is not None:
                frames, actions, scene_vecs, labels, seq_lengths = batch_transform_fn(batch, device)
            else:
                frames, actions, scene_vecs, labels, seq_lengths = batch
                frames = frames.to(device)
                actions = actions.to(device)
                labels = labels.to(device)
                scene_vecs = scene_vecs.to(device)
                seq_lengths = seq_lengths.to(device)

            sl = seq_lengths if pass_seq_lengths else None
            logits = model(frames, actions, seq_lengths=sl,
                           scene_vec=scene_vecs)
            loss = criterion(logits, labels)
            per_sample_loss = nn.functional.cross_entropy(
                logits, labels, reduction='none')

            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

            for lbl, pred, sl_val in zip(labels.cpu().tolist(),
                                         preds.cpu().tolist(),
                                         per_sample_loss.cpu().tolist()):
                val_class_total[lbl] += 1
                val_class_correct[lbl] += int(pred == lbl)
                val_class_loss_sum[lbl] += sl_val

    val_avg = val_loss / len(dataloader)
    val_acc = 100 * val_correct / val_total if val_total > 0 else 0

    # Macro recall
    per_class_recall = []
    for cid in range(num_verbs):
        n = val_class_total.get(cid, 0)
        tp = val_class_correct.get(cid, 0)
        per_class_recall.append(tp / n if n > 0 else 0)
    macro_recall = np.mean(per_class_recall) * 100

    # Per-class metrics dict
    per_class_val = {}
    for cid in range(num_verbs):
        verb = id_to_verb.get(cid, str(cid))
        vt = val_class_total.get(cid, 0)
        per_class_val[verb] = {
            "loss": val_class_loss_sum.get(cid, 0) / vt if vt > 0 else 0,
            "acc": 100 * val_class_correct.get(cid, 0) / vt if vt > 0 else 0,
            "count": vt,
        }

    return {
        "loss": val_avg,
        "acc": val_acc,
        "macro_recall": macro_recall,
        "per_class_val": per_class_val,
        "class_correct": dict(val_class_correct),
        "class_total": dict(val_class_total),
    }


# ======================================================================
# Checkpoint and logging
# ======================================================================

def save_checkpoint(state_dict, metadata, path):
    """Save a model checkpoint with metadata."""
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    ckpt = {"state_dict": state_dict}
    ckpt.update(metadata)
    torch.save(ckpt, path)


def save_training_log(config_dict, training_log, log_path):
    """Write training log JSON (overwrites each epoch for partial results)."""
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({"config": config_dict, "epochs": training_log}, f, indent=2)


def build_per_class_train_metrics(class_correct, class_total, class_loss_sum,
                                  id_to_verb, all_cids):
    """Build per-class train metrics dict from accumulators."""
    per_class_train = {}
    for cid in sorted(all_cids):
        verb = id_to_verb.get(cid, str(cid))
        t = class_total.get(cid, 0)
        entry = {
            "acc": 100 * class_correct.get(cid, 0) / t if t > 0 else 0,
            "count": t,
        }
        if class_loss_sum is not None:
            entry["loss"] = class_loss_sum.get(cid, 0) / t if t > 0 else 0
        per_class_train[verb] = entry
    return per_class_train


# ======================================================================
# Full training loop
# ======================================================================

def run_training_loop(model, train_loader, val_loader, criterion,
                      optimizer, scheduler, device, args,
                      num_verbs, id_to_verb,
                      checkpoint_metadata_fn,
                      use_aux=False, pass_seq_lengths=True,
                      track_per_class_loss=False,
                      attn_frac_fn=None,
                      batch_transform_fn=None):
    """Full training loop.

    Best checkpoint selected by lowest val loss. Saves val_loss,
    val_acc, and macro_recall in the checkpoint metadata.
    """
    grad_clip = getattr(args, 'grad_clip', 1.0)
    aux_weight = getattr(args, 'aux_loss_weight', 0.0)
    patience = getattr(args, 'patience', 0)

    training_log = []
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        train_result = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, grad_clip, epoch, args.epochs,
            use_aux=use_aux, aux_weight=aux_weight,
            track_per_class_loss=track_per_class_loss,
            batch_transform_fn=batch_transform_fn)

        print(f"--- Epoch {epoch}: Loss={train_result['loss']:.4f} "
              f"Acc={train_result['acc']:.1f}% LR={train_result['lr']:.2e}")

        # --- Validate ---
        val_result = validate(
            model, val_loader, criterion, device, num_verbs, id_to_verb,
            pass_seq_lengths=pass_seq_lengths,
            batch_transform_fn=batch_transform_fn)

        print(f"    Val: Loss={val_result['loss']:.4f} "
              f"Acc={val_result['acc']:.1f}% "
              f"MacroRecall={val_result['macro_recall']:.1f}%")

        # --- Attention fractions (optional) ---
        attn_fracs = {}
        if attn_frac_fn is not None:
            try:
                attn_fracs = attn_frac_fn(model, val_loader, device)
            except Exception:
                pass

        # --- Build per-class train metrics ---
        all_cids = set(list(train_result["class_total"].keys()) +
                       list(val_result["class_total"].keys()))
        per_class_train = build_per_class_train_metrics(
            train_result["class_correct"], train_result["class_total"],
            train_result.get("class_loss_sum"), id_to_verb, all_cids)

        # --- Save best checkpoint (by val loss) ---
        val_acc = val_result["acc"]
        val_loss = val_result["loss"]
        val_macro_recall = val_result["macro_recall"]
        if val_loss < best_val_loss and args.save_path:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_macro_recall = val_macro_recall
            best_epoch = epoch
            metadata = checkpoint_metadata_fn(model, args, best_val_acc, best_epoch)
            metadata["best_val_loss"] = best_val_loss
            metadata["best_val_macro_recall"] = best_macro_recall
            save_checkpoint(model.state_dict(), metadata, args.save_path)
            print(f"    * Best val loss: {val_loss:.4f} "
                  f"Acc={val_acc:.1f}% "
                  f"MacroRecall={val_macro_recall:.1f}% "
                  f"@ epoch {epoch}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience > 0 and patience_counter >= patience:
                print(f"    Early stopping after {patience} epochs no improvement")
                break

        # --- Log ---
        epoch_metrics = {
            "epoch": epoch,
            "lr": train_result["lr"],
            "train_loss": train_result["loss"],
            "train_acc": train_result["acc"],
            "val_loss": val_result["loss"],
            "val_acc": val_acc,
            "macro_recall": val_result["macro_recall"],
            "per_class_train": per_class_train,
            "per_class_val": val_result["per_class_val"],
        }
        if attn_fracs:
            epoch_metrics["attn_fracs"] = attn_fracs
        training_log.append(epoch_metrics)

        if args.log_path:
            config_dict = {k: v for k, v in vars(args).items()
                           if isinstance(v, (str, int, float, bool, type(None)))}
            save_training_log(config_dict, training_log, args.log_path)
            print(f"    Log saved to {args.log_path}")

    print(f"\nBest val loss: {best_val_loss:.4f} "
          f"Acc={best_val_acc:.1f}% @ epoch {best_epoch}")

    # Generate training curves if log exists
    if args.log_path and os.path.exists(args.log_path):
        try:
            from verb_probe.analysis import plot_training_curves
            curves_path = args.log_path.replace(".json", "_curves.png")
            plot_training_curves(args.log_path, curves_path)
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception as e:
            print(f"Warning: could not generate training curves: {e}")

    return best_val_acc, best_epoch


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
    """Build CalvinTokenizerDataset + frozen tokenizer for on-the-fly encoding."""
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
