"""Training utilities for verb probe.

Contains the generic training loop, validation, loss/optimizer builders,
and checkpoint/logging helpers. Nothing here is specific to dataset
construction or tokenizer loading.
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# ======================================================================
# Loss, optimizer, scheduler
# ======================================================================

def build_criterion(verb_counts, verb_to_id, num_verbs, device):
    """Build vanilla CE loss (no class weighting, no label smoothing).

    Probes are reported in MI bits via held-out NLL, which assumes the
    softmax is a calibrated posterior estimate. Class weighting and label
    smoothing both bias the softmax and are intentionally not exposed.
    """
    return nn.CrossEntropyLoss()


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
                    track_per_class_loss=False,
                    tok_model=None, tok_type=None, tok_mode=None,
                    vla_embed_info=None):
    """Run one training epoch.

    Standard batch format: (frames, actions, scene_vecs, labels, seq_lengths)
    Tokenizer batch format: dict from CalvinTokenizerDataset — encoded on-the-fly.
    """
    from verb_probe.load_tokenizer import encode_tokenizer_batch

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
        if tok_model is not None:
            actions, labels, seq_lengths = encode_tokenizer_batch(
                batch, device, tok_model, tok_type, tok_mode,
                vla_embed_info=vla_embed_info)
            frames = torch.zeros(labels.size(0), 1, device=device)
            scene_vecs = frames
        else:
            frames, actions, scene_vecs, labels, seq_lengths = batch
            frames = frames.to(device)
            actions = actions.to(device)
            labels = labels.to(device)
            scene_vecs = scene_vecs.to(device)
            seq_lengths = seq_lengths.to(device)

        optimizer.zero_grad()

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
             pass_seq_lengths=True,
             tok_model=None, tok_type=None, tok_mode=None,
             vla_embed_info=None):
    """Run validation.

    Standard batch format: (frames, actions, scene_vecs, labels, seq_lengths)
    Tokenizer batch format: dict from CalvinTokenizerDataset — encoded on-the-fly.
    """
    from verb_probe.load_tokenizer import encode_tokenizer_batch

    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    val_class_correct = defaultdict(int)
    val_class_total = defaultdict(int)
    val_class_loss_sum = defaultdict(float)
    all_preds_list = []
    all_labels_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Validating"):
            if tok_model is not None:
                actions, labels, seq_lengths = encode_tokenizer_batch(
                    batch, device, tok_model, tok_type, tok_mode,
                    vla_embed_info=vla_embed_info)
                frames = torch.zeros(labels.size(0), 1, device=device)
                scene_vecs = frames
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
            all_preds_list.append(preds.cpu())
            all_labels_list.append(labels.cpu())

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

    # Macro F1
    from sklearn.metrics import f1_score
    all_preds_cat = torch.cat(all_preds_list).numpy()
    all_labels_cat = torch.cat(all_labels_list).numpy()
    macro_f1 = 100 * f1_score(all_labels_cat, all_preds_cat,
                               average='macro', zero_division=0)

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
        "macro_f1": macro_f1,
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
                      pass_seq_lengths=True,
                      track_per_class_loss=False,
                      attn_frac_fn=None):
    """Full training loop.

    Best checkpoint selected by lowest val loss. Saves val_loss,
    val_acc, and macro_f1 in the checkpoint metadata.

    If args._tok_model is set, batches are encoded on-the-fly through
    the frozen tokenizer (same pattern as train_tokenizer.py).
    """
    grad_clip = getattr(args, 'grad_clip', 1.0)
    patience = getattr(args, 'patience', 0)
    tok_model = getattr(args, '_tok_model', None)
    tok_type = getattr(args, 'tokenizer_type', None)
    tok_mode = getattr(args, '_tok_mode', None)
    vla_embed_info = getattr(args, '_vla_embed_info', None)

    training_log = []
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_macro_f1 = 0.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        train_result = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, grad_clip, epoch, args.epochs,
            track_per_class_loss=track_per_class_loss,
            tok_model=tok_model, tok_type=tok_type, tok_mode=tok_mode,
            vla_embed_info=vla_embed_info)

        print(f"--- Epoch {epoch}: Loss={train_result['loss']:.4f} "
              f"Acc={train_result['acc']:.1f}% LR={train_result['lr']:.2e}")

        # --- Validate ---
        val_result = validate(
            model, val_loader, criterion, device, num_verbs, id_to_verb,
            pass_seq_lengths=pass_seq_lengths,
            tok_model=tok_model, tok_type=tok_type, tok_mode=tok_mode,
            vla_embed_info=vla_embed_info)

        print(f"    Val: Loss={val_result['loss']:.4f} "
              f"Acc={val_result['acc']:.1f}% "
              f"MacroF1={val_result['macro_f1']:.1f}%")

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
        val_macro_f1 = val_result["macro_f1"]
        if val_loss < best_val_loss and args.save_path:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            metadata = checkpoint_metadata_fn(model, args, best_val_acc, best_epoch)
            metadata["best_val_loss"] = best_val_loss
            metadata["best_val_macro_f1"] = best_macro_f1
            save_checkpoint(model.state_dict(), metadata, args.save_path)
            print(f"    * Best val loss: {val_loss:.4f} "
                  f"Acc={val_acc:.1f}% "
                  f"MacroF1={val_macro_f1:.1f}% "
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
            "macro_f1": val_result["macro_f1"],
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
          f"Acc={best_val_acc:.1f}% "
          f"MacroF1={best_macro_f1:.1f}% "
          f"@ epoch {best_epoch}")

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
