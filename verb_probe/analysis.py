"""Verb probe analysis utilities.

Provides:
  - plot_training_curves: loss, accuracy, macro recall/F1 from training log JSON
  - plot_tsne: t-SNE of latent embeddings with DBSCAN cluster labels
  - plot_symmetric_confusion: symmetrized confusion matrix (avg of CM and CM^T)
  - plot_all: convenience wrapper that generates all plots for a trained probe

Usage:
    # From command line (standalone):
    python verb_probe/analysis.py --log_path results/verb_probe/log.json \
        --preds_path results/verb_probe/preds.json \
        --embeddings_path results/verb_probe/embeddings.pt \
        --output_dir results/verb_probe/figures/

    # From code:
    from verb_probe.analysis import plot_training_curves, plot_tsne, plot_symmetric_confusion
"""

import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score,
)
from collections import defaultdict

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None


# ── Training curves ──────────────────────────────────────────────────────────

def plot_training_curves(log_path, output_path=None, title_prefix=""):
    """Plot train/val loss, accuracy, and macro recall from a training log JSON.

    Args:
        log_path: path to JSON file with {"config": ..., "epochs": [...]}.
                  Each epoch dict has: epoch, train_loss, train_acc, val_loss,
                  val_acc, macro_recall, and optionally per_class_val.
        output_path: if provided, save figure to this path.
        title_prefix: prepended to subplot titles.

    Returns:
        matplotlib Figure.
    """
    with open(log_path) as f:
        data = json.load(f)

    epochs_data = data.get("epochs", data)  # support both wrapped and raw list
    if isinstance(epochs_data, dict):
        epochs_data = epochs_data.get("epochs", [])

    epochs = [e["epoch"] for e in epochs_data]
    train_loss = [e["train_loss"] for e in epochs_data]
    val_loss = [e["val_loss"] for e in epochs_data]
    train_acc = [e["train_acc"] for e in epochs_data]
    val_acc = [e["val_acc"] for e in epochs_data]
    macro_recall = [e.get("macro_recall", 0) for e in epochs_data]

    # Compute macro F1 from per-class val metrics if available
    macro_f1 = []
    for e in epochs_data:
        pcv = e.get("per_class_val", {})
        if pcv:
            f1s = [v.get("f1", v.get("recall", 0)) for k, v in pcv.items()
                   if isinstance(v, dict) and k not in ("accuracy", "macro avg", "weighted avg")]
            macro_f1.append(np.mean(f1s) * 100 if f1s else 0)
        else:
            macro_f1.append(0)

    n_metrics = 3 if any(v > 0 for v in macro_f1) else 2
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 4.5))
    if n_metrics == 2:
        axes = list(axes) + [None]

    # Loss
    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train", color="tab:blue")
    ax.plot(epochs, val_loss, label="Val", color="tab:orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CE Loss")
    ax.set_title(f"{title_prefix}Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # Accuracy + macro recall
    ax = axes[1]
    ax.plot(epochs, train_acc, label="Train Acc", color="tab:blue")
    ax.plot(epochs, val_acc, label="Val Acc", color="tab:orange")
    ax.plot(epochs, macro_recall, label="Val Macro Recall", color="tab:green", ls="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("%")
    ax.set_title(f"{title_prefix}Accuracy & Macro Recall")
    ax.legend()
    ax.grid(alpha=0.3)

    # Macro F1 (if available)
    if axes[2] is not None and any(v > 0 for v in macro_f1):
        ax = axes[2]
        ax.plot(epochs, macro_f1, label="Val Macro F1", color="tab:red")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("%")
        ax.set_title(f"{title_prefix}Macro F1")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to {output_path}")
    return fig


# ── t-SNE visualization ─────────────────────────────────────────────────────

def _darken(hex_color, factor=0.65):
    r, g, b = mcolors.hex2color(hex_color)
    return (r * factor, g * factor, b * factor)


def _contrast_colors(coords, labels, unique_labels, k_neighbors=4):
    """Assign colors so spatially nearby verbs get maximally different hues."""
    centroids = {lbl: coords[labels == lbl].mean(axis=0) for lbl in unique_labels}
    lbl_list = list(unique_labels)
    n = len(lbl_list)

    cent_arr = np.array([centroids[l] for l in lbl_list])
    dists = np.linalg.norm(cent_arr[:, None] - cent_arr[None, :], axis=-1)
    K = min(k_neighbors, n - 1)
    adj = {l: set() for l in lbl_list}
    for i, li in enumerate(lbl_list):
        for j in np.argsort(dists[i])[1:K + 1]:
            lj = lbl_list[j]
            adj[li].add(lj)
            adj[lj].add(li)

    order = sorted(lbl_list, key=lambda l: -len(adj[l]))
    color_cls = {}
    for lbl in order:
        used = {color_cls[nb] for nb in adj[lbl] if nb in color_cls}
        c = 0
        while c in used:
            c += 1
        color_cls[lbl] = c

    interleaved = list(range(0, 20, 2)) + list(range(1, 20, 2))
    palette = plt.cm.get_cmap("tab20")

    slot_groups = defaultdict(list)
    for lbl in unique_labels:
        slot = interleaved[color_cls[lbl] % 20]
        slot_groups[slot].append(lbl)

    result = {}
    for slot, group in slot_groups.items():
        base_rgba = palette(slot / 19.0)
        base_hsv = rgb_to_hsv(np.array(base_rgba[:3]))
        if len(group) == 1:
            result[group[0]] = base_rgba
        else:
            for lbl, v in zip(group, np.linspace(0.45, 1.0, len(group))):
                hsv = base_hsv.copy()
                hsv[2] = v
                rgb = hsv_to_rgb(hsv)
                result[lbl] = (*rgb, 1.0)
    return result


def plot_tsne(embeddings, labels, id_to_verb, output_path=None, title="t-SNE",
              max_points=5000, perplexity=30):
    """t-SNE visualization with DBSCAN cluster labels and contrast colors.

    Args:
        embeddings: (N, D) numpy array of latent vectors.
        labels: (N,) integer class labels.
        id_to_verb: dict mapping label int -> verb string.
        output_path: save path (optional).
        title: plot title.
        max_points: subsample if N > this.
        perplexity: t-SNE perplexity.

    Returns:
        matplotlib Figure.
    """
    n = len(labels)
    if n > max_points:
        idx = np.random.RandomState(42).choice(n, max_points, replace=False)
        embeddings, labels = embeddings[idx], labels[idx]
        n = max_points

    tsne = TSNE(n_components=2, perplexity=min(perplexity, n // 5),
                random_state=42, n_iter=1000, learning_rate="auto", init="pca")
    coords = tsne.fit_transform(embeddings)

    unique_labels = sorted(set(labels))
    colors = _contrast_colors(coords, labels, unique_labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    for lbl in unique_labels:
        mask = labels == lbl
        verb = id_to_verb.get(lbl, str(lbl))
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[colors[lbl]], s=6, alpha=0.5, label=verb)

    ncol = 2 if len(unique_labels) <= 20 else 4
    ax.legend(markerscale=3, fontsize=8, ncol=ncol, loc="upper right",
              framealpha=0.8, handletextpad=0.3, columnspacing=0.5, borderpad=0.3)

    # DBSCAN cluster labels
    texts = []
    for lbl in unique_labels:
        verb = id_to_verb.get(lbl, str(lbl))
        mask = labels == lbl
        n_verb = mask.sum()
        if n_verb < 5:
            continue
        verb_coords = coords[mask]
        hex_c = mcolors.to_hex(colors[lbl])
        tc = _darken(hex_c)

        db = DBSCAN(eps=2.5, min_samples=max(3, int(n_verb * 0.05))).fit(verb_coords)
        cluster_ids = set(db.labels_) - {-1}
        cents = ([(verb_coords[:, 0].mean(), verb_coords[:, 1].mean())]
                 if not cluster_ids
                 else [(verb_coords[db.labels_ == cid, 0].mean(),
                        verb_coords[db.labels_ == cid, 1].mean())
                       for cid in cluster_ids])
        for cx, cy in cents:
            t = ax.text(cx, cy, verb, fontsize=8, color=tc,
                        ha="center", va="center", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.7))
            texts.append(t)

    if adjust_text is not None and texts:
        adjust_text(texts, ax=ax,
                    expand_text=(1.2, 1.4), force_text=(0.5, 0.8),
                    force_points=(0.1, 0.2),
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5))

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"t-SNE saved to {output_path}")
    return fig


# ── Symmetric confusion matrix ──────────────────────────────────────────────

def plot_symmetric_confusion(labels, preds, id_to_verb, output_path=None,
                             title="Symmetric Confusion Matrix", normalize=True):
    """Plot symmetrized confusion matrix: C_sym = (CM + CM^T) / 2.

    Symmetrization makes confusion(A,B) == confusion(B,A), which is useful
    for identifying confusable verb pairs regardless of prediction direction.

    Args:
        labels: (N,) ground truth integer labels.
        preds: (N,) predicted integer labels.
        id_to_verb: dict mapping label int -> verb string.
        output_path: save path (optional).
        title: plot title.
        normalize: if True, normalize rows to sum to 1 (before symmetrization).

    Returns:
        (fig, cm_sym) — matplotlib Figure and the symmetric confusion matrix.
    """
    present_labels = sorted(set(labels) | set(preds))
    target_names = [id_to_verb.get(i, str(i)) for i in present_labels]

    cm = confusion_matrix(labels, preds, labels=present_labels).astype(float)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums

    # Symmetrize: average CM and its transpose
    cm_sym = (cm + cm.T) / 2.0

    # Zero out diagonal for off-diagonal focus
    np.fill_diagonal(cm_sym, 0)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_sym, cmap="Reds", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = len(target_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(target_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(target_names, fontsize=8)

    # Annotate cells with values > threshold
    thresh = cm_sym.max() * 0.15
    for i in range(n):
        for j in range(i + 1, n):  # upper triangle only (symmetric)
            if cm_sym[i, j] > thresh:
                ax.text(j, i, f"{cm_sym[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if cm_sym[i, j] > cm_sym.max() * 0.5 else "black")
                ax.text(i, j, f"{cm_sym[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if cm_sym[i, j] > cm_sym.max() * 0.5 else "black")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Verb")
    ax.set_ylabel("Verb")
    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Symmetric confusion matrix saved to {output_path}")
    return fig, cm_sym


# ── Top confusable pairs ────────────────────────────────────────────────────

def get_top_confused_pairs(cm_sym, id_to_verb, present_labels, top_k=10):
    """Extract top-K most confused verb pairs from symmetric confusion matrix.

    Returns list of (verb_a, verb_b, confusion_score) tuples, sorted descending.
    """
    pairs = []
    n = cm_sym.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if cm_sym[i, j] > 0:
                va = id_to_verb.get(present_labels[i], str(present_labels[i]))
                vb = id_to_verb.get(present_labels[j], str(present_labels[j]))
                pairs.append((va, vb, cm_sym[i, j]))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:top_k]


# ── Checkpoint loading ─────────────────────────────────────────────────────

def load_checkpoint(model_path, device="cpu"):
    """Load a verb-probe checkpoint and extract state_dict + metadata.

    Handles both new-format ``{state_dict: ..., **meta}`` and legacy
    bare state_dict files.  Missing metadata fields are filled with
    sensible defaults so callers can rely on a stable key set.

    Returns:
        (state_dict, meta) where *meta* is a plain dict.
    """
    import torch
    raw = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(raw, dict) and "state_dict" in raw:
        meta = {k: v for k, v in raw.items() if k != "state_dict"}
        state_dict = raw["state_dict"]
    else:
        state_dict = raw
        classifier_bias_keys = [
            k for k in state_dict
            if k.startswith("classifier.") and k.endswith(".bias")]
        last_bias_key = sorted(
            classifier_bias_keys,
            key=lambda k: int(k.split(".")[1]))[-1]
        meta = {"num_verbs": state_dict[last_bias_key].shape[0]}

    defaults = {
        "num_verbs": None, "verb_to_id": None, "id_to_verb": None,
        "d_model": 128, "nhead": 8, "num_layers": 4,
        "action_dim": 7, "max_action_len": 64,
        "modality": "action_only", "action_rep": "native",
        "action_vocab_size": None,
        "image_encoder": "dinov2_s", "num_frames": 2, "delta_patches": 0,
        "dataset": "calvin",
    }
    for k, v in defaults.items():
        meta.setdefault(k, v)

    # Backward compat: old key names
    if meta["action_vocab_size"] is None:
        meta["action_vocab_size"] = meta.pop("fast_vocab_size", None)

    return state_dict, meta


# ── Evaluation report from predictions ────────────────────────────────────

def evaluate_predictions(all_labels, all_preds, id_to_verb,
                         title="", save_metrics=None, save_preds=None,
                         save_cm=None, analysis_dir=None,
                         embeddings=None):
    """Generate full evaluation output from raw predictions.

    Prints a classification report to stdout and optionally saves:
      - per-class metrics JSON (``save_metrics``)
      - raw predictions JSON (``save_preds``)
      - confusion matrix PNG (``save_cm``)
      - symmetric confusion + t-SNE in ``analysis_dir``

    Args:
        all_labels: list[int] ground-truth class ids.
        all_preds:  list[int] predicted class ids.
        id_to_verb: dict int -> str.
        title: descriptive string for plot titles.
        save_metrics / save_preds / save_cm: optional output paths.
        analysis_dir: if set, generate symmetric confusion + t-SNE there.
        embeddings: optional (N, D) numpy array for t-SNE.
    """
    all_labels = np.asarray(all_labels)
    all_preds = np.asarray(all_preds)
    accuracy = 100 * (all_labels == all_preds).mean()

    present_labels = sorted(set(all_labels.tolist()) | set(all_preds.tolist()))
    target_names = [id_to_verb.get(i, str(i)) for i in present_labels]

    print(f"\n{'=' * 60}")
    print(f"EVALUATION  {title}")
    print(f"Total examples: {len(all_preds)}")
    print(f"Overall accuracy: {accuracy:.2f}%")
    print(f"{'=' * 60}")

    report_str = classification_report(
        all_labels, all_preds, labels=present_labels,
        target_names=target_names, digits=3)
    print(f"\n{report_str}")

    # Metrics JSON
    if save_metrics:
        report_dict = classification_report(
            all_labels, all_preds, labels=present_labels,
            target_names=target_names, digits=4, output_dict=True)
        metrics = {
            "accuracy": accuracy,
            "num_examples": len(all_preds),
            "per_class": report_dict,
        }
        os.makedirs(os.path.dirname(save_metrics) or ".", exist_ok=True)
        with open(save_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {save_metrics}")

    # Predictions JSON
    if save_preds:
        os.makedirs(os.path.dirname(save_preds) or ".", exist_ok=True)
        with open(save_preds, "w") as f:
            json.dump({
                "labels": all_labels.tolist(),
                "preds": all_preds.tolist(),
                "id_to_verb": {str(k): v for k, v in id_to_verb.items()},
            }, f)
        print(f"Predictions saved to {save_preds}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=present_labels)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix {title}")
    plt.tight_layout()
    if save_cm:
        os.makedirs(os.path.dirname(save_cm) or ".", exist_ok=True)
        plt.savefig(save_cm, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_cm}")
    plt.close(fig)

    # Extended analysis
    if analysis_dir:
        os.makedirs(analysis_dir, exist_ok=True)
        id_to_verb_int = {int(k): v for k, v in id_to_verb.items()}

        fig_sym, cm_sym = plot_symmetric_confusion(
            all_labels, all_preds, id_to_verb_int,
            os.path.join(analysis_dir, "symmetric_confusion.png"),
            title=f"Symmetric Confusion {title}")
        plt.close(fig_sym)

        top_pairs = get_top_confused_pairs(
            cm_sym, id_to_verb_int,
            sorted(set(all_labels.tolist()) | set(all_preds.tolist())))
        print("\nTop confused pairs:")
        for va, vb, score in top_pairs:
            print("  {:15s} <-> {:15s}  {:.3f}".format(va, vb, score))

        if embeddings is not None:
            plot_tsne(embeddings, all_labels, id_to_verb_int,
                      os.path.join(analysis_dir, "tsne.png"),
                      title=f"t-SNE {title}")

    return accuracy


# ── Convenience: generate all plots ─────────────────────────────────────────

def plot_all(log_path=None, preds_path=None, embeddings_path=None,
             output_dir=".", title_prefix=""):
    """Generate all available plots from saved artifacts.

    Args:
        log_path: path to training log JSON (for training curves).
        preds_path: path to predictions JSON with {labels, preds, id_to_verb}.
        embeddings_path: path to .pt file with {embeddings: (N,D), labels: (N,)}.
        output_dir: directory to save figures.
        title_prefix: prepended to all titles.
    """
    os.makedirs(output_dir, exist_ok=True)

    if log_path and os.path.exists(log_path):
        plot_training_curves(log_path,
                             os.path.join(output_dir, "training_curves.png"),
                             title_prefix=title_prefix)

    id_to_verb = None
    if preds_path and os.path.exists(preds_path):
        with open(preds_path) as f:
            pred_data = json.load(f)
        labels = np.array(pred_data["labels"])
        preds = np.array(pred_data["preds"])
        id_to_verb = {int(k): v for k, v in pred_data["id_to_verb"].items()}

        fig, cm_sym = plot_symmetric_confusion(
            labels, preds, id_to_verb,
            os.path.join(output_dir, "symmetric_confusion.png"),
            title=f"{title_prefix}Symmetric Confusion")

        top_pairs = get_top_confused_pairs(
            cm_sym, id_to_verb, sorted(set(labels) | set(preds)))
        print(f"\nTop confused pairs:")
        for va, vb, score in top_pairs:
            print(f"  {va:15s} <-> {vb:15s}  {score:.3f}")

        plt.close(fig)

    if embeddings_path and os.path.exists(embeddings_path):
        import torch
        data = torch.load(embeddings_path, map_location="cpu")
        emb = data["embeddings"].numpy() if hasattr(data["embeddings"], "numpy") else data["embeddings"]
        lab = data["labels"].numpy() if hasattr(data["labels"], "numpy") else data["labels"]
        if id_to_verb is None:
            id_to_verb = data.get("id_to_verb", {i: str(i) for i in set(lab)})

        plot_tsne(emb, lab, id_to_verb,
                  os.path.join(output_dir, "tsne.png"),
                  title=f"{title_prefix}t-SNE")

    plt.close("all")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verb probe analysis plots")
    parser.add_argument("--log_path", type=str, default=None,
                        help="Training log JSON (for curves)")
    parser.add_argument("--preds_path", type=str, default=None,
                        help="Predictions JSON with {labels, preds, id_to_verb}")
    parser.add_argument("--embeddings_path", type=str, default=None,
                        help=".pt file with {embeddings, labels} (for t-SNE)")
    parser.add_argument("--output_dir", type=str, default="results/verb_probe/figures")
    parser.add_argument("--title_prefix", type=str, default="")
    args = parser.parse_args()

    plot_all(log_path=args.log_path, preds_path=args.preds_path,
             embeddings_path=args.embeddings_path, output_dir=args.output_dir,
             title_prefix=args.title_prefix)
