"""Visualize training logs and compare experiments.

Per-model output (one figure each):
  - 3-panel: overall loss/acc curves + per-class val accuracy + per-class val loss

Cross-model comparison (grid organized by action tokenization method):
  - comparison_bars.png: grouped bar chart grid, rows = action rep
  - comparison_curves.png: overlay curves grid, rows = action rep, cols = val loss / val acc
  - per_verb_accuracy.png: per-verb bar grid, rows = action rep, cols = modalities
  - per_verb_loss.png: per-verb loss bar grid, rows = action rep, cols = modalities
  - Console summary table

Usage:
    # Per-model plots only (legacy)
    python visualize_training.py --logs results/full_native_log.json

    # Structured comparison across tokenization methods
    python visualize_training.py \\
        --exp native full results/full_native_j1234_log.json results/full_native_j1234_metrics.json \\
        --exp native action_only results/action_only_j1234_log.json results/action_only_j1234_metrics.json \\
        --exp native vision_only results/vision_only_j1234_log.json results/vision_only_j1234_metrics.json \\
        --exp fast_v256 full results/full_fast_v256_j1235_log.json results/full_fast_v256_j1235_metrics.json \\
        --exp fast_v256 action_only results/ao_fast_v256_j1235_log.json results/ao_fast_v256_j1235_metrics.json \\
        --out_dir ./figures
"""

import os
import json
import argparse
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt


MODALITY_ORDER = ["full", "action_only", "vision_only"]
MODALITY_LABELS = {
    "full": "Action + Vision",
    "action_only": "Action Only",
    "vision_only": "Vision Only",
}
MODALITY_COLORS = {"full": "C0", "action_only": "C1", "vision_only": "C2"}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _safe_label(label):
    return label.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")


def _save_or_show(fig, out_dir, filename):
    if out_dir:
        fig.savefig(os.path.join(out_dir, filename), dpi=150, bbox_inches='tight')
        print(f"Saved {filename}")
    else:
        plt.show()
    plt.close(fig)


# ── Per-model: 3-panel figure ──────────────────────────────────────────────

def _get_per_class_series(log, split, metric, top_n):
    """Extract per-class time series for a given split/metric, ranked by count."""
    epochs_data = log["epochs"]
    key = f"per_class_{split}"

    all_verbs = set()
    for e in epochs_data:
        all_verbs.update(e[key].keys())

    last = epochs_data[-1][key]
    verbs_ranked = sorted(all_verbs,
                          key=lambda v: last.get(v, {}).get("count", 0),
                          reverse=True)
    verbs_show = verbs_ranked[:top_n]
    epochs = [e["epoch"] for e in epochs_data]

    series = {}
    for verb in verbs_show:
        vals = []
        for e in epochs_data:
            d = e[key].get(verb, {})
            vals.append(d.get(metric, np.nan) if d.get("count", 0) > 0 else np.nan)
        series[verb] = vals
    return epochs, series


def plot_per_model(log, label, top_n=10, out_dir=None):
    """3-panel figure: overall curves, per-class val acc lines, per-class val loss lines."""
    epochs_data = log["epochs"]
    epochs = [e["epoch"] for e in epochs_data]
    train_loss = [e["train_loss"] for e in epochs_data]
    val_loss = [e["val_loss"] for e in epochs_data]
    train_acc = [e["train_acc"] for e in epochs_data]
    val_acc = [e["val_acc"] for e in epochs_data]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(label, fontsize=14, y=1.02)

    # Panel 0: overall loss + accuracy (dual y-axis)
    ax_loss = axes[0]
    ax_acc = ax_loss.twinx()
    l1, = ax_loss.plot(epochs, train_loss, '-o', markersize=3, color='tab:red',
                       alpha=0.7, label='Train loss')
    l2, = ax_loss.plot(epochs, val_loss, '--s', markersize=3, color='tab:red',
                       label='Val loss')
    l3, = ax_acc.plot(epochs, train_acc, '-o', markersize=3, color='tab:blue',
                      alpha=0.7, label='Train acc')
    l4, = ax_acc.plot(epochs, val_acc, '--s', markersize=3, color='tab:blue',
                      label='Val acc')
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss", color='tab:red')
    ax_acc.set_ylabel("Accuracy (%)", color='tab:blue')
    ax_loss.set_title("Overall Loss & Accuracy")
    lines = [l1, l2, l3, l4]
    ax_loss.legend(lines, [l.get_label() for l in lines], fontsize=7, loc='center right')
    ax_loss.grid(True, alpha=0.3)

    # Panel 1: per-class val accuracy
    ep, acc_series = _get_per_class_series(log, "val", "acc", top_n)
    for verb, vals in acc_series.items():
        axes[1].plot(ep, vals, '-o', markersize=3, label=verb)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title(f"Per-Class Val Accuracy (top {top_n})")
    axes[1].legend(fontsize=7, ncol=2, loc='best')
    axes[1].grid(True, alpha=0.3)

    # Panel 2: per-class val loss
    ep, loss_series = _get_per_class_series(log, "val", "loss", top_n)
    for verb, vals in loss_series.items():
        axes[2].plot(ep, vals, '-o', markersize=3, label=verb)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_title(f"Per-Class Val Loss (top {top_n})")
    axes[2].legend(fontsize=7, ncol=2, loc='best')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, out_dir, f"model_{_safe_label(label)}.png")


# ── Grid helpers ───────────────────────────────────────────────────────────

def organize_experiments(exp_args):
    """Organize --exp arguments into {action_rep: {modality: {log, metrics}}}."""
    grid = OrderedDict()
    for action_rep, modality, log_path, metrics_path in exp_args:
        if action_rep not in grid:
            grid[action_rep] = {}
        log_data = load_json(log_path) if log_path.lower() not in ("none", "") else None
        met_data = load_json(metrics_path) if metrics_path.lower() not in ("none", "") else None
        grid[action_rep][modality] = {"log": log_data, "metrics": met_data}
    return grid


def _find_vision_only(grid):
    """Find vision_only experiment data from any group."""
    for exps in grid.values():
        if "vision_only" in exps:
            return exps["vision_only"]
    return None


def _get_cell_data(grid, action_rep, modality, row_idx):
    """Get experiment data for a grid cell, handling vision_only special case."""
    exps = grid[action_rep]
    if modality in exps:
        return exps[modality]
    # vision_only only falls through to other groups for the first row
    if modality == "vision_only" and row_idx == 0:
        return _find_vision_only(grid)
    return None


def _get_top_verbs(grid, top_n):
    """Get top verbs by average support across all experiments."""
    verb_support = {}
    for exps in grid.values():
        for data in exps.values():
            if data is None or data.get("metrics") is None:
                continue
            for verb, stats in data["metrics"]["per_class"].items():
                if verb in ("accuracy", "macro avg", "weighted avg"):
                    continue
                verb_support.setdefault(verb, []).append(stats.get("support", 0))
    return sorted(verb_support.keys(),
                  key=lambda v: np.mean(verb_support[v]),
                  reverse=True)[:top_n]


# ── comparison_bars.png ────────────────────────────────────────────────────

def plot_comparison_bars(grid, out_dir=None):
    """Grid of grouped bar charts: rows = action rep, bars = modalities x metrics."""
    action_reps = list(grid.keys())
    n_rows = len(action_reps)

    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 5 * n_rows), squeeze=False)

    for row_idx, action_rep in enumerate(action_reps):
        ax = axes[row_idx, 0]
        modalities_shown = []
        accs, macro_f1s, weighted_f1s = [], [], []

        for modality in MODALITY_ORDER:
            # vision_only only in first row
            if modality == "vision_only" and row_idx > 0:
                continue
            data = _get_cell_data(grid, action_rep, modality, row_idx)
            if data is None or data.get("metrics") is None:
                continue
            met = data["metrics"]
            modalities_shown.append(modality)
            accs.append(met["accuracy"])
            macro_f1s.append(met["per_class"]["macro avg"]["f1-score"] * 100)
            weighted_f1s.append(met["per_class"]["weighted avg"]["f1-score"] * 100)

        if not modalities_shown:
            ax.set_visible(False)
            continue

        labels = [MODALITY_LABELS[m] for m in modalities_shown]
        x = np.arange(len(modalities_shown))
        width = 0.25

        bars1 = ax.bar(x - width, accs, width, label='Accuracy (%)')
        bars2 = ax.bar(x, macro_f1s, width, label='Macro F1 (%)')
        bars3 = ax.bar(x + width, weighted_f1s, width, label='Weighted F1 (%)')

        ax.set_ylabel('Score (%)')
        ax.set_title(f'Action Representation: {action_rep}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        all_vals = accs + macro_f1s + weighted_f1s
        ax.set_ylim(0, max(all_vals) * 1.15 if all_vals else 100)

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=7)

    fig.suptitle("Model Comparison", fontsize=14)
    plt.tight_layout()
    _save_or_show(fig, out_dir, "comparison_bars.png")


# ── comparison_curves.png ──────────────────────────────────────────────────

def plot_comparison_curves(grid, out_dir=None):
    """Grid: rows = action rep, 2 cols (val loss, val acc), overlaying modalities."""
    action_reps = list(grid.keys())
    n_rows = len(action_reps)

    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows), squeeze=False)

    for row_idx, action_rep in enumerate(action_reps):
        for modality in MODALITY_ORDER:
            if modality == "vision_only" and row_idx > 0:
                continue
            data = _get_cell_data(grid, action_rep, modality, row_idx)
            if data is None or data.get("log") is None:
                continue

            log = data["log"]
            epochs_data = log["epochs"]
            epochs = [e["epoch"] for e in epochs_data]
            val_loss = [e["val_loss"] for e in epochs_data]
            val_acc = [e["val_acc"] for e in epochs_data]
            color = MODALITY_COLORS[modality]
            label = MODALITY_LABELS[modality]

            axes[row_idx, 0].plot(epochs, val_loss, '-o', markersize=3,
                                  color=color, label=label)
            axes[row_idx, 1].plot(epochs, val_acc, '-o', markersize=3,
                                  color=color, label=label)

        for col in range(2):
            axes[row_idx, col].set_xlabel("Epoch")
            axes[row_idx, col].grid(True, alpha=0.3)
            axes[row_idx, col].legend(fontsize=8)

        axes[row_idx, 0].set_ylabel("Loss")
        axes[row_idx, 0].set_title(f"Val Loss \u2014 {action_rep}")
        axes[row_idx, 1].set_ylabel("Accuracy (%)")
        axes[row_idx, 1].set_title(f"Val Accuracy \u2014 {action_rep}")

    fig.suptitle("Validation Curves Comparison", fontsize=14)
    plt.tight_layout()
    _save_or_show(fig, out_dir, "comparison_curves.png")


# ── Per-verb bar plots ─────────────────────────────────────────────────────

def plot_per_verb(grid, metric="accuracy", top_n=15, out_dir=None):
    """Grid: rows = action rep, cols = modalities. Each subplot: bar chart per verb."""
    action_reps = list(grid.keys())
    n_rows = len(action_reps)
    n_cols = len(MODALITY_ORDER)
    verbs = _get_top_verbs(grid, top_n)

    if not verbs:
        print(f"No verb data for per-verb {metric} plot.")
        return

    if metric == "accuracy":
        ylabel = "Recall / Per-Class Acc (%)"
        title = "Per-Verb Accuracy"
        filename = "per_verb_accuracy.png"
    else:
        ylabel = "Loss"
        title = "Per-Verb Loss"
        filename = "per_verb_loss.png"

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 4.5 * n_rows), squeeze=False)

    for row_idx, action_rep in enumerate(action_reps):
        for col_idx, modality in enumerate(MODALITY_ORDER):
            ax = axes[row_idx, col_idx]

            # vision_only only in first row
            if modality == "vision_only" and row_idx > 0:
                ax.set_visible(False)
                continue

            data = _get_cell_data(grid, action_rep, modality, row_idx)
            if data is None:
                ax.set_visible(False)
                continue

            if metric == "accuracy":
                if data.get("metrics") is None:
                    ax.set_visible(False)
                    continue
                pc = data["metrics"]["per_class"]
                values = [pc.get(v, {}).get("recall", 0) * 100 for v in verbs]
            else:  # loss
                if data.get("log") is None:
                    ax.set_visible(False)
                    continue
                last_epoch = data["log"]["epochs"][-1]
                pc_val = last_epoch.get("per_class_val", {})
                values = [pc_val.get(v, {}).get("loss", 0) for v in verbs]

            x = np.arange(len(verbs))
            color = MODALITY_COLORS[modality]
            ax.bar(x, values, color=color, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(verbs, rotation=45, ha='right', fontsize=7)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{MODALITY_LABELS[modality]} \u2014 {action_rep}", fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    _save_or_show(fig, out_dir, filename)


# ── Console table ──────────────────────────────────────────────────────────

def print_summary_table(grid):
    """Print console summary table of all experiments."""
    header = f"{'Action Rep':<15} {'Modality':<18} {'Acc%':>6} {'MacroF1%':>9} {'WtdF1%':>8} {'N':>6}"
    print("\n" + "=" * len(header))
    print("MODEL COMPARISON SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for action_rep, exps in grid.items():
        for modality in MODALITY_ORDER:
            if modality not in exps:
                continue
            data = exps[modality]
            if data is None or data.get("metrics") is None:
                continue
            met = data["metrics"]
            acc = met["accuracy"]
            macro_f1 = met["per_class"]["macro avg"]["f1-score"] * 100
            wtd_f1 = met["per_class"]["weighted avg"]["f1-score"] * 100
            n = met["num_examples"]
            print(f"{action_rep:<15} {MODALITY_LABELS[modality]:<18} "
                  f"{acc:>6.2f} {macro_f1:>9.2f} {wtd_f1:>8.2f} {n:>6d}")
    print("=" * len(header) + "\n")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize training logs and compare models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Per-model plots (legacy / standalone)
    parser.add_argument("--logs", nargs="+", default=None,
                        help="Training log JSON(s) for per-model 3-panel plots")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Display labels for --logs (default: from filename)")

    # Structured comparison grid
    parser.add_argument("--exp", nargs=4, action="append", default=None,
                        metavar=("ACTION_REP", "MODALITY", "LOG", "METRICS"),
                        help="Add experiment: action_rep modality log_path metrics_path "
                             "(use 'none' for missing log or metrics)")

    parser.add_argument("--out_dir", type=str, default=None,
                        help="Save plots to this directory instead of showing")
    parser.add_argument("--top_n", type=int, default=15,
                        help="Number of top verbs in per-class/per-verb plots")
    args = parser.parse_args()

    if not args.logs and not args.exp:
        parser.error("Provide --logs and/or --exp")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    # --- Per-model 3-panel figures (from --logs) ---
    if args.logs:
        if args.labels is None:
            args.labels = [os.path.splitext(os.path.basename(p))[0].replace("_log", "")
                           for p in args.logs]
        for path, label in zip(args.logs, args.labels):
            log = load_json(path)
            plot_per_model(log, label, top_n=args.top_n, out_dir=args.out_dir)

    # --- Structured grid comparison (from --exp) ---
    if args.exp:
        grid = organize_experiments(args.exp)

        # Per-model plots for each experiment in the grid
        for action_rep, exps in grid.items():
            for modality, data in exps.items():
                if data.get("log") is not None:
                    label = f"{MODALITY_LABELS[modality]} ({action_rep})"
                    plot_per_model(data["log"], label,
                                   top_n=args.top_n, out_dir=args.out_dir)

        # Grid comparisons
        plot_comparison_bars(grid, out_dir=args.out_dir)
        plot_comparison_curves(grid, out_dir=args.out_dir)
        plot_per_verb(grid, metric="accuracy", top_n=args.top_n, out_dir=args.out_dir)
        plot_per_verb(grid, metric="loss", top_n=args.top_n, out_dir=args.out_dir)
        print_summary_table(grid)


if __name__ == "__main__":
    main()
