"""Plot training curves for CALVIN-D tokenizer sweep."""
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

CKPT_DIR = "checkpoints/calvind_sweep"
OUT_DIR = "lab_notebooks/calvind_sweep"

def load_metrics(pattern):
    """Load all metrics.csv matching glob pattern, return {name: df}."""
    results = {}
    for path in sorted(glob.glob(os.path.join(CKPT_DIR, pattern, "metrics.csv"))):
        name = os.path.basename(os.path.dirname(path))
        try:
            df = pd.read_csv(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if len(df) > 1:  # skip empty/header-only
            results[name] = df
    return results


def plot_vanilla_comparison(save_path):
    """Row 1: Vanilla recon curves for all 3 tokenizers."""
    data = load_metrics("*_vanilla")
    if not data:
        print("No vanilla data found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = {"vq_bet_vanilla": "#1f77b4", "oat_vanilla": "#ff7f0e", "quest_vanilla": "#2ca02c"}
    labels = {"vq_bet_vanilla": "VQ-BeT (57K)", "oat_vanilla": "OAT (5.8M)", "quest_vanilla": "QueST (6.5M)"}

    for name, df in data.items():
        c = colors.get(name, "gray")
        lb = labels.get(name, name)
        axes[0].plot(df["epoch"], df["train_recon"], color=c, alpha=0.4, linewidth=0.8)
        axes[0].plot(df["epoch"], df["val_recon"], color=c, label=lb, linewidth=1.5)
        axes[1].semilogy(df["epoch"], df["val_recon"], color=c, label=lb, linewidth=1.5)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Reconstruction Loss")
    axes[0].set_title("Train (faint) / Val (solid) Recon Loss")
    axes[0].legend()
    axes[0].set_ylim(bottom=0)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Recon Loss (log)")
    axes[1].set_title("Val Recon Loss (log scale)")
    axes[1].legend()

    fig.suptitle("Vanilla Tokenizer Training — CALVIN D→D", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_verb_sweep(save_path):
    """Row 2: Verb lambda sweep — recon vs verb acc trade-off per tokenizer."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    tokenizers = [("vq_bet", "VQ-BeT"), ("oat", "OAT"), ("quest", "QueST")]
    cmap = plt.cm.viridis

    for ax, (tok, tok_label) in zip(axes, tokenizers):
        # Vanilla baseline
        vanilla = load_metrics(f"{tok}_vanilla")
        verb_runs = load_metrics(f"{tok}_verb*")

        all_runs = {}
        if vanilla:
            key = list(vanilla.keys())[0]
            all_runs["λ=0 (vanilla)"] = vanilla[key]
        for name, df in verb_runs.items():
            # Extract lambda from name like "vq_bet_verb0.5_verb0.5"
            parts = name.split("_verb")
            if len(parts) >= 2:
                lam = parts[1].split("_")[0]
                all_runs[f"λ={lam}"] = df

        if not all_runs:
            ax.set_title(f"{tok_label} — no data yet")
            continue

        n = len(all_runs)
        for i, (label, df) in enumerate(sorted(all_runs.items())):
            color = cmap(i / max(n - 1, 1))

            # Plot val recon
            ax.plot(df["epoch"], df["val_recon"], color=color, label=label, linewidth=1.2)

            # If has verb acc, plot on twin axis
            if df["val_verb_acc"].max() > 0:
                ax2 = ax.twinx()
                ax2.plot(df["epoch"], df["val_verb_acc"], color=color,
                         linestyle='--', alpha=0.6, linewidth=1.0)
                ax2.set_ylabel("Val Verb Acc (%)", color='gray')
                ax2.set_ylim(0, 50)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val Recon Loss")
        ax.set_title(f"{tok_label} — Verb λ Sweep")
        ax.legend(fontsize=7, loc='upper right')
        ax.set_ylim(bottom=0)

    fig.suptitle("Verb Classification λ Sweep — Recon (solid) / Verb Acc (dashed)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_clip_sweep(save_path):
    """Row 3: CLIP lambda sweep — recon vs CLIP loss per tokenizer."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    tokenizers = [("vq_bet", "VQ-BeT"), ("oat", "OAT"), ("quest", "QueST")]
    cmap = plt.cm.plasma

    for ax, (tok, tok_label) in zip(axes, tokenizers):
        vanilla = load_metrics(f"{tok}_vanilla")
        clip_runs = load_metrics(f"{tok}_clip*")

        all_runs = {}
        if vanilla:
            key = list(vanilla.keys())[0]
            all_runs["λ=0 (vanilla)"] = vanilla[key]
        for name, df in clip_runs.items():
            parts = name.split("_clip")
            if len(parts) >= 2:
                lam = parts[1].split("_")[0]
                all_runs[f"λ={lam}"] = df

        if not all_runs:
            ax.set_title(f"{tok_label} — no data yet")
            continue

        n = len(all_runs)
        for i, (label, df) in enumerate(sorted(all_runs.items())):
            color = cmap(i / max(n - 1, 1))
            ax.plot(df["epoch"], df["val_recon"], color=color, label=label, linewidth=1.2)

            if df["val_clip"].max() > 0:
                ax2 = ax.twinx()
                ax2.plot(df["epoch"], df["val_clip"], color=color,
                         linestyle='--', alpha=0.6, linewidth=1.0)
                ax2.set_ylabel("Val CLIP Loss", color='gray')

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val Recon Loss")
        ax.set_title(f"{tok_label} — CLIP λ Sweep")
        ax.legend(fontsize=7, loc='upper right')
        ax.set_ylim(bottom=0)

    fig.suptitle("CLIP Contrastive λ Sweep — Recon (solid) / CLIP Loss (dashed)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_tradeoff_summary(save_path):
    """Scatter: best val recon vs best val verb acc for all completed runs."""
    all_data = load_metrics("*")
    if not all_data:
        print("No data found")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    markers = {"vq_bet": "o", "oat": "s", "quest": "D"}
    colors_type = {"vanilla": "#888888", "verb": "#d62728", "clip": "#1f77b4"}

    for name, df in all_data.items():
        # Determine tokenizer and run type
        tok = name.split("_vanilla")[0].split("_verb")[0].split("_clip")[0]
        if "vanilla" in name:
            rtype = "vanilla"
        elif "verb" in name:
            rtype = "verb"
        elif "clip" in name:
            rtype = "clip"
        else:
            rtype = "vanilla"

        best_recon = df["val_recon"].min()
        best_verb = df["val_verb_acc"].max()

        marker = markers.get(tok, "x")
        color = colors_type.get(rtype, "gray")

        ax.scatter(best_recon, best_verb, marker=marker, color=color, s=80,
                   edgecolors='black', linewidth=0.5, zorder=3)
        # Label with lambda
        if rtype != "vanilla":
            parts = name.split(f"_{rtype}")
            if len(parts) >= 2:
                lam = parts[1].split("_")[0]
                ax.annotate(f"λ={lam}", (best_recon, best_verb),
                           fontsize=6, ha='left', va='bottom',
                           xytext=(3, 3), textcoords='offset points')

    # Legend
    from matplotlib.lines import Line2D
    tok_handles = [Line2D([0], [0], marker=m, color='gray', linestyle='',
                          markersize=8, label=t.replace("_", "-").upper())
                   for t, m in markers.items()]
    type_handles = [Line2D([0], [0], marker='o', color=c, linestyle='',
                           markersize=8, label=t)
                    for t, c in colors_type.items()]
    ax.legend(handles=tok_handles + type_handles, loc='upper left', fontsize=8)

    ax.set_xlabel("Best Val Reconstruction Loss")
    ax.set_ylabel("Best Val Verb Accuracy (%)")
    ax.set_title("Recon vs Verb Decodability Trade-off", fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    plot_vanilla_comparison(os.path.join(OUT_DIR, "vanilla_training.png"))
    plot_verb_sweep(os.path.join(OUT_DIR, "verb_sweep.png"))
    plot_clip_sweep(os.path.join(OUT_DIR, "clip_sweep.png"))
    plot_tradeoff_summary(os.path.join(OUT_DIR, "recon_vs_verb_tradeoff.png"))

    print("\nAll plots saved to", OUT_DIR)
