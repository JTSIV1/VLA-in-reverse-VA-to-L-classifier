"""t-SNE visualization of DROID embeddings colored by verb.

Generates 2x2 grid: action d128, action d256, goal DINOv2-S, goal VC-1.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_droid import DroidVerbDataset, DROID_ACTION_DIM
from train_droid_goal import DroidGoalDataset, build_frames_index
from train_transformer import ActionToVerbTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPTS = [
    ("checkpoints/droid_ao_native_spwt_j6656090_best.pth", "Action d128 (4L)", "action"),
    ("checkpoints/droid_ao_native_d256_v2_j6657183_best.pth", "Action d256 (6L)", "action"),
    ("checkpoints/droid_goal_dinov2_s_j6660457_best.pth", "Goal DINOv2-S", "goal"),
    ("checkpoints/droid_goal_vc1_j6660458_best.pth", "Goal VC-1", "goal"),
]


def load_action_val_data(verb_to_id):
    """Load filtered CSV, split, and load action shards for val set."""
    df = pd.read_csv("data/droid_episodes_filtered.csv")
    vc = df["verb"].value_counts()
    keep = set(vc[vc >= 30].index) & set(verb_to_id.keys())
    df = df[df["verb"].isin(keep)].reset_index(drop=True)

    _, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["verb"])
    val_df = val_df.reset_index(drop=True)

    needed_eps = set(val_df["episode_idx"].tolist())
    actions_cache = {}
    shard_dir = "/data/user_data/wenjiel2/datasets/droid_actions"
    shard_files = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))
    global_idx = 0
    for sf in tqdm(shard_files, desc="Loading action shards"):
        data = np.load(sf, allow_pickle=True)
        n_eps = int(data["n_episodes"])
        for i in range(n_eps):
            if global_idx in needed_eps:
                actions_cache[f"actions_{global_idx}"] = data[f"actions_{i}"]
            global_idx += 1
        if len(actions_cache) >= len(needed_eps):
            break

    return val_df, actions_cache


def load_goal_val_data(verb_to_id):
    """Load filtered CSV, split, and build frames index for val set."""
    df = pd.read_csv("data/droid_episodes_filtered.csv")
    vc = df["verb"].value_counts()
    keep = set(vc[vc >= 30].index) & set(verb_to_id.keys())
    df = df[df["verb"].isin(keep)].reset_index(drop=True)

    frames_index = build_frames_index("/data/user_data/wenjiel2/datasets/droid_frames")
    has_frames = df["episode_idx"].isin(frames_index)
    df = df[has_frames].reset_index(drop=True)

    _, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["verb"])
    val_df = val_df.reset_index(drop=True)

    return val_df, frames_index


def extract_action_embeddings(ckpt_path):
    """Extract CLS embeddings from an action-only model."""
    ckpt = torch.load(ckpt_path, map_location=device)
    verb_to_id = ckpt["verb_to_id"]
    id_to_verb = ckpt["id_to_verb"]
    d_model = ckpt["d_model"]
    num_layers = ckpt.get("num_layers", 4)
    print(f"\n--- {ckpt_path} ---")
    print(f"d_model={d_model}, num_layers={num_layers}, "
          f"num_verbs={ckpt['num_verbs']}, best_val_acc={ckpt.get('best_val_acc', '?'):.1f}%")

    val_df, actions_cache = load_action_val_data(verb_to_id)
    print(f"Val episodes: {len(val_df)}")

    dataset = DroidVerbDataset(val_df, actions_cache, max_seq_len=512, verb_to_id=verb_to_id)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    model = ActionToVerbTransformer(
        num_verbs=ckpt["num_verbs"], d_model=d_model,
        num_layers=num_layers, action_dim=DROID_ACTION_DIM,
        max_action_len=512, img_size=224, modality="action_only",
        action_rep="native", cross_layers=0, image_encoder="scratch",
        action_vocab_size=None, freeze_vision=True, num_frames=2,
        delta_patches=0, modal_dropout=0.0, aux_loss_weight=0.0, scene_dim=0,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    all_embeds, all_labels = [], []
    with torch.no_grad():
        for frames, actions, scene_vecs, labels, seq_lengths in tqdm(loader, desc="Extracting"):
            actions = actions.to(device)
            seq_lengths = seq_lengths.to(device)

            B = actions.shape[0]
            action_tokens = model.action_proj(actions)
            action_tokens = action_tokens + model.action_pos[:, :action_tokens.size(1), :]
            action_tokens = action_tokens + model.type_action

            cls_tokens = model.cls_token.expand(B, -1, -1) + model.cls_pos + model.type_cls
            x = torch.cat([cls_tokens, action_tokens], dim=1)

            max_len = x.size(1)
            mask = torch.arange(max_len, device=device).unsqueeze(0) >= seq_lengths.unsqueeze(1)

            for layer in model.layers:
                x = layer(x, src_key_padding_mask=mask)

            all_embeds.append(x[:, 0, :].cpu().numpy())
            all_labels.extend(labels.cpu().tolist())

    embeds = np.concatenate(all_embeds, axis=0)
    verbs = [id_to_verb[l] for l in all_labels]
    print(f"Embeddings: {embeds.shape}")
    return embeds, verbs, ckpt


def extract_goal_embeddings(ckpt_path):
    """Extract CLS embeddings from a goal (vision-only) model."""
    ckpt = torch.load(ckpt_path, map_location=device)
    verb_to_id = ckpt["verb_to_id"]
    id_to_verb = ckpt["id_to_verb"]
    d_model = ckpt["d_model"]
    num_layers = ckpt.get("num_layers", 4)
    image_encoder = ckpt.get("image_encoder", "dinov2_s")
    delta_patches = ckpt.get("delta_patches", 0)
    img_size = ckpt.get("img_size", 224)
    print(f"\n--- {ckpt_path} ---")
    print(f"d_model={d_model}, num_layers={num_layers}, encoder={image_encoder}, "
          f"num_verbs={ckpt['num_verbs']}, best_val_acc={ckpt.get('best_val_acc', '?'):.1f}%")

    val_df, frames_index = load_goal_val_data(verb_to_id)
    print(f"Val episodes: {len(val_df)}")

    dataset = DroidGoalDataset(val_df, frames_index, img_size=img_size,
                                verb_to_id=verb_to_id)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4,
                        pin_memory=True)

    model = ActionToVerbTransformer(
        num_verbs=ckpt["num_verbs"], d_model=d_model,
        num_layers=num_layers, action_dim=7, max_action_len=1,
        img_size=img_size, modality="vision_only", action_rep="native",
        cross_layers=0, image_encoder=image_encoder,
        action_vocab_size=None, freeze_vision=True, num_frames=2,
        delta_patches=delta_patches, modal_dropout=0.0,
        aux_loss_weight=0.0, scene_dim=0,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    all_embeds = []
    all_labels = []
    cls_hook_output = []

    # Register hook on classifier's first layer to capture CLS embedding input
    hook = model.classifier[0].register_forward_hook(
        lambda m, inp, out: cls_hook_output.append(inp[0].cpu())
    )

    with torch.no_grad():
        for frames, actions, scene_vecs, labels, seq_lengths in tqdm(loader, desc="Extracting (goal)"):
            frames = frames.to(device)
            actions = actions.to(device)
            scene_vecs = scene_vecs.to(device)
            cls_hook_output.clear()
            # Vision-only models don't use seq_lengths for masking
            _ = model(frames, actions, seq_lengths=None, scene_vec=scene_vecs)
            all_embeds.append(cls_hook_output[0].numpy())
            all_labels.extend(labels.cpu().tolist())

    hook.remove()

    embeds = np.concatenate(all_embeds, axis=0)
    verbs = [id_to_verb[l] for l in all_labels]
    print(f"Embeddings: {embeds.shape}")
    return embeds, verbs, ckpt


def _labels_overlap(x1, y1, x2, y2, min_dx=8.0, min_dy=4.0):
    """Check if two label positions are too close."""
    return abs(x1 - x2) < min_dx and abs(y1 - y2) < min_dy


def _nudge_label(cx, cy, placed, min_dx=8.0, min_dy=4.0, max_tries=20):
    """Nudge a label position to avoid overlaps with already-placed labels."""
    if not any(_labels_overlap(cx, cy, px, py, min_dx, min_dy) for px, py in placed):
        return cx, cy
    # Try shifting in expanding spiral
    for r in range(1, max_tries):
        for dx, dy in [(r*2, 0), (-r*2, 0), (0, r*2), (0, -r*2),
                        (r*2, r*2), (-r*2, r*2), (r*2, -r*2), (-r*2, -r*2)]:
            nx, ny = cx + dx, cy + dy
            if not any(_labels_overlap(nx, ny, px, py, min_dx, min_dy) for px, py in placed):
                return nx, ny
    return cx, cy  # give up


def plot_tsne(ax, coords, verbs, top_verbs, colors, title):
    """Plot t-SNE on a given axis with text labels at cluster centers."""
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    verb_series = pd.Series(verbs)

    other_mask = ~verb_series.isin(top_verbs)
    ax.scatter(coords[other_mask, 0], coords[other_mask, 1],
               c="lightgray", s=4, alpha=0.25, zorder=1)

    placed_labels = []  # list of (x, y) for collision detection

    for i, verb in enumerate(top_verbs):
        mask = (verb_series == verb).values
        verb_coords = coords[mask]
        ax.scatter(verb_coords[:, 0], verb_coords[:, 1],
                   c=[colors[i]], s=15, alpha=0.5, label=verb, zorder=2)

        # Find clusters using DBSCAN
        verb_count = mask.sum()
        min_samples = max(5, int(verb_count * 0.05))
        nn = NearestNeighbors(n_neighbors=min(min_samples, len(verb_coords)))
        nn.fit(verb_coords)
        dists, _ = nn.kneighbors(verb_coords)
        eps = np.percentile(dists[:, -1], 70)
        eps = max(eps, 2.0)

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(verb_coords)
        cluster_labels = db.labels_
        unique_clusters = sorted(set(cluster_labels) - {-1})

        if len(unique_clusters) == 0:
            cluster_centers = [(verb_coords[:, 0].mean(),
                                verb_coords[:, 1].mean())]
        else:
            cluster_centers = []
            for cl in unique_clusters:
                cl_mask = cluster_labels == cl
                cluster_centers.append((
                    verb_coords[cl_mask, 0].mean(),
                    verb_coords[cl_mask, 1].mean()))

        for cx, cy in cluster_centers:
            cx, cy = _nudge_label(cx, cy, placed_labels)
            placed_labels.append((cx, cy))
            ax.text(cx, cy, verb, fontsize=13, fontweight="bold",
                    color=colors[i], ha="center", va="center", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="none", alpha=0.8))

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.legend(loc="upper right", fontsize=10, ncol=2, markerscale=2,
              framealpha=0.9)


# --- Main ---
results = []
for ckpt_path, label, modality in CKPTS:
    if modality == "action":
        embeds, verbs, ckpt = extract_action_embeddings(ckpt_path)
    else:
        embeds, verbs, ckpt = extract_goal_embeddings(ckpt_path)

    # Save embeddings for downstream analysis
    save_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    embed_path = f"results/{save_name}_embeddings.npz"
    np.savez_compressed(embed_path, embeddings=embeds, verbs=verbs)
    print(f"Saved embeddings to {embed_path}")

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    coords = tsne.fit_transform(embeds)
    results.append((coords, verbs, ckpt, label))

# Use top verbs from the d256 (post-merge) model for consistent coloring
_, d256_verbs, _, _ = results[1]
top_verbs = pd.Series(d256_verbs).value_counts().head(15).index.tolist()
colors = plt.cm.tab20(np.linspace(0, 1, 15))

fig, axes = plt.subplots(2, 2, figsize=(22, 20))
axes_flat = axes.flatten()

for ax, (coords, verbs, ckpt, label) in zip(axes_flat, results):
    acc = ckpt.get("best_val_acc", 0)
    title = f"{label}\n{len(verbs)} val episodes, best acc={acc:.1f}%"
    plot_tsne(ax, coords, verbs, top_verbs, colors, title)

fig.suptitle("DROID Embeddings (t-SNE) — Action vs Goal Models", fontsize=18, y=1.01)
plt.tight_layout()
plt.savefig("figures/droid_tsne_all.png", dpi=200, bbox_inches="tight")
print("\nSaved to figures/droid_tsne_all.png")
