"""Generate confusion matrices for all 4 DROID verb classifiers.

Produces a 2x2 grid of confusion matrices (action d128, d256, goal DINOv2-S, VC-1).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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


def load_action_val(verb_to_id):
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


def load_goal_val(verb_to_id):
    df = pd.read_csv("data/droid_episodes_filtered.csv")
    vc = df["verb"].value_counts()
    keep = set(vc[vc >= 30].index) & set(verb_to_id.keys())
    df = df[df["verb"].isin(keep)].reset_index(drop=True)
    frames_index = build_frames_index("/data/user_data/wenjiel2/datasets/droid_frames")
    df = df[df["episode_idx"].isin(frames_index)].reset_index(drop=True)
    _, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["verb"])
    val_df = val_df.reset_index(drop=True)
    return val_df, frames_index


def get_predictions(ckpt_path, modality):
    ckpt = torch.load(ckpt_path, map_location=device)
    verb_to_id = ckpt["verb_to_id"]
    id_to_verb = ckpt["id_to_verb"]
    d_model = ckpt["d_model"]
    num_layers = ckpt.get("num_layers", 4)

    if modality == "action":
        val_df, actions_cache = load_action_val(verb_to_id)
        dataset = DroidVerbDataset(val_df, actions_cache, max_seq_len=512, verb_to_id=verb_to_id)
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
        model = ActionToVerbTransformer(
            num_verbs=ckpt["num_verbs"], d_model=d_model, num_layers=num_layers,
            action_dim=DROID_ACTION_DIM, max_action_len=512, img_size=224,
            modality="action_only", action_rep="native", cross_layers=0,
            image_encoder="scratch", action_vocab_size=None, freeze_vision=True,
            num_frames=2, delta_patches=0, modal_dropout=0.0,
            aux_loss_weight=0.0, scene_dim=0,
        ).to(device)
    else:
        image_encoder = ckpt.get("image_encoder", "dinov2_s")
        delta_patches = ckpt.get("delta_patches", 0)
        img_size = ckpt.get("img_size", 224)
        val_df, frames_index = load_goal_val(verb_to_id)
        dataset = DroidGoalDataset(val_df, frames_index, img_size=img_size, verb_to_id=verb_to_id)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        model = ActionToVerbTransformer(
            num_verbs=ckpt["num_verbs"], d_model=d_model, num_layers=num_layers,
            action_dim=7, max_action_len=1, img_size=img_size,
            modality="vision_only", action_rep="native", cross_layers=0,
            image_encoder=image_encoder, action_vocab_size=None, freeze_vision=True,
            num_frames=2, delta_patches=delta_patches, modal_dropout=0.0,
            aux_loss_weight=0.0, scene_dim=0,
        ).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for frames, actions, scene_vecs, labels, seq_lengths in tqdm(loader, desc=f"Predicting"):
            frames = frames.to(device)
            actions = actions.to(device)
            scene_vecs = scene_vecs.to(device)
            if modality == "action":
                seq_lengths = seq_lengths.to(device)
            else:
                seq_lengths = None
            logits = model(frames, actions, seq_lengths=seq_lengths, scene_vec=scene_vecs)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Convert to verb names
    pred_verbs = [id_to_verb[p] for p in all_preds]
    true_verbs = [id_to_verb[l] for l in all_labels]
    return true_verbs, pred_verbs, ckpt


def plot_confusion(ax, true_verbs, pred_verbs, title, verb_order):
    cm = confusion_matrix(true_verbs, pred_verbs, labels=verb_order)
    # Normalize by row (true class)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums
    # Symmetrize: average of A->B and B->A
    cm_sym = (cm_norm + cm_norm.T) / 2.0

    im = ax.imshow(cm_sym, interpolation="nearest", cmap="Blues", vmin=0, vmax=0.4)
    ax.set_title(title, fontsize=18, pad=14)
    ax.set_xticks(range(len(verb_order)))
    ax.set_yticks(range(len(verb_order)))
    ax.set_xticklabels(verb_order, rotation=90, fontsize=11, ha="center")
    ax.set_yticklabels(verb_order, fontsize=11)
    ax.set_xlabel("Verb B", fontsize=14)
    ax.set_ylabel("Verb A", fontsize=14)
    return im


# --- Main ---
print("Collecting predictions from all 4 models...")
results = []
for ckpt_path, label, modality in CKPTS:
    print(f"\n=== {label} ===")
    true_v, pred_v, ckpt = get_predictions(ckpt_path, modality)
    acc = sum(t == p for t, p in zip(true_v, pred_v)) / len(true_v) * 100
    print(f"  Val acc: {acc:.2f}%")
    results.append((true_v, pred_v, ckpt, label))

# Use consistent verb ordering (sorted, from d256 which has post-merge verbs)
_, _, d256_ckpt, _ = results[1]
verb_order = sorted(d256_ckpt["verb_to_id"].keys())
print(f"\nVerb order ({len(verb_order)} classes): {verb_order}")

fig, axes = plt.subplots(2, 2, figsize=(28, 24))
axes_flat = axes.flatten()

for ax, (true_v, pred_v, ckpt, label) in zip(axes_flat, results):
    acc = sum(t == p for t, p in zip(true_v, pred_v)) / len(true_v) * 100
    # Use this model's own verb set for its confusion matrix
    model_verbs = sorted(ckpt["verb_to_id"].keys())
    title = f"{label} (acc={acc:.1f}%)"
    im = plot_confusion(ax, true_v, pred_v, title, model_verbs)

fig.suptitle("DROID Verb Classification — Symmetrized Confusion Matrices",
             fontsize=22, y=0.99)
plt.tight_layout(rect=[0, 0, 0.88, 0.96])
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label="Avg(P(A→B), P(B→A))")
cbar_ax.set_ylabel("Avg(P(A→B), P(B→A))", fontsize=14)
cbar_ax.tick_params(labelsize=12)
plt.savefig("figures/droid_confusion_all.png", dpi=150, bbox_inches="tight")
print("\nSaved to figures/droid_confusion_all.png")
