"""
Cluster analysis of CALVIN action trajectories.
Builds features from action sequences, runs PCA + K-Means,
and saves plots + metrics to checkpoints/.

Usage:
    python cluster_analysis.py [--max_len 64] [--workers 8]
    sbatch run_cluster.sh
"""
import argparse
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import Counter
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

from config import TRAIN_DIR, ACTION_KEY, EPISODE_TEMPLATE, ACTION_DIM
from utils import load_calvin_to_dataframe


def _load_action(idx, data_dir, action_key, template):
    """Load a single frame's action vector (worker function for multiprocessing)."""
    path = os.path.join(data_dir, template.format(idx))
    return np.load(path)[action_key].astype(np.float32)


def load_all_actions(df, num_workers=8):
    # Collect all unique frame indices we actually need
    needed_indices = set()
    for s, e in zip(df['start_idx'].values, df['end_idx'].values):
        needed_indices.update(range(s, e + 1))
    needed_indices = sorted(needed_indices)
    print(f"Need to load {len(needed_indices)} unique frames using {num_workers} workers ...")

    t0 = time.time()
    load_fn = partial(
        _load_action,
        data_dir=TRAIN_DIR,
        action_key=ACTION_KEY,
        template=EPISODE_TEMPLATE,
    )

    with Pool(num_workers) as pool:
        # imap_unordered yields results as they finish -> tqdm can update
        it = pool.imap_unordered(load_fn, needed_indices, chunksize=1024)
        actions_list = list(tqdm(it, total=len(needed_indices), desc="Loading actions"))

    all_actions = np.asarray(actions_list, dtype=np.float32)
    elapsed = time.time() - t0
    print(f"Loaded {len(all_actions)} actions in {elapsed:.1f}s")
    print(f"Loaded {all_actions.shape} in {elapsed:.1f}s ({len(needed_indices)/elapsed:.0f} frames/s)")
    return all_actions, needed_indices


def build_features(df, max_len, num_workers=8):
    """Load action trajectories in parallel and build fixed-length feature matrix."""
    
    all_actions, needed_indices = load_all_actions(df, num_workers=num_workers)

    # Build index mapping
    idx_to_pos = {idx: i for i, idx in enumerate(needed_indices)}

    # Build feature matrix via pure slicing
    N = len(df)
    features = np.zeros((N, max_len * ACTION_DIM), dtype=np.float32)
    starts = df['start_idx'].values
    ends = df['end_idx'].values

    for i in range(N):
        s = idx_to_pos[starts[i]]
        e = idx_to_pos[ends[i]] + 1
        T = e - s
        traj = all_actions[s:e]
        if T >= max_len:
            traj = traj[:max_len]
        else:
            traj = np.pad(traj, ((0, max_len - T), (0, 0)), mode='constant')
        features[i] = traj.ravel()

    verb_labels = df['primary_verb'].tolist()
    return features, verb_labels


def run_pca(features, verb_labels, out_dir):
    """PCA to 2D and save scatter plot."""
    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(features)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    unique_verbs = sorted(set(verb_labels))
    cmap = plt.cm.get_cmap("tab20", len(unique_verbs))

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, v in enumerate(unique_verbs):
        mask = np.array([vl == v for vl in verb_labels])
        ax.scatter(pca_2d[mask, 0], pca_2d[mask, 1],
                   label=v, color=cmap(i), alpha=0.6, s=15)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of action trajectories (colored by ground-truth verb)")
    ax.legend(fontsize="small", ncol=3, loc="best", markerscale=2)
    plt.tight_layout()
    path = os.path.join(out_dir, "pca_trajectories.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    return pca_2d, unique_verbs, cmap


def run_kmeans(features, verb_labels, pca_2d, unique_verbs, cmap, out_dir):
    """K-Means clustering, metrics, and comparison plot."""
    n_clusters = len(unique_verbs)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(features)

    verb_ids = np.array([unique_verbs.index(v) for v in verb_labels])
    ari = adjusted_rand_score(verb_ids, cluster_ids)
    nmi = normalized_mutual_info_score(verb_ids, cluster_ids)
    print(f"K-Means (k={n_clusters})")
    print(f"  Adjusted Rand Index:          {ari:.3f}")
    print(f"  Normalized Mutual Information: {nmi:.3f}")

    # Side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for i, v in enumerate(unique_verbs):
        mask = np.array([vl == v for vl in verb_labels])
        axes[0].scatter(pca_2d[mask, 0], pca_2d[mask, 1],
                        label=v, color=cmap(i), alpha=0.6, s=15)
    axes[0].set_title("Ground-truth verb labels")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(fontsize="x-small", ncol=3, markerscale=2)

    scatter = axes[1].scatter(pca_2d[:, 0], pca_2d[:, 1],
                              c=cluster_ids, cmap="tab20", alpha=0.6, s=15)
    axes[1].set_title(f"K-Means clusters (k={n_clusters}, ARI={ari:.2f}, NMI={nmi:.2f})")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    plt.colorbar(scatter, ax=axes[1], label="Cluster ID")

    plt.tight_layout()
    path = os.path.join(out_dir, "kmeans_vs_ground_truth.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # Dominant verb per cluster
    print("\nDominant verb per cluster:\n")
    for c in range(n_clusters):
        mask = cluster_ids == c
        verbs_in_cluster = [verb_labels[j] for j in range(len(verb_labels)) if mask[j]]
        counts = Counter(verbs_in_cluster).most_common(3)
        top = ", ".join(f"{v} ({n})" for v, n in counts)
        print(f"  Cluster {c:2d} ({mask.sum():4d} samples): {top}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading annotations ...")
    df = load_calvin_to_dataframe(TRAIN_DIR)
    print(f"Episodes: {len(df)}, Unique verbs: {df['primary_verb'].nunique()}\n")

    features, verb_labels = build_features(df, args.max_len, args.workers)
    print(f"Feature matrix: {features.shape}  ({len(set(verb_labels))} unique verbs)\n")

    pca_2d, unique_verbs, cmap = run_pca(features, verb_labels, args.out_dir)
    print()
    run_kmeans(features, verb_labels, pca_2d, unique_verbs, cmap, args.out_dir)


if __name__ == "__main__":
    main()
