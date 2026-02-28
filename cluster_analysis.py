"""
Cluster analysis of CALVIN action trajectories and image embeddings.
Builds features from action sequences or frozen image encoder outputs,
runs PCA + clustering, and saves plots + metrics.

Usage:
    # Action-based clustering (existing)
    python cluster_analysis.py [--max_len 64] [--workers 8]

    # Image-based clustering (new)
    python cluster_analysis.py --feature_source images --image_encoder resnet18 --delta_patches 16

    sbatch run_cluster.sh
"""

import argparse
import os
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    confusion_matrix,
)
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
from multiprocessing import Pool
from functools import partial

from config import (
    TRAIN_DIR,
    ACTION_KEY,
    IMAGE_KEY,
    EPISODE_TEMPLATE,
    ACTION_DIM,
    FAST_TOKENIZER_PATH,
    TOKENIZER_HORIZON,
    IMAGE_SIZE,
)
from utils import load_calvin_to_dataframe
from fast_tokenizer import load_fast_tokenizer, tokenize_trajectory


def _load_action(idx, data_dir, action_key, template):
    """Load a single frame's action vector (worker function for multiprocessing)."""
    path = os.path.join(data_dir, template.format(idx))
    return np.load(path)[action_key].astype(np.float32)


def load_all_actions(df, num_workers=8):
    """Load every unique action frame referenced by df.

    Returns:
        all_actions  : np.ndarray of shape (N_unique_frames, ACTION_DIM)
        idx_to_pos   : dict mapping original frame index → row in all_actions
    """
    needed_indices = set()
    for s, e in zip(df["start_idx"].values, df["end_idx"].values):
        needed_indices.update(range(s, e + 1))
    needed_indices = sorted(needed_indices)

    load_fn = partial(
        _load_action,
        data_dir=TRAIN_DIR,
        action_key=ACTION_KEY,
        template=EPISODE_TEMPLATE,
    )
    with Pool(num_workers) as pool:
        actions_list = pool.map(load_fn, needed_indices, chunksize=1024)

    all_actions = np.array(actions_list)
    idx_to_pos = {idx: i for i, idx in enumerate(needed_indices)}
    return all_actions, idx_to_pos


def build_features(df, max_len, num_workers, action_rep="native", tokenizer=None):
    """Load action trajectories in parallel and build fixed-length feature matrix."""
    # Collect all unique frame indices we actually need
    needed_indices = set()
    for s, e in zip(df["start_idx"].values, df["end_idx"].values):
        needed_indices.update(range(s, e + 1))
    needed_indices = sorted(needed_indices)
    print(
        f"Need to load {len(needed_indices)} unique frames using {num_workers} workers ..."
    )

    t0 = time.time()
    load_fn = partial(
        _load_action,
        data_dir=TRAIN_DIR,
        action_key=ACTION_KEY,
        template=EPISODE_TEMPLATE,
    )

    with Pool(num_workers) as pool:
        actions_list = pool.map(load_fn, needed_indices, chunksize=1024)

    all_actions = np.array(actions_list)  # (num_unique_frames, 7)
    elapsed = time.time() - t0
    print(
        f"Loaded {all_actions.shape} in {elapsed:.1f}s ({len(needed_indices) / elapsed:.0f} frames/s)"
    )

    # Build index mapping
    idx_to_pos = {idx: i for i, idx in enumerate(needed_indices)}

    # Build feature matrix via pure slicing
    N = len(df)
    if action_rep == "native":
        features = np.zeros((N, max_len * ACTION_DIM), dtype=np.float32)
    elif action_rep == "fast":
        features = np.zeros((N, max_len), dtype=np.float32)
    else:
        dummy_traj = np.zeros((max_len, ACTION_DIM), dtype=np.float32)
        dummy_tokens = tokenizer(dummy_traj)
        if (
            isinstance(dummy_tokens, list)
            and len(dummy_tokens) > 0
            and isinstance(dummy_tokens[0], list)
        ):
            dummy_tokens = dummy_tokens[0]
        feat_len = len(dummy_tokens)
        features = np.zeros((N, feat_len), dtype=np.float32)

    starts = df["start_idx"].values
    ends = df["end_idx"].values

    for i in range(N):
        s = idx_to_pos[starts[i]]
        e = idx_to_pos[ends[i]] + 1
        T = e - s
        traj = all_actions[s:e]

        if action_rep == "fast":
            tokens = tokenize_trajectory(tokenizer, traj)
            L = len(tokens)
            if L >= max_len:
                tokens = tokens[:max_len]
            else:
                tokens = tokens + [0] * (max_len - L)
            features[i] = tokens
        elif action_rep in ("bin", "quest", "oat"):
            token_ids = tokenizer(traj)
            if (
                isinstance(token_ids, list)
                and len(token_ids) > 0
                and isinstance(token_ids[0], list)
            ):
                token_ids = token_ids[0]
            features[i] = token_ids
        else:
            if T >= max_len:
                traj = traj[:max_len]
            else:
                traj = np.pad(traj, ((0, max_len - T), (0, 0)), mode="constant")
            features[i] = traj.ravel()

    verb_labels = df["primary_verb"].tolist()
    return features, verb_labels


# ---------------------------------------------------------------------------
# Image-based feature extraction (frozen encoders)
# ---------------------------------------------------------------------------


def _build_encoder(encoder_name):
    """Build a frozen image encoder and return (encoder, embed_dim, num_tokens).

    Uses build_image_encoder from image_encoders.py.
    We call encoder.get_embeddings() for raw backbone features (no random proj).
    """
    from image_encoders import build_image_encoder

    encoder = build_image_encoder(
        encoder_name, d_model=128
    )  # d_model irrelevant for get_embeddings
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    embed_dim = getattr(encoder, "embed_dim", 128)  # raw backbone dim
    num_tokens = encoder.num_tokens
    return encoder, embed_dim, num_tokens


def build_image_features(df, encoder_name, delta_patches=16, batch_size=64):
    """Extract frozen image-encoder features for first+last frames of each episode.

    Returns:
        features   : np.ndarray (N_episodes, feature_dim)
        verb_labels: list[str]
    """
    import torch
    from PIL import Image
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Image feature extraction on device: {device}")

    encoder, embed_dim, num_tokens = _build_encoder(encoder_name)
    encoder.to(device)

    # Image transform: match what train_transformer.py uses
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE[0], IMAGE_SIZE[1])),
            transforms.ToTensor(),
        ]
    )

    N = len(df)
    starts = df["start_idx"].values
    ends = df["end_idx"].values

    use_delta = delta_patches > 0 and encoder_name != "r3m"
    if encoder_name == "r3m" and delta_patches > 0:
        print(
            "  [NOTE] R3M produces a global vector; ignoring delta_patches, using full mode."
        )

    if use_delta:
        feat_dim = delta_patches * embed_dim
        print(
            f"  Delta mode: top-{delta_patches} patches × {embed_dim}d = {feat_dim}d per episode"
        )
    else:
        # Concatenate both frames
        feat_dim = 2 * num_tokens * embed_dim
        print(
            f"  Full mode: 2 frames × {num_tokens} tokens × {embed_dim}d = {feat_dim}d per episode"
        )

    features = np.zeros((N, feat_dim), dtype=np.float32)

    print(f"  Extracting features for {N} episodes (batch_size={batch_size}) ...")
    t0 = time.time()

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        first_frames = []
        last_frames = []

        for i in range(batch_start, batch_end):
            s_idx, e_idx = int(starts[i]), int(ends[i])
            # Load first frame
            path_first = os.path.join(TRAIN_DIR, EPISODE_TEMPLATE.format(s_idx))
            img_first = Image.fromarray(np.load(path_first)[IMAGE_KEY]).convert("RGB")
            first_frames.append(transform(img_first))
            # Load last frame
            path_last = os.path.join(TRAIN_DIR, EPISODE_TEMPLATE.format(e_idx))
            img_last = Image.fromarray(np.load(path_last)[IMAGE_KEY]).convert("RGB")
            last_frames.append(transform(img_last))

        first_batch = torch.stack(first_frames).to(device)  # (B,3,H,W)
        last_batch = torch.stack(last_frames).to(device)

        with torch.no_grad():
            emb_first = encoder.get_embeddings(
                first_batch
            )  # (B, num_tokens, embed_dim)
            emb_last = encoder.get_embeddings(last_batch)

        if use_delta:
            diff = emb_last - emb_first  # (B, num_tokens, embed_dim)
            mag = diff.norm(dim=-1)  # (B, num_tokens)
            K = min(delta_patches, diff.size(1))
            topk_idx = mag.topk(K, dim=-1).indices  # (B, K)
            idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, embed_dim)
            selected = torch.gather(diff, 1, idx_exp)  # (B, K, embed_dim)
            feat = selected.reshape(selected.size(0), -1).cpu().numpy()
        else:
            feat = torch.cat(
                [emb_first, emb_last], dim=1
            )  # (B, 2*num_tokens, embed_dim)
            feat = feat.reshape(feat.size(0), -1).cpu().numpy()

        features[batch_start:batch_end] = feat

        if (batch_start // batch_size) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    {batch_end}/{N} episodes ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Done: {N} episodes in {elapsed:.1f}s ({N / elapsed:.0f} eps/s)")

    verb_labels = df["primary_verb"].tolist()
    return features, verb_labels


def run_pca(features, verb_labels, out_dir, prefix="native"):
    """Scale data, compute full PCA for 99% variance, then run 2D PCA for scatter plot.

    Returns scaled_features, pca_99_features, pca_2d, unique_verbs, cmap, pca_metrics.
    """
    # Scale data (Standardization)
    print("Normalizing features via StandardScaler...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Check components needed for 99% variance AND project into that subspace
    print("Computing PCA components needed for 99% variance...")
    pca_99 = PCA(n_components=0.99, svd_solver="full")
    pca_99_features = pca_99.fit_transform(scaled_features)
    comps_99 = pca_99.n_components_
    print(f"PCA components needed for 99% variance: {comps_99}")

    # 2D PCA for visualization
    print("Computing 2D PCA for visualization...")
    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(scaled_features)
    variance_2d = pca.explained_variance_ratio_.sum()
    print(f"2D PCA explained variance: {variance_2d:.1%}")

    unique_verbs = sorted(set(verb_labels))
    cmap = plt.cm.get_cmap("tab20", len(unique_verbs))

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, v in enumerate(unique_verbs):
        mask = np.array([vl == v for vl in verb_labels])
        ax.scatter(
            pca_2d[mask, 0], pca_2d[mask, 1], label=v, color=cmap(i), alpha=0.6, s=15
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of action trajectories (colored by ground-truth verb)")
    ax.legend(fontsize="small", ncol=3, loc="best", markerscale=2)
    plt.tight_layout()
    path = os.path.join(out_dir, f"pca_trajectories_{prefix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    return (
        scaled_features,
        pca_99_features,
        pca_2d,
        unique_verbs,
        cmap,
        {
            "variance_99_comps": int(comps_99) if comps_99 is not None else -1,
            "variance_2d": float(variance_2d),
        },
    )


def run_clustering(
    features,
    verb_labels,
    pca_2d,
    unique_verbs,
    cmap,
    out_dir,
    prefix="native",
    cluster_method="kmeans",
):
    """Cluster features with the chosen algorithm, compute metrics, and save plot.

    cluster_method: 'kmeans' | 'agglomerative'  (Ward linkage)
    """
    n_clusters = len(unique_verbs)

    if cluster_method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif cluster_method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    else:
        raise ValueError(f"Unknown cluster_method: {cluster_method!r}")

    cluster_ids = model.fit_predict(features)

    verb_ids = np.array([unique_verbs.index(v) for v in verb_labels])
    ari = adjusted_rand_score(verb_ids, cluster_ids)
    nmi = normalized_mutual_info_score(verb_ids, cluster_ids)

    # Purity
    cm = confusion_matrix(verb_ids, cluster_ids)
    purity = np.sum(np.amax(cm, axis=0)) / np.sum(cm)

    # Silhouette Score
    silhouette = silhouette_score(features, cluster_ids)

    print(f"K-Means (k={n_clusters})")
    print(f"  Adjusted Rand Index:          {ari:.3f}")
    print(f"  Normalized Mutual Information: {nmi:.3f}")
    print(f"  Silhouette Score:             {silhouette:.3f}")
    print(f"  Purity:                       {purity:.3f}")

    # Side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for i, v in enumerate(unique_verbs):
        mask = np.array([vl == v for vl in verb_labels])
        axes[0].scatter(
            pca_2d[mask, 0], pca_2d[mask, 1], label=v, color=cmap(i), alpha=0.6, s=15
        )
    axes[0].set_title("Ground-truth verb labels")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(fontsize="x-small", ncol=3, markerscale=2)

    scatter = axes[1].scatter(
        pca_2d[:, 0], pca_2d[:, 1], c=cluster_ids, cmap="tab20", alpha=0.6, s=15
    )
    axes[1].set_title(
        f"K-Means clusters (k={n_clusters}, ARI={ari:.2f}, NMI={nmi:.2f})"
    )
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    plt.colorbar(scatter, ax=axes[1], label="Cluster ID")

    plt.tight_layout()
    path = os.path.join(out_dir, f"clusters_vs_ground_truth_{prefix}.png")
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

    return {
        "num_clusters": n_clusters,
        "ari": float(ari),
        "nmi": float(nmi),
        "silhouette": float(silhouette),
        "purity": float(purity),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./results/clustering",
        help="Base output dir. Results go to <out_dir>/<feature_source>/<pca|raw>/<cluster_method>/",
    )
    parser.add_argument(
        "--feature_source",
        type=str,
        choices=["actions", "images"],
        default="actions",
        help="Feature source: 'actions' (default) or 'images' (frozen encoder embeddings)",
    )
    # ── Action-specific args ──
    parser.add_argument(
        "--action_rep",
        type=str,
        choices=["native", "fast", "bin", "quest", "oat"],
        default="native",
    )
    parser.add_argument("--tokenizer_path", type=str, default=FAST_TOKENIZER_PATH)
    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--scale", type=float, default=10)
    # ── Image-specific args ──
    parser.add_argument(
        "--image_encoder",
        type=str,
        choices=["resnet18", "dinov2", "dinov2_s", "dinov2_b", "vc1", "r3m"],
        default="resnet18",
        help="Image encoder backbone (only used when --feature_source images)",
    )
    parser.add_argument(
        "--delta_patches",
        type=int,
        default=16,
        help="Top-K changed patches between first/last frame (0 = use all patches from both frames)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for image feature extraction",
    )
    # ── Clustering args ──
    parser.add_argument(
        "--cluster_method",
        type=str,
        choices=["kmeans", "agglomerative"],
        default="kmeans",
        help="Clustering algorithm: 'kmeans' or 'agglomerative' (Ward linkage)",
    )
    parser.add_argument(
        "--use_pca",
        action="store_true",
        default=True,
        help="Cluster on 99%%-variance PCA-projected features (recommended, default True)",
    )
    parser.add_argument(
        "--no_use_pca",
        dest="use_pca",
        action="store_false",
        help="Cluster on raw StandardScaled features instead of PCA-projected",
    )
    args = parser.parse_args()

    # Build structured output dir: <base>/<feature_source>/<pca_or_raw>/<cluster_method>/
    pca_tag = "pca" if args.use_pca else "raw"
    run_out_dir = os.path.join(
        args.out_dir, args.feature_source, pca_tag, args.cluster_method
    )
    os.makedirs(run_out_dir, exist_ok=True)
    print(f"Output dir: {run_out_dir}")

    print("Loading annotations ...")
    df = load_calvin_to_dataframe(TRAIN_DIR)
    print(f"Episodes: {len(df)}, Unique verbs: {df['primary_verb'].nunique()}\n")

    # ── Feature extraction ─────────────────────────────────────────────────
    if args.feature_source == "images":
        print(
            f"Image encoder: {args.image_encoder}, delta_patches: {args.delta_patches}"
        )
        features, verb_labels = build_image_features(
            df,
            encoder_name=args.image_encoder,
            delta_patches=args.delta_patches,
            batch_size=args.batch_size,
        )
        # Build prefix for filenames
        if args.delta_patches > 0 and args.image_encoder != "r3m":
            prefix = f"{args.image_encoder}_delta{args.delta_patches}"
        else:
            prefix = f"{args.image_encoder}_full"
    else:
        # Existing action-based feature extraction
        tokenizer = None
        if args.action_rep == "fast":
            print(f"Loading FAST tokenizer from {args.tokenizer_path} ...")
            tokenizer = load_fast_tokenizer(args.tokenizer_path)
        elif args.action_rep in ("bin", "quest", "oat"):
            from action_tokenizers import load_action_tokenizer

            print(f"Loading {args.action_rep} tokenizer ...")
            tokenizer = load_action_tokenizer(
                args.action_rep,
                train_dir=TRAIN_DIR,
                horizon=TOKENIZER_HORIZON,
                max_tokens=args.max_len,
            )

        features, verb_labels = build_features(
            df,
            args.max_len,
            args.workers,
            action_rep=args.action_rep,
            tokenizer=tokenizer,
        )

        if args.action_rep == "fast":
            scale_str = (
                str(int(args.scale))
                if args.scale == int(args.scale)
                else str(args.scale)
            )
            prefix = f"{args.action_rep}_v{args.vocab_size}_s{scale_str}"
        else:
            prefix = args.action_rep

    print(f"Feature matrix: {features.shape}  ({len(set(verb_labels))} unique verbs)\n")

    # ── PCA + Clustering ───────────────────────────────────────────────────
    scaled_features, pca_99_features, pca_2d, unique_verbs, cmap, pca_metrics = run_pca(
        features, verb_labels, run_out_dir, prefix=prefix
    )
    print()

    cluster_features = pca_99_features if args.use_pca else scaled_features
    print(
        f"Clustering on {'PCA-99% features' if args.use_pca else 'raw scaled features'} "
        f"(shape {cluster_features.shape}) with {args.cluster_method} ..."
    )

    cluster_metrics = run_clustering(
        cluster_features,
        verb_labels,
        pca_2d,
        unique_verbs,
        cmap,
        run_out_dir,
        prefix=prefix,
        cluster_method=args.cluster_method,
    )

    # ── Save metrics ───────────────────────────────────────────────────────
    combined_metrics = {
        "feature_source": args.feature_source,
        "cluster_method": args.cluster_method,
        "use_pca": args.use_pca,
        "pca": pca_metrics,
        "clustering": cluster_metrics,
    }
    if args.feature_source == "images":
        combined_metrics["image_encoder"] = args.image_encoder
        combined_metrics["delta_patches"] = args.delta_patches
        combined_metrics["representation"] = args.image_encoder
    else:
        combined_metrics["representation"] = args.action_rep
        combined_metrics["vocab_size"] = (
            args.vocab_size if args.action_rep == "fast" else None
        )
        combined_metrics["scale"] = args.scale if args.action_rep == "fast" else None

    metrics_path = os.path.join(run_out_dir, f"metrics_{prefix}.json")
    with open(metrics_path, "w") as f:
        json.dump(combined_metrics, f, indent=4)
    print(f"\nSaved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
