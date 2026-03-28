"""
GAMPT intrinsic metrics matching the report's 4-metric framework.

Computes for each gampt_k*.pkl checkpoint:
  1. Codebook utilization  — fraction of k clusters used on the val set
  2. Token-ID verb probe   — linear probe Macro-F1 from token-ID histograms
  3. Encoder latent probe  — linear probe Macro-F1 from mean-pooled 14-D features
     (also computed with VC-1 features concatenated for the +VC-1 multimodal variants)

Reconstruction loss is not applicable (no learned decoder).

Usage:
    python scripts/gampt_intrinsic_probes.py
    python scripts/gampt_intrinsic_probes.py --k 64 --vc1 --data_dir /path/to/val
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VAL_DIR, EPISODE_TEMPLATE, ACTION_KEY, CHECKPOINT_DIR
from utils import load_calvin_to_dataframe


def load_val_trajectories(data_dir, max_trajs=None):
    df = load_calvin_to_dataframe(data_dir)
    if max_trajs:
        df = df.head(max_trajs)
    verbs = sorted(df["primary_verb"].unique())
    verb_to_id = {v: i for i, v in enumerate(verbs)}
    trajectories, verb_labels = [], []
    for _, row in df.iterrows():
        s, e = int(row["start_idx"]), int(row["end_idx"])
        traj = np.stack([
            np.load(os.path.join(data_dir, EPISODE_TEMPLATE.format(i)))[ACTION_KEY].astype(np.float32)
            for i in range(s, e + 1)
        ])
        trajectories.append(traj)
        verb_labels.append(verb_to_id[row["primary_verb"]])
    return trajectories, verb_labels, verb_to_id, df


def extract_vc1_features(df, data_dir):
    """Extract VC-1 delta-16 features for each trajectory (one vector per trajectory)."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analysis.cluster_analysis import build_image_features
    print("Extracting VC-1 delta-16 features ...")
    feats, _ = build_image_features(df, "vc1", delta_patches=16, data_dir=data_dir)
    return feats  # (N, 16*768)


def compute_metrics(tok, trajectories, verb_labels, vc1_feats=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import normalize

    k = tok.k
    n = len(trajectories)

    # --- Extract per-trajectory representations ---
    token_id_hists = np.zeros((n, k), dtype=np.float32)   # histogram of token IDs
    latent_feats   = np.zeros((n, 14), dtype=np.float32)  # mean-pooled 14-D primitives
    used_clusters  = set()

    for i, traj in enumerate(trajectories):
        ids = tok._tokenize_raw(traj)                      # List[int], no PAD
        for t in ids:
            token_id_hists[i, t] += 1
            used_clusters.add(t)
        _, feats_scaled = tok.get_primitive_features(traj) # (P, 14) scaled
        latent_feats[i] = feats_scaled.mean(axis=0)

    # Normalize histograms to relative frequencies
    row_sums = token_id_hists.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    token_id_hists /= row_sums

    # --- Metric 1: Codebook utilization ---
    utilization = len(used_clusters) / k

    # --- Metrics 2 & 3: Linear probes (5-fold cross-val) ---
    y = np.array(verb_labels)

    def linear_probe_macro_f1(X, y, max_iter=1000):
        from sklearn.model_selection import StratifiedKFold
        X = normalize(X)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        f1s = []
        for train_idx, val_idx in skf.split(X, y):
            clf = LogisticRegression(max_iter=max_iter, C=1.0, random_state=42)
            clf.fit(X[train_idx], y[train_idx])
            preds = clf.predict(X[val_idx])
            f1s.append(f1_score(y[val_idx], preds, average="macro", zero_division=0))
        return float(np.mean(f1s))

    token_id_f1 = linear_probe_macro_f1(token_id_hists, y)
    latent_f1   = linear_probe_macro_f1(latent_feats, y)

    result = {
        "k":                    k,
        "codebook_utilization": round(utilization * 100, 1),
        "token_id_probe_mf1":   round(token_id_f1 * 100, 1),
        "latent_probe_mf1":     round(latent_f1 * 100, 1),
    }

    # +VC-1: concatenate normalized VC-1 features with normalized latent features
    if vc1_feats is not None:
        combined = np.concatenate([normalize(latent_feats), normalize(vc1_feats)], axis=1)
        result["latent_probe_mf1_vc1"] = round(linear_probe_macro_f1(combined, y) * 100, 1)

    return result


def main(args):
    data_dir = args.data_dir or VAL_DIR
    print(f"Loading val trajectories from {data_dir} ...")
    trajectories, verb_labels, verb_to_id, df = load_val_trajectories(data_dir, args.max_trajs)
    print(f"Loaded {len(trajectories)} trajectories, {len(verb_to_id)} verbs")

    vc1_feats = None
    if args.vc1:
        vc1_feats = extract_vc1_features(df, data_dir)

    from tokenization.gampt import GAMPTTokenizer

    results = []
    for k in args.k:
        ckpt = os.path.join(CHECKPOINT_DIR, f"gampt_k{k}.pkl")
        if not os.path.exists(ckpt):
            print(f"  Skipping k={k}: checkpoint not found at {ckpt}")
            continue
        print(f"\n=== GAMPT k={k} ===")
        tok = GAMPTTokenizer.load(ckpt)
        metrics = compute_metrics(tok, trajectories, verb_labels, vc1_feats=vc1_feats)
        results.append(metrics)
        print(f"  Codebook utilization:    {metrics['codebook_utilization']}%")
        print(f"  Token-ID probe MF1:      {metrics['token_id_probe_mf1']}%")
        print(f"  Latent probe MF1:        {metrics['latent_probe_mf1']}%")
        if "latent_probe_mf1_vc1" in metrics:
            print(f"  Latent probe MF1 +VC-1: {metrics['latent_probe_mf1_vc1']}%")

    print("\n" + "=" * 70)
    header = f"{'k':<8} {'Utilization':>14} {'Token-ID F1':>13} {'Latent F1':>11}"
    if args.vc1:
        header += f" {'Latent+VC-1 F1':>15}"
    print(header)
    print("-" * 70)
    for r in results:
        row = f"{r['k']:<8} {r['codebook_utilization']:>13}% {r['token_id_probe_mf1']:>12}% {r['latent_probe_mf1']:>10}%"
        if args.vc1:
            row += f" {r.get('latent_probe_mf1_vc1', '---'):>14}%"
        print(row)
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", nargs="+", type=int, default=[21, 32, 64, 128, 256])
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--max_trajs", type=int, default=None)
    parser.add_argument("--vc1", action="store_true", help="Also compute latent probe with VC-1 features concatenated")
    args = parser.parse_args()
    main(args)
