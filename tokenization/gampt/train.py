"""Train GAMPT tokenizers for all vocabulary sizes.

Implementation order:
    1. Threshold calibration: plot ‖v_t‖ and cosine-sim distributions,
        confirm theta_stop=0.005 sits in the trough, compute v95.
    2. Segmentation sanity check: 50 random trajectories,
            report mean/std primitive count and length.
    3. Feature sanity check: 'lift' trajectories, verify dz dominates.
    4. Fit tokenizers for k in {21, 32, 64, 128, 256}.

Usage:
    python -m tokenization.gampt.train # all steps
    python -m tokenization.gampt.train --steps 1 2 # calibration + sanity only
    python -m tokenization.gampt.train --steps 4 # fit tokenizers only
    python -m tokenization.gampt.train --k 64 128 # fit specific k values only
    python -m tokenization.gampt.train --data_dir /path/to/training
"""

import os
import sys
import argparse
import time
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import partial
from multiprocessing import Pool

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import (
    TRAIN_DIR, ACTION_KEY, EPISODE_TEMPLATE,
    CHECKPOINT_DIR,
)
from utils import load_calvin_to_dataframe

FIGURES_DIR = os.path.join(_PROJECT_ROOT, "figures")
K_VALUES = [21, 32, 64, 128, 256]


# Data loading helpers
def _load_action_frame(idx, data_dir, action_key, template):
    path = os.path.join(data_dir, template.format(idx))
    return np.load(path, allow_pickle=False)[action_key].astype(np.float32)


def load_trajectories(df, data_dir, num_workers=8):
    """Load all training trajectories as a list of np.ndarray (T_i, 7).

    Uses the same parallel frame-loading pattern as cluster_analysis.py.
    """
    needed_indices = set()
    for s, e in zip(df["start_idx"].values, df["end_idx"].values):
        needed_indices.update(range(s, e + 1))
    needed_indices = sorted(needed_indices)

    print(f"Loading {len(needed_indices)} unique frames with {num_workers} workers ...")
    t0 = time.time()
    load_fn = partial(_load_action_frame,
                      data_dir=data_dir,
                      action_key=ACTION_KEY,
                      template=EPISODE_TEMPLATE)
    with Pool(num_workers) as pool:
        frames = pool.map(load_fn, needed_indices, chunksize=1024)
    print(f"Loaded in {time.time() - t0:.1f}s")

    idx_to_pos = {idx: i for i, idx in enumerate(needed_indices)}
    frames_arr = np.array(frames)   # (N_frames, 7)

    trajectories = []
    for _, row in df.iterrows():
        s = idx_to_pos[row["start_idx"]]
        e = idx_to_pos[row["end_idx"]] + 1
        trajectories.append(frames_arr[s:e].copy())

    return trajectories


# 1. threshold calibration
def step1_calibration(trajectories, theta_stop=0.005, save_dir=FIGURES_DIR):
    """Plot speed and cosine-similarity distributions; compute and return v95."""
    os.makedirs(save_dir, exist_ok=True)

    all_speeds = []
    all_cosines = []

    for traj in trajectories:
        v = traj[:, :3]
        speed = np.linalg.norm(v, axis=1)
        all_speeds.append(speed)

        for t in range(1, len(traj)):
            s0, s1 = speed[t - 1], speed[t]
            if s0 > theta_stop and s1 > theta_stop:
                cos = float(np.dot(v[t - 1], v[t]) / (s0 * s1 + 1e-9))
                all_cosines.append(cos)

    speeds = np.concatenate(all_speeds)
    cosines = np.array(all_cosines)

    v95 = float(np.percentile(speeds, 95))
    v50 = float(np.percentile(speeds, 50))
    print(f"\n[Step 1] Speed stats:  median={v50:.5f}  v95={v95:.5f}")
    print(f"[Step 1] theta_stop={theta_stop:.5f} vs median={v50:.5f}  "
          f"({'OK: theta_stop << median' if theta_stop < v50 * 0.5 else 'WARN: theta_stop may be too high'})")
    print(f"[Step 1] Cosine-sim stats:  mean={cosines.mean():.3f}  "
          f"std={cosines.std():.3f}  "
          f"p10={np.percentile(cosines, 10):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("GAMPT Threshold Calibration", fontweight="bold")

    # Speed distribution
    ax = axes[0]
    ax.hist(speeds, bins=200, color="#4393c3", edgecolor="none", alpha=0.8)
    ax.axvline(theta_stop, color="red", lw=1.5, label=f"θ_stop={theta_stop}")
    ax.axvline(v95, color="orange", lw=1.5, linestyle="--", label=f"v95={v95:.4f}")
    ax.set_xlim(0, v95 * 2)
    ax.set_xlabel("‖v_t‖  (translation speed, m/step)")
    ax.set_ylabel("Frame count")
    ax.set_title("Translation speed distribution")
    ax.legend(fontsize=9)

    # Cosine-similarity distribution
    ax = axes[1]
    ax.hist(cosines, bins=100, color="#d6604d", edgecolor="none", alpha=0.8)
    ax.axvline(0.8, color="red", lw=1.5, label="θ_dir=0.8")
    ax.set_xlabel("cos_sim(v_t, v_{t+1})  (consecutive translation)")
    ax.set_ylabel("Step count")
    ax.set_title("Direction change cosine-similarity distribution")
    ax.legend(fontsize=9)

    path = os.path.join(save_dir, "gampt_calibration.png")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[Step 1] Saved calibration plot → {path}")

    return v95



# 2. segmentation sanity check
def step2_segmentation_sanity(trajectories, segmenter, n_sample=50, seed=42):
    """Run segmenter on n_sample random trajectories and report statistics."""
    rng = random.Random(seed)
    sample = rng.sample(trajectories, min(n_sample, len(trajectories)))

    prim_counts = []
    prim_lengths = []

    for traj in sample:
        prims = segmenter.segment(traj)
        prim_counts.append(len(prims))
        prim_lengths.extend([(e - s + 1) for s, e in prims])

    prim_counts = np.array(prim_counts)
    prim_lengths = np.array(prim_lengths)

    print(f"\n[Step 2] Segmentation sanity check on {len(sample)} trajectories:")
    print(f"  Primitive count:  mean={prim_counts.mean():.1f}  "
          f"std={prim_counts.std():.1f}  "
          f"min={prim_counts.min()}  max={prim_counts.max()}")
    print(f"  Primitive length: mean={prim_lengths.mean():.1f}  "
          f"std={prim_lengths.std():.1f}  "
          f"min={prim_lengths.min()}  max={prim_lengths.max()}")

    if prim_counts.mean() > 10:
        print("  WARN: mean count > 10 -- θ_dir may be too tight (increase toward 0.9) "
              "or θ_speed_mult too low (increase toward 0.2)")
    elif prim_counts.mean() < 3:
        print("  WARN: mean count < 3 -- segmentation too coarse "
              "(decrease θ_dir toward 0.7 or θ_stop toward 0.002)")
    else:
        print("  OK: mean primitive count in target range [3, 10]")

    return prim_counts, prim_lengths


# 3. feature sanity check
def step3_feature_sanity(df, trajectories, segmenter):
    """For 'lift' trajectories, verify dz dominates displacement vector."""
    from tokenization.gampt.tokenizer import GAMPTFeaturizer
    featurizer = GAMPTFeaturizer()

    lift_mask = df["primary_verb"] == "lift"
    lift_indices = df.index[lift_mask].tolist()

    if not lift_indices:
        print("\n[Step 3] No 'lift' trajectories found in training set — skipping.")
        return

    sample_size = min(10, len(lift_indices))
    sample_idxs = lift_indices[:sample_size]

    print(f"\n[Step 3] Feature sanity check on {sample_size} 'lift' trajectories:")
    print(f"  {'prim':>4}  {'len':>4}  {'dx':>7}  {'dy':>7}  {'dz':>7}  "
          f"{'dz_dom':>7}  {'g_chg':>5}")

    for traj_idx in sample_idxs:
        traj = trajectories[traj_idx]
        prims = segmenter.segment(traj)
        feats = featurizer.extract(traj, prims)

        for p_i, ((s, e), fv) in enumerate(zip(prims, feats)):
            dx, dy, dz = fv[0], fv[1], fv[2]
            disp_mag = fv[3]
            dz_dominant = abs(dz) > abs(dx) and abs(dz) > abs(dy) if disp_mag > 1e-4 else None
            g_chg = int(fv[13])
            print(f"  {p_i:>4}  {e-s+1:>4}  {dx:>7.4f}  {dy:>7.4f}  {dz:>7.4f}  "
                  f"{'YES' if dz_dominant else ('NO' if dz_dominant is False else 'n/a'):>7}  "
                  f"{g_chg:>5}")


# 4. fit tokenizers
def step4_fit_tokenizers(trajectories, v95, k_values, out_dir,
                          theta_dir=0.8, theta_stop=0.005,
                          theta_speed_mult=0.1, L_max=15,
                          max_primitives=10):
    """Fit and save one GAMPTTokenizer per k value."""
    from tokenization.gampt.tokenizer import GAMPTTokenizer
    os.makedirs(out_dir, exist_ok=True)

    saved_paths = {}
    for k in k_values:
        print(f"\n{'='*60}")
        print(f"Fitting GAMPT tokenizer  k={k}")
        print(f"{'='*60}")
        tok = GAMPTTokenizer.fit(
            trajectories, k=k,
            theta_dir=theta_dir, theta_stop=theta_stop,
            theta_speed_mult=theta_speed_mult, L_max=L_max,
            max_primitives=max_primitives,
        )
        path = os.path.join(out_dir, f"gampt_k{k}.pkl")
        tok.save(path)
        saved_paths[k] = path

    print(f"\n[Step 4] All tokenizers saved to {out_dir}/")
    return saved_paths


# Main
def parse_args():
    parser = argparse.ArgumentParser(description="Train GAMPT tokenizers")
    parser.add_argument("--data_dir", type=str, default=TRAIN_DIR)
    parser.add_argument("--out_dir", type=str, default=CHECKPOINT_DIR,
                        help="Directory for saved tokenizer .pkl files")
    parser.add_argument("--figures_dir", type=str, default=FIGURES_DIR)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 4],
                        choices=[1, 2, 3, 4],
                        help="Which steps to run (default: all)")
    parser.add_argument("--k", type=int, nargs="+", default=K_VALUES,
                        help="Vocabulary sizes to fit (step 4 only)")
    # Segmentation hyperparameters
    parser.add_argument("--theta_dir", type=float, default=0.8)
    parser.add_argument("--theta_stop", type=float, default=0.005)
    parser.add_argument("--theta_speed_mult", type=float, default=0.1)
    parser.add_argument("--L_max", type=int, default=15)
    parser.add_argument("--max_primitives", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading CALVIN training annotations from {args.data_dir} ...")
    df = load_calvin_to_dataframe(args.data_dir)
    print(f"Loaded {len(df)} training trajectories across {df['primary_verb'].nunique()} verb classes")

    print("\nLoading trajectory action arrays ...")
    trajs = load_trajectories(df, args.data_dir, num_workers=args.num_workers)

    v95 = None

    if 1 in args.steps:
        v95 = step1_calibration(trajs, theta_stop=args.theta_stop,
                                 save_dir=args.figures_dir)

    if 2 in args.steps or 3 in args.steps or 4 in args.steps:
        # Build segmenter with calibrated v95 (run step 1 implicitly if needed)
        if v95 is None:
            all_speeds = np.concatenate([
                np.linalg.norm(t[:, :3], axis=1) for t in trajs])
            v95 = float(np.percentile(all_speeds, 95))
            print(f"[v95 computed] v95={v95:.6f}")

        from tokenization.gampt.tokenizer import GAMPTSegmenter
        segmenter = GAMPTSegmenter(
            theta_dir=args.theta_dir,
            theta_stop=args.theta_stop,
            v95=v95,
            theta_speed_mult=args.theta_speed_mult,
            L_max=args.L_max,
        )

        if 2 in args.steps:
            step2_segmentation_sanity(trajs, segmenter)

        if 3 in args.steps:
            step3_feature_sanity(df, trajs, segmenter)

        if 4 in args.steps:
            step4_fit_tokenizers(
                trajs, v95, k_values=args.k, out_dir=args.out_dir,
                theta_dir=args.theta_dir, theta_stop=args.theta_stop,
                theta_speed_mult=args.theta_speed_mult, L_max=args.L_max,
                max_primitives=args.max_primitives,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
