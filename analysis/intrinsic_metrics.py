"""
Intrinsic metrics for action tokenizer evaluation on CALVIN task_D_D.

Computes three metrics across tokenizers:
  1. Verb entropy per token     — semantic purity of each discrete token (discrete only)
  2. Verb consistency ratio     — within-verb vs cross-verb similarity (all methods)
  3. Token transition entropy   — structure of sequential token dependencies (discrete only)

Usage:
    python analysis/intrinsic_metrics.py --methods native gampt_k64 gampt_continuous
    python analysis/intrinsic_metrics.py --methods all --output results/intrinsic_metrics.json
"""

import argparse
import json
import os
import sys
import numpy as np
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    VAL_DIR,
    TRAIN_DIR,
    ACTION_KEY,
    EPISODE_TEMPLATE,
    ACTION_DIM,
    CHECKPOINT_DIR,
    FAST_TOKENIZER_PATH,
    QUEST_TOKENIZER_CKPT,
    OAT_TOKENIZER_CKPT,
    TOKENIZER_HORIZON,
    MAX_SEQ_LEN,
)
from utils import load_calvin_to_dataframe


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

def verb_entropy_per_token(token_assignments, verb_labels, k, num_verbs=21):
    """
    Metric 1: Semantic purity of each discrete token.
    Applicable to discrete tokenizers only (FAST, VQ-VLA, GAMPT k=*).

    Args:
        token_assignments : list of int, one token ID per primitive/timestep
        verb_labels       : list of int, corresponding verb label per entry
        k                 : vocabulary size
        num_verbs         : number of verb classes

    Returns dict with mean_H, max_H, frac_pure, entropies.
    """
    buckets = defaultdict(list)
    for token, verb in zip(token_assignments, verb_labels):
        buckets[int(token)].append(int(verb))

    entropies = []
    for token_id in range(k):
        if token_id not in buckets:
            continue
        verbs = buckets[token_id]
        counts = np.bincount(verbs, minlength=num_verbs).astype(float)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        h = -np.sum(probs * np.log2(probs))
        entropies.append(h)

    if not entropies:
        return {"mean_H": 0.0, "max_H": 0.0, "frac_pure": 0.0, "entropies": []}

    return {
        "mean_H":    float(np.mean(entropies)),
        "max_H":     float(np.max(entropies)),
        "frac_pure": float(np.mean([e < 1.0 for e in entropies])),
        "entropies": [float(e) for e in entropies],
    }


def verb_consistency_ratio(features, verb_labels, mode="continuous"):
    """
    Metric 2: Within-verb similarity vs cross-verb similarity.
    Applicable to all methods. Mode selects similarity function:
      "discrete"   — Jaccard on token sets
      "continuous" — mean cosine similarity on mean-pooled feature vectors
      "vision"     — cosine similarity on pooled patch features (same as continuous)

    Args:
        features    : list of np.ndarray (continuous/vision) or list of set (discrete)
        verb_labels : list of int, one per trajectory
        mode        : "discrete" | "continuous" | "vision"

    Returns dict with within_verb, cross_verb, consistency_ratio.
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    n = len(features)
    within, cross = [], []

    # For large datasets, subsample pairs to keep runtime manageable
    indices = list(range(n))
    if n > 500:
        rng = np.random.default_rng(42)
        indices = rng.choice(n, size=500, replace=False).tolist()

    pairs = list(combinations(indices, 2))
    if len(pairs) > 50000:
        rng = np.random.default_rng(42)
        pair_idx = rng.choice(len(pairs), size=50000, replace=False)
        pairs = [pairs[i] for i in pair_idx]

    for i, j in pairs:
        if mode == "discrete":
            fi, fj = features[i], features[j]
            union = len(fi | fj)
            s = len(fi & fj) / union if union > 0 else 0.0
        else:
            fi = features[i].reshape(1, -1)
            fj = features[j].reshape(1, -1)
            s = float(cos_sim(fi, fj)[0, 0])

        if verb_labels[i] == verb_labels[j]:
            within.append(s)
        else:
            cross.append(s)

    w = float(np.mean(within)) if within else 0.0
    c = float(np.mean(cross)) if cross else 0.0
    ratio = w / c if c > 0 else 0.0

    return {
        "within_verb":       w,
        "cross_verb":        c,
        "consistency_ratio": ratio,
    }


def token_transition_entropy(token_sequences, k):
    """
    Metric 3: Structure of sequential token dependencies.
    Applicable to discrete tokenizers only.

    Args:
        token_sequences : list of list of int (one list per trajectory)
        k               : vocabulary size

    Returns dict with transition_entropy, normalized_transition_entropy, max_possible.
    """
    counts = np.zeros((k, k), dtype=np.float64)
    for seq in token_sequences:
        for a, b in zip(seq[:-1], seq[1:]):
            if 0 <= a < k and 0 <= b < k:
                counts[a][b] += 1

    unigram = counts.sum(axis=1)
    total = unigram.sum()
    if total == 0:
        return {"transition_entropy": 0.0, "normalized_transition_entropy": 0.0, "max_possible": float(np.log2(k))}

    transition_H = 0.0
    for a in range(k):
        if unigram[a] == 0:
            continue
        row = counts[a] / unigram[a]
        row = row[row > 0]
        h_a = -np.sum(row * np.log2(row))
        transition_H += (unigram[a] / total) * h_a

    max_H = float(np.log2(k))
    return {
        "transition_entropy":            float(transition_H),
        "normalized_transition_entropy": float(transition_H / max_H) if max_H > 0 else 0.0,
        "max_possible":                  max_H,
    }


# Data loading

def load_val_trajectories(data_dir, max_trajs=None):
    """
    Load val set trajectories as raw action arrays with verb labels.

    Returns:
        trajectories : list of np.ndarray, shape (T, 7) each
        verb_labels  : list of str
        verb_to_id   : dict str -> int
    """
    df = load_calvin_to_dataframe(data_dir)
    if max_trajs is not None:
        df = df.head(max_trajs)

    verbs = sorted(df["primary_verb"].unique())
    verb_to_id = {v: i for i, v in enumerate(verbs)}

    trajectories = []
    verb_labels = []

    for _, row in df.iterrows():
        s, e = int(row["start_idx"]), int(row["end_idx"])
        traj = []
        for idx in range(s, e + 1):
            path = os.path.join(data_dir, EPISODE_TEMPLATE.format(idx))
            traj.append(np.load(path)[ACTION_KEY].astype(np.float32))
        trajectories.append(np.stack(traj))
        verb_labels.append(verb_to_id[row["primary_verb"]])

    return trajectories, verb_labels, verb_to_id


# Per-method feature extraction

def extract_native(trajectories, max_len=64):
    """Mean-pooled raw action vectors — one vector per trajectory."""
    features = []
    for traj in trajectories:
        features.append(traj.mean(axis=0))  # (7,)
    return features, "continuous"


def extract_gampt_discrete(trajectories, tokenizer):
    """
    Returns:
        token_seqs    : list of list of int (variable length, PAD excluded)
        token_sets    : list of set of int (for Jaccard)
        flat_tokens   : flat list of token IDs (for verb entropy + transition entropy)
        flat_verbs    : flat list of verb labels aligned with flat_tokens
        vocab_size    : k
    """
    token_seqs = []
    for traj in trajectories:
        ids = tokenizer(traj)
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        # strip PAD
        pad_id = tokenizer.pad_id
        ids = [t for t in ids if t != pad_id]
        token_seqs.append(ids)
    return token_seqs, tokenizer.vocab_size - 1  # k without PAD


def extract_gampt_continuous(trajectories, tokenizer):
    """Mean-pooled scaled primitive features — one vector per trajectory."""
    features = []
    for traj in trajectories:
        _, feats = tokenizer.get_primitive_features(traj)
        if feats is not None and len(feats) > 0:
            features.append(feats.mean(axis=0))  # (14,)
        else:
            features.append(np.zeros(14, dtype=np.float32))
    return features, "continuous"


def extract_discrete_tokenizer(trajectories, tokenizer, max_len=64):
    """Generic extraction for FAST / QueST / OAT / VQ-VLA."""
    token_seqs = []
    for traj in trajectories:
        ids = tokenizer(traj)
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        token_seqs.append([int(t) for t in ids])
    # infer vocab size from max token id
    all_ids = [t for seq in token_seqs for t in seq]
    vocab_size = max(all_ids) + 1 if all_ids else 1
    return token_seqs, vocab_size


# Run metrics for one method

def run_metrics(name, trajectories, verb_labels, tokenizer=None,
                mode="discrete", vocab_size=None, num_verbs=21):
    """
    Compute all applicable metrics for one method.
    mode: "discrete" | "continuous"
    """
    print(f"\n  [{name}] Extracting features ...")
    results = {"method": name, "mode": mode}

    if mode == "discrete":
        if name.startswith("gampt_k"):
            token_seqs, k = extract_gampt_discrete(trajectories, tokenizer)
        else:
            token_seqs, k = extract_discrete_tokenizer(trajectories, tokenizer)

        if vocab_size is not None:
            k = vocab_size

        # Metric 1: verb entropy per token
        print(f"  [{name}] Computing verb entropy per token (k={k}) ...")
        flat_tokens = [t for seq in token_seqs for t in seq]
        flat_verbs_expanded = []
        for seq, vl in zip(token_seqs, verb_labels):
            flat_verbs_expanded.extend([vl] * len(seq))

        results["verb_entropy_per_token"] = verb_entropy_per_token(
            flat_tokens, flat_verbs_expanded, k, num_verbs=num_verbs)

        # Metric 2: verb consistency ratio (Jaccard on token sets)
        print(f"  [{name}] Computing verb consistency ratio ...")
        token_sets = [set(seq) for seq in token_seqs]
        results["verb_consistency_ratio"] = verb_consistency_ratio(
            token_sets, verb_labels, mode="discrete")

        # Metric 3: token transition entropy
        print(f"  [{name}] Computing token transition entropy ...")
        results["token_transition_entropy"] = token_transition_entropy(token_seqs, k)

        results["vocab_size"] = k
        results["mean_seq_len"] = float(np.mean([len(s) for s in token_seqs]))

    else:
        # Continuous / vision
        if name == "native":
            features, _ = extract_native(trajectories)
        elif name == "gampt_continuous":
            features, _ = extract_gampt_continuous(trajectories, tokenizer)
        else:
            raise ValueError(f"Unknown continuous method: {name}")

        # Metric 2 only
        print(f"  [{name}] Computing verb consistency ratio ...")
        results["verb_consistency_ratio"] = verb_consistency_ratio(
            features, verb_labels, mode="continuous")

    return results


# CLI

SUPPORTED_METHODS = [
    "native",
    "gampt_k21", "gampt_k32", "gampt_k64", "gampt_k128", "gampt_k256",
    "gampt_continuous",
    "fast_pretrained",
    "fast",
    "quest",
    "oat",
    "scratch",
    "r3m",
    "dinov2_s",
    "vc1",
]

VISION_ENCODER_MAP = {
    "scratch":  ("scratch",   0),   # (encoder_name, delta_patches)
    "r3m":      ("r3m",       0),
    "dinov2_s": ("dinov2_s", 16),
    "vc1":      ("vc1",       16),
}


def main(args):
    data_dir = args.data_dir or VAL_DIR
    print(f"Loading val trajectories from {data_dir} ...")
    trajectories, verb_labels, verb_to_id = load_val_trajectories(
        data_dir, max_trajs=args.max_trajs)
    num_verbs = len(verb_to_id)
    print(f"Loaded {len(trajectories)} trajectories, {num_verbs} verbs")

    methods = args.methods
    if "all" in methods:
        methods = SUPPORTED_METHODS

    all_results = {}

    for method in methods:
        print(f"\n=== {method} ===")
        try:
            if method == "native":
                result = run_metrics(
                    "native", trajectories, verb_labels,
                    mode="continuous", num_verbs=num_verbs)

            elif method == "gampt_continuous":
                from tokenization.gampt import GAMPTTokenizer
                ckpt = os.path.join(CHECKPOINT_DIR, f"gampt_k{args.gampt_k}.pkl")
                tok = GAMPTTokenizer.load(ckpt)
                result = run_metrics(
                    "gampt_continuous", trajectories, verb_labels,
                    tokenizer=tok, mode="continuous", num_verbs=num_verbs)

            elif method.startswith("gampt_k"):
                k = int(method.split("gampt_k")[1])
                from tokenization.gampt import GAMPTTokenizer
                ckpt = os.path.join(CHECKPOINT_DIR, f"gampt_k{k}.pkl")
                tok = GAMPTTokenizer.load(ckpt)
                result = run_metrics(
                    method, trajectories, verb_labels,
                    tokenizer=tok, mode="discrete",
                    vocab_size=k, num_verbs=num_verbs)

            elif method == "fast_pretrained":
                import zarr
                import torch as _torch
                from tokenization.action_tokenizers import TokenizerAdapter
                from oat.tokenizer.fast.tokenizer_wrapper import FASTTok
                from oat.model.common.normalizer import LinearNormalizer as _LN
                tok_raw = FASTTok("physical-intelligence/fast")
                # Fit normalizer from training data (same data the model was trained on)
                root = zarr.open(os.path.join(os.path.dirname(CHECKPOINT_DIR), "data/calvin_N500.zarr"), "r")
                actions = _torch.from_numpy(root["data/action"][:].astype("float32"))
                _norm = _LN()
                _norm.fit({"action": actions}, last_n_dims=1, mode="limits")
                tok_raw.set_normalizer(_norm)
                tok = TokenizerAdapter(tok_raw, "fast", horizon=TOKENIZER_HORIZON, max_tokens=MAX_SEQ_LEN)
                result = run_metrics(
                    "fast_pretrained", trajectories, verb_labels,
                    tokenizer=tok, mode="discrete", num_verbs=num_verbs)

            elif method == "fast":
                import dill
                import torch as _torch
                from tokenization.action_tokenizers import TokenizerAdapter
                from oat.tokenizer.fast.tokenizer_wrapper import FASTTok
                from oat.model.common.normalizer import LinearNormalizer as _LN
                from transformers import PreTrainedTokenizerFast
                # Initialize with HF hub (gets custom FAST processor code)
                tok_raw = FASTTok("physical-intelligence/fast")
                # Load model state dict from trained checkpoint (includes fitted normalizer)
                fast_ckpt = os.path.join(CHECKPOINT_DIR, "fast_trained.ckpt")
                payload = _torch.load(fast_ckpt, pickle_module=dill, map_location="cpu", weights_only=False)
                tok_raw.load_state_dict(payload["state_dicts"]["model"])
                # Swap in trained tokenizer vocab
                trained_path = os.path.join(CHECKPOINT_DIR, "my_fast")
                trained_inner = PreTrainedTokenizerFast(tokenizer_file=os.path.join(trained_path, "tokenizer.json"))
                tok_raw.fast_tok.tokenizer = trained_inner
                tok = TokenizerAdapter(tok_raw, "fast", horizon=TOKENIZER_HORIZON, max_tokens=MAX_SEQ_LEN)
                result = run_metrics(
                    "fast", trajectories, verb_labels,
                    tokenizer=tok, mode="discrete", num_verbs=num_verbs)

            elif method == "quest":
                from tokenization.action_tokenizers import load_action_tokenizer
                tok = load_action_tokenizer("quest", TRAIN_DIR)
                result = run_metrics(
                    "quest", trajectories, verb_labels,
                    tokenizer=tok, mode="discrete", num_verbs=num_verbs)

            elif method == "oat":
                from tokenization.action_tokenizers import load_action_tokenizer
                tok = load_action_tokenizer("oat", TRAIN_DIR)
                result = run_metrics(
                    "oat", trajectories, verb_labels,
                    tokenizer=tok, mode="discrete", num_verbs=num_verbs)

            elif method in VISION_ENCODER_MAP:
                from analysis.cluster_analysis import build_image_features
                from utils import load_calvin_to_dataframe
                encoder_name, delta_patches = VISION_ENCODER_MAP[method]
                df = load_calvin_to_dataframe(data_dir)
                if args.max_trajs:
                    df = df.head(args.max_trajs)
                print(f"  [{method}] Extracting image features (encoder={encoder_name}, delta={delta_patches}) ...")
                feats, _ = build_image_features(df, encoder_name, delta_patches=delta_patches, data_dir=data_dir)
                features = [feats[i] for i in range(len(feats))]
                result = {"method": method, "mode": "vision"}
                print(f"  [{method}] Computing verb consistency ratio ...")
                result["verb_consistency_ratio"] = verb_consistency_ratio(features, verb_labels, mode="continuous")
                vcr = result["verb_consistency_ratio"]
                print(f"  consistency_ratio={vcr['consistency_ratio']:.4f}  within={vcr['within_verb']:.4f}  cross={vcr['cross_verb']:.4f}")

            else:
                print(f"  Skipping unknown method: {method}")
                continue

            all_results[method] = result
            _print_summary(method, result)

        except FileNotFoundError as e:
            print(f"  Skipping {method}: checkpoint not found — {e}")
        except Exception as e:
            print(f"  Error on {method}: {e}")
            import traceback; traceback.print_exc()

    # Save
    output_path = args.output or "results/intrinsic_metrics.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print comparison table
    _print_table(all_results)


def _print_summary(name, result):
    mode = result.get("mode", "?")
    vcr = result.get("verb_consistency_ratio", {})
    print(f"  consistency_ratio={vcr.get('consistency_ratio', 'N/A'):.4f}  "
          f"within={vcr.get('within_verb', 'N/A'):.4f}  "
          f"cross={vcr.get('cross_verb', 'N/A'):.4f}")
    if mode == "discrete":
        vep = result.get("verb_entropy_per_token", {})
        tte = result.get("token_transition_entropy", {})
        print(f"  mean_H={vep.get('mean_H', 'N/A'):.3f}  "
              f"frac_pure={vep.get('frac_pure', 'N/A'):.3f}  "
              f"norm_transition_H={tte.get('normalized_transition_entropy', 'N/A'):.3f}")


def _print_table(all_results):
    print("\n" + "=" * 90)
    print(f"{'Method':<22} {'Consistency':>12} {'Within':>8} {'Cross':>8} "
          f"{'mean_H':>8} {'frac_pure':>10} {'norm_TE':>8}")
    print("-" * 90)
    for name, r in all_results.items():
        vcr = r.get("verb_consistency_ratio", {})
        vep = r.get("verb_entropy_per_token", {})
        tte = r.get("token_transition_entropy", {})
        cons  = f"{vcr.get('consistency_ratio', float('nan')):.4f}"
        with_ = f"{vcr.get('within_verb', float('nan')):.4f}"
        cross = f"{vcr.get('cross_verb', float('nan')):.4f}"
        mH    = f"{vep.get('mean_H', float('nan')):.3f}" if vep else "  —  "
        fp    = f"{vep.get('frac_pure', float('nan')):.3f}" if vep else "  —  "
        nte   = f"{tte.get('normalized_transition_entropy', float('nan')):.3f}" if tte else "  —  "
        print(f"{name:<22} {cons:>12} {with_:>8} {cross:>8} {mH:>8} {fp:>10} {nte:>8}")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intrinsic tokenizer metrics on CALVIN val set")
    parser.add_argument("--methods", nargs="+", default=["native", "gampt_k64", "gampt_continuous"],
                        help=f"Methods to evaluate. Use 'all' for all. Choices: {SUPPORTED_METHODS}")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to CALVIN val split (default: VAL_DIR from config.py)")
    parser.add_argument("--output", type=str, default="results/intrinsic_metrics.json",
                        help="Path to save JSON results")
    parser.add_argument("--max_trajs", type=int, default=None,
                        help="Limit number of trajectories (useful for quick testing)")
    parser.add_argument("--gampt_k", type=int, default=64,
                        help="k used when loading gampt_continuous tokenizer")
    args = parser.parse_args()
    main(args)
