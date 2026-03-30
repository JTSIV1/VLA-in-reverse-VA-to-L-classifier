"""Quick linear probe (logistic regression) on tokenizer codes/latents.

Tests whether a simple linear model can do better than the transformer probe
on discrete token IDs — to diagnose if the bottleneck is the representation
or the probe architecture.

Usage:
    python verb_probe/linear_probe.py \
        --tokenizer_type quest \
        --tokenizer_ckpt checkpoints/calvin_sweep/tokenizers/quest_16_4444_2/full.pth

    # All 9 quest conditions at once:
    python verb_probe/linear_probe.py --sweep_quest
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder

from config import TRAIN_DIR, VAL_DIR, TOK_DIR
from datasets.calvin_dataset import build_calvin_tokenizer_data
from verb_probe.load_tokenizer import load_frozen_tokenizer, get_tokenizer_chunk_params, get_vocab_size
from tokenization.train_utils import extract_episode_batch


def encode_all(dataset, tok_model, tok_type, device, max_chunks=8):
    """Encode all episodes → (codes, latents, verb_labels)."""
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    all_codes, all_latents, all_labels = [], [], []

    for batch in loader:
        with torch.no_grad():
            result = extract_episode_batch(tok_model, batch, device, tok_type)

        codes = result['codes']       # (B, seq_len) or (B, K, groups)
        latents = result['latents']   # (B, seq_len, dim) or (B, K, dim)
        n_valid = result['n_valid']   # (B,)
        labels = result['verb_ids']   # (B,)

        B = codes.shape[0]
        for i in range(B):
            n = n_valid[i].item()
            if codes.ndim == 3:
                # VQ-BeT: (B, K, groups) → flatten to (K*groups,)
                c = codes[i, :n].reshape(-1).cpu().numpy()
                l = latents[i, :n].reshape(-1).cpu().numpy()
            else:
                # OAT/QueST: (B, seq_len) codes, (B, seq_len, dim) latents
                c = codes[i, :n].cpu().numpy()
                l = latents[i, :n].reshape(-1).cpu().numpy()

            all_codes.append(c)
            all_latents.append(l)
            all_labels.append(labels[i].item())

    return all_codes, all_latents, np.array(all_labels)


def codes_to_features(codes_list, vocab_size, method="onehot_flat"):
    """Convert variable-length code sequences to fixed-size feature vectors."""
    if method == "onehot_flat":
        # One-hot encode each position, then flatten
        max_len = max(len(c) for c in codes_list)
        features = np.zeros((len(codes_list), max_len * vocab_size), dtype=np.float32)
        for i, codes in enumerate(codes_list):
            for t, c in enumerate(codes):
                c = int(np.clip(c, 0, vocab_size - 1))
                features[i, t * vocab_size + c] = 1.0
        return features
    elif method == "bag_of_codes":
        # Histogram of code usage (position-independent)
        features = np.zeros((len(codes_list), vocab_size), dtype=np.float32)
        for i, codes in enumerate(codes_list):
            for c in codes:
                c = int(np.clip(c, 0, vocab_size - 1))
                features[i, c] += 1.0
        return features
    elif method == "embed_mean":
        # Just use raw code values as floats, zero-padded, then mean
        max_len = max(len(c) for c in codes_list)
        features = np.zeros((len(codes_list), max_len), dtype=np.float32)
        for i, codes in enumerate(codes_list):
            features[i, :len(codes)] = codes.astype(np.float32)
        return features


def latents_to_features(latents_list):
    """Pad variable-length latent sequences to fixed size and flatten."""
    max_len = max(l.shape[0] for l in latents_list)
    features = np.zeros((len(latents_list), max_len), dtype=np.float32)
    for i, l in enumerate(latents_list):
        features[i, :l.shape[0]] = l
    return features


def run_probe(name, X_train, y_train, X_val, y_val):
    """Fit logistic regression and report results."""
    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                             multi_class='multinomial', n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred) * 100
    mf1 = f1_score(y_val, y_pred, average='macro') * 100
    print(f"  {name:30s}  Acc={acc:.1f}%  MacroF1={mf1:.1f}%")
    return acc, mf1


def run_one_condition(tok_type, ckpt_path, device):
    """Run all linear probes for one tokenizer checkpoint."""
    name = os.path.basename(os.path.dirname(ckpt_path))
    print(f"\n{'='*60}")
    print(f"  {name}  ({tok_type})")
    print(f"{'='*60}")

    tok_model = load_frozen_tokenizer(tok_type, ckpt_path).to(device)
    chunk_params = get_tokenizer_chunk_params(ckpt_path)
    vocab_size = get_vocab_size(tok_model, tok_type)
    print(f"  vocab_size={vocab_size}, chunk_size={chunk_params['chunk_size']}")

    train_ds, val_ds, num_verbs, _, _, _ = build_calvin_tokenizer_data(
        TRAIN_DIR, VAL_DIR, min_class_count=30, cache_actions=True,
        **chunk_params)

    print(f"  {len(train_ds)} train / {len(val_ds)} val, {num_verbs} classes")
    print(f"  Encoding episodes...")

    train_codes, train_latents, y_train = encode_all(
        train_ds, tok_model, tok_type, device)
    val_codes, val_latents, y_val = encode_all(
        val_ds, tok_model, tok_type, device)

    print(f"  Code lengths: {[len(c) for c in train_codes[:3]]}")
    print(f"  Latent lengths: {[l.shape[0] for l in train_latents[:3]]}")

    results = {}

    # Token ID probes
    for method in ["onehot_flat", "bag_of_codes", "embed_mean"]:
        X_tr = codes_to_features(train_codes, vocab_size, method)
        X_va = codes_to_features(val_codes, vocab_size, method)
        acc, mf1 = run_probe(f"tokid ({method})", X_tr, y_train, X_va, y_val)
        results[f"tokid_{method}"] = {"acc": acc, "mf1": mf1}

    # Latent probe
    X_tr = latents_to_features(train_latents)
    X_va = latents_to_features(val_latents)
    acc, mf1 = run_probe("latent (flatten)", X_tr, y_train, X_va, y_val)
    results["latent_flatten"] = {"acc": acc, "mf1": mf1}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_type", type=str, default="quest")
    parser.add_argument("--tokenizer_ckpt", type=str, default=None)
    parser.add_argument("--sweep_quest", action="store_true",
                        help="Run all 9 quest conditions")
    parser.add_argument("--sweep_vqbet", action="store_true",
                        help="Run all vq_bet conditions with checkpoints")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.sweep_quest or args.sweep_vqbet:
        conditions = []
        if args.sweep_quest:
            for base in ["quest_16_4444_2", "quest_32_8555_4", "quest_16_4444_4"]:
                for suffix in ["", "_verb0.1", "_clip0.1"]:
                    name = base + suffix
                    ckpt = os.path.join(TOK_DIR, name, "full.pth")
                    if os.path.exists(ckpt):
                        conditions.append(("quest", ckpt))
        if args.sweep_vqbet:
            for name in os.listdir(TOK_DIR):
                if name.startswith("vq_bet_"):
                    ckpt = os.path.join(TOK_DIR, name, "full.pth")
                    if os.path.exists(ckpt):
                        conditions.append(("vq_bet", ckpt))

        all_results = {}
        for tok_type, ckpt in sorted(conditions, key=lambda x: x[1]):
            name = os.path.basename(os.path.dirname(ckpt))
            all_results[name] = run_one_condition(tok_type, ckpt, device)

        # Summary table
        print(f"\n{'='*80}")
        print(f"  SUMMARY")
        print(f"{'='*80}")
        print(f"  {'Condition':40s} {'tokid(oh)':>12s} {'tokid(bag)':>12s} {'latent':>12s}")
        for name, res in all_results.items():
            oh = res.get("tokid_onehot_flat", {})
            bag = res.get("tokid_bag_of_codes", {})
            lat = res.get("latent_flatten", {})
            print(f"  {name:40s} "
                  f"{oh.get('mf1', 0):5.1f}% MF1   "
                  f"{bag.get('mf1', 0):5.1f}% MF1   "
                  f"{lat.get('mf1', 0):5.1f}% MF1")
    else:
        if not args.tokenizer_ckpt:
            parser.error("Provide --tokenizer_ckpt or --sweep_quest/--sweep_vqbet")
        run_one_condition(args.tokenizer_type, args.tokenizer_ckpt, device)


if __name__ == "__main__":
    main()
