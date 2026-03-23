#!/usr/bin/env python3
"""Measure codebook utilization for all HP sweep checkpoints."""
import sys, os, argparse, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tokenization.train_tokenizer import fit_normalizer, build_oat, build_quest
from tokenization.vqbet_tokenizer import VQBeTTokenizer

DATA_DIR = '/data/user_data/yashagar/task_D_D/training/'
CKPT_BASE = 'checkpoints/calvind_hp_sweep'


def load_actions(data_dir):
    ann = np.load(os.path.join(data_dir, 'lang_annotations', 'auto_lang_ann.npy'),
                  allow_pickle=True).item()
    starts = ann['info']['indx']
    cache_path = os.path.join(data_dir, '_action_cache.npz')
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        actions = data['actions']
        if 'min_id' in data:
            return starts, actions, int(data['min_id']), int(data['max_id'])
        min_id = int(data['offset']) if 'offset' in data else 0
        max_id = min_id + len(actions) - 1
        return starts, actions, min_id, max_id
    raise RuntimeError("No action cache found. Run codebook_utilization.py first.")


def get_chunks(actions, min_id, starts, chunk_size, stride=1):
    """Get all overlapping chunks of given size from all episodes."""
    chunks = []
    for s, e in starts:
        ep_len = e - s + 1
        for i in range(0, max(1, ep_len - chunk_size + 1), stride):
            idx = s - min_id + i
            c = actions[idx:idx + chunk_size]
            if len(c) < chunk_size:
                c = np.pad(c, ((0, chunk_size - len(c)), (0, 0)))
            chunks.append(c)
    return np.array(chunks, dtype=np.float32)


def measure_vqbet(ckpt_path, normalizer, actions, min_id, starts):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args = ckpt['args']
    chunk_size = args.get('chunk_size', 5)
    n_embed = args.get('num_codes', 16)
    groups = args.get('vq_groups', 2)

    model = VQBeTTokenizer(
        action_dim=7, chunk_size=chunk_size,
        latent_dim=args.get('latent_dim', 512),
        n_embed=n_embed, groups=groups,
        hidden_dim=args.get('hidden_dim', 128),
        num_layers=args.get('num_mlp_layers', 1))
    model.set_normalizer(normalizer)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    chunks = get_chunks(actions, min_id, starts, chunk_size, stride=1)
    # Flatten for VQ-BeT
    chunks_flat = chunks.reshape(len(chunks), -1)

    codes_per_group = [set() for _ in range(groups)]
    code_tuples = set()
    BS = 512
    for i in range(0, len(chunks_flat), BS):
        batch = torch.from_numpy(chunks_flat[i:i+BS])
        with torch.no_grad():
            _, indices, _ = model.encode(batch)
        idx_np = indices.numpy()
        for row in idx_np:
            t = tuple(row.tolist())
            code_tuples.add(t)
            for g in range(groups):
                codes_per_group[g].add(int(row[g]))

    total_combos = n_embed ** groups
    return len(code_tuples), total_combos, len(chunks_flat)


def measure_oat(ckpt_path, normalizer, actions, min_id, starts):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args = ckpt['args']
    model = build_oat(argparse.Namespace(**args))
    model.set_normalizer(normalizer)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    horizon = args.get('horizon', 32)
    num_regs = args.get('num_registers', 8)
    fsq_levels = args.get('fsq_levels', [8, 5, 5, 5])
    codebook_size = 1
    for l in fsq_levels:
        codebook_size *= l

    chunks = get_chunks(actions, min_id, starts, horizon, stride=horizon)
    unique_codes = set()
    BS = 256
    for i in range(0, len(chunks), BS):
        batch = torch.from_numpy(chunks[i:i+BS])
        with torch.no_grad():
            _, tokens = model.encode(batch)  # tokens: (B, T') scalar indices
        for row in tokens.cpu().numpy():
            for t in row:
                unique_codes.add(int(t))

    return len(unique_codes), codebook_size, len(chunks) * num_regs


def measure_quest(ckpt_path, normalizer, actions, min_id, starts):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args = ckpt['args']
    model = build_quest(argparse.Namespace(**args))
    model.set_normalizer(normalizer)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    horizon = args.get('horizon', 32)
    ds = args.get('downsample_factor', 4)
    fsq_levels = args.get('fsq_levels', [8, 5, 5, 5])
    codebook_size = 1
    for l in fsq_levels:
        codebook_size *= l
    n_tokens = horizon // ds

    chunks = get_chunks(actions, min_id, starts, horizon, stride=horizon)
    unique_codes = set()
    BS = 256
    for i in range(0, len(chunks), BS):
        batch = torch.from_numpy(chunks[i:i+BS])
        with torch.no_grad():
            codes = model.encode_fsq_codes(batch)  # (B, T', D) quantized floats
            indices = model.vq.codes_to_indices(codes)  # (B, T') scalar indices
        for row in indices.cpu().numpy():
            for t in row:
                unique_codes.add(int(t))

    return len(unique_codes), codebook_size, len(chunks) * n_tokens


def main():
    starts, actions, min_id, max_id = load_actions(DATA_DIR)
    normalizer = fit_normalizer(DATA_DIR)
    print(f"Episodes: {len(starts)}")

    configs = [
        # VQ-BeT
        ("vq_bet", "vq_bet_c5_e16_g2"),
        ("vq_bet", "vq_bet_c5_e16_g4"),
        ("vq_bet", "vq_bet_c5_e64_g2"),
        ("vq_bet", "vq_bet_c10_e16_g2"),
        ("vq_bet", "vq_bet_c10_e16_g4"),
        ("vq_bet", "vq_bet_c10_e64_g2"),
        # OAT
        ("oat", "oat_h32_f1000_r8"),
        ("oat", "oat_h32_f256_r8"),
        ("oat", "oat_h32_f256_r4"),
        ("oat", "oat_h32_f64_r4"),
        ("oat", "oat_h16_f256_r4"),
        ("oat", "oat_h16_f256_r8"),
        # QueST
        ("quest", "quest_h32_f1000_d4"),
        ("quest", "quest_h32_f256_d4"),
        ("quest", "quest_h32_f256_d8"),
        ("quest", "quest_h32_f64_d4"),
        ("quest", "quest_h16_f256_d4"),
        ("quest", "quest_h16_f256_d2"),
    ]

    print(f"\n{'Config':<25} {'Unique':>8} {'Total':>8} {'Util%':>8} {'Tokens':>10}")
    print("-" * 65)

    for tok_type, name in configs:
        ckpt_path = f'{CKPT_BASE}/{name}/full.pth'
        if not os.path.exists(ckpt_path):
            print(f"{name:<25} MISSING")
            continue

        if tok_type == "vq_bet":
            unique, total, n_tokens = measure_vqbet(ckpt_path, normalizer, actions, min_id, starts)
        elif tok_type == "oat":
            unique, total, n_tokens = measure_oat(ckpt_path, normalizer, actions, min_id, starts)
        elif tok_type == "quest":
            unique, total, n_tokens = measure_quest(ckpt_path, normalizer, actions, min_id, starts)

        pct = 100 * unique / total
        print(f"{name:<25} {unique:>8} {total:>8} {pct:>7.1f}% {n_tokens:>10}")


if __name__ == '__main__':
    main()
