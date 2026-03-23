#!/usr/bin/env python3
"""Measure codebook utilization for all BridgeV2 tokenizer sweep checkpoints."""
import sys, os, argparse, glob, torch, numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tokenization'))

from tokenization.train_tokenizer import build_oat, build_quest
from tokenization.vqbet_tokenizer import VQBeTTokenizer

SHARD_DIR = '/data/user_data/wenjiel2/datasets/bridge_actions'
CKPT_BASE = 'checkpoints/bridge_sweep'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_bridge_actions():
    shard_files = sorted(glob.glob(os.path.join(SHARD_DIR, 'shard_*.npz')))
    actions_list = []
    for sf in tqdm(shard_files, desc='Loading shards'):
        data = np.load(sf, allow_pickle=True)
        n_eps = int(data['n_episodes'])
        for i in range(n_eps):
            actions_list.append(data[f'actions_{i}'].astype(np.float32))
    return actions_list


def get_chunks(actions_list, chunk_size, stride=None):
    """Get all non-overlapping chunks from all episodes."""
    if stride is None:
        stride = chunk_size
    chunks = []
    for ep in actions_list:
        T = len(ep)
        if T >= chunk_size:
            for start in range(0, T - chunk_size + 1, stride):
                chunks.append(ep[start:start + chunk_size])
        elif T >= 2:
            padded = np.pad(ep, ((0, chunk_size - T), (0, 0)), mode='edge')
            chunks.append(padded)
    return np.array(chunks, dtype=np.float32)


def fit_bridge_normalizer(actions_list):
    """Fit a LinearNormalizer on bridge actions."""
    from oat.model.common.normalizer import LinearNormalizer
    all_actions = np.concatenate(actions_list, axis=0)
    normalizer = LinearNormalizer()
    normalizer.fit({'action': all_actions}, mode='limits')
    return normalizer


def measure_oat(ckpt_path, actions_list, normalizer):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Handle two checkpoint formats:
    # 1. train_oat_bridge.py: top-level keys (horizon, num_registers, etc.) + 'state_dict'
    # 2. train_tokenizer.py: 'args' dict + 'model_state_dict'
    if 'args' in ckpt:
        args_dict = ckpt['args']
        state_key = 'model_state_dict'
    else:
        # train_oat_bridge.py format
        vocab_size = ckpt.get('vocab_size', 1000)
        # Reverse-engineer fsq_levels from vocab_size
        fsq_map = {1000: [8, 5, 5, 5], 200: [8, 5, 5], 512: [8, 8, 8], 125: [5, 5, 5]}
        args_dict = {
            'horizon': ckpt.get('horizon', 32),
            'num_registers': ckpt.get('num_registers', 8),
            'action_dim': ckpt.get('action_dim', 7),
            'fsq_levels': fsq_map.get(vocab_size, [8, 5, 5, 5]),
        }
        state_key = 'state_dict'

    args = argparse.Namespace(**args_dict)
    model = build_oat(args)
    model.set_normalizer(normalizer)
    model.load_state_dict(ckpt[state_key])
    model.to(DEVICE).eval()

    horizon = args_dict.get('horizon', 32)
    fsq_levels = args_dict.get('fsq_levels', [8, 5, 5, 5])
    codebook_size = 1
    for l in fsq_levels:
        codebook_size *= l

    chunks = get_chunks(actions_list, horizon)
    unique_codes = set()
    BS = 256
    for i in range(0, len(chunks), BS):
        batch = torch.from_numpy(chunks[i:i+BS]).to(DEVICE)
        with torch.no_grad():
            _, tokens = model.encode(batch)  # tokens: (B, T') scalar indices
        for row in tokens.cpu().numpy():
            for t in row:
                unique_codes.add(int(t))

    return len(unique_codes), codebook_size, len(chunks)


def measure_quest(ckpt_path, actions_list, normalizer):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args_dict = ckpt.get('args', {})
    args = argparse.Namespace(**args_dict)
    model = build_quest(args)
    model.set_normalizer(normalizer)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE).eval()

    horizon = args_dict.get('horizon', 32)
    fsq_levels = args_dict.get('fsq_levels', [8, 5, 5, 5])
    codebook_size = 1
    for l in fsq_levels:
        codebook_size *= l

    chunks = get_chunks(actions_list, horizon)
    unique_codes = set()
    BS = 256
    for i in range(0, len(chunks), BS):
        batch = torch.from_numpy(chunks[i:i+BS]).to(DEVICE)
        with torch.no_grad():
            codes = model.encode_fsq_codes(batch)  # (B, T', D) quantized floats
            indices = model.vq.codes_to_indices(codes)  # (B, T') scalar indices
        for row in indices.cpu().numpy():
            for t in row:
                unique_codes.add(int(t))

    return len(unique_codes), codebook_size, len(chunks)


def measure_vqbet(ckpt_path, actions_list, normalizer):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args_dict = ckpt.get('args', {})
    chunk_size = args_dict.get('chunk_size', 5)
    n_embed = args_dict.get('num_codes', 16)
    groups = args_dict.get('vq_groups', 2)

    model = VQBeTTokenizer(
        action_dim=7, chunk_size=chunk_size,
        latent_dim=args_dict.get('latent_dim', 512),
        n_embed=n_embed, groups=groups,
        hidden_dim=args_dict.get('hidden_dim', 128),
        num_layers=args_dict.get('num_mlp_layers', 1))
    model.set_normalizer(normalizer)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE).eval()

    chunks = get_chunks(actions_list, chunk_size)
    chunks_flat = chunks.reshape(len(chunks), -1)

    codes_per_group = [set() for _ in range(groups)]
    code_tuples = set()
    BS = 512
    for i in range(0, len(chunks_flat), BS):
        batch = torch.from_numpy(chunks_flat[i:i+BS]).to(DEVICE)
        with torch.no_grad():
            _, indices, _ = model.encode(batch)  # (B, groups)
        idx_np = indices.cpu().numpy()
        for row in idx_np:
            code_tuples.add(tuple(row.tolist()))
            for g in range(groups):
                codes_per_group[g].add(int(row[g]))

    total_combos = n_embed ** groups
    per_group = ', '.join(f'{len(s)}/{n_embed}' for s in codes_per_group)
    return len(code_tuples), total_combos, len(chunks_flat), per_group


def main():
    print('Loading BridgeV2 actions...')
    actions_list = load_bridge_actions()
    print(f'Loaded {len(actions_list)} episodes')

    print('Fitting normalizer...')
    normalizer = fit_bridge_normalizer(actions_list)
    print()

    # Auto-discover checkpoints
    configs = []
    for d in sorted(os.listdir(CKPT_BASE)):
        dirpath = os.path.join(CKPT_BASE, d)
        if not os.path.isdir(dirpath):
            continue
        # Find checkpoint file
        for candidate in ['full_best.pth', 'full.pth']:
            # Check top-level and one level down
            for ckpt in [os.path.join(dirpath, candidate)] + \
                        glob.glob(os.path.join(dirpath, '*', candidate)):
                if os.path.exists(ckpt):
                    if d.startswith('oat'):
                        configs.append(('oat', d, ckpt))
                    elif d.startswith('quest'):
                        configs.append(('quest', d, ckpt))
                    elif d.startswith('vqbet'):
                        configs.append(('vq_bet', d, ckpt))
                    break
            else:
                continue
            break

    print(f"{'Config':<30} {'Unique':>8} {'Total':>8} {'Util%':>8} {'Chunks':>10} {'Per-group':>20}")
    print('-' * 90)

    for tok_type, name, ckpt_path in configs:
        try:
            if tok_type == 'oat':
                unique, total, n_chunks = measure_oat(ckpt_path, actions_list, normalizer)
                per_group = ''
            elif tok_type == 'quest':
                unique, total, n_chunks = measure_quest(ckpt_path, actions_list, normalizer)
                per_group = ''
            elif tok_type == 'vq_bet':
                unique, total, n_chunks, per_group = measure_vqbet(ckpt_path, actions_list, normalizer)

            pct = 100 * unique / total
            print(f'{name:<30} {unique:>8} {total:>8} {pct:>7.1f}% {n_chunks:>10} {per_group:>20}')
        except Exception as e:
            print(f'{name:<30} ERROR: {e}')


if __name__ == '__main__':
    main()
