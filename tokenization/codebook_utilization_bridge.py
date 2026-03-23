#!/usr/bin/env python3
"""Measure OAT codebook utilization on BridgeV2 action shards."""
import sys, os, glob, torch, numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tokenization'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tokenization', 'oat'))

from oat.tokenizer.oat.tokenizer import OATTok
from oat.tokenizer.oat.encoder.register_encoder import RegisterEncoder
from oat.tokenizer.oat.decoder.single_pass_decoder import SinglePassDecoder
from oat.tokenizer.oat.quantizer.fsq import FSQ
from oat.model.common.normalizer import LinearNormalizer

SHARD_DIR = '/data/user_data/wenjiel2/datasets/bridge_actions'
CKPT = 'checkpoints/oat_bridge_j6660959_best.pth'


def main():
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    print(f"Checkpoint: epoch {ckpt['epoch']}, recon MSE {ckpt['best_recon_mse']:.6f}")

    # Build model
    fsq_levels = (8, 5, 5, 5)
    latent_dim = len(fsq_levels)
    encoder = RegisterEncoder(sample_dim=7, sample_horizon=32, emb_dim=256, head_dim=64,
                              depth=2, pdropout=0.1, latent_dim=latent_dim, num_registers=8)
    decoder = SinglePassDecoder(sample_dim=7, sample_horizon=32, emb_dim=256, head_dim=64,
                                depth=4, pdropout=0.1, token_dropout_mode='pow2',
                                latent_dim=latent_dim, latent_horizon=8, use_causal_decoder=True)
    quantizer = FSQ(levels=list(fsq_levels))
    model = OATTok(encoder=encoder, decoder=decoder, quantizer=quantizer)
    model.load_state_dict(ckpt['state_dict'])

    # Load actions
    shard_files = sorted(glob.glob(os.path.join(SHARD_DIR, 'shard_*.npz')))
    actions_list = []
    for sf in tqdm(shard_files, desc='Loading shards'):
        data = np.load(sf, allow_pickle=True)
        for i in range(int(data['n_episodes'])):
            actions_list.append(data[f'actions_{i}'].astype(np.float32))
    print(f'Loaded {len(actions_list)} episodes')

    # Compute normalizer (same split as training)
    np.random.seed(42)
    perm = np.random.permutation(len(actions_list))
    n_val = max(1, int(len(actions_list) * 0.1))
    train_actions = [actions_list[i] for i in perm[n_val:]]
    all_train = np.concatenate(train_actions, axis=0)
    normalizer = LinearNormalizer()
    normalizer.fit({'action': all_train}, mode='limits')
    model.set_normalizer(normalizer)
    model.eval()

    # Encode non-overlapping chunks
    horizon = 32
    unique_codes = set()
    n_chunks = 0
    per_position_codes = [set() for _ in range(8)]
    code_counts = {}

    with torch.no_grad():
        for ep_actions in tqdm(actions_list, desc='Encoding'):
            T = len(ep_actions)
            if T < 2:
                continue
            for start in range(0, max(1, T - horizon + 1), horizon):
                chunk = ep_actions[start:start + horizon]
                if len(chunk) < horizon:
                    chunk = np.pad(chunk, ((0, horizon - len(chunk)), (0, 0)), mode='edge')
                x = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
                latents, _ = model.encode(x)
                codes = latents[0].numpy().astype(int)
                for r in range(8):
                    ct = tuple(codes[r])
                    unique_codes.add(ct)
                    per_position_codes[r].add(ct)
                    code_counts[ct] = code_counts.get(ct, 0) + 1
                n_chunks += 1

    max_codes = int(np.prod(fsq_levels))  # 1000
    print(f'\n=== BridgeV2 OAT Codebook Utilization ===')
    print(f'Total episodes: {len(actions_list)}')
    print(f'Total chunks: {n_chunks}')
    print(f'Total tokens: {n_chunks * 8}')
    print(f'Unique codes (global): {len(unique_codes)}/{max_codes} ({100*len(unique_codes)/max_codes:.1f}%)')
    for r in range(8):
        print(f'  Position {r}: {len(per_position_codes[r])}/{max_codes} ({100*len(per_position_codes[r])/max_codes:.1f}%)')

    counts = np.array(list(code_counts.values()))
    print(f'\nCode frequency stats:')
    print(f'  Mean: {counts.mean():.1f}, Median: {np.median(counts):.0f}, Max: {counts.max()}, Min: {counts.min()}')
    print(f'  Codes used only once: {(counts == 1).sum()}')
    print(f'  Codes used >= 10: {(counts >= 10).sum()}')
    print(f'  Codes used >= 100: {(counts >= 100).sum()}')

    print(f'\nTop-10 most used codes:')
    for ct, count in sorted(code_counts.items(), key=lambda x: -x[1])[:10]:
        print(f'  {ct}: {count} ({100*count/(n_chunks*8):.2f}%)')


if __name__ == '__main__':
    main()
