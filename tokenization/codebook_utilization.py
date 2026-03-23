#!/usr/bin/env python3
"""Measure codebook utilization for VQ-BeT, OAT, and QueST on CALVIN D training data."""
import sys, os, argparse, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tokenization.train_tokenizer import fit_normalizer, build_oat, build_quest
from tokenization.vqbet_tokenizer import VQBeTTokenizer

DATA_DIR = '/data/user_data/yashagar/task_D_D/training/'
CKPT_BASE = 'checkpoints/calvind_sweep'


def load_actions_from_episodes(data_dir):
    """Load all actions from per-episode npz files, indexed by frame ID."""
    ann = np.load(os.path.join(data_dir, 'lang_annotations', 'auto_lang_ann.npy'),
                  allow_pickle=True).item()
    starts = ann['info']['indx']

    # Build action cache
    cache_path = os.path.join(data_dir, '_action_cache.npz')
    if os.path.exists(cache_path):
        print(f"Loading action cache from {cache_path}")
        data = np.load(cache_path)
        actions = data['actions']
        if 'min_id' in data:
            return starts, actions, int(data['min_id']), int(data['max_id'])
        # Older cache format: offset = min_id, infer max_id from array length
        min_id = int(data['offset']) if 'offset' in data else 0
        max_id = min_id + len(actions) - 1
        return starts, actions, min_id, max_id

    # Find frame range
    all_ids = set()
    for s, e in starts:
        all_ids.update(range(s, e + 1))
    min_id, max_id = min(all_ids), max(all_ids)
    print(f"Frame range: {min_id} - {max_id} ({max_id - min_id + 1} frames)")

    # Load actions into array
    actions = np.zeros((max_id - min_id + 1, 7), dtype=np.float32)
    loaded = 0
    for frame_id in range(min_id, max_id + 1):
        ep_file = os.path.join(data_dir, f'episode_{frame_id:07d}.npz')
        if os.path.exists(ep_file):
            actions[frame_id - min_id] = np.load(ep_file)['actions']
            loaded += 1
    print(f"Loaded {loaded} frames")

    np.savez(cache_path, actions=actions, min_id=min_id, max_id=max_id)
    return starts, actions, min_id, max_id


def get_action_chunk(actions, min_id, start, chunk_size):
    """Extract action chunk from frame start, pad if needed."""
    idx = start - min_id
    acts = actions[idx:idx + chunk_size]
    if len(acts) < chunk_size:
        acts = np.pad(acts, ((0, chunk_size - len(acts)), (0, 0)))
    return acts


def main():
    starts, actions, min_id, max_id = load_actions_from_episodes(DATA_DIR)
    normalizer = fit_normalizer(DATA_DIR)
    N = len(starts)
    print(f"Total episodes: {N}")

    # --- VQ-BeT ---
    print("\n=== VQ-BeT ===")
    ckpt = torch.load(f'{CKPT_BASE}/vq_bet_vanilla/full.pth', map_location='cpu', weights_only=False)
    model = VQBeTTokenizer(action_dim=7, chunk_size=5, latent_dim=512, n_embed=16, groups=2)
    model.set_normalizer(normalizer)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    code_pairs = set()
    codes_per_group = [set(), set()]
    n_chunks_vqbet = 0
    for i in range(N):
        s, e = starts[i]
        # Slide over ALL chunks in the episode, not just the first
        for t in range(s, e + 1 - 4):  # need 5 steps
            acts = get_action_chunk(actions, min_id, t, 5)
            x = torch.tensor(acts.flatten()).unsqueeze(0).float()
            with torch.no_grad():
                _, indices, _ = model.encode(x)
            g0, g1 = indices[0, 0].item(), indices[0, 1].item()
            codes_per_group[0].add(g0)
            codes_per_group[1].add(g1)
            code_pairs.add((g0, g1))
            n_chunks_vqbet += 1
    print(f"  Group 0: {len(codes_per_group[0])}/16 ({100*len(codes_per_group[0])/16:.0f}%)")
    print(f"  Group 1: {len(codes_per_group[1])}/16 ({100*len(codes_per_group[1])/16:.0f}%)")
    print(f"  Combined pairs: {len(code_pairs)}/256 ({100*len(code_pairs)/256:.1f}%)")
    print(f"  Total chunks encoded: {n_chunks_vqbet}")
    del model

    # --- OAT ---
    print("\n=== OAT ===")
    ckpt = torch.load(f'{CKPT_BASE}/oat_vanilla/full.pth', map_location='cpu', weights_only=False)
    model_oat = build_oat(argparse.Namespace(**ckpt['args']))
    model_oat.set_normalizer(normalizer)
    model_oat.load_state_dict(ckpt['model_state_dict'])
    model_oat.eval()

    codes_oat = set()
    n_chunks_oat = 0
    for i in range(N):
        s, e = starts[i]
        for t in range(s, e + 1 - 31):  # need 32 steps
            acts = get_action_chunk(actions, min_id, t, 32)
            x = torch.tensor(acts).unsqueeze(0).float()
            with torch.no_grad():
                latents, _ = model_oat.encode(x)
            c = latents[0].numpy()
            for r in range(8):
                codes_oat.add(tuple(c[r].astype(int)))
            n_chunks_oat += 1
    print(f"  Unique codes: {len(codes_oat)}/1000 ({100*len(codes_oat)/1000:.1f}%)")
    print(f"  Total chunks encoded: {n_chunks_oat}, total tokens: {n_chunks_oat * 8}")
    del model_oat

    # --- QueST ---
    print("\n=== QueST ===")
    ckpt = torch.load(f'{CKPT_BASE}/quest_vanilla/full.pth', map_location='cpu', weights_only=False)
    model_quest = build_quest(argparse.Namespace(**ckpt['args']))
    model_quest.set_normalizer(normalizer)
    model_quest.load_state_dict(ckpt['model_state_dict'])
    model_quest.eval()

    codes_quest = set()
    n_chunks_quest = 0
    for i in range(N):
        s, e = starts[i]
        for t in range(s, e + 1 - 31):
            acts = get_action_chunk(actions, min_id, t, 32)
            x = torch.tensor(acts).unsqueeze(0).float()
            with torch.no_grad():
                c = model_quest.encode_fsq_codes(x)
            c = c[0].numpy()
            for r in range(8):
                codes_quest.add(tuple(c[r].astype(int)))
            n_chunks_quest += 1
    print(f"  Unique codes: {len(codes_quest)}/1000 ({100*len(codes_quest)/1000:.1f}%)")
    print(f"  Total chunks encoded: {n_chunks_quest}, total tokens: {n_chunks_quest * 8}")

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"VQ-BeT: {len(codes_per_group[0])}/16 g0, {len(codes_per_group[1])}/16 g1, {len(code_pairs)}/256 pairs ({n_chunks_vqbet} chunks)")
    print(f"OAT:    {len(codes_oat)}/1000 unique codes ({n_chunks_oat} chunks)")
    print(f"QueST:  {len(codes_quest)}/1000 unique codes ({n_chunks_quest} chunks)")


if __name__ == '__main__':
    main()
