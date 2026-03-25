"""Codebook utilization measurement for action tokenizers.

Works with OAT (scalar token indices), QueST (FSQ float codes), and VQ-BeT
(ResidualVQ group indices). All are converted to scalar indices for counting.

Usage from training loop:
    from analysis.codebook_util import count_unique_codes
    n_unique = count_unique_codes(all_codes_list)

Usage standalone:
    python analysis/codebook_util.py --ckpt_dir checkpoints/bridge_sweep --dataset bridge
"""
import torch


def codes_to_unique_count(codes_list):
    """Count unique discrete codes from a list of code tensors.

    Args:
        codes_list: list of tensors, each (N_i, D) where D >= 1.
            For scalar indices (OAT tokens, FSQ indices): D=1, values are ints.
            For VQ-BeT group indices: D=groups, values are ints.

    Returns:
        int: number of unique code tuples.
    """
    if not codes_list:
        return None
    codes_cat = torch.cat(codes_list, dim=0)  # (N, D)
    unique = set(map(tuple, codes_cat.long().cpu().tolist()))
    return len(unique)
