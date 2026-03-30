"""Evaluation functions for action tokenizer training.

eval_epoch: compute recon/VQ/verb/CLIP losses and codebook utilization.
eval_clip_retrieval: action-text top-k retrieval recall.
"""

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from tokenization.aux_heads import contrastive_loss


def codes_to_unique_count(codes_list, is_residual_vq=False):
    """Count unique discrete codes from a list of code tensors.

    OAT/QueST (is_residual_vq=False):
        Codes are (B, T) scalar FSQ indices. Count unique scalars.
    VQ-BeT (is_residual_vq=True):
        Codes are (B*K, groups) per-group indices. Count unique tuples
        (each tuple = one effective token in the combinatorial vocabulary).

    Returns int or None if empty.
    """
    if not codes_list:
        return None
    codes_cat = torch.cat(codes_list, dim=0)
    if is_residual_vq:
        return len(set(map(tuple, codes_cat.long().numpy().tolist())))
    else:
        return codes_cat.unique().numel()


def _extract_episode_batch_import():
    """Lazy import to avoid circular dependency."""
    from tokenization.train_utils import extract_episode_batch
    return extract_episode_batch


@torch.no_grad()
def eval_epoch(model, loader, device, args,
               verb_head=None, verb_criterion=None,
               clip_head=None, text_encoder=None, text_proj=None):
    _extract_episode_batch = _extract_episode_batch_import()
    model.eval()
    if verb_head is not None:
        verb_head.eval()
    if clip_head is not None:
        clip_head.eval()

    totals = {'recon': 0, 'vq': 0, 'verb': 0, 'clip': 0}
    correct = total = 0
    all_preds, all_labels = [], []
    all_codes = []
    n_batches = 0

    for batch in loader:
        result = _extract_episode_batch(
            model, batch, device, args.tokenizer)

        # Select aux input: post-FSQ 4d codes or pre-FSQ 256d latents
        aux_input = (result['fsq_codes'] if getattr(args, 'aux_target', 'latent') == 'post_fsq'
                     else result['latents'])

        # Verb classification
        if verb_head is not None and args.verb_cls_lambda > 0:
            verb_logits = verb_head(aux_input, result['n_valid'],
                                    positions=result['positions'])
            verb_ids = result['verb_ids']
            valid = verb_ids >= 0
            if valid.any():
                verb_loss = verb_criterion(verb_logits[valid], verb_ids[valid])
                totals['verb'] += verb_loss.item()
                preds = verb_logits.argmax(dim=1)
                correct += (preds[valid] == verb_ids[valid]).sum().item()
                total += valid.sum().item()
                all_preds.append(preds[valid].cpu())
                all_labels.append(verb_ids[valid].cpu())

        # CLIP contrastive
        if clip_head is not None and args.clip_lambda > 0:
            action_emb = clip_head(aux_input, result['n_valid'],
                                   positions=result['positions'])
            instructions = result['instructions']
            text_features = text_encoder(list(instructions))
            text_emb = F.normalize(text_proj(text_features), dim=-1)
            clip_loss = contrastive_loss(
                action_emb, text_emb, list(instructions), clip_head.temperature)
            totals['clip'] += clip_loss.item()

        if result.get('codes') is not None:
            codes = result['codes']
            if codes.ndim == 3:
                codes = codes.reshape(-1, codes.size(-1))
            all_codes.append(codes.cpu())

        totals['recon'] += result['recon_loss'].item()
        totals['vq'] += result.get('vq_loss', torch.tensor(0)).item()
        n_batches += 1

    macro_f1 = 0.0
    if all_preds:
        macro_f1 = 100.0 * f1_score(
            torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy(),
            average='macro', zero_division=0)

    codebook_util = codes_to_unique_count(
        all_codes, is_residual_vq=(args.tokenizer == 'vq_bet'))

    return {k: v / max(n_batches, 1) for k, v in totals.items()} | {
        'verb_acc': 100.0 * correct / max(total, 1),
        'verb_macro_f1': macro_f1,
        'codebook_util': codebook_util,
    }


@torch.no_grad()
def eval_clip_retrieval(model, loader, device, args,
                        clip_head, text_encoder, text_proj, ks=(1, 5, 10)):
    """Action->text and text->action top-k retrieval on the full val set.

    Returns dict: {'r@1': float, 'r@5': float, 'r@10': float}  (percentages)
    """
    _extract_episode_batch = _extract_episode_batch_import()
    model.eval()
    clip_head.eval()

    all_action_emb, all_text_emb = [], []

    for batch in loader:
        result = _extract_episode_batch(
            model, batch, device, args.tokenizer)
        aux_input = (result['fsq_codes'] if getattr(args, 'aux_target', 'latent') == 'post_fsq'
                     else result['latents'])
        action_emb = clip_head(aux_input, result['n_valid'],
                               positions=result['positions'])
        instructions = result['instructions']
        text_features = text_encoder(list(instructions))
        text_emb = F.normalize(text_proj(text_features), dim=-1)

        all_action_emb.append(action_emb.cpu())
        all_text_emb.append(text_emb.cpu())

    if not all_action_emb:
        return {f'r@{k}': 0.0 for k in ks}

    A = torch.cat(all_action_emb, 0)  # (N, proj_dim)
    T = torch.cat(all_text_emb,   0)  # (N, proj_dim)
    N = A.shape[0]

    sim = A @ T.t()

    results = {}
    for k in ks:
        k_clamped = min(k, N)
        topk_idx = sim.topk(k_clamped, dim=1).indices
        gt = torch.arange(N).unsqueeze(1)
        hit_a2t = (topk_idx == gt).any(dim=1).float().mean().item()
        topk_idx_t = sim.t().topk(k_clamped, dim=1).indices
        hit_t2a = (topk_idx_t == gt).any(dim=1).float().mean().item()
        results[f'r@{k}'] = 100.0 * (hit_a2t + hit_t2a) / 2.0

    return results
