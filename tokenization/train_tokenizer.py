"""Unified action tokenizer training / fitting script.

Supports all tokenizer types:
  Gradient-based (iterative training):
    vq_bet   - chunk-based MLP + ResidualVQ (VQ-BeT paper)
    oat      - register encoder + FSQ (from oat/)
    quest    - causal conv + FSQ (from oat/)

  Fit-once (non-gradient):
    fast     - DCT + BPE fitting
    bin      - analytical binning (no training, eval only)

Optional auxiliary losses (for gradient-based tokenizers):
    --aux_head verb  - verb classification head on pooled latents
    --aux_head clip  - contrastive action-language head

Usage:
    # VQ-BeT from scratch
    python tokenization/train_tokenizer.py --tokenizer vq_bet --epochs 100

    # OAT from scratch with verb head
    python tokenization/train_tokenizer.py --tokenizer oat --aux_head verb --aux_lambda 0.5 --epochs 50

    # FAST fit
    python tokenization/train_tokenizer.py --tokenizer fast --fast_vocab_size 1024

    # Resume from checkpoint
    python tokenization/train_tokenizer.py --tokenizer vq_bet --resume checkpoints/vq_bet_vanilla/full.pth
"""

import os
import sys
import time as time_mod
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# Project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_TOKENIZATION_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOKENIZATION_DIR not in sys.path:
    sys.path.insert(0, _TOKENIZATION_DIR)

from utils import load_calvin_to_dataframe
from config import (
    DATA_DIR, VAL_DIR, ACTION_KEY, EPISODE_TEMPLATE, ACTION_DIM,
    TOKENIZER_DOWNSAMPLE_FACTOR, OAT_NUM_REGISTERS,
)
from tokenization.train_utils import (
    resume_checkpoint, setup_output_dir, open_csv_logger,
    log_epoch, write_csv_row, save_best_checkpoint, save_final_config,
)
from datasets.bridge_dataset import (
    BridgeTokenizerDataset,
    load_bridge_actions, load_bridge_verb_labels, load_bridge_instructions,
    fit_bridge_normalizer,
)
from tokenization.aux_heads import contrastive_loss, build_aux_heads
from tokenization.eval_tokenizer import eval_epoch, eval_clip_retrieval


# ======================================================================
# Lazy imports (heavy deps)
# ======================================================================

def _import_vqbet():
    from tokenization.vq_bet_official.vqvae.tokenizer import VQBeTTokenizer
    return VQBeTTokenizer

def _import_oat():
    from oat.tokenizer.oat.tokenizer import OATTok
    from oat.tokenizer.oat.encoder.register_encoder import RegisterEncoder
    from oat.tokenizer.oat.decoder.single_pass_decoder import SinglePassDecoder
    from oat.tokenizer.oat.quantizer.fsq import FSQ
    return OATTok, RegisterEncoder, SinglePassDecoder, FSQ

def _import_quest():
    from oat.tokenizer.quest.tokenizer import QueSTTok
    return QueSTTok

def _import_fast():
    from tokenization.fast.fast_tokenizer import FASTTokenizer, collect_trajectories
    return FASTTokenizer, collect_trajectories


# ======================================================================
# Tokenizer builders
# ======================================================================

def build_vqbet(args):
    VQBeTTokenizer = _import_vqbet()
    model = VQBeTTokenizer(
        action_dim=ACTION_DIM, chunk_size=args.chunk_size,
        latent_dim=args.latent_dim, n_embed=args.num_codes,
        groups=args.vq_groups, hidden_dim=args.hidden_dim,
        num_layers=args.num_mlp_layers)
    return model


def build_oat(args):
    OATTok, RegisterEncoder, SinglePassDecoder, FSQ = _import_oat()
    levels = getattr(args, 'fsq_levels', [8, 5, 5, 5])
    num_registers = getattr(args, 'num_registers', OAT_NUM_REGISTERS)
    latent_dim = len(levels)
    cs = getattr(args, 'horizon', getattr(args, 'chunk_size', 32))
    enc = RegisterEncoder(
        sample_dim=ACTION_DIM, sample_horizon=cs,
        emb_dim=256, head_dim=64, depth=2, pdropout=0.1,
        latent_dim=latent_dim, num_registers=num_registers)
    dec = SinglePassDecoder(
        sample_dim=ACTION_DIM, sample_horizon=cs,
        emb_dim=256, head_dim=64, depth=4, pdropout=0.1,
        token_dropout_mode="pow2", latent_dim=latent_dim,
        latent_horizon=num_registers, use_causal_decoder=True)
    q = FSQ(levels=levels)
    tok = OATTok(encoder=enc, decoder=dec, quantizer=q)
    return tok


def build_quest(args):
    QueSTTok = _import_quest()
    levels = getattr(args, 'fsq_levels', [8, 5, 5, 5])
    ds = getattr(args, 'downsample_factor', TOKENIZER_DOWNSAMPLE_FACTOR)
    vq_type = getattr(args, 'vq_type', 'fsq')
    horizon = getattr(args, 'horizon', getattr(args, 'chunk_size', 32))
    tok = QueSTTok(
        action_dim=ACTION_DIM, horizon=horizon,
        vq_type=vq_type, fsq_level=levels,
        vq_codebook_size=getattr(args, 'vq_codebook_size', 256),
        vq_codebook_dim=getattr(args, 'vq_codebook_dim', 256),
        downsample_factor=ds)
    return tok


# ======================================================================
# Training loop
# ======================================================================

# Shared batch encoding — lives in datasets/ next to CalvinTokenizerDataset
from tokenization.train_utils import extract_episode_batch as _extract_episode_batch


def train_epoch(model, loader, optimizer, device, args,
                verb_head=None, verb_criterion=None,
                clip_head=None, text_encoder=None, text_proj=None):
    model.train()
    if verb_head is not None:
        verb_head.train()
    if clip_head is not None:
        clip_head.train()

    totals = {'recon': 0, 'vq': 0, 'verb': 0, 'clip': 0}
    correct = total = 0
    all_preds, all_labels = [], []
    n_batches = 0

    for batch in loader:
        result = _extract_episode_batch(
            model, batch, device, args.tokenizer)
        loss = result['recon_loss']
        loss = loss + args.vq_weight * result.get('vq_loss', torch.tensor(0.0, device=device))

        # Select aux input: post-FSQ 4d codes or pre-FSQ 256d latents
        aux_input = (result['fsq_codes'] if args.aux_target == 'post_fsq'
                     else result['latents'])

        # Verb classification
        if verb_head is not None and args.verb_cls_lambda > 0:
            verb_logits = verb_head(aux_input, result['n_valid'],
                                    positions=result['positions'])
            verb_ids = result['verb_ids']
            valid = verb_ids >= 0
            if valid.any():
                verb_loss = verb_criterion(verb_logits[valid], verb_ids[valid])
                loss = loss + args.verb_cls_lambda * verb_loss
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
            with torch.set_grad_enabled(text_encoder.lora_r > 0):
                text_features = text_encoder(list(instructions))
            text_emb = F.normalize(text_proj(text_features), dim=-1)
            clip_loss = contrastive_loss(
                action_emb, text_emb, list(instructions), clip_head.temperature)
            loss = loss + args.clip_lambda * clip_loss
            totals['clip'] += clip_loss.item()

        optimizer.zero_grad()
        loss.backward()
        if args.max_grad_norm > 0:
            all_params = [p for p in model.parameters() if p.requires_grad]
            if verb_head is not None:
                all_params += list(verb_head.parameters())
            if clip_head is not None:
                all_params += list(clip_head.parameters())
                all_params += list(text_proj.parameters())
                if text_encoder.lora_r > 0:
                    all_params += [p for p in text_encoder.parameters()
                                   if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(all_params, args.max_grad_norm)
        optimizer.step()

        totals['recon'] += result['recon_loss'].item()
        totals['vq'] += result.get('vq_loss', torch.tensor(0)).item()
        n_batches += 1

    macro_f1 = 0.0
    if all_preds:
        macro_f1 = 100.0 * f1_score(
            torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy(),
            average='macro', zero_division=0)
    return {k: v / max(n_batches, 1) for k, v in totals.items()} | {
        'verb_acc': 100.0 * correct / max(total, 1),
        'verb_macro_f1': macro_f1,
    }



# ======================================================================
# Normalizer fitting (for OAT/QueST)
# ======================================================================

def fit_normalizer(data_dir, max_trajs=2000):
    """Fit oat LinearNormalizer on CALVIN actions."""
    from oat.model.common.normalizer import LinearNormalizer
    from analysis.cluster_analysis import load_all_actions

    df = load_calvin_to_dataframe(data_dir)
    if max_trajs:
        df = df.head(min(max_trajs, len(df))).copy()
    all_actions, _ = load_all_actions(df, num_workers=8)
    actions_t = torch.from_numpy(all_actions)

    normalizer = LinearNormalizer()
    normalizer.fit({"action": actions_t}, last_n_dims=1, mode="limits",
                   output_min=-1.0, output_max=1.0)
    return normalizer


# ======================================================================
# Data loading
# ======================================================================

def _compute_verb_class_weights(num_verbs, verb_to_id, train_verb_ids=None,
                                train_df=None, verb_col=None):
    """Compute inverse-frequency class weights for verb CE loss.

    Bridge path: uses train_verb_ids (list of int).
    Calvin path: uses train_df + verb_col + verb_to_id.
    """
    from collections import Counter
    weights = torch.zeros(num_verbs)
    if train_verb_ids is not None:
        id_counts = Counter(v for v in train_verb_ids if v >= 0)
        for cid, cnt in id_counts.items():
            weights[cid] = 1.0 / cnt
    elif train_df is not None and verb_col is not None:
        class_counts = train_df[verb_col].value_counts()
        for verb, cid in verb_to_id.items():
            weights[cid] = 1.0 / class_counts.get(verb, 1)
    weights = weights / weights.sum() * num_verbs
    return weights


def build_dataloaders(args):
    """Build train/val datasets, loaders, normalizer, and verb class weights.

    Returns dict with keys:
        train_ds, val_ds, train_loader, val_loader, normalizer,
        num_verbs (int, 0 if no verb head),
        verb_class_weights (Tensor or None)
    """
    aux_head = args.aux_head

    verb_class_weights = None
    num_verbs = 0

    if args.dataset == "bridge":
        import pandas as pd

        # Bridge: load shards, then filter to episodes in the CSV
        all_actions, all_keys = load_bridge_actions(args.shard_dir)
        csv_df = pd.read_csv(args.bridge_csv)
        csv_key_set = set(csv_df["episode_key"])

        # Keep only episodes that appear in the CSV
        n_total = len(all_actions)
        keep_idx = [i for i, k in enumerate(all_keys) if k in csv_key_set]
        all_actions = [all_actions[i] for i in keep_idx]
        all_keys = [all_keys[i] for i in keep_idx]
        print(f"Filtered to {len(all_actions)}/{n_total} episodes "
              f"using {args.bridge_csv}")

        np.random.seed(42)
        perm = np.random.permutation(len(all_actions))
        n_val = max(1, int(len(all_actions) * args.val_fraction))
        train_actions = [all_actions[i] for i in perm[n_val:]]
        val_actions = [all_actions[i] for i in perm[:n_val]]
        print(f"Train: {len(train_actions)} episodes, Val: {len(val_actions)} episodes")

        train_verb_ids, val_verb_ids, verb_to_id = None, None, None
        if aux_head == 'verb':
            all_verb_ids, verb_to_id = load_bridge_verb_labels(
                args.bridge_csv, all_keys, min_class_count=args.min_class_count)
            train_verb_ids = [all_verb_ids[i] for i in perm[n_val:]]
            val_verb_ids = [all_verb_ids[i] for i in perm[:n_val]]

        train_instructions, val_instructions = None, None
        if aux_head == 'clip':
            all_instructions = load_bridge_instructions(args.bridge_csv, all_keys)
            train_instructions = [all_instructions[i] for i in perm[n_val:]]
            val_instructions = [all_instructions[i] for i in perm[:n_val]]

        sampling = args.sampling
        max_k = args.max_chunks if aux_head else 1

        train_ds = BridgeTokenizerDataset(
            train_actions, chunk_size=args.chunk_size, max_chunks=max_k,
            sampling=sampling,
            verb_ids=train_verb_ids, verb_to_id=verb_to_id,
            instructions=train_instructions)
        val_ds = BridgeTokenizerDataset(
            val_actions, chunk_size=args.chunk_size, max_chunks=max_k,
            sampling=sampling,
            verb_ids=val_verb_ids, verb_to_id=verb_to_id,
            instructions=val_instructions)
        train_ds.verb_to_id = verb_to_id or {}
        train_ds.id_to_verb = {v: k for k, v in train_ds.verb_to_id.items()}
        val_ds.verb_to_id = train_ds.verb_to_id
        val_ds.id_to_verb = train_ds.id_to_verb

        normalizer = fit_bridge_normalizer(train_actions)

        if aux_head == 'verb' and verb_to_id:
            num_verbs = len(verb_to_id)
            verb_class_weights = _compute_verb_class_weights(
                num_verbs, verb_to_id, train_verb_ids=train_verb_ids)
    else:
        # CALVIN
        from datasets.calvin_dataset import build_calvin_tokenizer_data

        max_k = args.max_chunks if aux_head else 1
        include_instr = (aux_head == 'clip')
        min_cc = args.min_class_count if aux_head == 'verb' else 0

        train_ds, val_ds, num_verbs, _, _, _ = \
            build_calvin_tokenizer_data(
                args.data_dir, args.val_dir,
                chunk_size=args.chunk_size, max_chunks=max_k,
                sampling=args.sampling, min_class_count=min_cc,
                cache_actions=True, include_instruction=include_instr)

        normalizer = fit_normalizer(args.data_dir)

        if aux_head == 'verb' and train_ds.verb_to_id:
            num_verbs = len(train_ds.verb_to_id)
            verb_class_weights = _compute_verb_class_weights(
                num_verbs, train_ds.verb_to_id,
                train_df=train_ds.df,
                verb_col=train_ds._verb_col)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    return dict(
        train_ds=train_ds, val_ds=val_ds,
        train_loader=train_loader, val_loader=val_loader,
        normalizer=normalizer,
        num_verbs=num_verbs,
        verb_class_weights=verb_class_weights,
    )


# ======================================================================
# FAST fitting (non-gradient)
# ======================================================================

def fit_fast(args):
    """Fit FAST tokenizer (DCT + BPE). No gradient training."""
    FASTTokenizer, collect_trajectories = _import_fast()

    train_df = load_calvin_to_dataframe(args.data_dir)
    trajectories = collect_trajectories(train_df, args.data_dir)
    print(f"Collected {len(trajectories)} trajectories for FAST fitting")

    tok = FASTTokenizer.fit(trajectories, scale=args.fast_scale,
                            vocab_size=args.fast_vocab_size)

    save_dir = args.save_dir or os.path.join("checkpoints", f"fast_s{args.fast_scale}_v{args.fast_vocab_size}")
    os.makedirs(save_dir, exist_ok=True)
    tok.save(os.path.join(save_dir, "fast_tokenizer"))
    print(f"FAST tokenizer saved to {save_dir}")
    return tok


# ======================================================================
# Main
# ======================================================================

_DEFAULT_TOKENIZER_CONFIGS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "tokenizers")


def _load_tokenizer_config(tokenizer_type, config_path=None):
    """Load tokenizer-specific defaults from YAML.

    Resolution order:
        1. Explicit --config path (if given)
        2. configs/tokenizers/{tokenizer_type}.yaml (auto-discovered)
        3. Empty dict (fall back to argparse defaults)
    """
    import yaml

    if config_path:
        path = config_path
    else:
        path = os.path.join(_DEFAULT_TOKENIZER_CONFIGS, f"{tokenizer_type}.yaml")

    if os.path.isfile(path):
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        print(f"Loaded tokenizer config: {path}")
        return cfg

    if config_path:
        raise FileNotFoundError(f"Config not found: {config_path}")
    return {}


def parse_args():
    """Parse command-line arguments for tokenizer training.

    Tokenizer-specific hyper-parameters (chunk_size, fsq_levels, etc.) are
    loaded from a YAML config file and can be overridden on the CLI via
    ``--set key=value``.  See ``configs/tokenizers/*.yaml`` for defaults.
    """
    parser = argparse.ArgumentParser(description="Unified action tokenizer training")

    # Core
    parser.add_argument("--tokenizer", type=str, required=True,
                        choices=["vq_vae", "vq_bet", "vqvla", "oat", "quest", "fast", "bin"])
    parser.add_argument("--config", type=str, default=None,
                        help="Path to tokenizer YAML config (default: auto-discover "
                             "from configs/tokenizers/{tokenizer}.yaml)")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides", metavar="KEY=VAL",
                        help="Override any config/arg value, e.g. --set chunk_size=8 lr=3e-4")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional suffix appended to auto-generated run name")
    parser.add_argument("--dataset", type=str, default="calvin",
                        choices=["calvin", "bridge"],
                        help="Dataset to train on (default: calvin)")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--val_dir", type=str, default=VAL_DIR)
    parser.add_argument("--shard_dir", type=str,
                        default="/data/user_data/wenjiel2/datasets/bridge_actions",
                        help="BridgeV2 action shard directory (only used with --dataset bridge)")
    parser.add_argument("--bridge_csv", type=str,
                        default="data/bridge_episodes_filtered.csv",
                        help="BridgeV2 episode CSV with verb labels (Bridge + verb head)")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Fraction of episodes for validation (Bridge only)")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for model initialization (default: None = no seed)")
    parser.add_argument("--num_workers", type=int, default=4)

    # Aux head
    parser.add_argument("--aux_head", type=str, default="none",
                        choices=["none", "verb", "clip"],
                        help="Auxiliary head type (default: none)")
    parser.add_argument("--aux_lambda", type=float, default=0.5,
                        help="Loss weight for the auxiliary head (default: 0.5)")
    parser.add_argument("--aux_target", type=str, default="latent",
                        choices=["latent", "post_fsq"],
                        help="Which representation to feed to aux head: "
                             "'latent' = 256d pre-FSQ, 'post_fsq' = 4d post-round with STE")
    parser.add_argument("--min_class_count", type=int, default=30,
                        help="Min samples per verb class (sparse filtering)")
    parser.add_argument("--max_chunks", type=int, default=8,
                        help="Max chunks per episode for aux head")
    parser.add_argument("--sampling", type=str, default="random",
                        choices=["random", "sequential"],
                        help="Chunk sampling strategy (random=OAT/QueST, sequential=VQ-BeT)")

    # CLIP-specific
    parser.add_argument("--text_model", type=str,
                        default='laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
    parser.add_argument("--text_type", type=str, default='clip',
                        choices=['clip', 'gpt2'])
    parser.add_argument("--text_lora_r", type=int, default=0)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--clip_d_model", type=int, default=128)
    parser.add_argument("--clip_transformer_layers", type=int, default=2)

    # Checkpoint
    parser.add_argument("--max_episodes_per_epoch", type=int, default=None,
                        help="Subsample training episodes per epoch (for large datasets)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to full.pth checkpoint to resume from")
    
    # Semantic Loss
    parser.add_argument("--loss_function", type=str, default="ce", choices=["ce", "semantic"],
                        help="Loss function to use for the verb head")
    parser.add_argument("--semantic_temp", type=float, default=0.1,
                        help="Temperature for the semantic similarity softmax")

    args = parser.parse_args()

    # ── Load tokenizer config YAML and merge ───────────────────────────
    tok_cfg = _load_tokenizer_config(args.tokenizer, args.config)
    for key, value in tok_cfg.items():
        if not hasattr(args, key):
            setattr(args, key, value)
        # YAML provides defaults; don't overwrite CLI-explicit values.
        # Since tokenizer params aren't in argparse, they're always "new".
        # For shared keys that ARE in argparse (like vq_weight), YAML wins
        # only if the user didn't pass them on the CLI.
        # We handle this below via --set overrides.

    # ── Apply --set overrides (highest priority) ───────────────────────
    for item in args.overrides:
        if "=" not in item:
            parser.error(f"--set values must be KEY=VALUE, got: {item}")
        key, val = item.split("=", 1)
        # Auto-cast: list, int, float, bool, or string
        if val.startswith("[") and val.endswith("]"):
            import ast
            val = ast.literal_eval(val)
        else:
            for cast in (int, float):
                try:
                    val = cast(val)
                    break
                except ValueError:
                    continue
            else:
                if val.lower() in ("true", "false"):
                    val = val.lower() == "true"
        setattr(args, key, val)

    # ── Ensure all expected tokenizer attrs exist with sensible fallbacks
    _defaults = dict(
        chunk_size=4, num_codes=512, latent_dim=64, vq_groups=4,
        hidden_dim=128, num_mlp_layers=1, fsq_levels=[8, 5, 5, 5],
        num_registers=OAT_NUM_REGISTERS,
        downsample_factor=TOKENIZER_DOWNSAMPLE_FACTOR,
        vq_type="fsq", vq_codebook_size=256, vq_codebook_dim=256,
        vqvla_config_dir=None, vqvla_pretrained=None,
        fast_vocab_size=1024, fast_scale=10.0, vq_weight=5.0,
    )
    for key, default in _defaults.items():
        if not hasattr(args, key):
            setattr(args, key, default)

    # For OAT/QueST, chunk_size must equal horizon (dataset chunking = model horizon)
    if args.tokenizer in ('quest', 'oat') and hasattr(args, 'horizon'):
        args.chunk_size = args.horizon

    # Derive verb_cls_lambda / clip_lambda from aux_head + aux_lambda
    args.verb_cls_lambda = args.aux_lambda if args.aux_head == 'verb' else 0.0
    args.clip_lambda = args.aux_lambda if args.aux_head == 'clip' else 0.0

    return args


def main():
    args = parse_args()

    # ── Handle non-gradient tokenizers ──────────────────────────────────
    if args.tokenizer == "fast":
        fit_fast(args)
        return

    if args.tokenizer == "bin":
        print("Bin tokenizer is analytical — no training needed.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"Torch seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Tokenizer: {args.tokenizer}")

    data = build_dataloaders(args)
    train_ds = data['train_ds']
    val_ds = data['val_ds']
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    normalizer = data['normalizer']
    aux_head = args.aux_head

    # ── Build tokenizer model ───────────────────────────────────────────
    if args.tokenizer == 'vq_bet':
        model = build_vqbet(args)
    elif args.tokenizer == 'oat':
        model = build_oat(args)
    elif args.tokenizer == 'quest':
        model = build_quest(args)
    model.set_normalizer(normalizer)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tokenizer model: {n_params:,} trainable params")

    # ── Token distribution diagnostic ──────────────────────────────────
    with torch.no_grad():
        model.eval()
        n_valid_list = []
        for batch in train_loader:
            n_valid_list.append(batch['n_valid'])
            if len(n_valid_list) * args.batch_size >= len(train_ds):
                break
        all_n_valid = torch.cat(n_valid_list)
        # Run one batch to get latent shape
        result = _extract_episode_batch(model, batch, device, args.tokenizer)
        lat = result['latents']
        print(f"\n--- Token distribution ({args.tokenizer}) ---")
        print(f"  Chunks per episode:  min={all_n_valid.min().item()}, "
              f"mean={all_n_valid.float().mean().item():.1f}, "
              f"max={all_n_valid.max().item()}")
        print(f"  Latent shape (batch): {tuple(lat.shape)}")
        print(f"  Tokens per episode:  {result['n_valid'][0].item()} "
              f"(n_valid after expansion)")
        print(f"  Latent dim:          {lat.shape[-1]}")
        if result.get('codes') is not None:
            print(f"  Codes shape:         {tuple(result['codes'].shape)}")
        print()
        model.train()

    # ── Build aux heads ─────────────────────────────────────────────────
    clip_config = None
    if aux_head == 'clip':
        clip_config = dict(
            d_model=args.clip_d_model,
            transformer_layers=args.clip_transformer_layers,
            proj_dim=args.proj_dim,
            text_model=args.text_model,
            text_type=args.text_type,
            text_lora_r=args.text_lora_r,
        )

    # FSQ dim = number of FSQ levels (e.g. [8,5,5,5] → 4)
    fsq_dim = len(args.fsq_levels) if hasattr(args, 'fsq_levels') else None
    heads = build_aux_heads(
        args.tokenizer, device,
        latent_dim=args.latent_dim,
        num_verbs=data['num_verbs'],
        verb_class_weights=data['verb_class_weights'],
        clip_config=clip_config,
        loss_function=getattr(args, 'loss_function', 'ce'),
        semantic_temp=getattr(args, 'semantic_temp', 0.1),
        id_to_verb=getattr(train_ds, 'id_to_verb', None),
        aux_target=args.aux_target,
        fsq_dim=fsq_dim,
    )
    verb_head = heads['verb_head']
    verb_criterion = heads['verb_criterion']
    clip_head = heads['clip_head']
    text_encoder = heads['text_encoder']
    text_proj = heads['text_proj']

    # ── Precompute text embeddings if encoder is frozen ────────────────
    if text_encoder is not None and text_encoder.lora_r == 0:
        all_instructions = set()
        for ds in (train_ds, val_ds):
            if hasattr(ds, 'df') and 'instruction' in ds.df.columns:
                all_instructions.update(ds.df['instruction'].dropna().unique())
            elif hasattr(ds, 'instructions') and ds.instructions:
                all_instructions.update(ds.instructions)
        # Remove empty strings — no embedding needed for missing instructions
        all_instructions.discard("")
        if all_instructions:
            text_encoder.precompute_cache(list(all_instructions))

    # ── Optimizer ───────────────────────────────────────────────────────
    params = [p for p in model.parameters() if p.requires_grad]
    if verb_head is not None:
        params += list(verb_head.parameters())
    if clip_head is not None:
        params += list(clip_head.parameters())
        params += list(text_proj.parameters())
        if text_encoder.lora_r > 0:
            params += [p for p in text_encoder.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # ── Resume from checkpoint ──────────────────────────────────────────
    start_epoch, best_metric, best_verb_acc = resume_checkpoint(
        args, model, optimizer, verb_head, clip_head, text_proj, device)
    patience_counter = 0

    # ── Output directory (auto-generated name) ─────────────────────────
    save_dir, run_name = setup_output_dir(args)
    csv_writer, csv_file = open_csv_logger(save_dir, resume=bool(args.resume))

    # ── Training ────────────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs")
    print(f"  aux_head={args.aux_head}, aux_lambda={args.aux_lambda}")
    print(f"  vq_weight={args.vq_weight}")
    print(f"  Save dir: {save_dir}")
    print("=" * 80)

    for epoch in range(start_epoch, args.epochs):
        t0 = time_mod.time()

        # Subsample training data if requested
        if args.max_episodes_per_epoch and len(train_ds) > args.max_episodes_per_epoch:
            from torch.utils.data import Subset
            subset_idx = np.random.choice(len(train_ds), args.max_episodes_per_epoch, replace=False)
            epoch_loader = DataLoader(Subset(train_ds, subset_idx),
                                      batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, pin_memory=True)
        else:
            epoch_loader = train_loader

        train_m = train_epoch(
            model, epoch_loader, optimizer, device, args,
            verb_head=verb_head, verb_criterion=verb_criterion,
            clip_head=clip_head, text_encoder=text_encoder, text_proj=text_proj)

        val_m = eval_epoch(
            model, val_loader, device, args,
            verb_head=verb_head, verb_criterion=verb_criterion,
            clip_head=clip_head, text_encoder=text_encoder, text_proj=text_proj)

        # CLIP retrieval metrics
        retrieval = {}
        if aux_head == 'clip':
            retrieval = eval_clip_retrieval(
                model, val_loader, device, args,
                clip_head, text_encoder, text_proj, ks=(1, 5, 10))

        scheduler.step()
        dt = time_mod.time() - t0

        log_epoch(epoch, args.epochs, dt, train_m, val_m, aux_head, retrieval)

        # Best checkpoint: monitor total weighted val loss (lower = better)
        val_total = val_m['recon'] + args.vq_weight * val_m['vq']
        if aux_head == 'verb':
            v = val_m['verb']
            val_total += args.verb_cls_lambda * (v if not np.isnan(v) else 0.0)
        if aux_head == 'clip':
            val_total += args.clip_lambda * val_m['clip']
        is_best = val_total < best_metric
        if is_best:
            best_metric = val_total
            if aux_head == 'verb':
                best_verb_acc = val_m['verb_acc']

        if is_best:
            patience_counter = 0
            save_best_checkpoint(
                save_dir, epoch, model, optimizer, train_m, val_m,
                args, best_metric, best_verb_acc,
                verb_head=verb_head, train_ds=train_ds,
                clip_head=clip_head, text_proj=text_proj)
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1} "
                      f"({patience_counter} epochs without improvement)")
                break

        write_csv_row(csv_writer, csv_file, epoch, train_m, val_m,
                      retrieval, optimizer.param_groups[0]['lr'], dt)

    csv_file.close()

    save_final_config(save_dir, args, run_name, epoch, best_metric, best_verb_acc)


if __name__ == "__main__":
    main()
