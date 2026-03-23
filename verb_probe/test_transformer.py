"""Evaluate a trained verb classifier checkpoint.

Loads a saved checkpoint, runs inference on the validation set, and outputs:
per-class precision/recall/F1, overall accuracy, macro/weighted F1,
confusion matrix PNG, and metrics JSON.

Modality and action_rep are read from checkpoint metadata automatically.
Supports CALVIN, Bridge, and DROID datasets.

Usage:
    python verb_probe/test_transformer.py --model_path ./checkpoints/model_best.pth \\
        --save_cm ./figures/cm.png --save_metrics ./results/metrics.json
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
try:
    from torchvision import transforms
except (ImportError, RuntimeError):
    transforms = None
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from verb_probe.models import ActionToVerbTransformer, SCENE_FUSION_MODALITIES
from config import (
    VAL_DIR, D_MODEL, NHEAD, NUM_LAYERS, CROSS_LAYERS, ACTION_DIM,
    SCENE_OBS_DIM, PATCH_SIZE, IMAGE_SIZE, IMG_MEAN, IMG_STD,
    BATCH_SIZE, MAX_SEQ_LEN, NUM_WORKERS, FAST_TOKENIZER_PATH,
    IMAGE_ENCODER, QUEST_TOKENIZER_CKPT, OAT_TOKENIZER_CKPT,
    TOKENIZER_HORIZON, TOKENIZER_FIT_NORM_MAX_TRAJS,
)


def _load_checkpoint(model_path, device):
    """Load checkpoint and extract metadata."""
    raw = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(raw, dict) and 'state_dict' in raw:
        meta = {k: v for k, v in raw.items() if k != 'state_dict'}
        state_dict = raw['state_dict']
    else:
        # Legacy bare state_dict
        state_dict = raw
        classifier_bias_keys = [
            k for k in state_dict
            if k.startswith('classifier.') and k.endswith('.bias')]
        last_bias_key = sorted(classifier_bias_keys,
                               key=lambda k: int(k.split('.')[1]))[-1]
        meta = {'num_verbs': state_dict[last_bias_key].shape[0]}

    # Defaults for missing metadata
    defaults = {
        'num_verbs': None, 'verb_to_id': None, 'id_to_verb': None,
        'd_model': D_MODEL, 'nhead': NHEAD, 'num_layers': NUM_LAYERS,
        'action_dim': ACTION_DIM, 'patch_size': PATCH_SIZE,
        'img_size': IMAGE_SIZE[0], 'max_action_len': MAX_SEQ_LEN,
        'modality': 'action_only', 'action_rep': 'native',
        'action_vocab_size': None, 'cross_layers': CROSS_LAYERS,
        'image_encoder': IMAGE_ENCODER, 'freeze_vision': True,
        'num_frames': 2, 'delta_patches': 0,
        'modal_dropout': 0.0, 'aux_loss_weight': 0.0, 'scene_dim': 0,
        'dataset': 'calvin',
    }
    for k, v in defaults.items():
        meta.setdefault(k, v)

    # Backward compat: old key names
    if meta['action_vocab_size'] is None:
        meta['action_vocab_size'] = meta.pop('fast_vocab_size', None)
    if meta.get('image_encoder') is None:
        _map = {'patch': 'scratch', 'r3m': 'r3m', 'dinov2_s': 'dinov2_s',
                'dinov2_b': 'dinov2_b', 'vc1': 'vc1'}
        meta['image_encoder'] = _map.get(meta.pop('vision_encoder', 'patch'), 'scratch')

    # Override action_dim for oracle modality
    if meta['modality'] == 'scene_obs':
        meta['action_dim'] = SCENE_OBS_DIM

    return state_dict, meta


def _load_action_tokenizer(meta, args):
    """Load action tokenizer based on checkpoint metadata."""
    action_rep = meta['action_rep']
    max_action_len = meta['max_action_len']

    if action_rep == 'native':
        return None

    if action_rep in ('bin', 'quest', 'oat'):
        from tokenization.action_tokenizers import load_action_tokenizer
        tok = load_action_tokenizer(
            action_rep, train_dir=args.data_dir,
            horizon=TOKENIZER_HORIZON, max_tokens=max_action_len,
            quest_ckpt=args.quest_ckpt, oat_ckpt=args.oat_ckpt,
            fit_norm_max_trajs=TOKENIZER_FIT_NORM_MAX_TRAJS)
        meta['action_vocab_size'] = tok.vocab_size
        return tok

    if action_rep == 'fast':
        from tokenization.fast_tokenizer import load_fast_tokenizer, tokenize_trajectory
        _fast_tok = load_fast_tokenizer(args.fast_tokenizer_path)
        def _fast_wrapper(actions_batch):
            return [tokenize_trajectory(_fast_tok, actions_batch[0])]
        _fast_wrapper.vocab_size = _fast_tok.vocab_size
        meta['action_vocab_size'] = _fast_tok.vocab_size
        return _fast_wrapper

    if action_rep == 'vq_vae':
        from tokenization.vqvae_tokenizer import load_vqvae_tokenizer, tokenize_trajectory_vqvae
        from functools import partial
        _vq = load_vqvae_tokenizer(args.tokenizer_ckpt)
        tok = partial(tokenize_trajectory_vqvae, _vq)
        tok.vocab_size = _vq.num_codes
        meta['action_vocab_size'] = _vq.num_codes
        return tok

    if action_rep == 'vq_bet':
        from tokenization.vqbet_tokenizer import load_vqbet_tokenizer, tokenize_trajectory_vqbet
        from functools import partial
        _vqb = load_vqbet_tokenizer(args.tokenizer_ckpt)
        tok = partial(tokenize_trajectory_vqbet, _vqb)
        tok.vocab_size = _vqb.total_vocab_size
        meta['action_vocab_size'] = _vqb.total_vocab_size
        return tok

    if action_rep == 'vqvla':
        from tokenization.vqvae_tokenizer import (
            load_vqvla_tokenizer, tokenize_trajectory_vqvla, VQVLA_VOCAB_SIZE)
        from functools import partial
        _vqvla = load_vqvla_tokenizer(
            config_dir=args.vqvla_config_dir,
            checkpoint_path=args.vqvla_checkpoint_path)
        tok = partial(tokenize_trajectory_vqvla, _vqvla)
        tok.vocab_size = VQVLA_VOCAB_SIZE
        meta['action_vocab_size'] = VQVLA_VOCAB_SIZE
        return tok

    raise ValueError(f"Unknown action_rep: {action_rep}")


def _build_eval_dataset(args, meta, tok, transform, img_size):
    """Build evaluation dataset based on dataset type in checkpoint."""
    dataset_name = meta.get('dataset', 'calvin')
    modality = meta['modality']

    if dataset_name == 'calvin':
        from utils import load_calvin_to_dataframe
        from datasets.calvin_dataset import CalvinVerbProbeDataset
        df = load_calvin_to_dataframe(args.data_dir)
        ds = CalvinVerbProbeDataset(
            args.data_dir, df, modality=modality,
            action_tokenizer=tok,
            max_seq_len=meta['max_action_len'],
            num_frames=meta['num_frames'],
            delta_patches=meta['delta_patches'],
            image_encoder=meta['image_encoder'],
            transform=transform, img_size=img_size,
            cache_actions=True)
        verb_col = 'primary_verb'

    elif dataset_name in ('bridge', 'bridge_v2_subtask'):
        from datasets.bridge_dataset import BridgeVerbDataset, BRIDGE_CSV, BRIDGE_ACTIONS_NPZ
        csv_path = args.csv_path or BRIDGE_CSV
        df = pd.read_csv(csv_path)
        _, val_df = train_test_split(
            df, test_size=0.15, random_state=42, stratify=df['verb'])
        val_df = val_df.reset_index(drop=True)
        df = val_df
        npz = np.load(args.actions_npz or BRIDGE_ACTIONS_NPZ, allow_pickle=True)
        actions_cache = {k: npz[k] for k in npz.files if k.startswith("actions_")}
        npz.close()
        ds = BridgeVerbDataset(df, actions_cache,
                                max_seq_len=meta['max_action_len'])
        verb_col = 'verb'

    elif dataset_name == 'droid':
        from datasets.droid_dataset import (
            DroidVerbDataset, DroidGoalDataset, load_actions_cache,
            build_frames_index, DROID_CSV, DROID_ACTIONS_DIR, DROID_FRAMES_DIR)
        csv_path = args.csv_path or DROID_CSV
        df = pd.read_csv(csv_path)
        _, val_df = train_test_split(
            df, test_size=0.15, random_state=42, stratify=df['verb'])
        val_df = val_df.reset_index(drop=True)
        df = val_df
        if modality == 'vision_only':
            frames_index = build_frames_index(args.frames_dir or DROID_FRAMES_DIR)
            df = df[df['episode_idx'].isin(frames_index)].reset_index(drop=True)
            ds = DroidGoalDataset(df, frames_index, img_size=img_size)
        else:
            actions_cache, _ = load_actions_cache(args.actions_dir or DROID_ACTIONS_DIR)
            ds = DroidVerbDataset(df, actions_cache,
                                   max_seq_len=meta['max_action_len'])
        verb_col = 'verb'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return ds, df, verb_col


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Could not find model weights at {args.model_path}")

    # Load checkpoint
    state_dict, meta = _load_checkpoint(args.model_path, device)
    modality = meta['modality']
    action_rep = meta['action_rep']
    print(f"Checkpoint: {meta['num_verbs']} verbs, d_model={meta['d_model']}, "
          f"modality={modality}, action_rep={action_rep}, "
          f"image_encoder={meta['image_encoder']}, dataset={meta.get('dataset', 'calvin')}")

    # Image setup
    img_size = 224 if meta['image_encoder'] in ('r3m', 'dinov2_s', 'dinov2_b', 'vc1', 'dinov2') else meta['img_size']
    transform = None
    if transforms is not None:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)])

    # Action tokenizer
    tok = _load_action_tokenizer(meta, args)

    # Dataset
    ds, df, verb_col = _build_eval_dataset(args, meta, tok, transform, img_size)

    if args.debug:
        n = min(args.debug, len(df))
        ds.df = df.head(n).copy()

    # Override vocab from checkpoint
    if meta['verb_to_id'] is not None:
        ds.verb_to_id = meta['verb_to_id']
        ds.id_to_verb = meta['id_to_verb']
        valid_mask = df[verb_col].isin(meta['verb_to_id'].keys())
        if (~valid_mask).sum() > 0:
            print(f"Dropping {(~valid_mask).sum()} samples with verbs not in model vocab")
            ds.df = df[valid_mask].reset_index(drop=True)

    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    # Build model
    model = ActionToVerbTransformer(
        num_verbs=meta['num_verbs'], d_model=meta['d_model'],
        nhead=meta['nhead'], num_layers=meta['num_layers'],
        action_dim=meta['action_dim'], img_size=img_size,
        patch_size=meta['patch_size'],
        max_action_len=meta['max_action_len'],
        modality=modality, action_rep=action_rep,
        action_vocab_size=meta['action_vocab_size'],
        cross_layers=meta['cross_layers'],
        image_encoder=meta['image_encoder'],
        freeze_vision=meta['freeze_vision'],
        num_frames=meta['num_frames'],
        delta_patches=meta['delta_patches'],
        modal_dropout=meta['modal_dropout'],
        aux_loss_weight=meta['aux_loss_weight'],
        scene_dim=meta['scene_dim'])

    # Backward compat: remap state dict keys
    state_dict = {k.replace("transformer.layers.", "layers."): v
                  for k, v in state_dict.items()}
    state_dict = {k.replace("vision_enc.", "patch_embed."): v
                  for k, v in state_dict.items()}
    if "type_img_start" in state_dict and "frame_pos" not in state_dict:
        d = state_dict["type_img_start"].shape[-1]
        state_dict["type_img"] = state_dict.pop("type_img_start")
        type_end = state_dict.pop("type_img_end")
        frame_pos = torch.zeros(1, meta['num_frames'], 1, d)
        if meta['num_frames'] >= 2:
            frame_pos[0, 1, 0, :] = type_end.squeeze() - state_dict["type_img"].squeeze()
        state_dict["frame_pos"] = frame_pos

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Sync num_patches
    if hasattr(ds, 'num_patches') and hasattr(model, 'num_patches'):
        if modality not in ("action_only", "scene_obs") + SCENE_FUSION_MODALITIES:
            ds.num_patches = model.num_patches

    # Evaluation
    label_map = meta['id_to_verb'] or ds.id_to_verb
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            frames, actions, scene_vecs, labels, seq_lengths = batch
            frames = frames.to(device)
            actions, labels = actions.to(device), labels.to(device)
            scene_vecs = scene_vecs.to(device)
            seq_lengths = seq_lengths.to(device)

            logits = model(frames, actions, seq_lengths=seq_lengths, scene_vec=scene_vecs)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            if (batch_idx + 1) % 50 == 0:
                running_acc = 100 * sum(
                    p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
                print(f"  Processed {batch_idx + 1} batches... acc: {running_acc:.2f}%")

    # Report
    accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
    present_labels = sorted(set(all_labels + all_preds))
    target_names = [label_map[i] for i in present_labels]

    print(f"\n{'=' * 60}")
    print(f"EVALUATION COMPLETE  [{modality} / {action_rep}]")
    print(f"Total examples: {len(all_preds)}")
    print(f"Overall accuracy: {accuracy:.2f}%")
    print(f"{'=' * 60}")

    report_str = classification_report(all_labels, all_preds,
                                       labels=present_labels,
                                       target_names=target_names, digits=3)
    print(f"\n{report_str}")

    if args.save_metrics:
        report_dict = classification_report(all_labels, all_preds,
                                            labels=present_labels,
                                            target_names=target_names,
                                            digits=4, output_dict=True)
        metrics = {
            "modality": modality, "action_rep": action_rep,
            "dataset": meta.get('dataset', 'calvin'),
            "accuracy": accuracy, "num_examples": len(all_preds),
            "per_class": report_dict,
        }
        os.makedirs(os.path.dirname(args.save_metrics) or '.', exist_ok=True)
        with open(args.save_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.save_metrics}")

    if args.save_preds:
        os.makedirs(os.path.dirname(args.save_preds) or '.', exist_ok=True)
        with open(args.save_preds, "w") as f:
            json.dump({"labels": all_labels, "preds": all_preds,
                       "id_to_verb": {str(v): k for k, v in ds.verb_to_id.items()}}, f)
        print(f"Predictions saved to {args.save_preds}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=present_labels)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, xticks_rotation=45, cmap='Blues', values_format='d')
    ax.set_title(f"Confusion Matrix [{modality} / {action_rep}]")
    plt.tight_layout()

    if args.save_cm:
        plt.savefig(args.save_cm, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {args.save_cm}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate verb classifier checkpoint")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=VAL_DIR,
                        help="CALVIN val dir, or override for other datasets")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="CSV path for bridge/droid")
    parser.add_argument("--actions_npz", type=str, default=None)
    parser.add_argument("--actions_dir", type=str, default=None)
    parser.add_argument("--frames_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--save_cm", type=str, default=None)
    parser.add_argument("--save_metrics", type=str, default=None)
    parser.add_argument("--save_preds", type=str, default=None)
    parser.add_argument("--debug", type=int, default=0, metavar="N")
    # Tokenizer paths (used when checkpoint specifies a tokenized action_rep)
    parser.add_argument("--tokenizer_ckpt", type=str, default=None)
    parser.add_argument("--fast_tokenizer_path", type=str, default=FAST_TOKENIZER_PATH)
    parser.add_argument("--quest_ckpt", type=str, default=QUEST_TOKENIZER_CKPT)
    parser.add_argument("--oat_ckpt", type=str, default=OAT_TOKENIZER_CKPT)
    parser.add_argument("--vqvla_config_dir", type=str, default="./tokenization/vqvla/config")
    parser.add_argument("--vqvla_checkpoint_path", type=str,
                        default="./checkpoints/vqvla_pretrained/action_tokenizer_weight/all_data_vq.pth")

    args = parser.parse_args()
    main(args)
