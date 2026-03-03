"""Sample-level prediction agreement analysis for modality contribution decomposition.

Loads three checkpoints (action-only, scene-only, fusion) trained on the same
label set and computes:
  1. Per-sample correctness for each model
  2. 4-way agreement table (both correct / only-A / only-S / neither)
  3. Fusion model rescue rates in each quadrant
  4. Per-class breakdown of modality contributions
"""

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import defaultdict

from train_transformer import ActionToVerbTransformer, CalvinVerbDataset, SCENE_FUSION_MODALITIES
from utils import load_calvin_to_dataframe
from config import (
    VAL_DIR, ACTION_DIM, PATCH_SIZE, IMAGE_SIZE, IMG_MEAN, IMG_STD,
    BATCH_SIZE, MAX_SEQ_LEN, NUM_WORKERS, SCENE_OBS_DIM, SCENE_REP_DIM,
)


def load_model_from_checkpoint(ckpt_path, device):
    """Load model from checkpoint, return (model, metadata dict)."""
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = {
        'num_verbs': raw['num_verbs'],
        'd_model': raw.get('d_model', 128),
        'nhead': raw.get('nhead', 8),
        'num_layers': raw.get('num_layers', 4),
        'action_dim': raw.get('action_dim', ACTION_DIM),
        'patch_size': raw.get('patch_size', PATCH_SIZE),
        'img_size': raw.get('img_size', IMAGE_SIZE[0]),
        'max_action_len': raw.get('max_action_len', MAX_SEQ_LEN),
        'modality': raw['modality'],
        'action_rep': raw.get('action_rep', 'native'),
        'action_vocab_size': raw.get('action_vocab_size', None),
        'cross_layers': raw.get('cross_layers', 4),
        'image_encoder': raw.get('image_encoder', 'scratch'),
        'freeze_vision': raw.get('freeze_vision', True),
        'num_frames': raw.get('num_frames', 2),
        'delta_patches': raw.get('delta_patches', 0),
        'modal_dropout': raw.get('modal_dropout', 0.0),
        'aux_loss_weight': raw.get('aux_loss_weight', 0.0),
        'scene_dim': raw.get('scene_dim', 0),
        'verb_to_id': raw.get('verb_to_id', None),
        'id_to_verb': raw.get('id_to_verb', None),
    }
    state_dict = raw.get('model_state_dict', raw.get('state_dict'))

    model = ActionToVerbTransformer(
        num_verbs=meta['num_verbs'], d_model=meta['d_model'], nhead=meta['nhead'],
        num_layers=meta['num_layers'], action_dim=meta['action_dim'],
        img_size=meta['img_size'], patch_size=meta['patch_size'],
        max_action_len=meta['max_action_len'],
        modality=meta['modality'], action_rep=meta['action_rep'],
        action_vocab_size=meta['action_vocab_size'],
        cross_layers=meta['cross_layers'],
        image_encoder=meta['image_encoder'],
        freeze_vision=meta['freeze_vision'],
        num_frames=meta['num_frames'],
        delta_patches=meta['delta_patches'],
        modal_dropout=meta['modal_dropout'],
        aux_loss_weight=meta['aux_loss_weight'],
        scene_dim=meta['scene_dim'])

    state_dict = {k.replace("transformer.layers.", "layers."): v
                  for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, meta


def get_predictions(model, dataloader, device):
    """Run inference and return (preds, labels) as numpy arrays."""
    all_preds, all_labels = [], []
    with torch.no_grad():
        for frames, actions, scene_vecs, labels, seq_lengths in dataloader:
            frames = frames.to(device)
            actions = actions.to(device)
            scene_vecs = scene_vecs.to(device)
            seq_lengths = seq_lengths.to(device)
            logits = model(frames, actions, seq_lengths=seq_lengths, scene_vec=scene_vecs)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_ckpt", required=True, help="Action-only best checkpoint")
    parser.add_argument("--scene_ckpt", required=True, help="Scene MLP-only best checkpoint")
    parser.add_argument("--fusion_ckpt", required=True, help="Fusion (scene_token) best checkpoint")
    parser.add_argument("--data_dir", default=VAL_DIR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--min_class_count", type=int, default=30)
    parser.add_argument("--save_json", default="results/modality_contribution.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all 3 models
    print("Loading action-only model...")
    action_model, action_meta = load_model_from_checkpoint(args.action_ckpt, device)
    print("Loading scene-only model...")
    scene_model, scene_meta = load_model_from_checkpoint(args.scene_ckpt, device)
    print("Loading fusion model...")
    fusion_model, fusion_meta = load_model_from_checkpoint(args.fusion_ckpt, device)

    # Verify same verb set
    assert action_meta['verb_to_id'] == scene_meta['verb_to_id'] == fusion_meta['verb_to_id'], \
        "Models must have the same verb vocabulary"
    id_to_verb = fusion_meta['id_to_verb']
    verb_to_id = fusion_meta['verb_to_id']

    # Build shared transform (no vision encoder needed for these models)
    img_size = fusion_meta['img_size']
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
    ])

    # Load val dataframe
    df = load_calvin_to_dataframe(args.data_dir)
    if args.min_class_count > 0:
        counts = df['primary_verb'].value_counts()
        keep = counts[counts >= args.min_class_count].index
        df = df[df['primary_verb'].isin(keep)].reset_index(drop=True)

    # Create 3 datasets (same data, different modality flags)
    common = dict(transform=transform, max_seq_len=fusion_meta['max_action_len'],
                  image_encoder='scratch', img_size=img_size,
                  num_frames=2, delta_patches=0)

    ds_action = CalvinVerbDataset(df, args.data_dir, modality="action_only",
                                  scene_rep=False, **common)
    ds_scene = CalvinVerbDataset(df, args.data_dir, modality="scene_mlp",
                                 scene_rep=True, **common)
    ds_fusion = CalvinVerbDataset(df, args.data_dir, modality="scene_token",
                                  scene_rep=True, **common)

    # Override verb maps to match checkpoints
    for ds in [ds_action, ds_scene, ds_fusion]:
        ds.verb_to_id = verb_to_id
        ds.id_to_verb = id_to_verb
        valid = df['primary_verb'].isin(verb_to_id.keys())
        if (~valid).sum() > 0:
            ds.df = df[valid].reset_index(drop=True)

    dl_action = DataLoader(ds_action, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers)
    dl_scene = DataLoader(ds_scene, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers)
    dl_fusion = DataLoader(ds_fusion, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers)

    # Get predictions
    print("\nRunning action-only inference...")
    a_preds, a_labels = get_predictions(action_model, dl_action, device)
    print("Running scene-only inference...")
    s_preds, s_labels = get_predictions(scene_model, dl_scene, device)
    print("Running fusion inference...")
    f_preds, f_labels = get_predictions(fusion_model, dl_fusion, device)

    # Verify labels match
    assert np.array_equal(a_labels, s_labels) and np.array_equal(a_labels, f_labels), \
        "Label ordering must match across datasets"
    labels = a_labels

    N = len(labels)
    a_correct = (a_preds == labels)
    s_correct = (s_preds == labels)
    f_correct = (f_preds == labels)

    # ===== 1. Overall agreement table =====
    both = a_correct & s_correct           # redundant
    only_a = a_correct & ~s_correct        # unique action
    only_s = ~a_correct & s_correct        # unique scene
    neither = ~a_correct & ~s_correct      # hard

    print("\n" + "=" * 70)
    print("SAMPLE-LEVEL PREDICTION AGREEMENT ANALYSIS")
    print("=" * 70)
    print(f"\nTotal samples: {N}")
    print(f"Action-only accuracy:  {a_correct.sum()}/{N} = {100*a_correct.mean():.1f}%")
    print(f"Scene-only accuracy:   {s_correct.sum()}/{N} = {100*s_correct.mean():.1f}%")
    print(f"Fusion accuracy:       {f_correct.sum()}/{N} = {100*f_correct.mean():.1f}%")

    print(f"\n{'Category':<25} {'Count':>6} {'%':>7}  {'Fusion correct':>15} {'Fusion %':>9}")
    print("-" * 70)
    for name, mask in [("Both correct", both),
                       ("Only action correct", only_a),
                       ("Only scene correct", only_s),
                       ("Neither correct", neither)]:
        n = mask.sum()
        fc = (f_correct & mask).sum()
        pct = 100 * n / N
        fpct = 100 * fc / n if n > 0 else 0
        print(f"{name:<25} {n:>6} {pct:>6.1f}%  {fc:>15} {fpct:>8.1f}%")

    # Derived metrics
    union_correct = a_correct | s_correct  # samples where at least one unimodal model is right
    complementary = f_correct & neither    # fusion rescues from neither
    fusion_loss_both = ~f_correct & both   # fusion loses despite both being right

    print(f"\n--- Derived metrics ---")
    print(f"Union unimodal acc (either correct): {100*union_correct.mean():.1f}%")
    print(f"Fusion complementary gains (neither→correct): {complementary.sum()} / {neither.sum()} = {100*complementary.mean():.1f}% of total")
    print(f"Fusion regressions (both correct→fusion wrong): {fusion_loss_both.sum()} / {both.sum()}")

    # ===== 2. Per-class breakdown =====
    print(f"\n{'Class':<15} {'N':>4} {'A%':>5} {'S%':>5} {'F%':>5} | {'Both':>5} {'OnlyA':>5} {'OnlyS':>5} {'Neither':>7} | {'F|Neither':>9}")
    print("-" * 95)
    results_per_class = {}
    for vid in sorted(id_to_verb.keys()):
        vname = id_to_verb[vid]
        mask = (labels == vid)
        n = mask.sum()
        if n == 0:
            continue
        ac = a_correct[mask].sum()
        sc = s_correct[mask].sum()
        fc = f_correct[mask].sum()
        b = (both & mask).sum()
        oa = (only_a & mask).sum()
        os_ = (only_s & mask).sum()
        ne = (neither & mask).sum()
        fne = (f_correct & neither & mask).sum()
        print(f"{vname:<15} {n:>4} {100*ac/n:>5.1f} {100*sc/n:>5.1f} {100*fc/n:>5.1f} | "
              f"{b:>5} {oa:>5} {os_:>5} {ne:>7} | {fne:>9}")
        results_per_class[vname] = {
            'n': int(n),
            'action_correct': int(ac), 'scene_correct': int(sc), 'fusion_correct': int(fc),
            'both_correct': int(b), 'only_action': int(oa), 'only_scene': int(os_),
            'neither': int(ne), 'fusion_rescue_from_neither': int(fne),
        }

    # ===== 3. Confusion analysis: what does fusion fix? =====
    # Cases where fusion is right but action-only is wrong
    fusion_fixes_a = f_correct & ~a_correct
    # Cases where fusion is right but scene-only is wrong
    fusion_fixes_s = f_correct & ~s_correct

    print(f"\n--- Fusion rescue analysis ---")
    print(f"Fusion fixes action-only errors:  {fusion_fixes_a.sum()} / {(~a_correct).sum()} "
          f"({100*fusion_fixes_a.sum()/(~a_correct).sum():.1f}%)")
    print(f"Fusion fixes scene-only errors:   {fusion_fixes_s.sum()} / {(~s_correct).sum():.0f} "
          f"({100*fusion_fixes_s.sum()/(~s_correct).sum():.1f}%)")

    # ===== 4. Agreement between unimodal predictions (even when wrong) =====
    a_s_agree = (a_preds == s_preds)
    print(f"\n--- Prediction agreement (regardless of correctness) ---")
    print(f"Action and scene predict same class: {a_s_agree.sum()}/{N} ({100*a_s_agree.mean():.1f}%)")
    agree_correct = (a_s_agree & a_correct).sum()
    agree_wrong = (a_s_agree & ~a_correct).sum()
    print(f"  Agree & correct: {agree_correct}  |  Agree & wrong: {agree_wrong}")

    # Save results
    output = {
        'n_samples': int(N),
        'accuracy': {
            'action_only': float(100 * a_correct.mean()),
            'scene_only': float(100 * s_correct.mean()),
            'fusion': float(100 * f_correct.mean()),
            'union_unimodal': float(100 * union_correct.mean()),
        },
        'agreement': {
            'both_correct': int(both.sum()),
            'only_action': int(only_a.sum()),
            'only_scene': int(only_s.sum()),
            'neither': int(neither.sum()),
        },
        'fusion_in_quadrants': {
            'both_correct_fusion_correct': int((f_correct & both).sum()),
            'only_action_fusion_correct': int((f_correct & only_a).sum()),
            'only_scene_fusion_correct': int((f_correct & only_s).sum()),
            'neither_fusion_correct': int((f_correct & neither).sum()),
        },
        'per_class': results_per_class,
    }
    os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
    with open(args.save_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.save_json}")


if __name__ == "__main__":
    main()
