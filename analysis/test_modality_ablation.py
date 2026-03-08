"""
Modality ablation: evaluate a trained full multimodal model with one modality
zeroed + attention-masked at inference time.

Three conditions:
  both        — normal inference (both modalities)
  action_only — vision tokens zeroed + masked  (does the model rely on vision?)
  vision_only — action tokens zeroed + masked  (does the model rely on actions?)

Usage:
    python test_modality_ablation.py \
        --model_path checkpoints/full_vc1_d16_late2_sp_wt_j6459079_best.pth \
        --save_metrics results/ablation_full_vc1_d16_late2_sp_wt.json
"""

import os, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report

from train_transformer import ActionToVerbTransformer, CalvinVerbDataset
from utils import load_calvin_to_dataframe
from config import (
    VAL_DIR, D_MODEL, NHEAD, NUM_LAYERS, CROSS_LAYERS, ACTION_DIM, PATCH_SIZE,
    IMAGE_SIZE, IMG_MEAN, IMG_STD, BATCH_SIZE, MAX_SEQ_LEN, NUM_WORKERS,
    FAST_TOKENIZER_PATH, FAST_VOCAB_SIZE,
)

SKIP = {'accuracy', 'macro avg', 'weighted avg'}


def ablated_forward(model, frames, actions, seq_lengths, ablate, device):
    """Re-implement model.forward() with one modality zeroed + attention-masked.

    ablate: 'vision'  → zero & mask vision tokens [1, v_end)
            'action'  → zero & mask action tokens  [v_end, total_len)
    """
    B = actions.size(0)
    nf = frames.size(1)

    # Vision token boundary (mirrors train_transformer.py forward)
    if model.delta_patches > 0:
        n_vis_tokens = max(nf - 1, 1) * model.num_patches
    else:
        n_vis_tokens = nf * model.num_patches
    v_start, v_end = 1, 1 + n_vis_tokens

    # ---- Build full_seq (mirrors model.forward) ----
    cls = model.cls_token.expand(B, -1, -1) + model.cls_pos + model.type_cls
    parts = [cls]

    if model.modality != "action_only":
        if model.delta_patches > 0:
            all_patches = []
            for fi in range(nf):
                all_patches.append(model.vision_enc(frames[:, fi]))
            K, d = model.delta_patches, all_patches[0].size(-1)
            for pi in range(nf - 1):
                diff = all_patches[pi + 1] - all_patches[pi]
                topk_idx = diff.norm(dim=-1).topk(K, dim=-1).indices
                idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, d)
                selected = torch.gather(diff, 1, idx_exp)
                pos = torch.gather(model.patch_pos.expand(B, -1, -1), 1, idx_exp)
                selected = selected + pos + model.frame_pos[:, pi] + model.type_img
                parts.append(selected)
        else:
            for fi in range(nf):
                frame_i = frames[:, fi]
                if model.vision_encoder_type in ("r3m", "dinov2_s", "dinov2_b", "vc1"):
                    patches = model.vision_enc(frame_i)
                else:
                    patches = model.patch_embed(frame_i)
                patches = (patches + model.patch_pos
                           + model.frame_pos[:, fi] + model.type_img)
                parts.append(patches)

    if model.modality != "vision_only":
        action_len = actions.size(1)
        if model.action_rep == "native":
            action_emb = model.action_proj(actions)
        else:
            action_emb = model.action_embed(actions)
        action_emb = (action_emb
                      + model.action_pos[:, :action_len, :]
                      + model.type_action)
        parts.append(action_emb)

    full_seq = torch.cat(parts, dim=1)
    total_len = full_seq.size(1)

    # ---- Build padding mask (base: action padding) ----
    positions = torch.arange(total_len, device=device).unsqueeze(0)
    padding_mask = positions >= seq_lengths.unsqueeze(1)

    # ---- Apply ablation: zero tokens + extend padding mask ----
    full_seq = full_seq.clone()
    abl_mask = torch.zeros(B, total_len, dtype=torch.bool, device=device)
    if ablate == 'vision':
        full_seq[:, v_start:v_end, :] = 0.0
        abl_mask[:, v_start:v_end] = True
    elif ablate == 'action':
        full_seq[:, v_end:, :] = 0.0
        abl_mask[:, v_end:] = True
    padding_mask = padding_mask | abl_mask

    # ---- Build self-only block mask for early layers (mirrors model.forward) ----
    self_mask = None
    num_self_layers = model.num_layers - model.cross_layers
    if num_self_layers > 0 and model.modality == "full":
        self_mask = torch.full((total_len, total_len), float('-inf'), device=device)
        self_mask[0, 0] = 0.0
        self_mask[v_start:v_end, v_start:v_end] = 0.0
        self_mask[v_end:, v_end:] = 0.0

    # ---- Forward through transformer layers ----
    x = full_seq
    for i, layer in enumerate(model.layers):
        mask = self_mask if i < num_self_layers else None
        x = layer(x, src_mask=mask, src_key_padding_mask=padding_mask)

    return model.classifier(x[:, 0, :])


def evaluate(model, dataloader, label_map, ablate, device):
    """Run one evaluation pass; return (accuracy, macro_f1, n_active_classes)."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for frames, actions, labels, seq_lengths in dataloader:
            frames = frames.to(device)
            actions = actions.to(device)
            labels = labels.to(device)
            seq_lengths = seq_lengths.to(device)

            if ablate is None:
                logits = model(frames, actions, seq_lengths=seq_lengths)
            else:
                logits = ablated_forward(
                    model, frames, actions, seq_lengths, ablate, device)

            all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
    present = sorted(set(all_labels + all_preds))
    names = [label_map[i] for i in present]
    report = classification_report(all_labels, all_preds, labels=present,
                                   target_names=names, digits=4, output_dict=True)
    f1s = [v['f1-score'] for k, v in report.items()
           if k not in SKIP and isinstance(v, dict)]
    macro_f1 = np.mean(f1s) * 100 if f1s else 0.0
    active = len(set(all_preds))
    return acc, macro_f1, active


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load checkpoint ----
    raw = torch.load(args.model_path, map_location=device, weights_only=False)
    sd            = raw['state_dict']
    num_verbs     = raw['num_verbs']
    verb_to_id    = raw['verb_to_id']
    id_to_verb    = raw['id_to_verb']
    d_model       = raw.get('d_model', D_MODEL)
    nhead         = raw.get('nhead', NHEAD)
    num_layers    = raw.get('num_layers', NUM_LAYERS)
    action_dim    = raw.get('action_dim', ACTION_DIM)
    patch_size    = raw.get('patch_size', PATCH_SIZE)
    img_size      = raw.get('img_size', IMAGE_SIZE[0])
    max_action_len = raw.get('max_action_len', MAX_SEQ_LEN)
    modality      = raw.get('modality', 'full')
    action_rep    = raw.get('action_rep', 'native')
    fast_vocab_size = raw.get('fast_vocab_size', FAST_VOCAB_SIZE)
    cross_layers  = raw.get('cross_layers', num_layers)
    vision_encoder = raw.get('vision_encoder', 'patch')
    freeze_vision = raw.get('freeze_vision', True)
    num_frames    = raw.get('num_frames', 2)
    delta_patches = raw.get('delta_patches', 0)

    assert modality == "full", (
        f"Ablation requires a 'full' multimodal checkpoint; got modality={modality}")
    print(f"Checkpoint: {num_verbs} verbs, d_model={d_model}, "
          f"encoder={vision_encoder}, delta_patches={delta_patches}, "
          f"num_frames={num_frames}, cross_layers={cross_layers}")

    # ---- Dataset ----
    eff_size = 224 if vision_encoder in ("r3m", "dinov2_s", "dinov2_b", "vc1") else img_size
    transform = transforms.Compose([
        transforms.Resize((eff_size, eff_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])

    df = load_calvin_to_dataframe(args.data_dir)
    if args.debug:
        df = df.head(args.debug).copy()

    dataset = CalvinVerbDataset(df, args.data_dir, transform=transform,
                                max_seq_len=max_action_len, modality=modality,
                                vision_encoder=vision_encoder, img_size=eff_size,
                                num_frames=num_frames, delta_patches=delta_patches)
    dataset.verb_to_id = verb_to_id
    dataset.id_to_verb = id_to_verb
    valid_mask = df['primary_verb'].isin(verb_to_id.keys())
    if (~valid_mask).sum() > 0:
        dataset.df = df[valid_mask].reset_index(drop=True)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    # ---- Build model ----
    model = ActionToVerbTransformer(
        num_verbs=num_verbs, d_model=d_model, nhead=nhead,
        num_layers=num_layers, action_dim=action_dim,
        img_size=eff_size, patch_size=patch_size, max_action_len=max_action_len,
        modality=modality, action_rep=action_rep, fast_vocab_size=fast_vocab_size,
        cross_layers=cross_layers, vision_encoder=vision_encoder,
        freeze_vision=freeze_vision, num_frames=num_frames, delta_patches=delta_patches)
    sd = {k.replace("transformer.layers.", "layers."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device)

    label_map = id_to_verb

    # ---- Run three conditions ----
    CONDITIONS = [
        ('both',        None,     'Both modalities (normal inference)'),
        ('action_only', 'vision', 'Action-only  (vision tokens zeroed + masked)'),
        ('vision_only', 'action', 'Vision-only  (action tokens zeroed + masked)'),
    ]

    results = {}
    for tag, ablate, desc in CONDITIONS:
        print(f"\n{'='*60}\n{desc}")
        acc, mf1, active = evaluate(model, dataloader, label_map, ablate, device)
        results[tag] = dict(accuracy=round(acc, 2),
                            macro_f1=round(mf1, 2),
                            active_classes=active)
        print(f"  Accuracy:       {acc:.2f}%")
        print(f"  Macro F1:       {mf1:.2f}%")
        print(f"  Active classes: {active}/{num_verbs}")

    print(f"\n{'='*60}\nSUMMARY")
    print(f"  {'Condition':<20} {'Accuracy':>10} {'Macro F1':>10} {'Active':>8}")
    print(f"  {'-'*50}")
    for tag, res in results.items():
        print(f"  {tag:<20} {res['accuracy']:>9.1f}% {res['macro_f1']:>9.1f}% "
              f"{res['active_classes']:>6}/{num_verbs}")

    if args.save_metrics:
        with open(args.save_metrics, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved → {args.save_metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Modality ablation for a trained multimodal transformer.")
    parser.add_argument("--model_path", required=True,
                        help="Path to a 'full' modality checkpoint (.pth)")
    parser.add_argument("--data_dir", default=VAL_DIR,
                        help="Validation data directory")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--save_metrics", default=None,
                        help="Path to save ablation results JSON")
    parser.add_argument("--debug", type=int, default=0,
                        help="Debug: use only N samples")
    args = parser.parse_args()
    main(args)
