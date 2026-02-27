import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from train_transformer import ActionToVerbTransformer, CalvinVerbDataset
from utils import load_calvin_to_dataframe
from config import (
    VAL_DIR, D_MODEL, NHEAD, NUM_LAYERS, CROSS_LAYERS, ACTION_DIM, PATCH_SIZE,
    IMAGE_SIZE, IMG_MEAN, IMG_STD, R3M_IMG_SIZE,
    BATCH_SIZE, MAX_SEQ_LEN, NUM_WORKERS, FAST_TOKENIZER_PATH, FAST_VOCAB_SIZE,
    VQVAE_TOKENIZER_PATH,
)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Could not find model weights at {args.model_path}")

    # --- Load checkpoint ---
    print(f"Loading weights from {args.model_path}...")
    raw = torch.load(args.model_path, map_location=device, weights_only=False)

    if isinstance(raw, dict) and 'state_dict' in raw:
        state_dict = raw['state_dict']
        num_verbs = raw['num_verbs']
        verb_to_id = raw['verb_to_id']
        id_to_verb = raw['id_to_verb']
        d_model = raw.get('d_model', D_MODEL)
        nhead = raw.get('nhead', NHEAD)
        num_layers = raw.get('num_layers', NUM_LAYERS)
        action_dim = raw.get('action_dim', ACTION_DIM)
        patch_size = raw.get('patch_size', PATCH_SIZE)
        img_size = raw.get('img_size', IMAGE_SIZE[0])
        max_action_len = raw.get('max_action_len', args.max_seq_len)
        # Read modality/action_rep from checkpoint, with CLI override
        modality = raw.get('modality', args.modality)
        action_rep = raw.get('action_rep', args.action_rep)
        fast_vocab_size = raw.get('fast_vocab_size', FAST_VOCAB_SIZE)
        cross_layers = raw.get('cross_layers', num_layers)
        vision_encoder = raw.get('vision_encoder', 'patch')
        freeze_vision = raw.get('freeze_vision', True)
        num_frames = raw.get('num_frames', 2)
        delta_patches = raw.get('delta_patches', 0)
        vqvae_chunk_size = raw.get('vqvae_chunk_size', args.vqvae_chunk_size)
        modal_dropout = raw.get('modal_dropout', 0.0)
        aux_loss_weight = raw.get('aux_loss_weight', 0.0)
        print(f"Loaded checkpoint: {num_verbs} verbs, d_model={d_model}, "
              f"modality={modality}, action_rep={action_rep}, "
              f"vision_encoder={vision_encoder}, num_frames={num_frames}, "
              f"delta_patches={delta_patches}, cross_layers={cross_layers}, "
              f"modal_dropout={modal_dropout}, aux_loss_weight={aux_loss_weight}")
    else:
        # Legacy bare state_dict
        state_dict = raw
        classifier_bias_keys = [
            k for k in state_dict
            if k.startswith('classifier.') and k.endswith('.bias')
        ]
        last_bias_key = sorted(classifier_bias_keys,
                               key=lambda k: int(k.split('.')[1]))[-1]
        num_verbs = state_dict[last_bias_key].shape[0]
        verb_to_id = None
        id_to_verb = None
        d_model = D_MODEL
        nhead = NHEAD
        num_layers = NUM_LAYERS
        action_dim = ACTION_DIM
        patch_size = PATCH_SIZE
        img_size = IMAGE_SIZE[0]
        max_action_len = args.max_seq_len
        modality = args.modality
        action_rep = args.action_rep
        fast_vocab_size = FAST_VOCAB_SIZE
        cross_layers = args.cross_layers
        vision_encoder = args.vision_encoder
        freeze_vision = True
        num_frames = args.num_frames
        delta_patches = 0
        vqvae_chunk_size = args.vqvae_chunk_size
        modal_dropout = 0.0
        aux_loss_weight = 0.0
        print(f"Loaded legacy state_dict: {num_verbs} verbs (from '{last_bias_key}')")

    print(f"Modality: {modality} | Action rep: {action_rep} | Vision: {vision_encoder}")

    # Image size depends on vision encoder
    if vision_encoder in ("r3m", "dinov2_s", "dinov2_b", "vc1"):
        effective_img_size = 224
    else:
        effective_img_size = img_size
    transform = transforms.Compose([
        transforms.Resize((effective_img_size, effective_img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
    ])

    # --- Load FAST / VQ-VAE tokenizer if needed ---
    fast_tok = None
    vqvae_tok = None
    if action_rep == "fast":
        from fast_tokenizer import load_fast_tokenizer
        fast_tok = load_fast_tokenizer(args.fast_tokenizer_path)
        print(f"Loaded FAST tokenizer from {args.fast_tokenizer_path}")
    elif action_rep == "vq_vae":
        from vqvae_tokenizer import load_vqvae_tokenizer
        vqvae_tok = load_vqvae_tokenizer(args.vqvae_tokenizer_path)
        print(f"Loaded VQ-VAE tokenizer from {args.vqvae_tokenizer_path} "
              f"(num_codes={vqvae_tok.num_codes}, chunk_size={vqvae_tok.chunk_size})")

    # --- Load test dataset ---
    print(f"Loading test dataset from {args.data_dir}...")
    df = load_calvin_to_dataframe(args.data_dir)

    if args.debug:
        n = min(args.debug, len(df))
        df = df.head(n).copy()
        print(f"[DEBUG] Using {n} test samples")

    # Use max_action_len from checkpoint so dataset padding matches model's action_pos size
    dataset = CalvinVerbDataset(df, args.data_dir, transform=transform,
                                max_seq_len=max_action_len,
                                modality=modality,
                                fast_tokenizer=fast_tok,
                                vision_encoder=vision_encoder,
                                img_size=effective_img_size,
                                num_frames=num_frames,
                                delta_patches=delta_patches,
                                vqvae_tokenizer=vqvae_tok,
                                vqvae_chunk_size=vqvae_chunk_size)

    # Override vocab from checkpoint if available
    if verb_to_id is not None:
        dataset.verb_to_id = verb_to_id
        dataset.id_to_verb = id_to_verb
        valid_mask = df['primary_verb'].isin(verb_to_id.keys())
        if (~valid_mask).sum() > 0:
            print(f"Dropping {(~valid_mask).sum()} samples with verbs not in model vocab")
            dataset.df = df[valid_mask].reset_index(drop=True)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    # --- Initialize and load model ---
    model = ActionToVerbTransformer(
        num_verbs=num_verbs, d_model=d_model, nhead=nhead,
        num_layers=num_layers, action_dim=action_dim,
        img_size=effective_img_size, patch_size=patch_size,
        max_action_len=max_action_len,
        modality=modality, action_rep=action_rep,
        fast_vocab_size=fast_vocab_size,
        cross_layers=cross_layers,
        vision_encoder=vision_encoder,
        freeze_vision=freeze_vision,
        num_frames=num_frames,
        delta_patches=delta_patches,
        modal_dropout=modal_dropout,
        aux_loss_weight=aux_loss_weight)
    # Backward compat: handle old nn.TransformerEncoder key prefix
    state_dict = {k.replace("transformer.layers.", "layers."): v
                  for k, v in state_dict.items()}
    # Backward compat: old checkpoints used type_img_start/type_img_end instead of frame_pos/type_img
    if "type_img_start" in state_dict and "frame_pos" not in state_dict:
        d = state_dict["type_img_start"].shape[-1]
        # Use type_img_start as the shared type_img
        state_dict["type_img"] = state_dict.pop("type_img_start")
        type_end = state_dict.pop("type_img_end")
        # Construct frame_pos: frame 0 gets zeros, frame 1 gets (end - start) difference
        import torch as _torch
        frame_pos = _torch.zeros(1, num_frames, 1, d)
        if num_frames >= 2:
            frame_pos[0, 1, 0, :] = type_end.squeeze() - state_dict["type_img"].squeeze()
        state_dict["frame_pos"] = frame_pos
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # --- Evaluation ---
    print("\nStarting Evaluation...\n")
    label_map = id_to_verb if id_to_verb else dataset.id_to_verb
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (frames, actions, labels, seq_lengths) in enumerate(dataloader):
            frames = frames.to(device)
            actions, labels = actions.to(device), labels.to(device)
            seq_lengths = seq_lengths.to(device)

            logits = model(frames, actions, seq_lengths=seq_lengths)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            if (batch_idx + 1) % 50 == 0:
                running_acc = 100 * sum(
                    p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
                print(f"  Processed {batch_idx + 1} batches... "
                      f"running acc: {running_acc:.2f}%")

    # --- Report ---
    accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
    present_labels = sorted(set(all_labels + all_preds))
    target_names = [label_map[i] for i in present_labels]

    print("\n" + "=" * 60)
    print(f"EVALUATION COMPLETE  [{modality} / {action_rep}]")
    print(f"Total examples: {len(all_preds)}")
    print(f"Overall accuracy: {accuracy:.2f}%")
    print("=" * 60)

    # Per-class metrics
    report_str = classification_report(all_labels, all_preds,
                                       labels=present_labels,
                                       target_names=target_names,
                                       digits=3)
    print("\nPer-class metrics:")
    print(report_str)

    # Save metrics JSON for programmatic comparison
    if args.save_metrics:
        report_dict = classification_report(all_labels, all_preds,
                                            labels=present_labels,
                                            target_names=target_names,
                                            digits=4, output_dict=True)
        metrics = {
            "modality": modality,
            "action_rep": action_rep,
            "accuracy": accuracy,
            "num_examples": len(all_preds),
            "per_class": report_dict,
        }
        metrics_dir = os.path.dirname(args.save_metrics)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        with open(args.save_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {args.save_metrics}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=VAL_DIR,
                        help="Path to CALVIN dataset for testing (default: validation split)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved checkpoint (.pth)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--save_cm", type=str, default=None,
                        help="Path to save confusion matrix image (e.g., cm.png)")
    parser.add_argument("--save_metrics", type=str, default=None,
                        help="Path to save metrics JSON (e.g., metrics.json)")
    parser.add_argument("--debug", type=int, default=0, metavar="N",
                        help="Debug mode: use only N samples for quick smoke testing")
    parser.add_argument("--modality", type=str, default="full",
                        choices=["full", "action_only", "vision_only"],
                        help="Fallback if not in checkpoint")
    parser.add_argument("--action_rep", type=str, default="native",
                        choices=["native", "fast", "vq_vae"],
                        help="Fallback if not in checkpoint")
    parser.add_argument("--fast_tokenizer_path", type=str, default=FAST_TOKENIZER_PATH,
                        help="Path to fitted FAST tokenizer")
    parser.add_argument("--vqvae_tokenizer_path", type=str, default=VQVAE_TOKENIZER_PATH,
                        help="Path to fitted VQ-VAE tokenizer")
    parser.add_argument("--vqvae_chunk_size", type=int, default=4,
                        help="Fallback chunk size if not in checkpoint")
    parser.add_argument("--cross_layers", type=int, default=CROSS_LAYERS,
                        help="Fallback if not in checkpoint")
    parser.add_argument("--vision_encoder", type=str, default="patch",
                        choices=["patch", "r3m", "dinov2_s", "dinov2_b", "vc1"],
                        help="Fallback if not in checkpoint")
    parser.add_argument("--num_frames", type=int, default=2,
                        help="Fallback if not in checkpoint")

    args = parser.parse_args()
    main(args)
