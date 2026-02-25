import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from train_transformer import ActionToVerbTransformer, CalvinVerbDataset
from utils import load_calvin_to_dataframe
from config import (
    VAL_DIR, D_MODEL, NHEAD, NUM_LAYERS, ACTION_DIM, PATCH_SIZE,
    IMAGE_SIZE, IMG_MEAN, IMG_STD,
    BATCH_SIZE, MAX_SEQ_LEN, NUM_WORKERS,
)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Could not find model weights at {args.model_path}")

    # --- Load checkpoint (supports both new dict and legacy bare state_dict) ---
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
        print(f"Loaded checkpoint: {num_verbs} verbs, d_model={d_model}, "
              f"patch_size={patch_size}, img_size={img_size}")
    else:
        # Legacy bare state_dict — infer num_verbs from classifier bias
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
        print(f"Loaded legacy state_dict: {num_verbs} verbs (from '{last_bias_key}')")

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
    ])

    # --- Load test dataset ---
    print(f"Loading test dataset from {args.data_dir}...")
    df = load_calvin_to_dataframe(args.data_dir)

    if args.debug:
        n = min(args.debug, len(df))
        df = df.head(n).copy()
        print(f"[DEBUG] Using {n} test samples")

    dataset = CalvinVerbDataset(df, args.data_dir, transform=transform,
                                max_seq_len=args.max_seq_len)

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
        img_size=img_size, patch_size=patch_size,
        max_action_len=max_action_len)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # --- Evaluation ---
    print("\nStarting Evaluation...\n")
    label_map = id_to_verb if id_to_verb else dataset.id_to_verb
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (first, last, actions, labels, seq_lengths) in enumerate(dataloader):
            first, last = first.to(device), last.to(device)
            actions, labels = actions.to(device), labels.to(device)
            seq_lengths = seq_lengths.to(device)

            logits = model(first, last, actions, seq_lengths=seq_lengths)
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
    print(f"EVALUATION COMPLETE")
    print(f"Total examples: {len(all_preds)}")
    print(f"Overall accuracy: {accuracy:.2f}%")
    print("=" * 60)

    # Per-class metrics
    print("\nPer-class metrics:")
    print(classification_report(all_labels, all_preds,
                                labels=present_labels,
                                target_names=target_names,
                                digits=3))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=present_labels)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, xticks_rotation=45, cmap='Blues', values_format='d')
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    if args.save_cm:
        plt.savefig(args.save_cm, dpi=150, bbox_inches='tight')
        print(f"\nConfusion matrix saved to {args.save_cm}")
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
    parser.add_argument("--debug", type=int, default=0, metavar="N",
                        help="Debug mode: use only N samples for quick smoke testing")

    args = parser.parse_args()
    main(args)
