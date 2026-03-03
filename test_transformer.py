import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from train_transformer import ActionToVerbTransformer, CalvinVerbDataset, SCENE_FUSION_MODALITIES
from utils import load_calvin_to_dataframe
from config import (
    VAL_DIR, D_MODEL, NHEAD, NUM_LAYERS, CROSS_LAYERS, ACTION_DIM, PATCH_SIZE,
    SCENE_OBS_DIM, ROBOT_OBS_DIM,
    IMAGE_SIZE, IMG_MEAN, IMG_STD, R3M_IMG_SIZE,
    BATCH_SIZE, MAX_SEQ_LEN, NUM_WORKERS, FAST_TOKENIZER_PATH,
    IMAGE_ENCODER,
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
        modality = raw.get('modality', args.modality)
        # Override action_dim for oracle modalities (handles old checkpoints that saved ACTION_DIM=7)
        if modality == "scene_obs":
            action_dim = SCENE_OBS_DIM
        elif modality == "robot_obs":
            action_dim = ROBOT_OBS_DIM
        patch_size = raw.get('patch_size', PATCH_SIZE)
        img_size = raw.get('img_size', IMAGE_SIZE[0])
        max_action_len = raw.get('max_action_len', args.max_seq_len)
        action_rep = raw.get('action_rep', args.action_rep)
        # Support both old (fast_vocab_size) and new (action_vocab_size) checkpoint keys
        action_vocab_size = raw.get('action_vocab_size', raw.get('fast_vocab_size', None))
        cross_layers = raw.get('cross_layers', num_layers)
        # Support both old (vision_encoder) and new (image_encoder) checkpoint keys
        image_encoder = raw.get('image_encoder', None)
        if image_encoder is None:
            _venc_map = {'patch': 'scratch', 'r3m': 'r3m',
                         'dinov2_s': 'dinov2_s', 'dinov2_b': 'dinov2_b', 'vc1': 'vc1'}
            image_encoder = _venc_map.get(raw.get('vision_encoder', 'patch'), 'scratch')
        freeze_vision = raw.get('freeze_vision', True)
        num_frames = raw.get('num_frames', 2)
        delta_patches = raw.get('delta_patches', 0)
        vqvae_chunk_size = raw.get('vqvae_chunk_size', args.vqvae_chunk_size)
        modal_dropout = raw.get('modal_dropout', 0.0)
        aux_loss_weight = raw.get('aux_loss_weight', 0.0)
        scene_dim = raw.get('scene_dim', 0)
        print(f"Loaded checkpoint: {num_verbs} verbs, d_model={d_model}, "
              f"modality={modality}, action_rep={action_rep}, "
              f"image_encoder={image_encoder}, num_frames={num_frames}, "
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
        action_vocab_size = None
        cross_layers = args.cross_layers
        image_encoder = args.image_encoder
        freeze_vision = True
        num_frames = args.num_frames
        delta_patches = 0
        vqvae_chunk_size = args.vqvae_chunk_size
        modal_dropout = 0.0
        aux_loss_weight = 0.0
        scene_dim = 0
        print(f"Loaded legacy state_dict: {num_verbs} verbs (from '{last_bias_key}')")

    print(f"Modality: {modality} | Action rep: {action_rep} | Image encoder: {image_encoder}")

    # Pretrained encoders need 224×224; scratch/resnet18 can use native resolution
    if image_encoder in ("r3m", "dinov2_s", "dinov2_b", "vc1", "dinov2"):
        effective_img_size = 224
    else:
        effective_img_size = img_size
    transform = transforms.Compose([
        transforms.Resize((effective_img_size, effective_img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
    ])

    # --- Load tokenizer if needed ---
    action_tokenizer = None
    vqvla_tok = None
    if action_rep in ("bin", "quest", "oat"):
        from action_tokenizers import load_action_tokenizer
        from config import (QUEST_TOKENIZER_CKPT, OAT_TOKENIZER_CKPT,
                            TOKENIZER_HORIZON, TOKENIZER_FIT_NORM_MAX_TRAJS)
        action_tokenizer = load_action_tokenizer(
            action_rep, train_dir=args.data_dir,
            horizon=TOKENIZER_HORIZON, max_tokens=max_action_len,
            quest_ckpt=args.quest_ckpt, oat_ckpt=args.oat_ckpt,
            fit_norm_max_trajs=TOKENIZER_FIT_NORM_MAX_TRAJS)
        action_vocab_size = action_tokenizer.vocab_size
        print(f"Loaded {action_rep} tokenizer (vocab_size={action_vocab_size})")
    elif action_rep == "fast":
        from fast_tokenizer import load_fast_tokenizer
        action_tokenizer = load_fast_tokenizer(args.fast_tokenizer_path)
        print(f"Loaded FAST tokenizer from {args.fast_tokenizer_path}")
    elif action_rep == "vq_vae":
        from vqvae_tokenizer import load_vqvae_tokenizer, tokenize_trajectory_vqvae
        from functools import partial
        _vq = load_vqvae_tokenizer(args.vqvae_tokenizer_path)
        action_tokenizer = partial(tokenize_trajectory_vqvae, _vq)
        action_tokenizer.vocab_size = _vq.num_codes
        action_vocab_size = _vq.num_codes
        print(f"Loaded VQ-VAE tokenizer from {args.vqvae_tokenizer_path} "
              f"(num_codes={action_vocab_size}, chunk_size={_vq.chunk_size})")
    elif action_rep == "vqvla":
        from vqvae_tokenizer import load_vqvla_tokenizer, VQVLA_VOCAB_SIZE
        vqvla_tok = load_vqvla_tokenizer(
            config_dir=args.vqvla_config_dir,
            checkpoint_path=args.vqvla_checkpoint_path)
        action_vocab_size = VQVLA_VOCAB_SIZE

    # --- Load test dataset ---
    print(f"Loading test dataset from {args.data_dir}...")
    df = load_calvin_to_dataframe(args.data_dir)

    if args.debug:
        n = min(args.debug, len(df))
        df = df.head(n).copy()
        print(f"[DEBUG] Using {n} test samples")

    # Use max_action_len from checkpoint so dataset padding matches model's action_pos size
    use_scene_rep = modality in SCENE_FUSION_MODALITIES
    dataset = CalvinVerbDataset(df, args.data_dir, transform=transform,
                                max_seq_len=max_action_len,
                                modality=modality,
                                action_tokenizer=action_tokenizer,
                                image_encoder=image_encoder,
                                img_size=effective_img_size,
                                num_frames=num_frames,
                                delta_patches=delta_patches,
                                vqvae_chunk_size=vqvae_chunk_size,
                                vqvla_tokenizer=vqvla_tok,
                                scene_rep=use_scene_rep)

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
        action_vocab_size=action_vocab_size,
        cross_layers=cross_layers,
        image_encoder=image_encoder,
        freeze_vision=freeze_vision,
        num_frames=num_frames,
        delta_patches=delta_patches,
        modal_dropout=modal_dropout,
        aux_loss_weight=aux_loss_weight,
        scene_dim=scene_dim)

    # Backward compat: handle old nn.TransformerEncoder key prefix
    state_dict = {k.replace("transformer.layers.", "layers."): v
                  for k, v in state_dict.items()}
    # Backward compat: old checkpoints saved vision encoder as "vision_enc.*"; new code uses "patch_embed.*"
    state_dict = {k.replace("vision_enc.", "patch_embed."): v
                  for k, v in state_dict.items()}
    # Backward compat: old checkpoints used type_img_start/type_img_end instead of frame_pos/type_img
    if "type_img_start" in state_dict and "frame_pos" not in state_dict:
        d = state_dict["type_img_start"].shape[-1]
        state_dict["type_img"] = state_dict.pop("type_img_start")
        type_end = state_dict.pop("type_img_end")
        import torch as _torch
        frame_pos = _torch.zeros(1, num_frames, 1, d)
        if num_frames >= 2:
            frame_pos[0, 1, 0, :] = type_end.squeeze() - state_dict["type_img"].squeeze()
        state_dict["frame_pos"] = frame_pos
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Sync dataset num_patches with the loaded encoder
    if modality not in ("action_only", "scene_obs", "robot_obs") + SCENE_FUSION_MODALITIES:
        dataset.num_patches = model.num_patches

    # --- Evaluation ---
    print("\nStarting Evaluation...\n")
    label_map = id_to_verb if id_to_verb else dataset.id_to_verb
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (frames, actions, scene_vecs, labels, seq_lengths) in enumerate(dataloader):
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
                        choices=["full", "action_only", "vision_only",
                                 "scene_obs", "robot_obs",
                                 "scene_token", "scene_concat", "scene_film",
                                 "scene_mlp"],
                        help="Fallback if not in checkpoint")
    parser.add_argument("--action_rep", type=str, default="native",
                        choices=["native", "fast", "quest", "oat", "bin", "vq_vae", "vqvla"],
                        help="Fallback if not in checkpoint")
    parser.add_argument("--fast_tokenizer_path", type=str, default=FAST_TOKENIZER_PATH,
                        help="Path to fitted FAST tokenizer")
    parser.add_argument("--quest_ckpt", type=str, default=None,
                        help="Path to QueST tokenizer checkpoint")
    parser.add_argument("--oat_ckpt", type=str, default=None,
                        help="Path to OAT tokenizer checkpoint")
    parser.add_argument("--vqvae_tokenizer_path", type=str,
                        default="./checkpoints/vqvae_tokenizer",
                        help="Path to fitted VQ-VAE tokenizer")
    parser.add_argument("--vqvae_chunk_size", type=int, default=4,
                        help="Fallback chunk size if not in checkpoint")
    parser.add_argument("--vqvla_config_dir", type=str, default="./vqvla_config",
                        help="Directory containing VQ-VLA config.json")
    parser.add_argument("--vqvla_checkpoint_path", type=str,
                        default="./checkpoints/vqvla_pretrained/action_tokenizer_weight/all_data_vq.pth",
                        help="Path to VQ-VLA pretrained weights (all_data_vq.pth)")
    parser.add_argument("--cross_layers", type=int, default=CROSS_LAYERS,
                        help="Fallback if not in checkpoint")
    parser.add_argument("--image_encoder", type=str, default=IMAGE_ENCODER,
                        choices=["scratch", "patch", "resnet18", "dinov2",
                                 "dinov2_s", "dinov2_b", "vc1", "r3m"],
                        help="Fallback if not in checkpoint")
    parser.add_argument("--num_frames", type=int, default=2,
                        help="Fallback num_frames if not in checkpoint")

    args = parser.parse_args()
    main(args)
