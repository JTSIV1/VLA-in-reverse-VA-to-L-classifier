import os
import json
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from utils import load_calvin_to_dataframe
from image_encoders import build_image_encoder
from config import (
    DATA_DIR, VAL_DIR, IMAGE_KEY, ACTION_KEY, EPISODE_TEMPLATE,
    ACTION_DIM, D_MODEL, NHEAD, NUM_LAYERS, CROSS_LAYERS, DROPOUT_RATE, PATCH_SIZE,
    IMAGE_SIZE, IMG_MEAN, IMG_STD,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, MAX_SEQ_LEN, NUM_WORKERS,
    WARMUP_EPOCHS, GRAD_CLIP_NORM, FAST_VOCAB_SIZE, FAST_TOKENIZER_PATH,
)



class ActionToVerbTransformer(nn.Module):
    def __init__(self, num_verbs, d_model=D_MODEL, nhead=NHEAD,
                 num_layers=NUM_LAYERS, action_dim=ACTION_DIM,
                 dropout=DROPOUT_RATE, img_size=IMAGE_SIZE[0],
                 patch_size=PATCH_SIZE, max_action_len=MAX_SEQ_LEN,
                 modality="full", action_rep="native",
                 fast_vocab_size=FAST_VOCAB_SIZE,
                 cross_layers=CROSS_LAYERS, image_encoder="scratch"):
        super().__init__()
        self.modality = modality
        self.action_rep = action_rep
        self.num_layers = num_layers
        self.cross_layers = cross_layers

        # -- Vision branch (skip for action_only) --
        if modality != "action_only":
            self.patch_embed = build_image_encoder(image_encoder, d_model, img_size, patch_size)
            self.num_patches = self.patch_embed.num_tokens
            self.patch_pos = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
            self.type_img_start = nn.Parameter(torch.zeros(1, 1, d_model))
            self.type_img_end = nn.Parameter(torch.zeros(1, 1, d_model))

        # -- Action branch (skip for vision_only) --
        if modality != "vision_only":
            if action_rep == "native":
                self.action_proj = nn.Linear(action_dim, d_model)
            else:
                # FAST: discrete token IDs -> learned embeddings
                self.action_embed = nn.Embedding(fast_vocab_size, d_model)
            # Temporal position for action sequence
            self.action_pos = nn.Parameter(torch.zeros(1, max_action_len, d_model))
            self.type_action = nn.Parameter(torch.zeros(1, 1, d_model))

        # -- CLS token --
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # Separate positional embeddings per modality so spatial patch
        # positions and temporal action positions are independently learned,
        # rather than sharing a single flat 1D encoding
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, d_model))
        # Token type: like BERT segment embeddings
        self.type_cls = nn.Parameter(torch.zeros(1, 1, d_model))

        # Initialize all learned embeddings (ViT convention)
        for name, p in self.named_parameters():
            if 'type_' in name or '_pos' in name or 'cls_token' in name:
                nn.init.trunc_normal_(p, std=0.02)

        # -- Transformer encoder (ModuleList for per-layer attention masks) --
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, batch_first=True, activation='gelu'
            )
            for _ in range(num_layers)
        ])

        # -- Classification head --
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_verbs)
        )

    def forward(self, first_frame, last_frame, trajectories, seq_lengths=None):
        batch_size = trajectories.size(0)

        # CLS token + position + type
        cls = self.cls_token.expand(batch_size, -1, -1) + self.cls_pos + self.type_cls
        parts = [cls]

        # Vision: patch embeddings + spatial position + type
        if self.modality != "action_only":
            start_patches = (self.patch_embed(first_frame)
                             + self.patch_pos + self.type_img_start)
            end_patches = (self.patch_embed(last_frame)
                           + self.patch_pos + self.type_img_end)
            parts.extend([start_patches, end_patches])

        # Action embeddings + temporal position + type
        if self.modality != "vision_only":
            action_len = trajectories.size(1)
            if self.action_rep == "native":
                action_emb = self.action_proj(trajectories)
            else:
                action_emb = self.action_embed(trajectories)
            action_emb = action_emb + self.action_pos[:, :action_len, :] + self.type_action
            parts.append(action_emb)

        # Sequence layout: [CLS] [start patches?] [end patches?] [actions?]
        # Start/end patches adjacent for cross-attention; variable-length
        # actions at the end so padding is contiguous.
        full_seq = torch.cat(parts, dim=1)
        total_len = full_seq.size(1)

        # Padding mask: True = padded position to ignore
        src_key_padding_mask = None
        if seq_lengths is not None:
            positions = torch.arange(total_len, device=full_seq.device).unsqueeze(0)
            src_key_padding_mask = positions >= seq_lengths.unsqueeze(1)

        # Build block-diagonal self-only mask for early layers (full modality only)
        self_mask = None
        num_self_layers = self.num_layers - self.cross_layers
        if num_self_layers > 0 and self.modality == "full":
            # Additive mask: -inf blocks attention, 0.0 allows it
            self_mask = torch.full((total_len, total_len), float('-inf'),
                                   device=full_seq.device)
            # CLS attends only to itself
            self_mask[0, 0] = 0.0
            # Vision block: positions [1, 1+2*num_patches)
            v_start, v_end = 1, 1 + 2 * self.num_patches
            self_mask[v_start:v_end, v_start:v_end] = 0.0
            # Action block: positions [v_end, total_len)
            self_mask[v_end:, v_end:] = 0.0

        # Forward through layers with per-layer masking
        x = full_seq
        for i, layer in enumerate(self.layers):
            mask = self_mask if i < num_self_layers else None
            x = layer(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        # Classify from CLS token
        return self.classifier(x[:, 0, :])


class CalvinVerbDataset(Dataset):
    def __init__(self, df, data_dir, transform=None, max_seq_len=MAX_SEQ_LEN,
                 modality="full", fast_tokenizer=None, num_patches=64):
        """CALVIN dataset loader with modality ablation support.
        Args:
            modality: "full", "action_only", or "vision_only"
            fast_tokenizer: if provided, tokenize actions into FAST token IDs
            num_patches: number of image tokens per frame (from encoder.num_tokens)
        """
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        self.max_seq_len = max_seq_len
        self.modality = modality
        self.fast_tokenizer = fast_tokenizer
        self.num_patches = num_patches

        unique_verbs = sorted(list(set(df['primary_verb'].unique())))
        self.verb_to_id = {v: i for i, v in enumerate(unique_verbs)}
        self.id_to_verb = {i: v for v, i in self.verb_to_id.items()}
        print(f"Vocab mapped: {len(self.verb_to_id)} unique verbs found.")

    def __len__(self):
        return len(self.df)

    def _load_npz(self, idx):
        filename = EPISODE_TEMPLATE.format(idx)
        return np.load(os.path.join(self.data_dir, filename), mmap_mode='r')

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        start_idx = row['start_idx']
        end_idx = row['end_idx']
        verb = row['primary_verb']

        # -- Load images (skip for action_only to save I/O) --
        if self.modality != "action_only":
            start_data = self._load_npz(start_idx)
            end_data = self._load_npz(end_idx)
            img_start = Image.fromarray(np.array(start_data[IMAGE_KEY])).convert("RGB")
            img_end = Image.fromarray(np.array(end_data[IMAGE_KEY])).convert("RGB")
            if self.transform:
                first_frame = self.transform(img_start)
                last_frame = self.transform(img_end)
            else:
                first_frame = transforms.ToTensor()(img_start)
                last_frame = transforms.ToTensor()(img_end)
        else:
            # Dummy tensors — not used by model but keeps DataLoader shape consistent
            first_frame = torch.zeros(3, IMAGE_SIZE[0], IMAGE_SIZE[1])
            last_frame = torch.zeros(3, IMAGE_SIZE[0], IMAGE_SIZE[1])

        # -- Load actions (skip for vision_only) --
        if self.modality != "vision_only":
            all_actions = []
            for i in range(start_idx, end_idx + 1):
                step_data = self._load_npz(i)
                all_actions.append(np.array(step_data[ACTION_KEY]))
            actions = np.array(all_actions)  # (T, 7)
            L = actions.shape[0]

            if self.fast_tokenizer is not None:
                # FAST tokenization: (T, 7) -> list[int] token IDs
                token_ids = self.fast_tokenizer(actions[np.newaxis])
                if isinstance(token_ids, list) and len(token_ids) > 0:
                    if isinstance(token_ids[0], list):
                        token_ids = token_ids[0]
                token_ids = list(token_ids)
                L_tok = len(token_ids)
                if L_tok < self.max_seq_len:
                    token_ids = token_ids + [0] * (self.max_seq_len - L_tok)
                else:
                    token_ids = token_ids[:self.max_seq_len]
                actions_tensor = torch.tensor(token_ids, dtype=torch.long)
                action_real_len = min(L_tok, self.max_seq_len)
            else:
                # Native continuous actions
                if L < self.max_seq_len:
                    actions_padded = np.pad(actions, ((0, self.max_seq_len - L), (0, 0)),
                                            mode='constant')
                else:
                    actions_padded = actions[:self.max_seq_len]
                actions_tensor = torch.tensor(actions_padded, dtype=torch.float32)
                action_real_len = min(L, self.max_seq_len)
        else:
            # Vision-only: dummy action tensor, no variable-length actions
            actions_tensor = torch.zeros(self.max_seq_len, ACTION_DIM)
            action_real_len = 0

        label_id = self.verb_to_id.get(verb, self.verb_to_id.get("unknown", 0))
        label = torch.tensor(label_id, dtype=torch.long)

        # Compute actual sequence length for padding mask
        seq_len = 1  # CLS
        if self.modality != "action_only":
            seq_len += 2 * self.num_patches
        if self.modality != "vision_only":
            seq_len += action_real_len

        return first_frame, last_frame, actions_tensor, label, seq_len


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    print(f"Modality: {args.modality} | Action rep: {args.action_rep} | "
          f"Cross layers: {args.cross_layers}/{NUM_LAYERS}")

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
    ])

    # --- Load FAST tokenizer if needed ---
    fast_tok = None
    fast_vocab_size = FAST_VOCAB_SIZE
    if args.action_rep == "fast":
        from fast_tokenizer import load_fast_tokenizer
        fast_tok = load_fast_tokenizer(args.fast_tokenizer_path)
        fast_vocab_size = fast_tok.bpe_tokenizer.vocab_size
        print(f"Loaded FAST tokenizer from {args.fast_tokenizer_path} "
              f"(vocab_size={fast_vocab_size})")

    # --- Load dataframes ---
    print(f"Loading training data from {args.data_dir}...")
    df = load_calvin_to_dataframe(args.data_dir)

    print(f"Loading validation data from {args.val_dir}...")
    val_df = load_calvin_to_dataframe(args.val_dir)

    if args.debug:
        n = min(args.debug, len(df))
        df = df.head(n).copy()
        val_df = val_df.head(n).copy()
        args.epochs = min(args.epochs, 2)
        print(f"[DEBUG] Using {n} train samples, {len(val_df)} val samples, "
              f"{args.epochs} epochs")

    # --- Build datasets ---
    dataset = CalvinVerbDataset(df, args.data_dir, transform=transform,
                                max_seq_len=args.max_seq_len,
                                modality=args.modality,
                                fast_tokenizer=fast_tok)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers)

    val_dataset = CalvinVerbDataset(val_df, args.val_dir, transform=transform,
                                    max_seq_len=args.max_seq_len,
                                    modality=args.modality,
                                    fast_tokenizer=fast_tok)
    val_dataset.verb_to_id = dataset.verb_to_id
    val_dataset.id_to_verb = dataset.id_to_verb
    valid_mask = val_df['primary_verb'].isin(dataset.verb_to_id.keys())
    if (~valid_mask).sum() > 0:
        print(f"Dropping {(~valid_mask).sum()} val samples with unseen verbs")
        val_dataset.df = val_df[valid_mask].reset_index(drop=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers)

    # --- Model, optimizer, scheduler ---
    num_verbs = len(dataset.verb_to_id)
    model = ActionToVerbTransformer(
        num_verbs=num_verbs, max_action_len=args.max_seq_len,
        modality=args.modality, action_rep=args.action_rep,
        fast_vocab_size=fast_vocab_size,
        cross_layers=args.cross_layers,
        image_encoder=args.image_encoder).to(device)

    # Update datasets with the actual token count from the encoder
    if args.modality != "action_only":
        dataset.num_patches = model.num_patches
        val_dataset.num_patches = model.num_patches

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(dataloader) * args.epochs
    warmup_pct = min(args.warmup_epochs / args.epochs, 0.3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=warmup_pct, anneal_strategy='cos')

    # --- Training loop ---
    print("Starting training...")
    id_to_verb = dataset.id_to_verb
    training_log = []  # per-epoch metrics saved to JSON

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        # Per-class accumulators for train
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        class_loss_sum = defaultdict(float)

        for batch_idx, (first, last, actions, labels, seq_lengths) in enumerate(dataloader):
            first, last = first.to(device), last.to(device)
            actions, labels = actions.to(device), labels.to(device)
            seq_lengths = seq_lengths.to(device)

            optimizer.zero_grad()
            logits = model(first, last, actions, seq_lengths=seq_lengths)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Per-class stats (using per-sample loss)
            with torch.no_grad():
                per_sample_loss = nn.functional.cross_entropy(
                    logits, labels, reduction='none')
                for lbl, pred, sl in zip(labels.cpu().tolist(),
                                         preds.cpu().tolist(),
                                         per_sample_loss.cpu().tolist()):
                    class_total[lbl] += 1
                    class_correct[lbl] += int(pred == lbl)
                    class_loss_sum[lbl] += sl

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | "
                      f"Loss: {loss.item():.4f} | Acc: {100*correct/total:.2f}%")

        avg_loss = total_loss / len(dataloader)
        train_acc = 100 * correct / total
        current_lr = scheduler.get_last_lr()[0]
        print(f"--- Epoch {epoch+1}: Train Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | LR: {current_lr:.2e} ---")

        # --- Validation ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_class_correct = defaultdict(int)
        val_class_total = defaultdict(int)
        val_class_loss_sum = defaultdict(float)

        with torch.no_grad():
            for first, last, actions, labels, seq_lengths in val_dataloader:
                first, last = first.to(device), last.to(device)
                actions, labels = actions.to(device), labels.to(device)
                seq_lengths = seq_lengths.to(device)

                logits = model(first, last, actions, seq_lengths=seq_lengths)
                loss = criterion(logits, labels)
                per_sample_loss = nn.functional.cross_entropy(
                    logits, labels, reduction='none')

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                for lbl, pred, sl in zip(labels.cpu().tolist(),
                                         preds.cpu().tolist(),
                                         per_sample_loss.cpu().tolist()):
                    val_class_total[lbl] += 1
                    val_class_correct[lbl] += int(pred == lbl)
                    val_class_loss_sum[lbl] += sl

        val_avg = val_loss / len(val_dataloader)
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        print(f"    Val Loss: {val_avg:.4f} | Val Acc: {val_acc:.2f}%")

        # Build per-class metrics dict
        per_class_train = {}
        for cid in sorted(set(list(class_total.keys()) + list(val_class_total.keys()))):
            verb = id_to_verb.get(cid, str(cid))
            t = class_total.get(cid, 0)
            per_class_train[verb] = {
                "loss": class_loss_sum.get(cid, 0) / t if t > 0 else 0,
                "acc": 100 * class_correct.get(cid, 0) / t if t > 0 else 0,
                "count": t,
            }
        per_class_val = {}
        for cid in sorted(set(list(class_total.keys()) + list(val_class_total.keys()))):
            verb = id_to_verb.get(cid, str(cid))
            vt = val_class_total.get(cid, 0)
            per_class_val[verb] = {
                "loss": val_class_loss_sum.get(cid, 0) / vt if vt > 0 else 0,
                "acc": 100 * val_class_correct.get(cid, 0) / vt if vt > 0 else 0,
                "count": vt,
            }

        epoch_metrics = {
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": avg_loss,
            "train_acc": train_acc,
            "val_loss": val_avg,
            "val_acc": val_acc,
            "per_class_train": per_class_train,
            "per_class_val": per_class_val,
        }
        training_log.append(epoch_metrics)

        # Write log after each epoch so partial results are available
        if args.log_path:
            log_dir = os.path.dirname(args.log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(args.log_path, "w") as f:
                json.dump({"config": {"modality": args.modality,
                                      "action_rep": args.action_rep,
                                      "cross_layers": args.cross_layers,
                                      "lr": args.lr, "batch_size": args.batch_size,
                                      "max_seq_len": args.max_seq_len},
                           "epochs": training_log}, f, indent=2)
            print(f"    Training log saved to {args.log_path}")

    # --- Save checkpoint ---
    if args.save_path:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            'state_dict': model.state_dict(),
            'num_verbs': num_verbs,
            'verb_to_id': dataset.verb_to_id,
            'id_to_verb': dataset.id_to_verb,
            'd_model': D_MODEL,
            'action_dim': ACTION_DIM,
            'nhead': NHEAD,
            'num_layers': NUM_LAYERS,
            'patch_size': PATCH_SIZE,
            'img_size': IMAGE_SIZE[0],
            'max_action_len': args.max_seq_len,
            'modality': args.modality,
            'action_rep': args.action_rep,
            'fast_vocab_size': fast_vocab_size,
            'cross_layers': args.cross_layers,
            'image_encoder': args.image_encoder,
        }
        torch.save(checkpoint, args.save_path)
        print(f"\nCheckpoint saved to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DIR,
                        help="Path to CALVIN training split")
    parser.add_argument("--val_dir", type=str, default=VAL_DIR,
                        help="Path to CALVIN validation split")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS,
                        help="Number of warmup epochs for LR scheduler")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save checkpoint (e.g., model.pth)")
    parser.add_argument("--log_path", type=str, default=None,
                        help="Path to save training log JSON (e.g., training_log.json)")
    parser.add_argument("--debug", type=int, default=0, metavar="N",
                        help="Debug mode: use only N samples for quick smoke testing")
    parser.add_argument("--modality", type=str, default="full",
                        choices=["full", "action_only", "vision_only"],
                        help="Which input modalities to use")
    parser.add_argument("--action_rep", type=str, default="native",
                        choices=["native", "fast"],
                        help="Action representation: native continuous or FAST tokens")
    parser.add_argument("--fast_tokenizer_path", type=str, default=FAST_TOKENIZER_PATH,
                        help="Path to fitted FAST tokenizer")
    parser.add_argument("--cross_layers", type=int, default=CROSS_LAYERS,
                        help="Number of final layers with cross-modal attention "
                             "(default=NUM_LAYERS for early fusion)")

    args = parser.parse_args()
    main(args)
