import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from utils import load_calvin_to_dataframe
from config import (
    DATA_DIR, VAL_DIR, IMAGE_KEY, ACTION_KEY, EPISODE_TEMPLATE,
    ACTION_DIM, D_MODEL, NHEAD, NUM_LAYERS, DROPOUT_RATE, PATCH_SIZE,
    IMAGE_SIZE, IMG_MEAN, IMG_STD,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, MAX_SEQ_LEN, NUM_WORKERS,
    WARMUP_EPOCHS, GRAD_CLIP_NORM,
)


class PatchEmbed(nn.Module):
    """ViT-style patch embedding: split image into non-overlapping patches and
    linearly project each to d_model via Conv2d (equivalent to flatten + linear,
    but fused into a single GPU op)."""

    def __init__(self, img_size=200, patch_size=25, in_channels=3, embed_dim=128):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 3, H, W) -> (B, num_patches, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


class ActionToVerbTransformer(nn.Module):
    def __init__(self, num_verbs, d_model=D_MODEL, nhead=NHEAD,
                 num_layers=NUM_LAYERS, action_dim=ACTION_DIM,
                 dropout=DROPOUT_RATE, img_size=IMAGE_SIZE[0],
                 patch_size=PATCH_SIZE, max_action_len=MAX_SEQ_LEN):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2

        # -- Vision: ViT-style patch embedding (no pretrained CNN needed for
        # 200x200 synthetic CALVIN images; preserves spatial info unlike ResNet
        # global avg pool which collapses each image to a single token) --
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, d_model)

        # -- Action projection --
        self.action_proj = nn.Linear(action_dim, d_model)

        # -- CLS token --
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # -- Positional embeddings (separate per modality so spatial patch
        # positions and temporal action positions are independently learned,
        # rather than sharing a single flat 1D encoding) --
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, d_model))
        self.patch_pos = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        self.action_pos = nn.Parameter(torch.zeros(1, max_action_len, d_model))

        # -- Token type embeddings (lets the transformer distinguish CLS,
        # start-frame patches, end-frame patches, and action tokens regardless
        # of their position in the sequence — like BERT segment embeddings) --
        self.type_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.type_img_start = nn.Parameter(torch.zeros(1, 1, d_model))
        self.type_img_end = nn.Parameter(torch.zeros(1, 1, d_model))
        self.type_action = nn.Parameter(torch.zeros(1, 1, d_model))

        # Initialize all learned embeddings (ViT convention)
        for p in [self.cls_token, self.cls_pos,
                  self.patch_pos, self.action_pos,
                  self.type_cls, self.type_img_start,
                  self.type_img_end, self.type_action]:
            nn.init.trunc_normal_(p, std=0.02)

        # -- Transformer encoder --
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # -- Classification head --
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_verbs)
        )

    def forward(self, first_frame, last_frame, trajectories, seq_lengths=None):
        batch_size = first_frame.shape[0]

        # Patch embeddings + spatial position + type
        start_patches = (self.patch_embed(first_frame)
                         + self.patch_pos + self.type_img_start)
        end_patches = (self.patch_embed(last_frame)
                       + self.patch_pos + self.type_img_end)

        # Action embeddings + temporal position + type
        action_len = trajectories.size(1)
        action_emb = (self.action_proj(trajectories)
                      + self.action_pos[:, :action_len, :] + self.type_action)

        # CLS token + position + type
        cls = self.cls_token.expand(batch_size, -1, -1) + self.cls_pos + self.type_cls

        # Sequence layout: [CLS] [start patches] [end patches] [actions]
        # Start/end patches are adjacent so they can cross-attend in early layers;
        # variable-length actions are at the end so padding is contiguous.
        full_seq = torch.cat([cls, start_patches, end_patches, action_emb], dim=1)

        # Padding mask: True = padded position to ignore
        src_key_padding_mask = None
        if seq_lengths is not None:
            total_len = full_seq.size(1)
            positions = torch.arange(total_len, device=full_seq.device).unsqueeze(0)
            src_key_padding_mask = positions >= seq_lengths.unsqueeze(1)

        output = self.transformer(full_seq, src_key_padding_mask=src_key_padding_mask)

        # Classify from CLS token
        return self.classifier(output[:, 0, :])


class CalvinVerbDataset(Dataset):
    def __init__(self, df, data_dir, transform=None, max_seq_len=MAX_SEQ_LEN):
        """CALVIN dataset loader using a pandas DataFrame."""
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        self.max_seq_len = max_seq_len
        self.num_patches = (IMAGE_SIZE[0] // PATCH_SIZE) ** 2

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

        # Load start and end frames
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

        # Action trajectory
        all_actions = []
        for i in range(start_idx, end_idx + 1):
            step_data = self._load_npz(i)
            all_actions.append(np.array(step_data[ACTION_KEY]))

        actions = np.array(all_actions)  # (T, 7)
        L = actions.shape[0]

        if L < self.max_seq_len:
            actions_padded = np.pad(actions, ((0, self.max_seq_len - L), (0, 0)),
                                    mode='constant')
        else:
            actions_padded = actions[:self.max_seq_len]

        actions_tensor = torch.tensor(actions_padded, dtype=torch.float32)

        label_id = self.verb_to_id.get(verb, self.verb_to_id.get("unknown", 0))
        label = torch.tensor(label_id, dtype=torch.long)

        # Total real tokens: [CLS](1) + [start patches](P) + [end patches](P) + [actions](L)
        actual_seq_len = 1 + 2 * self.num_patches + min(L, self.max_seq_len)

        return first_frame, last_frame, actions_tensor, label, actual_seq_len


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
    ])

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
                                max_seq_len=args.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers)

    val_dataset = CalvinVerbDataset(val_df, args.val_dir, transform=transform,
                                    max_seq_len=args.max_seq_len)
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
        num_verbs=num_verbs, max_action_len=args.max_seq_len).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(dataloader) * args.epochs
    warmup_pct = min(args.warmup_epochs / args.epochs, 0.3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=warmup_pct, anneal_strategy='cos')

    # --- Training loop ---
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

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

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | "
                      f"Loss: {loss.item():.4f} | Acc: {100*correct/total:.2f}%")

        avg_loss = total_loss / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"--- Epoch {epoch+1}: Train Loss: {avg_loss:.4f} | "
              f"Train Acc: {100*correct/total:.2f}% | LR: {current_lr:.2e} ---")

        # --- Validation ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for first, last, actions, labels, seq_lengths in val_dataloader:
                first, last = first.to(device), last.to(device)
                actions, labels = actions.to(device), labels.to(device)
                seq_lengths = seq_lengths.to(device)

                logits = model(first, last, actions, seq_lengths=seq_lengths)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_avg = val_loss / len(val_dataloader)
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        print(f"    Val Loss: {val_avg:.4f} | Val Acc: {val_acc:.2f}%")

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
    parser.add_argument("--debug", type=int, default=0, metavar="N",
                        help="Debug mode: use only N samples for quick smoke testing")

    args = parser.parse_args()
    main(args)
