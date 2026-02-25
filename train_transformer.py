import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from PIL import Image

from utils import load_calvin_to_dataframe, visualize_frames
from config import (
    DATA_DIR, IMAGE_KEY, ACTION_KEY, EPISODE_TEMPLATE,
    ACTION_DIM, D_MODEL, NHEAD, NUM_LAYERS, MAX_SEQ_LENGTH, RESNET_FEATURE_DIM,
    IMAGE_SIZE, CLIP_MEAN, CLIP_STD,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, MAX_SEQ_LEN, NUM_WORKERS,
)

class ActionToVerbTransformer(nn.Module):
    def __init__(self, num_verbs, d_model=D_MODEL, max_seq_length=MAX_SEQ_LENGTH, nhead=NHEAD, num_layers=NUM_LAYERS, action_dim=ACTION_DIM):
        super().__init__()
        
        # Load ResNet18 & remove the final classification layer
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.visual_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        for param in self.visual_backbone.parameters():
            param.requires_grad = False
            
        # Project image and action embeddings
        self.vision_proj = nn.Linear(RESNET_FEATURE_DIM, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        
        # [CLS] token for classification and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoder = nn.Parameter(torch.randn(1, max_seq_length, d_model)) 
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head: projects the CLS token output to verb classes
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            # nn.Dropout(0.1), # adds random noise to prevent overfitting
            nn.Linear(d_model // 2, num_verbs)
        )

    def forward(self, first_frame, last_frame, trajectories):
        # first_frame/last_frame: (Batch, 3, 224, 224)
        # trajectories: (Batch, Seq_Len, action_dim)
        batch_size = first_frame.shape[0]
        
        v_start = self.visual_backbone(first_frame).flatten(2).transpose(1, 2)
        v_end = self.visual_backbone(last_frame).flatten(2).transpose(1, 2)
        
        # Project all inputs to d_model
        v_start_emb = self.vision_proj(v_start)
        v_end_emb = self.vision_proj(v_end)
        action_emb = self.action_proj(trajectories)
        
        # Expand CLS token for the batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Build sequence: [CLS] [Image_Start] [Actions...] [Image_End]
        full_seq = torch.cat([cls_tokens, v_start_emb, action_emb, v_end_emb], dim=1)
        
        # Add positional information
        seq_len = full_seq.size(1)
        full_seq += self.pos_encoder[:, :seq_len, :]
        
        # Pass through Transformer
        output = self.transformer(full_seq)
        
        # Classify based on the CLS token (index 0)
        verb_logits = self.classifier(output[:, 0, :])
        
        return verb_logits

class CalvinVerbDataset(Dataset):
    def __init__(self, df, data_dir, transform=None, max_seq_len=MAX_SEQ_LEN):
        """
        CALVIN specific dataset loader using a pandas DataFrame.
        """
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        self.max_seq_len = max_seq_len
        
        # Vocabulary is already built in the dataframe, just map it
        unique_verbs = sorted(list(set(df['primary_verb'].unique())))
        self.verb_to_id = {v: i for i, v in enumerate(unique_verbs)}
        self.id_to_verb = {i: v for v, i in self.verb_to_id.items()}
        print(f"Vocab mapped: {len(self.verb_to_id)} unique verbs found.")

    def __len__(self):
        return len(self.df)

    def _load_npz(self, idx):
        # CALVIN files are usually named episode_XXXXXXX.npz
        filename = EPISODE_TEMPLATE.format(idx)
        return np.load(os.path.join(self.data_dir, filename))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        start_idx = row['start_idx']
        end_idx = row['end_idx']
        verb = row['primary_verb']
        
        # Load start and end frames
        # CALVIN stores frames in individual step files or sequence files
        # Here we assume standard flat directory of .npz files
        start_data = self._load_npz(start_idx)
        end_data = self._load_npz(end_idx)
        
        # Image frames (CALVIN uses 'rgb_static' for the desk camera)
        img_start = Image.fromarray(start_data[IMAGE_KEY]).convert("RGB")
        img_end = Image.fromarray(end_data[IMAGE_KEY]).convert("RGB")
        
        if self.transform:
            first_frame = self.transform(img_start)
            last_frame = self.transform(img_end)
        else:
            first_frame = transforms.ToTensor()(img_start)
            last_frame = transforms.ToTensor()(img_end)

        # Collect actions from start to end
        # This requires loading all intermediate files in the sequence
        all_actions = []
        for i in range(start_idx, end_idx + 1):
            step_data = self._load_npz(i)
            all_actions.append(step_data[ACTION_KEY]) # rel_actions is 7D in CALVIN
        
        actions = np.array(all_actions) # (T, 7)
        
        # Pad/Truncate trajectory
        L = actions.shape[0]
        if L < self.max_seq_len:
            actions_padded = np.pad(actions, ((0, self.max_seq_len - L), (0, 0)), mode='constant')
        else:
            actions_padded = actions[:self.max_seq_len]
            
        actions_tensor = torch.tensor(actions_padded, dtype=torch.float32)
        
        # Label extraction
        label_id = self.verb_to_id.get(verb, self.verb_to_id.get("unknown", 0))
        label = torch.tensor(label_id, dtype=torch.long)
        
        return first_frame, last_frame, actions_tensor, label

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Standard CLIP normalization
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    ])

    print(f"Loading dataset from {args.data_dir}...")
    df = load_calvin_to_dataframe(args.data_dir)
    
    print("Visualizing some examples...")
    visualize_frames(df, args.data_dir, num_samples=3)

    dataset = CalvinVerbDataset(df, args.data_dir, transform=transform, max_seq_len=args.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    num_verbs = len(dataset.verb_to_id)
    model = ActionToVerbTransformer(num_verbs=num_verbs, max_seq_length=args.max_seq_len + 3).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (first, last, actions, labels) in enumerate(dataloader):
            first, last = first.to(device), last.to(device)
            actions, labels = actions.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(first, last, actions)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Accuracy tracking
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f} | Acc: {100*correct/total:.2f}%")

        avg_loss = total_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Summary: Avg Loss: {avg_loss:.4f} | Final Acc: {100*correct/total:.2f}% ---")

    print("\n=== Final Training Summary: Predictions vs Ground Truth (First 5) ===")
    model.eval()
    num_printed = 0
    with torch.no_grad():
        for first, last, actions, labels in dataloader:
            if num_printed >= 5:
                break
                
            first, last = first.to(device), last.to(device)
            actions = actions.to(device)
            
            logits = model(first, last, actions)
            preds = torch.argmax(logits, dim=1)
            
            for p, l in zip(preds, labels):
                if num_printed >= 5:
                    break
                pred_verb = dataset.id_to_verb[p.item()]
                true_verb = dataset.id_to_verb[l.item()]
                status = "✅" if p.item() == l.item() else "❌"
                print(f"{status} Target: {true_verb:<15} | Predicted: {pred_verb}")
                num_printed += 1

    # Save the model if a save path is provided
    if args.save_path:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        # We save the model's state dictionary (the weights)
        torch.save(model.state_dict(), args.save_path)
        print(f"\nModel weights successfully saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Path to CALVIN dataset (containing episode_XXXX.npz files)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--save_path", type=str, default=None, help="Optional: Path to save the trained model weights (e.g., my_model.pth)")
    
    args = parser.parse_args()
    main(args)