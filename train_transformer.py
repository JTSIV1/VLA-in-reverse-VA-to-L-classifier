import os
import argparse
import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPVisionModel
from PIL import Image

class ActionToVerbTransformer(nn.Module):
    def __init__(self, num_verbs, d_model=768, max_seq_length=512, nhead=12, num_layers=6, action_dim=7):
        super().__init__()
        
        # Vision encoder for first and last frames
        # We use CLIP's ViT-B/32 backbone
        self.visual_backbone = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.visual_backbone.parameters():
            param.requires_grad = False
            
        # Projections to align visual and action features to d_model
        self.vision_proj = nn.Linear(768, d_model)
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
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_verbs)
        )

    def forward(self, first_frame, last_frame, trajectories):
        # first_frame/last_frame: (Batch, 3, 224, 224)
        # trajectories: (Batch, Seq_Len, action_dim)
        batch_size = first_frame.shape[0]
        
        # Extract visual features (Pooler output gives global image representation)
        v_start = self.visual_backbone(first_frame).pooler_output.unsqueeze(1) # (B, 1, 768)
        v_end = self.visual_backbone(last_frame).pooler_output.unsqueeze(1)     # (B, 1, 768)
        
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
    def __init__(self, data_dir, transform=None, max_seq_len=64):
        """
        CALVIN specific dataset loader.
        Expects 'auto_lang_ann.npy' in a subfolder.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.max_seq_len = max_seq_len
        
        # 1. Load spaCy for verb extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # 2. Load Language Annotations
        # CALVIN structure: lang_annotations/auto_lang_ann.npy
        lang_path = os.path.join(data_dir, "lang_annotations", "auto_lang_ann.npy")
        if not os.path.exists(lang_path):
            raise FileNotFoundError(f"Could not find CALVIN language annotations at {lang_path}")
        
        self.lang_data = np.load(lang_path, allow_pickle=True).item()
        
        # In CALVIN, language data is often structured as a dict of lists
        # keys: 'info' (indices), 'language' (annotations)
        self.instructions = self.lang_data['language']['ann']
        self.indices = self.lang_data['info']['indx'] # List of (start_idx, end_idx) pairs
        
        # 3. Build Vocabulary
        self.verb_to_id = {}
        self._build_vocab()

    def _extract_verb(self, text):
        """Extracts the primary verb from a CALVIN instruction."""
        doc = self.nlp(text.lower())
        for token in doc:
            if token.pos_ == "VERB" and (token.dep_ in ("ROOT", "conj", "xcomp", "advcl")):
                parts = [t.text for t in token.children if t.dep_ == "prt"]
                return " ".join([token.text] + parts)
        return "unknown"

    def _build_vocab(self):
        print("Parsing verbs from instructions to build vocabulary...")
        verbs = set()
        for text in self.instructions:
            verb = self._extract_verb(text)
            verbs.add(verb)
        
        self.verb_to_id = {v: i for i, v in enumerate(sorted(list(verbs)))}
        print(f"Vocab built: {len(self.verb_to_id)} unique verbs found.")

    def __len__(self):
        return len(self.instructions)

    def _load_npz(self, idx):
        # CALVIN files are usually named ep_XXXXXXX.npz
        filename = f"episode_{idx:07d}.npz"
        return np.load(os.path.join(self.data_dir, filename))

    def __getitem__(self, idx):
        start_idx, end_idx = self.indices[idx]
        instruction = self.instructions[idx]
        
        # Load start and end frames
        # CALVIN stores frames in individual step files or sequence files
        # Here we assume standard flat directory of .npz files
        start_data = self._load_npz(start_idx)
        end_data = self._load_npz(end_idx)
        
        # Image frames (CALVIN uses 'rgb_static' for the desk camera)
        img_start = Image.fromarray(start_data['rgb_static']).convert("RGB")
        img_end = Image.fromarray(end_data['rgb_static']).convert("RGB")
        
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
            all_actions.append(step_data['rel_actions']) # rel_actions is 7D in CALVIN
        
        actions = np.array(all_actions) # (T, 7)
        
        # Pad/Truncate trajectory
        L = actions.shape[0]
        if L < self.max_seq_len:
            actions_padded = np.pad(actions, ((0, self.max_seq_len - L), (0, 0)), mode='constant')
        else:
            actions_padded = actions[:self.max_seq_len]
            
        actions_tensor = torch.tensor(actions_padded, dtype=torch.float32)
        
        # Label extraction
        verb = self._extract_verb(instruction)
        label_id = self.verb_to_id.get(verb, self.verb_to_id.get("unknown", 0))
        label = torch.tensor(label_id, dtype=torch.long)
        
        return first_frame, last_frame, actions_tensor, label

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Standard CLIP normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])

    print(f"Loading dataset from {args.data_dir}...")
    dataset = CalvinVerbDataset(args.data_dir, transform=transform, max_seq_len=args.max_seq_len)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to CALVIN dataset (containing episode_XXXX.npz files)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    main(args)
