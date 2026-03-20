"""Train OAT (Object Action Tokenizer) on BridgeV2 action trajectories.

Standalone script that bypasses the hydra/zarr/wandb infrastructure.
Uses the OAT model architecture (RegisterEncoder + FSQ + SinglePassDecoder)
directly on BridgeV2 action shards.

Usage:
    python train_oat_bridge.py --epochs 500 --batch_size 256
"""

import os
import sys
import json
import glob
import copy
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

# Ensure OAT imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tokenization"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tokenization", "oat"))

from oat.tokenizer.oat.tokenizer import OATTok
from oat.tokenizer.oat.encoder.register_encoder import RegisterEncoder
from oat.tokenizer.oat.decoder.single_pass_decoder import SinglePassDecoder
from oat.tokenizer.oat.quantizer.fsq import FSQ
from oat.model.common.normalizer import LinearNormalizer


# ---------- Dataset ----------

class BridgeActionChunkDataset(Dataset):
    """Dataset that yields random action chunks of fixed horizon from BridgeV2."""

    def __init__(self, actions_list, horizon=32):
        """
        Args:
            actions_list: list of (T_i, 7) numpy arrays, one per episode
            horizon: chunk length
        """
        self.horizon = horizon
        # Index: (episode_idx, start_step) for all valid chunks
        self.indices = []
        self.actions = actions_list
        for ep_idx, ep_actions in enumerate(actions_list):
            T = len(ep_actions)
            if T >= horizon:
                for start in range(T - horizon + 1):
                    self.indices.append((ep_idx, start))
            # Short episodes: pad and include as single chunk
            elif T >= 2:
                self.indices.append((ep_idx, 0))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, start = self.indices[idx]
        actions = self.actions[ep_idx]
        T = len(actions)

        if T >= self.horizon:
            chunk = actions[start:start + self.horizon]
        else:
            # Pad short episodes
            chunk = np.pad(actions, ((0, self.horizon - T), (0, 0)), mode="edge")

        return {"action": torch.tensor(chunk, dtype=torch.float32)}


def load_bridge_actions(shard_dir):
    """Load all BridgeV2 action trajectories from shards."""
    shard_files = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))
    print(f"Loading {len(shard_files)} action shards...")

    actions_list = []
    for sf in tqdm(shard_files, desc="Loading shards"):
        data = np.load(sf, allow_pickle=True)
        n_eps = int(data["n_episodes"])
        for i in range(n_eps):
            actions_list.append(data[f"actions_{i}"].astype(np.float32))

    print(f"Loaded {len(actions_list)} episodes")
    return actions_list


def compute_normalizer(actions_list):
    """Compute LinearNormalizer statistics from the dataset."""
    all_actions = np.concatenate(actions_list, axis=0)  # (N, 7)
    normalizer = LinearNormalizer()
    normalizer.fit({"action": all_actions}, mode="limits")
    return normalizer


# ---------- Model ----------

def build_oat(action_dim=7, horizon=32, num_registers=8,
              emb_dim=256, encoder_depth=2, decoder_depth=4,
              head_dim=64, dropout=0.1, fsq_levels=(8, 5, 5, 5)):
    """Build OAT model with default hyperparameters."""
    latent_dim = len(fsq_levels)

    encoder = RegisterEncoder(
        sample_dim=action_dim,
        sample_horizon=horizon,
        emb_dim=emb_dim,
        head_dim=head_dim,
        depth=encoder_depth,
        pdropout=dropout,
        latent_dim=latent_dim,
        num_registers=num_registers,
    )

    decoder = SinglePassDecoder(
        sample_dim=action_dim,
        sample_horizon=horizon,
        emb_dim=emb_dim,
        head_dim=head_dim,
        depth=decoder_depth,
        pdropout=dropout,
        token_dropout_mode="pow2",
        latent_dim=latent_dim,
        latent_horizon=num_registers,
        use_causal_decoder=True,
    )

    quantizer = FSQ(levels=list(fsq_levels))

    model = OATTok(encoder=encoder, decoder=decoder, quantizer=quantizer)
    return model


# ---------- Training ----------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    actions_list = load_bridge_actions(args.shard_dir)

    # Train/val split by episode
    np.random.seed(42)
    n_eps = len(actions_list)
    perm = np.random.permutation(n_eps)
    n_val = max(1, int(n_eps * args.val_fraction))
    val_indices = perm[:n_val]
    train_indices = perm[n_val:]

    train_actions = [actions_list[i] for i in train_indices]
    val_actions = [actions_list[i] for i in val_indices]
    print(f"Train: {len(train_actions)} episodes, Val: {len(val_actions)} episodes")

    train_dataset = BridgeActionChunkDataset(train_actions, horizon=args.horizon)
    val_dataset = BridgeActionChunkDataset(val_actions, horizon=args.horizon)
    print(f"Train chunks: {len(train_dataset)}, Val chunks: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False)

    # Build model
    model = build_oat(
        action_dim=args.action_dim,
        horizon=args.horizon,
        num_registers=args.num_registers,
        emb_dim=args.emb_dim,
    )

    # Compute normalizer (must be on same device as model)
    print("Computing normalizer...")
    normalizer = compute_normalizer(train_actions)
    model.set_normalizer(normalizer)
    model.to(device)  # move after normalizer is set (normalizer params start on CPU)
    print(f"Vocab size: {model.vocab_size}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # EMA (deep copy after model is on device)
    ema_model = copy.deepcopy(model)

    # Optimizer
    optimizer = model.get_optimizer(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Cosine LR scheduler
    scheduler = None
    if args.cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Resume from checkpoint
    start_epoch = 0
    training_log = []
    best_val_mse = float("inf")
    best_epoch = -1

    if args.resume_path and os.path.exists(args.resume_path):
        print(f"Resuming from {args.resume_path}")
        ckpt = torch.load(args.resume_path, map_location=device)
        ema_model.load_state_dict(ckpt["state_dict"])
        # Also load into model (ema and model should be close)
        model.load_state_dict(ckpt["state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_mse = ckpt.get("best_recon_mse", float("inf"))
        best_epoch = start_epoch
        if scheduler:
            for _ in range(start_epoch):
                scheduler.step()
        print(f"  Resumed at epoch {start_epoch}, best recon MSE: {best_val_mse:.6f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        # Subsample training data if requested (for large datasets)
        if args.max_chunks_per_epoch and len(train_dataset) > args.max_chunks_per_epoch:
            subset_idx = np.random.choice(len(train_dataset), args.max_chunks_per_epoch, replace=False)
            epoch_dataset = Subset(train_dataset, subset_idx)
            epoch_loader = DataLoader(epoch_dataset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers,
                                      pin_memory=True, drop_last=True)
        else:
            epoch_loader = train_loader

        pbar = tqdm(epoch_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # EMA update
            with torch.no_grad():
                decay = min(0.9999, (1 + epoch * len(train_loader) + n_batches) /
                            (10 + epoch * len(train_loader) + n_batches))
                for p_ema, p_model in zip(ema_model.parameters(), model.parameters()):
                    p_ema.data.mul_(decay).add_(p_model.data, alpha=1 - decay)

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_train_loss = total_loss / n_batches

        if scheduler:
            scheduler.step()

        # Validation (reconstruction MSE)
        ema_model.eval()
        val_loss = 0
        val_recon_mse = 0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = ema_model(batch)
                val_loss += loss.item() * batch["action"].size(0)

                # Reconstruction MSE in original space
                recon = ema_model.autoencode(batch["action"])
                mse = F.mse_loss(recon, batch["action"]).item()
                val_recon_mse += mse * batch["action"].size(0)
                val_n += batch["action"].size(0)

        avg_val_loss = val_loss / val_n
        avg_recon_mse = val_recon_mse / val_n

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1}: train_loss={avg_train_loss:.6f} "
              f"val_loss={avg_val_loss:.6f} recon_mse={avg_recon_mse:.6f} lr={cur_lr:.2e}")

        # Checkpoint
        if avg_recon_mse < best_val_mse and args.save_path:
            best_val_mse = avg_recon_mse
            best_epoch = epoch + 1
            best_path = args.save_path.replace(".pth", "_best.pth")
            os.makedirs(os.path.dirname(best_path) or ".", exist_ok=True)
            torch.save({
                "state_dict": ema_model.state_dict(),
                "epoch": epoch + 1,
                "best_recon_mse": best_val_mse,
                "horizon": args.horizon,
                "num_registers": args.num_registers,
                "action_dim": args.action_dim,
                "emb_dim": args.emb_dim,
                "vocab_size": model.vocab_size,
                "dataset": "bridge_v2",
            }, best_path)
            print(f"  * Best recon MSE: {avg_recon_mse:.6f} @ epoch {epoch+1}")

        training_log.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "recon_mse": avg_recon_mse,
            "lr": cur_lr,
        })

        if args.log_path:
            os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)
            with open(args.log_path, "w") as f:
                json.dump({"config": vars(args), "epochs": training_log}, f, indent=2)

    # Final checkpoint
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        torch.save({
            "state_dict": ema_model.state_dict(),
            "epoch": args.epochs,
            "best_recon_mse": best_val_mse,
            "horizon": args.horizon,
            "num_registers": args.num_registers,
            "action_dim": args.action_dim,
            "emb_dim": args.emb_dim,
            "vocab_size": model.vocab_size,
            "dataset": "bridge_v2",
        }, args.save_path)

    print(f"\nBest recon MSE: {best_val_mse:.6f} @ epoch {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_dir", default="/data/user_data/wenjiel2/datasets/bridge_actions")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--num_registers", type=int, default=8)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--resume_path", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--max_chunks_per_epoch", type=int, default=None,
                        help="Subsample training chunks per epoch (for large datasets)")
    parser.add_argument("--cosine_lr", action="store_true",
                        help="Use cosine LR decay")
    args = parser.parse_args()
    main(args)
