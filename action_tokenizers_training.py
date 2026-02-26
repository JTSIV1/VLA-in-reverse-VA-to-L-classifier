import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import load_calvin_to_dataframe
from config import ACTION_KEY, EPISODE_TEMPLATE, ACTION_DIM, DATA_DIR
from config import (
    CHECKPOINT_DIR,
    QUEST_TOKENIZER_CKPT,
    OAT_TOKENIZER_CKPT,
    TOKENIZER_HORIZON,
    TOKENIZER_FIT_NORM_MAX_TRAJS,
    ACTION_DIM,
    TOKENIZER_DOWNSAMPLE_FACTOR,
    OAT_NUM_REGISTERS,
)
from action_tokenizers import fit_calvin_normalizer  # from the file above

from oat.tokenizer.quest.tokenizer import QueSTTok
from oat.tokenizer.oat.tokenizer import OATTok
from oat.tokenizer.oat.encoder.register_encoder import RegisterEncoder
from oat.tokenizer.oat.decoder.single_pass_decoder import SinglePassDecoder
from oat.tokenizer.oat.quantizer.fsq import FSQ


class CalvinActionChunkDataset(Dataset):
    def __init__(self, data_dir, horizon=32, max_trajs=None):
        self.data_dir = data_dir
        self.horizon = horizon
        self.df = load_calvin_to_dataframe(data_dir)
        if max_trajs:
            self.df = self.df.head(min(max_trajs, len(self.df))).copy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        acts = []
        for i in range(row["start_idx"], row["end_idx"] + 1):
            p = os.path.join(self.data_dir, EPISODE_TEMPLATE.format(i))
            step = np.load(p, mmap_mode="r")
            acts.append(np.array(step[ACTION_KEY], dtype=np.float32))
        traj = np.stack(acts, axis=0)  # (T,D)

        # sample or pad to fixed horizon
        T = traj.shape[0]
        if T >= self.horizon:
            s = np.random.randint(0, T - self.horizon + 1)
            chunk = traj[s:s+self.horizon]
        else:
            pad = np.zeros((self.horizon - T, traj.shape[1]), dtype=np.float32)
            chunk = np.concatenate([traj, pad], axis=0)

        return {"action": torch.from_numpy(chunk).float()}


def build_oat(horizon):
    levels = [8, 5, 5, 5]
    latent_dim = len(levels)
    num_registers = OAT_NUM_REGISTERS

    enc = RegisterEncoder(
        sample_dim=ACTION_DIM, sample_horizon=horizon,
        emb_dim=256, head_dim=64, depth=2, pdropout=0.1,
        latent_dim=latent_dim, num_registers=num_registers
    )
    dec = SinglePassDecoder(
        sample_dim=ACTION_DIM, sample_horizon=horizon,
        emb_dim=256, head_dim=64, depth=4, pdropout=0.1,
        token_dropout_mode="pow2", latent_dim=latent_dim,
        latent_horizon=num_registers, use_causal_decoder=True
    )
    q = FSQ(levels=levels)
    return OATTok(encoder=enc, decoder=dec, quantizer=q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["quest", "oat"], required=True)
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--out", default="")  # .pt file
    ap.add_argument("--horizon", type=int, default=TOKENIZER_HORIZON)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_trajs", type=int, default=2000)
    args = ap.parse_args()

    if not args.out:
        args.out = QUEST_TOKENIZER_CKPT if args.kind == "quest" else OAT_TOKENIZER_CKPT
        os.makedirs(os.path.dirname(args.out), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = CalvinActionChunkDataset(args.data_dir, horizon=args.horizon, max_trajs=args.max_trajs)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)

    normalizer = fit_calvin_normalizer(args.data_dir, max_trajs=args.max_trajs)

    if args.kind == "quest":
        tok = QueSTTok(action_dim=ACTION_DIM, horizon=args.horizon, vq_type="fsq", fsq_level=[8,5,5,5], downsample_factor=TOKENIZER_DOWNSAMPLE_FACTOR)
    else:
        tok = build_oat(args.horizon)

    tok.set_normalizer(normalizer)
    tok = tok.to(device)

    opt = torch.optim.AdamW(tok.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    tok.train()
    for ep in range(args.epochs):
        tot = 0.0
        n = 0
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = tok(batch)   # both tokenizers implement forward() returning recon loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tok.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            n += 1
        print(f"epoch {ep}: loss={tot/max(n,1):.6f}")

    # save checkpoint
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": tok.state_dict(), "normalizer": normalizer}, args.out)
    print("saved:", args.out)


if __name__ == "__main__":
    main()