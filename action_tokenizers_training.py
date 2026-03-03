import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import load_calvin_to_dataframe
from cluster_analysis import load_all_actions
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
)  # from the file above

# QueST / OAT imports are heavy (require zarr, vector_quantize_pytorch); import lazily
def _import_quest():
    from oat.tokenizer.quest.tokenizer import QueSTTok
    return QueSTTok

def _import_oat():
    from oat.tokenizer.oat.tokenizer import OATTok
    from oat.tokenizer.oat.encoder.register_encoder import RegisterEncoder
    from oat.tokenizer.oat.decoder.single_pass_decoder import SinglePassDecoder
    from oat.tokenizer.oat.quantizer.fsq import FSQ
    return OATTok, RegisterEncoder, SinglePassDecoder, FSQ


def fit_calvin_normalizer(data_dir, max_trajs=None):
    """Fit oat LinearNormalizer on all CALVIN actions in data_dir."""
    print("Fitting tokenizer normalizer on actions from trajectories in", data_dir)
    df = load_calvin_to_dataframe(data_dir)
    if max_trajs:
        df = df.head(min(max_trajs, len(df))).copy()

    print("Reading actions...")
    all_actions, _ = load_all_actions(df, num_workers=8)
    actions_t = torch.from_numpy(all_actions)

    print("Fitting normalizer...")
    from oat.model.common.normalizer import LinearNormalizer
    normalizer = LinearNormalizer()
    normalizer.fit({"action": actions_t}, last_n_dims=1, mode="limits", output_min=-1.0, output_max=1.0)
    return normalizer


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
    OATTok, RegisterEncoder, SinglePassDecoder, FSQ = _import_oat()
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


def train_tokenizer(kind, data_dir, out_path, normalizer, horizon=TOKENIZER_HORIZON, batch=256, epochs=10, lr=5e-5, max_trajs=2000):
    print(f"Training {kind} tokenizer on data from {data_dir} for epochs {epochs}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = CalvinActionChunkDataset(data_dir, horizon=horizon, max_trajs=max_trajs)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=4, drop_last=True)

    if kind == "quest":
        QueSTTok = _import_quest()
        tok = QueSTTok(action_dim=ACTION_DIM, horizon=horizon, vq_type="fsq", fsq_level=[8,5,5,5], downsample_factor=TOKENIZER_DOWNSAMPLE_FACTOR)
    else:
        tok = build_oat(horizon)

    tok.set_normalizer(normalizer)
    tok = tok.to(device)

    opt = torch.optim.AdamW(tok.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    tok.train()
    pbar = tqdm(range(epochs), total=epochs, desc=f"Training {kind} tokenizer")
    for ep in pbar:
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
        pbar.set_postfix(loss=f"{tot/max(n,1):.6f}")

    # save checkpoint
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"model": tok.state_dict(), "normalizer": normalizer}, out_path)
    print("saved:", out_path)


if __name__ == "__main__":
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

    normalizer = fit_calvin_normalizer(args.data_dir, max_trajs=args.max_trajs)
    train_tokenizer(args.kind, args.data_dir, args.out, normalizer, args.horizon, args.batch, args.epochs, args.lr, args.max_trajs)