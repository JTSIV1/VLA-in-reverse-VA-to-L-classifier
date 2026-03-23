"""BridgeV2 verb classification datasets.

BridgeVerbDataset: per-segment verb classification from action trajectories.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------- BridgeV2 constants ----------
BRIDGE_ACTION_DIM = 7
BRIDGE_CSV = "data/bridge_verb_segments.csv"
BRIDGE_ACTIONS_NPZ = "/data/user_data/wenjiel2/datasets/bridge_actions/segment_actions.npz"
BRIDGE_MAX_SEQ_LEN = 64  # segments are short (mean=7, max=117)


class BridgeVerbDataset(Dataset):
    """Dataset for BridgeV2 subtask verb classification."""

    def __init__(self, df, actions_cache, max_seq_len=BRIDGE_MAX_SEQ_LEN,
                 verb_to_id=None):
        self.df = df.reset_index(drop=True)
        self.actions_cache = actions_cache
        self.max_seq_len = max_seq_len

        if verb_to_id is not None:
            self.verb_to_id = verb_to_id
        else:
            unique_verbs = sorted(self.df["verb"].unique())
            self.verb_to_id = {v: i for i, v in enumerate(unique_verbs)}
        self.id_to_verb = {i: v for v, i in self.verb_to_id.items()}
        print(f"Vocab: {len(self.verb_to_id)} verbs")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seg_idx = row["seg_idx"]
        verb = row["verb"]

        actions = self.actions_cache[f"actions_{seg_idx}"]
        L = actions.shape[0]

        if L < self.max_seq_len:
            actions_padded = np.pad(actions, ((0, self.max_seq_len - L), (0, 0)),
                                    mode="constant")
        else:
            actions_padded = actions[:self.max_seq_len]

        actions_tensor = torch.tensor(actions_padded, dtype=torch.float32)
        action_real_len = min(L, self.max_seq_len)
        label = torch.tensor(self.verb_to_id.get(verb, 0), dtype=torch.long)

        # Dummy frames and scene_vec for compatibility with standard batch format
        frames = torch.zeros(2, 3, 224, 224)
        scene_vec = torch.zeros(48)
        seq_len = 1 + action_real_len  # CLS + action tokens

        return frames, actions_tensor, scene_vec, label, seq_len
