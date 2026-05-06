"""BridgeV2 datasets and data-loading utilities.

BridgeVerbDataset: per-segment verb classification from action trajectories.
BridgeVisionDataset: first+last frame vision classification.
BridgeTokenizerDataset: episode-level tokenizer training (recon, verb, CLIP).
"""

import os
import glob as _glob

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image


# ---------- BridgeV2 constants ----------
BRIDGE_ACTION_DIM = 7
BRIDGE_CSV = "data/bridge_verb_segments.csv"
BRIDGE_ACTIONS_NPZ = "/data/user_data/wenjiel2/datasets/bridge_actions/segment_actions.npz"
BRIDGE_FRAMES_DIR = "/data/user_data/wenjiel2/datasets/bridge_frames"
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


# ======================================================================
# Vision dataset (first + last frame → verb classification)
# ======================================================================

class BridgeVisionDataset(Dataset):
    """Dataset for BridgeV2 vision-only verb classification.

    Loads pre-extracted first+last frames (from extract_bridge_frames.py)
    and returns them with the standard batch format expected by
    GoalVerbClassifier.

    Args:
        episode_keys: list of episode keys to include
        verb_ids: list of verb class IDs (parallel to episode_keys)
        frames_cache: dict mapping episode_key → (first_frame, last_frame)
                      where frames are uint8 numpy arrays (H, W, 3)
        verb_to_id: dict mapping verb string → class ID
        transform: torchvision transform to apply to each frame
    """

    def __init__(self, episode_keys, verb_ids, frames_cache,
                 verb_to_id=None, transform=None):
        # Filter to episodes that have both frames and valid verb labels
        self.samples = []
        for key, vid in zip(episode_keys, verb_ids):
            if vid >= 0 and key in frames_cache:
                self.samples.append((key, vid))

        self.frames_cache = frames_cache
        self.verb_to_id = verb_to_id or {}
        self.id_to_verb = {i: v for v, i in self.verb_to_id.items()}
        self.transform = transform
        print("BridgeVisionDataset: {} samples ({} verbs)".format(
            len(self.samples), len(self.verb_to_id)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, label = self.samples[idx]
        first_frame, last_frame = self.frames_cache[key]

        # Apply transform (resize, normalize) or default conversion
        if self.transform is not None:
            first_img = self.transform(Image.fromarray(first_frame))
            last_img = self.transform(Image.fromarray(last_frame))
        else:
            # Default: convert to float tensor, CHW, normalize to [0, 1]
            first_img = torch.from_numpy(first_frame).permute(2, 0, 1).float() / 255.0
            last_img = torch.from_numpy(last_frame).permute(2, 0, 1).float() / 255.0

        frames = torch.stack([first_img, last_img], dim=0)  # (2, 3, H, W)

        # Dummy tensors for batch-format compatibility
        actions = torch.zeros(BRIDGE_MAX_SEQ_LEN, BRIDGE_ACTION_DIM)
        scene_vec = torch.zeros(48)
        label_t = torch.tensor(label, dtype=torch.long)
        seq_len = torch.tensor(1, dtype=torch.long)  # unused for vision-only

        return frames, actions, scene_vec, label_t, seq_len


# ======================================================================
# Motion+Goal dataset (paired actions + first/last frame for the unified
# capacity-matched probe; see lab_notebooks/motion_goal_capacity_matched/)
# ======================================================================

class BridgeMotionGoalDataset(Dataset):
    """Pairs each Bridge episode with both its action trajectory and frames.

    Returns the standard 5-tuple expected by run_training_loop:
        (frames, actions, scene_vec, label, seq_len)
    where:
        frames    : (num_frames, 3, H, W) — first and last frame, transformed
        actions   : (max_seq_len, 7)      — action trajectory, zero-padded
        scene_vec : (48,)                 — dummy zeros, kept for batch compat
        label     : ()                    — verb class id
        seq_len   : int                   — number of real action timesteps

    Modality masking happens inside MotionGoalVerbClassifier; this dataset
    always returns both modalities so AO / VO / Both runs use identical
    batches.

    Args:
        episode_keys: list of episode keys parallel to actions_list / verb_ids
        actions_list: list of (T_i, 7) numpy arrays (variable length)
        verb_ids:     list of int verb class ids (parallel to episode_keys)
        frames_cache: dict episode_key -> (first_frame, last_frame) uint8 HWC
        verb_to_id:   dict verb -> id
        transform:    torchvision transform applied to each frame
        max_seq_len:  pad/truncate action sequences to this length
    """

    def __init__(self, episode_keys, actions_list, verb_ids, frames_cache,
                 verb_to_id=None, transform=None,
                 max_seq_len=BRIDGE_MAX_SEQ_LEN,
                 skip_frame_data=False):
        """
        Args:
            frames_cache: dict key -> (first, last) frame uint8 arrays, OR
                a set of keys when skip_frame_data=True (AO mode).
            skip_frame_data: if True, return zero-frame placeholders. The
                model still needs frame-shaped tensors for capacity-matched
                masking, but does not need real pixel data because the
                vision tokens are masked out.
        """
        assert len(episode_keys) == len(actions_list) == len(verb_ids)
        self.samples = []
        for key, actions, vid in zip(episode_keys, actions_list, verb_ids):
            if vid >= 0 and key in frames_cache:
                self.samples.append((key, actions, vid))
        self.frames_cache = frames_cache
        self.skip_frame_data = skip_frame_data
        self.verb_to_id = verb_to_id or {}
        self.id_to_verb = {i: v for v, i in self.verb_to_id.items()}
        self.transform = transform
        self.max_seq_len = max_seq_len
        print("BridgeMotionGoalDataset: {} samples ({} verbs){}".format(
            len(self.samples), len(self.verb_to_id),
            " [zero-frame mode]" if skip_frame_data else ""))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, actions, label = self.samples[idx]

        if self.skip_frame_data:
            frames = torch.zeros(2, 3, 224, 224)
        else:
            first_frame, last_frame = self.frames_cache[key]
            if self.transform is not None:
                first_img = self.transform(Image.fromarray(first_frame))
                last_img = self.transform(Image.fromarray(last_frame))
            else:
                first_img = torch.from_numpy(first_frame).permute(2, 0, 1).float() / 255.0
                last_img = torch.from_numpy(last_frame).permute(2, 0, 1).float() / 255.0
            frames = torch.stack([first_img, last_img], dim=0)  # (2, 3, H, W)

        L = actions.shape[0]
        if L < self.max_seq_len:
            actions_padded = np.pad(
                actions, ((0, self.max_seq_len - L), (0, 0)), mode="constant")
        else:
            actions_padded = actions[:self.max_seq_len]
        actions_tensor = torch.tensor(actions_padded, dtype=torch.float32)
        action_real_len = min(L, self.max_seq_len)

        scene_vec = torch.zeros(48)
        label_t = torch.tensor(label, dtype=torch.long)
        seq_len = torch.tensor(action_real_len, dtype=torch.long)
        return frames, actions_tensor, scene_vec, label_t, seq_len


# ======================================================================
# Episode-level dataset (unified for recon, verb, and CLIP training)
# ======================================================================

class BridgeTokenizerDataset(Dataset):
    """Yields K action chunks per episode with positions and metadata.

    Unified dataset for all training modes (recon-only, verb, CLIP).
    With max_chunks=1, degenerates to single-chunk training.

    Args:
        sampling: 'random' — sample K random overlapping windows (OAT/QueST).
                  'sequential' — tile non-overlapping chunks (VQ-BeT).
        instructions: per-episode instruction strings (for CLIP head).

    Each item: {
        'chunks':           (max_chunks, chunk_size, action_dim),
        'action_real_lens': (max_chunks,) — real timesteps per chunk (≤ chunk_size),
        'positions':        (max_chunks,) — normalized start position in [0, 1],
        'n_valid':          int — number of real chunks (≤ max_chunks),
        'verb_label':       int — episode-level verb class id (-1 if unknown),
        'instruction':      str — episode instruction (empty if not provided),
    }
    """

    def __init__(self, actions_list, chunk_size=32, max_chunks=8,
                 sampling='random',
                 verb_ids=None, verb_to_id=None, instructions=None):
        self.actions = actions_list
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self.sampling = sampling
        self.verb_ids = verb_ids
        self.verb_to_id = verb_to_id or {}
        self.instructions = instructions
        # Include all episodes; filter to verb-labeled only if verb_ids provided
        self.ep_indices = []
        for i in range(len(actions_list)):
            if verb_ids is None or verb_ids[i] >= 0:
                self.ep_indices.append(i)

    def __len__(self):
        return len(self.ep_indices)

    def __getitem__(self, idx):
        ep_idx = self.ep_indices[idx]
        actions = self.actions[ep_idx]
        T = len(actions)
        cs = self.chunk_size
        adim = actions.shape[1]

        if self.sampling == 'random':
            starts, n_valid = self._random_starts(T, cs)
        else:
            starts, n_valid = self._sequential_starts(T, cs)

        chunks = np.zeros((self.max_chunks, cs, adim), dtype=np.float32)
        action_real_lens = np.full(self.max_chunks, cs, dtype=np.int64)
        positions = np.zeros(self.max_chunks, dtype=np.float32)

        for i, s in enumerate(starts[:n_valid]):
            end = s + cs
            real_len = min(cs, T - s)
            if end <= T:
                raw = actions[s:end]
            elif T > s:
                raw = np.pad(actions[s:], ((0, end - T), (0, 0)), mode="edge")
            else:
                raw = np.pad(actions, ((0, cs - T), (0, 0)), mode="edge")
            chunks[i] = raw
            action_real_lens[i] = real_len
            positions[i] = s / max(T - 1, 1)

        verb_label = self.verb_ids[ep_idx] if self.verb_ids is not None else -1
        instruction = self.instructions[ep_idx] if self.instructions is not None else ""
        return {
            'chunks': torch.tensor(chunks, dtype=torch.float32),
            'action_real_lens': torch.tensor(action_real_lens, dtype=torch.long),
            'positions': torch.tensor(positions, dtype=torch.float32),
            'n_valid': torch.tensor(n_valid, dtype=torch.long),
            'verb_label': torch.tensor(verb_label, dtype=torch.long),
            'instruction': instruction,
        }

    def _random_starts(self, T, cs):
        """Sample up to max_chunks random window start positions.

        Only starts where the full chunk fits are allowed: [0, T - cs].
        Episodes shorter than cs are skipped (n_valid=0).
        This matches original OAT's SequenceSampler which drops short episodes.
        """
        n_possible = max(0, T - cs + 1)
        if n_possible == 0:
            return np.array([], dtype=np.int64), 0
        n_valid = min(self.max_chunks, n_possible)
        starts = np.sort(np.random.choice(n_possible, n_valid, replace=False))
        return starts, n_valid

    def _sequential_starts(self, T, cs):
        """Tile non-overlapping chunks across the episode."""
        n_chunks = max(1, -(-T // cs))  # ceiling division
        n_valid = min(n_chunks, self.max_chunks)
        starts = np.array([i * cs for i in range(n_valid)])
        return starts, n_valid


# ======================================================================
# Data loading utilities
# ======================================================================

def load_bridge_frames(frames_dir, keys_only=False):
    """Load pre-extracted first+last frames from sharded NPZ files.

    Args:
        frames_dir: directory containing shard_*.npz files.
        keys_only: if True, skip loading frame data and return frames_cache
            as a set of keys (used by AO probes that don't need image data
            but still need to know which episodes have available frames for
            consistent train/val membership across modalities).

    Returns:
        frames_cache: dict episode_key -> (first_frame, last_frame) of uint8
            numpy arrays (224, 224, 3); or, if keys_only=True, a set of keys.
        keys_list: list of all episode keys in order
    """
    shard_files = sorted(_glob.glob(os.path.join(frames_dir, "shard_*.npz")))
    if not shard_files:
        raise FileNotFoundError(
            "No frame shards found in {}. "
            "Run scripts/extract_bridge_frames.py first.".format(frames_dir))

    print("Loading {} frame shards from {}{}...".format(
        len(shard_files), frames_dir,
        " (keys only)" if keys_only else ""))
    frames_cache = set() if keys_only else {}
    keys_list = []
    for sf in tqdm(shard_files, desc="Loading frame shards"):
        data = np.load(sf, allow_pickle=True)
        n_eps = int(data["n_episodes"])
        for i in range(n_eps):
            key = str(data["episode_key_{}".format(i)])
            keys_list.append(key)
            if keys_only:
                frames_cache.add(key)
            else:
                first_frame = data["first_frame_{}".format(i)]
                last_frame = data["last_frame_{}".format(i)]
                frames_cache[key] = (first_frame, last_frame)
    print("Loaded {} episodes{}".format(
        len(frames_cache), " (keys only)" if keys_only else " with frames"))
    return frames_cache, keys_list


def load_bridge_actions(shard_dir):
    """Load all BridgeV2 action trajectories and episode keys from shards."""
    shard_files = sorted(_glob.glob(os.path.join(shard_dir, "shard_*.npz")))
    print(f"Loading {len(shard_files)} action shards...")
    actions_list = []
    keys_list = []
    for sf in tqdm(shard_files, desc="Loading shards"):
        data = np.load(sf, allow_pickle=True)
        for i in range(int(data["n_episodes"])):
            actions_list.append(data[f"actions_{i}"].astype(np.float32))
            keys_list.append(str(data[f"episode_key_{i}"]))
    print(f"Loaded {len(actions_list)} episodes")
    return actions_list, keys_list


def load_bridge_verb_labels(csv_path, episode_keys, min_class_count=30):
    """Build verb_to_id and per-episode verb_ids from the episode CSV."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    key_to_verb = dict(zip(df["episode_key"], df["verb"]))

    verb_counts = df["verb"].value_counts()
    keep_verbs = set(verb_counts[verb_counts >= min_class_count].index)
    verb_to_id = {v: i for i, v in enumerate(sorted(keep_verbs))}
    print(f"Bridge verb vocab: {len(verb_to_id)} classes (min_count={min_class_count})")

    verb_ids = []
    matched, unmatched = 0, 0
    for key in episode_keys:
        verb = key_to_verb.get(key)
        if verb and verb in verb_to_id:
            verb_ids.append(verb_to_id[verb])
            matched += 1
        else:
            verb_ids.append(-1)
            unmatched += 1
    print(f"  Matched {matched}/{matched + unmatched} episodes to verb labels")
    return verb_ids, verb_to_id


def load_bridge_instructions(csv_path, episode_keys):
    """Return per-episode instruction strings matched by episode_key."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    key_to_instr = dict(zip(df["episode_key"], df["instruction"]))
    instructions = [key_to_instr.get(key, "") for key in episode_keys]
    matched = sum(1 for i in instructions if i)
    print(f"  Matched {matched}/{len(episode_keys)} episodes to instructions")
    return instructions


def fit_bridge_normalizer(actions_list):
    """Fit LinearNormalizer on BridgeV2 actions."""
    from oat.model.common.normalizer import LinearNormalizer
    all_actions = np.concatenate(actions_list, axis=0)
    normalizer = LinearNormalizer()
    normalizer.fit({"action": all_actions}, mode="limits")
    return normalizer
