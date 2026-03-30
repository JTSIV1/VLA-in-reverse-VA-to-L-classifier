"""CALVIN dataset base class and subclasses.

CalvinDataset: base class that handles loading raw data from per-frame .npz
files given (start_idx, end_idx) time spans. Each row in the DataFrame is a
time span — could be a full episode (from auto_lang_ann.npy) or a subtask
segment (from Gemini decomposition).

Subclasses define __getitem__ for specific consumers:
- CalvinVerbProbeDataset: verb classification from full action sequences
- CalvinTokenizerDataset: action chunks for tokenizer training (recon, verb, CLIP)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
try:
    from torchvision import transforms
    from PIL import Image
except (ImportError, RuntimeError):
    transforms = None
    Image = None

from config import (
    IMAGE_KEY, ACTION_KEY, SCENE_OBS_KEY,
    SCENE_OBS_DIM, SCENE_REP_DIM, ACTION_DIM,
    PATCH_SIZE, IMAGE_SIZE, MAX_SEQ_LEN, EPISODE_TEMPLATE,
)


class CalvinDataset(Dataset):
    """Base CALVIN dataset — loads raw data from per-frame .npz files.

    Each row in df must have at least 'start_idx' and 'end_idx' columns
    (global frame indices). Additional columns (instruction, primary_verb,
    verb, task, etc.) are available to subclasses via self.df.iloc[idx].

    Also provides:
    - Verb vocabulary mapping (verb_to_id / id_to_verb) built from the df
    - Action chunking utility (_chunk_actions)

    Args:
        data_dir: path to CALVIN split (e.g. .../task_D_D/training/)
        df: DataFrame with (start_idx, end_idx, ...) per time span
        verb_to_id: optional pre-built verb mapping (for val set consistency)
        cache_actions: if True, preload all timestep actions into RAM
            (builds/loads a _action_cache.npz file for fast restarts)
        transform: torchvision transform for image loading
        img_size: image size for transforms / dummy tensors
    """

    def __init__(self, data_dir, df, verb_to_id=None, cache_actions=False,
                 transform=None, img_size=224):
        self.data_dir = data_dir
        self.df = df
        self.transform = transform
        self.img_size = img_size

        # Verb vocabulary
        self._verb_col = 'primary_verb' if 'primary_verb' in df.columns else 'verb'
        if verb_to_id is not None:
            self.verb_to_id = verb_to_id
        elif self._verb_col in df.columns:
            unique_verbs = sorted(df[self._verb_col].dropna().unique())
            self.verb_to_id = {v: i for i, v in enumerate(unique_verbs)}
        else:
            self.verb_to_id = {}
        self.id_to_verb = {i: v for v, i in self.verb_to_id.items()}

        if cache_actions:
            self._action_cache, self._cache_offset = self._build_action_cache()
        else:
            self._action_cache = None
            self._cache_offset = 0

    def __len__(self):
        return len(self.df)

    # ------------------------------------------------------------------
    # Raw data loading (protected methods for subclasses)
    # ------------------------------------------------------------------

    def _load_npz(self, frame_idx):
        """Load a single frame's .npz file."""
        path = os.path.join(self.data_dir, EPISODE_TEMPLATE.format(frame_idx))
        return np.load(path, mmap_mode='r')

    def _get_actions(self, idx):
        """Load raw action trajectory for row idx. Returns (T, action_dim)."""
        row = self.df.iloc[idx]
        start, end = int(row['start_idx']), int(row['end_idx'])

        if self._action_cache is not None:
            s = start - self._cache_offset
            e = end - self._cache_offset + 1
            return self._action_cache[s:e].copy()

        actions = []
        for i in range(start, end + 1):
            data = self._load_npz(i)
            actions.append(np.array(data[ACTION_KEY]))
        return np.array(actions, dtype=np.float32)

    def _get_frames(self, idx, frame_indices=None, num_frames=2):
        """Load and transform frames for row idx.

        Args:
            frame_indices: explicit list of global frame indices to load.
                If None, samples num_frames uniformly from [start, end].
        Returns:
            torch.Tensor of shape (num_frames, C, H, W)
        """
        row = self.df.iloc[idx]
        start, end = int(row['start_idx']), int(row['end_idx'])

        if frame_indices is None:
            total_steps = end - start + 1
            if num_frames == 2:
                frame_indices = [start, end]
            else:
                positions = np.linspace(0, total_steps - 1, num_frames, dtype=int)
                frame_indices = [start + p for p in positions]

        frame_list = []
        for fi in frame_indices:
            data = self._load_npz(fi)
            img = Image.fromarray(np.array(data[IMAGE_KEY])).convert("RGB")
            if self.transform:
                frame_list.append(self.transform(img))
            else:
                frame_list.append(transforms.ToTensor()(img))
        return torch.stack(frame_list)

    def _get_scene_obs(self, idx):
        """Load scene_obs at start and end of time span.

        Returns:
            (start_obs, end_obs) each np.ndarray of shape (scene_obs_dim,)
        """
        row = self.df.iloc[idx]
        start_data = self._load_npz(int(row['start_idx']))
        end_data = self._load_npz(int(row['end_idx']))
        return (np.array(start_data[SCENE_OBS_KEY], dtype=np.float32),
                np.array(end_data[SCENE_OBS_KEY], dtype=np.float32))

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    def _get_verb_id(self, idx):
        """Get verb label as integer for row idx."""
        verb = self.df.iloc[idx].get(self._verb_col, None)
        return self.verb_to_id.get(verb, 0) if verb else 0

    def _get_instruction(self, idx):
        """Get instruction string for row idx."""
        return self.df.iloc[idx]['instruction']

    # ------------------------------------------------------------------
    # Action chunking
    # ------------------------------------------------------------------

    def _chunk_actions(self, actions, window_size, max_windows):
        """Chunk raw (T, D) actions into fixed-size windows with padding.

        Returns: (padded_windows, positions, n_windows) where
            padded_windows: (max_windows, window_size, action_dim)
            positions: (max_windows,) normalized start positions in [0, 1]
        """
        T, action_dim = actions.shape

        n_windows = T // window_size
        if n_windows == 0:
            padded = np.pad(actions, ((0, window_size - T), (0, 0)), mode='edge')
            windows = padded.reshape(1, window_size, action_dim)
            n_windows = 1
        else:
            usable = n_windows * window_size
            windows = actions[:usable].reshape(n_windows, window_size, action_dim)

        if n_windows > max_windows:
            windows = windows[:max_windows]
            n_windows = max_windows

        padded_windows = np.zeros(
            (max_windows, window_size, action_dim), dtype=np.float32)
        padded_windows[:n_windows] = windows

        positions = np.zeros(max_windows, dtype=np.float32)
        for i in range(n_windows):
            positions[i] = (i * window_size) / max(T - 1, 1)

        return padded_windows, positions, n_windows

    # ------------------------------------------------------------------
    # Action caching
    # ------------------------------------------------------------------

    def _build_action_cache(self):
        """Preload all needed timestep actions into a contiguous array.

        Builds/loads a _action_cache.npz file in data_dir for fast restarts.
        """
        cache_path = os.path.join(self.data_dir, '_action_cache.npz')
        all_starts = self.df['start_idx'].values.astype(int)
        all_ends = self.df['end_idx'].values.astype(int)

        if os.path.exists(cache_path):
            print(f"  Loading action cache from {cache_path}...")
            cache = np.load(cache_path)
            return cache['actions'], int(cache['offset'])

        needed = set()
        for s, e in zip(all_starts, all_ends):
            needed.update(range(s, e + 1))
        needed = sorted(needed)
        offset = needed[0]
        size = needed[-1] - offset + 1
        print(f"  Building action cache: {len(needed)} timesteps "
              f"({offset}-{needed[-1]})...")
        all_actions = np.zeros((size, ACTION_DIM), dtype=np.float32)
        for j in needed:
            path = os.path.join(self.data_dir, EPISODE_TEMPLATE.format(j))
            data = np.load(path, mmap_mode='r')
            all_actions[j - offset] = data[ACTION_KEY]
        np.savez_compressed(cache_path, actions=all_actions,
                            offset=np.array(offset))
        print(f"  Saved cache to {cache_path}")
        return all_actions, offset

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__")


# ======================================================================
# Verb probe subclass
# ======================================================================

class CalvinVerbProbeDataset(CalvinDataset):
    """For verb classification: full action sequence + optional frames/scene_obs.

    Returns: (frames, actions_tensor, scene_vec, verb_label, seq_len)

    This is the standard batch format consumed by MotionVerbClassifier.
    Actions are native continuous vectors by default; FAST tokenization
    is supported via the action_tokenizer parameter. For VQ-BeT/OAT/QueST
    tokenized actions, use CalvinTokenizerDataset with on-the-fly encoding.
    """

    # Import here to avoid circular dependency at module level
    SCENE_FUSION_MODALITIES = ("scene_token", "scene_concat", "scene_film",
                               "scene_mlp")

    def __init__(self, data_dir, df, modality="action_only",
                 action_tokenizer=None,
                 max_seq_len=MAX_SEQ_LEN, num_frames=2, delta_patches=0,
                 image_encoder="scratch", num_patches=64,
                 verb_to_id=None,
                 transform=None, img_size=224,
                 cache_actions=False):
        super().__init__(data_dir, df, verb_to_id=verb_to_id,
                         cache_actions=cache_actions,
                         transform=transform, img_size=img_size)
        self.modality = modality
        self.action_tokenizer = action_tokenizer
        self.max_seq_len = max_seq_len
        self.num_frames = num_frames
        self.delta_patches = delta_patches

        # Oracle modality
        self.obs_key = SCENE_OBS_KEY if modality == "scene_obs" else None

        # Scene rep flag
        self.scene_rep = modality in self.SCENE_FUSION_MODALITIES

        # Num patches based on encoder
        if image_encoder in ("r3m",):
            self.num_patches = 49
        elif image_encoder in ("dinov2_s", "dinov2_b", "vc1"):
            self.num_patches = 49 if delta_patches == 0 else delta_patches
        elif image_encoder == "dinov2":
            self.num_patches = 64
        elif image_encoder == "resnet18":
            self.num_patches = 49
        else:
            self.num_patches = (IMAGE_SIZE[0] // PATCH_SIZE) ** 2
        if num_patches != 64:
            self.num_patches = num_patches

        print(f"Vocab mapped: {len(self.verb_to_id)} unique verbs.")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        start_idx, end_idx = int(row['start_idx']), int(row['end_idx'])

        # -- Frames --
        if self.modality not in ("action_only", "scene_obs") + self.SCENE_FUSION_MODALITIES:
            frames = self._get_frames(idx, num_frames=self.num_frames)
        else:
            frames = torch.zeros(self.num_frames, 3, self.img_size, self.img_size)

        # -- Oracle obs or actions --
        if self.obs_key is not None:
            actions_tensor, action_real_len = self._load_oracle_obs(
                idx, start_idx, end_idx)
        elif self.modality != "vision_only":
            actions_tensor, action_real_len = self._load_and_tokenize_actions(idx)
        else:
            actions_tensor = torch.zeros(self.max_seq_len, ACTION_DIM)
            action_real_len = 0

        label = torch.tensor(self._get_verb_id(idx), dtype=torch.long)

        # -- Scene vec --
        if self.scene_rep:
            start_obs, end_obs = self._get_scene_obs(idx)
            delta = end_obs - start_obs
            scene_vec = torch.from_numpy(np.concatenate([start_obs, delta]))
        else:
            scene_vec = torch.zeros(SCENE_REP_DIM)

        # -- Sequence length for padding mask --
        seq_len = 1  # CLS
        if self.modality == "scene_token":
            seq_len += 1
        elif self.modality not in ("action_only", "scene_obs") + self.SCENE_FUSION_MODALITIES:
            if self.delta_patches > 0:
                seq_len += max(self.num_frames - 1, 1) * self.delta_patches
            else:
                seq_len += self.num_frames * self.num_patches
        if self.modality != "vision_only":
            seq_len += action_real_len

        return frames, actions_tensor, scene_vec, label, seq_len

    def _load_and_tokenize_actions(self, idx):
        """Load actions and optionally FAST-tokenize. Returns (tensor, real_len)."""
        actions = self._get_actions(idx)
        L = actions.shape[0]

        if self.action_tokenizer is not None:
            # FAST tokenizer: (T, D) → List[int]
            token_ids = self.action_tokenizer(actions)[0]
            token_ids = list(token_ids)
            L_tok = len(token_ids)
            if L_tok < self.max_seq_len:
                token_ids = token_ids + [0] * (self.max_seq_len - L_tok)
            else:
                token_ids = token_ids[:self.max_seq_len]
            return torch.tensor(token_ids, dtype=torch.long), min(L_tok, self.max_seq_len)

        # Native continuous actions
        if L < self.max_seq_len:
            actions_padded = np.pad(actions, ((0, self.max_seq_len - L), (0, 0)),
                                    mode='constant')
        else:
            actions_padded = actions[:self.max_seq_len]
        return torch.tensor(actions_padded, dtype=torch.float32), min(L, self.max_seq_len)

    def _load_oracle_obs(self, idx, start_idx, end_idx):
        """Load oracle obs (scene_obs) as the action input."""
        total_steps = end_idx - start_idx + 1
        if self.num_frames == 0:
            sample_indices = list(range(start_idx, end_idx + 1))
        elif self.num_frames == 2:
            sample_indices = [start_idx, end_idx]
        else:
            positions = np.linspace(0, total_steps - 1, self.num_frames, dtype=int)
            sample_indices = [start_idx + p for p in positions]

        obs_list = [np.array(self._load_npz(si)[self.obs_key])
                    for si in sample_indices]
        obs = np.array(obs_list)
        L = obs.shape[0]
        if L < self.max_seq_len:
            obs_padded = np.pad(obs, ((0, self.max_seq_len - L), (0, 0)),
                                mode='constant')
        else:
            obs_padded = obs[:self.max_seq_len]
        return torch.tensor(obs_padded, dtype=torch.float32), min(L, self.max_seq_len)


# ======================================================================
# Tokenizer training subclass
# ======================================================================

class CalvinTokenizerDataset(CalvinDataset):
    """Unified CALVIN dataset for tokenizer training (recon, verb, CLIP).

    Always returns episode dict format matching BridgeTokenizerDataset:
    {
        'chunks':      (max_chunks, chunk_size, action_dim),
        'positions':   (max_chunks,) — normalized start positions in [0, 1],
        'n_valid':     int — number of real chunks,
        'verb_label':  int — verb class id (-1 if unknown),
        'instruction': str — episode instruction (empty if not provided),
    }

    Args:
        sampling: 'random' — sample K random overlapping windows.
                  'sequential' — tile non-overlapping chunks.
    """

    def __init__(self, data_dir, df, chunk_size=5, max_chunks=16,
                 sampling='random', verb_to_id=None, cache_actions=False,
                 include_instruction=False):
        super().__init__(data_dir, df, verb_to_id=verb_to_id,
                         cache_actions=cache_actions)
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self.sampling = sampling
        self.include_instruction = include_instruction

    def __getitem__(self, idx):
        actions = self._get_actions(idx)
        T = len(actions)
        cs = self.chunk_size
        adim = actions.shape[1]

        if self.sampling == 'random':
            starts, n_valid = self._random_starts(T, cs)
        else:
            starts, n_valid = self._sequential_starts(T, cs)

        chunks = np.zeros((self.max_chunks, cs, adim), dtype=np.float32)
        positions = np.zeros(self.max_chunks, dtype=np.float32)

        for i, s in enumerate(starts[:n_valid]):
            end = s + cs
            if end <= T:
                chunks[i] = actions[s:end]
            elif T > s:
                chunks[i] = np.pad(actions[s:], ((0, end - T), (0, 0)), mode='edge')
            else:
                chunks[i] = np.pad(actions, ((0, cs - T), (0, 0)), mode='edge')
            positions[i] = s / max(T - 1, 1)

        instruction = ""
        if self.include_instruction:
            instruction = self._get_instruction(idx)

        return {
            'chunks': torch.tensor(chunks, dtype=torch.float32),
            'positions': torch.tensor(positions, dtype=torch.float32),
            'n_valid': torch.tensor(n_valid, dtype=torch.long),
            'verb_label': torch.tensor(self._get_verb_id(idx), dtype=torch.long),
            'instruction': instruction,
        }

    def _random_starts(self, T, cs):
        max_start = max(0, T - cs)
        if max_start > 0:
            n_possible = max_start + 1
            n_valid = min(self.max_chunks, n_possible)
            starts = np.sort(np.random.choice(n_possible, n_valid, replace=False))
        else:
            starts = np.array([0])
            n_valid = 1
        return starts, n_valid

    def _sequential_starts(self, T, cs):
        n_chunks = max(1, -(-T // cs))  # ceiling division
        n_valid = min(n_chunks, self.max_chunks)
        starts = np.array([i * cs for i in range(n_valid)])
        return starts, n_valid


# ======================================================================
# Dataset builder helpers (shared by train_tokenizer.py & train_verb_probe.py)
# ======================================================================

def _load_and_filter_dfs(data_dir, val_dir, min_class_count=0):
    """Load CALVIN DataFrames and optionally filter sparse verb classes.

    Returns (train_df, val_df).
    """
    from utils import load_calvin_to_dataframe

    train_df = load_calvin_to_dataframe(data_dir)
    val_df = load_calvin_to_dataframe(val_dir)

    if min_class_count > 0:
        verb_col = 'primary_verb' if 'primary_verb' in train_df.columns else 'verb'
        vc = train_df[verb_col].value_counts()
        keep_verbs = set(vc[vc >= min_class_count].index)
        n_before = len(train_df)
        train_df = train_df[train_df[verb_col].isin(keep_verbs)].reset_index(drop=True)
        val_df = val_df[val_df[verb_col].isin(keep_verbs)].reset_index(drop=True)
        print(f"Filtered: {len(vc)}->{len(keep_verbs)} classes, "
              f"train {n_before}->{len(train_df)}, val->{len(val_df)}")

    return train_df, val_df


def _drop_unseen_val_verbs(val_ds, val_df, verb_to_id):
    """Drop val samples whose verb is not in verb_to_id."""
    verb_col = val_ds._verb_col
    valid_mask = val_df[verb_col].isin(verb_to_id.keys())
    n_drop = (~valid_mask).sum()
    if n_drop > 0:
        print(f"Dropping {n_drop} val samples with unseen verbs")
        val_ds.df = val_df[valid_mask].reset_index(drop=True)


def _verb_metadata(train_ds, train_df):
    """Extract verb_to_id, id_to_verb, verb_counts from a built dataset."""
    verb_to_id = train_ds.verb_to_id
    id_to_verb = train_ds.id_to_verb
    verb_col = train_ds._verb_col
    verb_counts = train_df[verb_col].value_counts().to_dict()
    num_verbs = len(verb_to_id)
    return num_verbs, id_to_verb, verb_to_id, verb_counts


def build_calvin_tokenizer_data(data_dir, val_dir, chunk_size, max_chunks,
                                sampling, min_class_count=0,
                                cache_actions=False,
                                include_instruction=False):
    """Build train/val CalvinTokenizerDataset pair.

    Used by both tokenizer training and verb probe (tokid/latent modes).

    Returns (train_ds, val_ds, num_verbs, id_to_verb, verb_to_id, verb_counts).
    """
    train_df, val_df = _load_and_filter_dfs(data_dir, val_dir, min_class_count)

    train_ds = CalvinTokenizerDataset(
        data_dir, train_df, chunk_size=chunk_size,
        max_chunks=max_chunks, sampling=sampling,
        cache_actions=cache_actions,
        include_instruction=include_instruction)
    val_ds = CalvinTokenizerDataset(
        val_dir, val_df, chunk_size=chunk_size,
        max_chunks=max_chunks, sampling=sampling,
        verb_to_id=train_ds.verb_to_id,
        cache_actions=cache_actions,
        include_instruction=include_instruction)

    _drop_unseen_val_verbs(val_ds, val_df, train_ds.verb_to_id)
    num_verbs, id_to_verb, verb_to_id, verb_counts = _verb_metadata(train_ds, train_df)

    return train_ds, val_ds, num_verbs, id_to_verb, verb_to_id, verb_counts


def build_calvin_verb_probe_data(data_dir, val_dir, min_class_count=0,
                                 cache_actions=False, **ds_kwargs):
    """Build train/val CalvinVerbProbeDataset pair.

    Extra kwargs are forwarded to CalvinVerbProbeDataset (modality,
    action_tokenizer, max_seq_len, num_frames, delta_patches,
    image_encoder, transform, img_size).

    Returns (train_ds, val_ds, num_verbs, id_to_verb, verb_to_id, verb_counts).
    """
    train_df, val_df = _load_and_filter_dfs(data_dir, val_dir, min_class_count)

    train_ds = CalvinVerbProbeDataset(
        data_dir, train_df, cache_actions=cache_actions, **ds_kwargs)
    val_ds = CalvinVerbProbeDataset(
        val_dir, val_df, verb_to_id=train_ds.verb_to_id,
        cache_actions=cache_actions, **ds_kwargs)

    _drop_unseen_val_verbs(val_ds, val_df, train_ds.verb_to_id)
    num_verbs, id_to_verb, verb_to_id, verb_counts = _verb_metadata(train_ds, train_df)

    return train_ds, val_ds, num_verbs, id_to_verb, verb_to_id, verb_counts
