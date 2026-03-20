"""Geometry-Aware Motion Primitive Tokenizer

Pipeline:
    trajectory (T, 7) :
    - segment into motion primitives,
    - extract 14-D geometric feature per primitive,
    - StandardScaler + KMeans -> token IDs

Classes:
    GAMPTSegmenter: detects primitive boundaries via the 5 kinematic conditions
    GAMPTFeaturizer: extracts 14-D feature vector per primitive
    GAMPTTokenizer: fits/applies scaler+kmeans; dataset-compatible callable

Usage (fit):
    from tokenization.gampt import GAMPTTokenizer
    trajs = [np.load(f)['actions'] for f in train_files]  # List[np.ndarray (T, 7)]
    tok = GAMPTTokenizer.fit(trajs, k=64)
    tok.save("checkpoints/gampt_k64.pkl")

Usage (inference):
    tok = GAMPTTokenizer.load("checkpoints/gampt_k64.pkl")
    ids = tok(actions_np[np.newaxis])   # (1, T, 7) → [[int, ...]]
"""

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Boundary conditions (priority order):
# gripper > stop/start > direction change > speed change > max length
class GAMPTSegmenter:
    """Splits a (T, 7) trajectory into motion primitives.

    Each action vector is (dx, dy, dz, droll, dpitch, dyaw, gripper).

    Args:
        theta_dir: cosine-similarity threshold for direction change (default 0.8 ≈ 37°)
        theta_stop: speed threshold below which robot is considered stationary (m/step)
        v95: 95th-percentile translation speed across training set (set by GAMPTTokenizer.fit)
        theta_speed_mult: fraction of v95 that counts as a significant speed change
        L_max: hard cap on primitive length (timesteps)
    """

    def __init__(self, theta_dir=0.8, theta_stop=0.005,
                 v95=1.0, theta_speed_mult=0.1, L_max=15):
        self.theta_dir = theta_dir
        self.theta_stop = theta_stop
        self.v95 = v95
        self.theta_speed_mult = theta_speed_mult
        self.L_max = L_max

    def segment(self, traj: np.ndarray):
        """Detect primitive boundaries and return list of (start, end) index pairs.

        Both start and end are inclusive. Boundary inserted at timestep t means
        primitive [s, t-1] ends and new primitive [t, ...] begins.

        Args:
            traj: np.ndarray of shape (T, 7)

        Returns:
            List[Tuple[int, int]] -- (start, end) pairs, inclusive, covering [0, T-1]
        """
        T = len(traj)
        if T <= 1:
            return [(0, T - 1)]

        v = traj[:, :3] # translation deltas
        g = traj[:, 6] # gripper state
        speed = np.linalg.norm(v, axis=1) # (T,)
        theta_speed = self.theta_speed_mult * self.v95

        boundary_starts = [0]
        prim_start = 0

        for t in range(1, T):
            fire = False

            # 1. Gripper event (highest priority)
            if abs(g[t] - g[t - 1]) > 0.5:
                fire = True

            # 2. Stop / Start: speed crosses the stationary threshold
            elif (speed[t - 1] < self.theta_stop) != (speed[t] < self.theta_stop):
                fire = True

            # 3. Direction change (only meaningful when both steps are moving)
            elif speed[t - 1] > self.theta_stop and speed[t] > self.theta_stop:
                cos_sim = np.dot(v[t - 1], v[t]) / (speed[t - 1] * speed[t] + 1e-9)
                if cos_sim < self.theta_dir:
                    fire = True

            # 4. Speed change (data-relative threshold)
            elif theta_speed > 0 and abs(speed[t] - speed[t - 1]) > theta_speed:
                fire = True

            # 5. Max length cap (always checked, can override lower-priority decisions)
            if (t - prim_start) >= self.L_max:
                fire = True

            if fire:
                boundary_starts.append(t)
                prim_start = t

        # Build (start, end) pairs from boundary start positions
        primitives = []
        for i, s in enumerate(boundary_starts):
            e = boundary_starts[i + 1] - 1 if i + 1 < len(boundary_starts) else T - 1
            primitives.append((s, e))

        return primitives


# 14-D feature vector per primitive
class GAMPTFeaturizer:
    """Converts a list of (start, end) primitive spans into a (N, 14) feature matrix.

    Feature layout:
        [0:3]  translation displacement vector  Σv_t          (3D)
        [3]    displacement magnitude            ‖Σv_t‖        (1D)
        [4]    mean translation speed            mean ‖v_t‖    (1D)
        [5]    max translation speed             max ‖v_t‖     (1D)
        [6]    duration                          e - s         (1D)
        [7]    # internal direction changes                     (1D)
        [8]    total rotation magnitude          Σ‖r_t‖        (1D)
        [9]    mean rotation per step            mean ‖r_t‖    (1D)
        [10]   dominant rotation axis            argmax(mean |r_axis|), in {0,1,2} (1D)
        [11]   gripper state at start            g[s] > 0.5    (1D)
        [12]   gripper state at end              g[e] > 0.5    (1D)
        [13]   gripper change flag               start ≠ end   (1D)
    """

    def extract(self, traj: np.ndarray, primitives):
        """
        Args:
            traj: np.ndarray (T, 7)
            primitives: List[Tuple[int, int]]  -- (start, end) inclusive pairs

        Returns:
            np.ndarray (len(primitives), 14)
        """
        features = np.zeros((len(primitives), 14), dtype=np.float32)

        for i, (s, e) in enumerate(primitives):
            seg = traj[s: e + 1] # (L, 7), L >= 1
            v = seg[:, :3] # translation deltas
            r = seg[:, 3:6] # rotation deltas
            g = seg[:, 6] # gripper

            speed = np.linalg.norm(v, axis=1) # (L,)

            # Translation (dims 0-5)
            disp_vec = v.sum(axis=0) # (3,) net displacement
            disp_mag = float(np.linalg.norm(disp_vec))
            mean_speed = float(speed.mean())
            max_speed = float(speed.max())

            features[i, 0:3] = disp_vec
            features[i, 3] = disp_mag
            features[i, 4] = mean_speed
            features[i, 5] = max_speed

            # Temporal (dims 6-7)
            duration = e - s # 0 for single-step primitives
            n_dir_changes = 0
            for t in range(1, len(seg)):
                if speed[t - 1] > 1e-6 and speed[t] > 1e-6:
                    cos = np.dot(v[t - 1], v[t]) / (speed[t - 1] * speed[t] + 1e-9)
                    if cos < 0.8:
                        n_dir_changes += 1

            features[i, 6] = float(duration)
            features[i, 7] = float(n_dir_changes)

            # Rotation (dims 8-10)
            rot_norms = np.linalg.norm(r, axis=1) # (L,)
            total_rot = float(rot_norms.sum())
            mean_rot = float(rot_norms.mean())
            mean_abs_per_axis = np.abs(r).mean(axis=0) # (3,)
            dominant_axis = float(np.argmax(mean_abs_per_axis))

            features[i, 8] = total_rot
            features[i, 9] = mean_rot
            features[i, 10] = dominant_axis

            # Gripper (dims 11-13)
            g_start = float(g[0] > 0.5)
            g_end = float(g[-1] > 0.5)
            g_change = float(g_start != g_end)

            features[i, 11] = g_start
            features[i, 12] = g_end
            features[i, 13] = g_change

        return features


# Full tokenizer: fit, save, load, __call__
class GAMPTTokenizer:
    """KMeans-based motion primitive tokenizer.

    vocab_size = k + 1 (k cluster IDs 0..k-1, plus pad_id = k)
    max_primitives: sequences are capped/padded to this length

    Dataset-compatible interface:
        tok(actions_np): where actions_np is (B, T, 7) or (T, 7)
        returns List[List[int]]: variable-length (no padding), for dataset to pad

    For fixed-length output (cluster analysis / ablations):
        tok.tokenize(traj): returns List[int] of length max_primitives
    """

    def __init__(self, k, segmenter, featurizer, scaler, kmeans, max_primitives=10):
        self.k = k
        self.segmenter = segmenter
        self.featurizer = featurizer
        self.scaler = scaler
        self.kmeans = kmeans
        self.max_primitives = max_primitives
        self.vocab_size = k + 1 # k cluster IDs + 1 PAD token
        self.pad_id = k

    # Fit
    @classmethod
    def fit(cls, trajectories, k,
            theta_dir=0.8, theta_stop=0.005,
            theta_speed_mult=0.1, L_max=15,
            max_primitives=10, n_init=20, max_iter=500, random_state=42):
        """Fit scaler + KMeans on training primitives.

        Args:
            trajectories: List[np.ndarray (T_i, 7)]
            k: vocabulary size (number of k-means clusters)
            theta_dir: cosine-sim threshold for direction change boundary
            theta_stop: stationary speed threshold (m/step)
            theta_speed_mult: speed-change threshold = theta_speed_mult * v95
            L_max: max primitive length (timesteps)
            max_primitives: cap on tokens per trajectory (longer → truncated)
            n_init: KMeans n_init
            max_iter: KMeans max_iter
            random_state: KMeans random_state

        Returns:
            Fitted GAMPTTokenizer
        """
        # Compute v95: 95th-percentile translation speed across all training frames
        all_speeds = []
        for traj in trajectories:
            v = traj[:, :3]
            all_speeds.append(np.linalg.norm(v, axis=1))
        v95 = float(np.percentile(np.concatenate(all_speeds), 95))
        print(f"[GAMPT] v95 = {v95:.6f}  (speed threshold = {theta_speed_mult * v95:.6f})")

        segmenter = GAMPTSegmenter(
            theta_dir=theta_dir, theta_stop=theta_stop,
            v95=v95, theta_speed_mult=theta_speed_mult, L_max=L_max,
        )
        featurizer = GAMPTFeaturizer()

        # Extract features from all primitives across all training trajectories
        all_features = []
        total_prims = 0
        for traj in trajectories:
            prims = segmenter.segment(traj)
            feats = featurizer.extract(traj, prims)
            all_features.append(feats)
            total_prims += len(prims)
        print(f"[GAMPT] Extracted {total_prims} primitives from {len(trajectories)} trajectories "
              f"(mean {total_prims / max(len(trajectories), 1):.1f} per traj)")

        X = np.vstack(all_features) # (total_prims, 14)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        print(f"[GAMPT] Fitting KMeans(k={k}, n_init={n_init}) on {len(X)} primitives ...")
        kmeans = KMeans(
            n_clusters=k, init='k-means++',
            n_init=n_init, max_iter=max_iter,
            random_state=random_state,
        )
        kmeans.fit(X_scaled)
        inertia = kmeans.inertia_
        print(f"[GAMPT] KMeans converged. Inertia = {inertia:.2f}")

        return cls(k, segmenter, featurizer, scaler, kmeans, max_primitives)

    # Tokenize
    def _tokenize_raw(self, traj: np.ndarray):
        """Segment -> featurize -> scale -> predict cluster IDs (variable length, capped).

        Args:
            traj: np.ndarray (T, 7)

        Returns:
            List[int] of length min(num_primitives, max_primitives)
        """
        prims = self.segmenter.segment(traj)
        feats = self.featurizer.extract(traj, prims)
        feats_scaled = self.scaler.transform(feats)
        ids = self.kmeans.predict(feats_scaled).tolist()
        return ids[:self.max_primitives]

    def tokenize(self, traj: np.ndarray):
        """Returns fixed-length sequence padded to max_primitives with pad_id.

        Use this for cluster analysis and ablation C (continuous baseline).

        Args:
            traj: np.ndarray (T, 7)

        Returns:
            List[int] of length max_primitives
        """
        ids = self._tokenize_raw(traj)
        n = len(ids)
        if n >= self.max_primitives:
            return ids[:self.max_primitives]
        return ids + [self.pad_id] * (self.max_primitives - n)

    def get_primitive_features(self, traj: np.ndarray):
        """Return raw (unscaled) 14-D feature matrix for a trajectory.

        Used for Ablation C: continuous primitive features instead of discrete tokens.

        Args:
            traj: np.ndarray (T, 7)

        Returns:
            np.ndarray (num_primitives, 14) - unscaled
            np.ndarray (num_primitives, 14) - scaled (ready for downstream linear layer)
        """
        prims = self.segmenter.segment(traj)
        feats = self.featurizer.extract(traj, prims)
        feats_scaled = self.scaler.transform(feats)
        return feats, feats_scaled

    def __call__(self, actions_np: np.ndarray):
        """Dataset-compatible interface.

        Args:
            actions_np: np.ndarray (B, T, 7) or (T, 7)

        Returns:
            List[List[int]]: variable-length token IDs per sample (no padding)
                                The dataset pads to max_seq_len and tracks seq_lengths.
        """
        if actions_np.ndim == 2:
            actions_np = actions_np[np.newaxis]    # (1, T, 7)
        return [self._tokenize_raw(actions_np[b]) for b in range(actions_np.shape[0])]

    # Persistence
    def save(self, path: str):
        """Serialize to disk with joblib."""
        import joblib
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self, path)
        print(f"[GAMPT] Saved tokenizer (k={self.k}) → {path}")

    @classmethod
    def load(cls, path: str):
        """Deserialize from disk."""
        import joblib
        tok = joblib.load(path)
        print(f"[GAMPT] Loaded tokenizer (k={tok.k}, vocab_size={tok.vocab_size}) from {path}")
        return tok

    # Diagnostics
    def describe_segmentation(self, traj: np.ndarray):
        """Return primitive boundaries and feature table for one trajectory (for sanity checks)."""
        prims = self.segmenter.segment(traj)
        feats = self.featurizer.extract(traj, prims)
        feats_scaled = self.scaler.transform(feats)
        ids = self.kmeans.predict(feats_scaled)

        rows = []
        for i, ((s, e), fv, tok_id) in enumerate(zip(prims, feats, ids)):
            rows.append({
                "prim": i,
                "start": s,
                "end": e,
                "length": e - s + 1,
                "token": int(tok_id),
                "disp_z": round(float(fv[2]), 4),
                "disp_mag": round(float(fv[3]), 4),
                "mean_speed": round(float(fv[4]), 4),
                "gripper_change": int(fv[13]),
            })
        return rows
