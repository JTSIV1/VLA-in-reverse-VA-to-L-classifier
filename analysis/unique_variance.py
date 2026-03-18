"""
Unique variance analysis via linear probes on learned embeddings.

Extracts:
  h_AO   (128-d): CLS token from frozen AO transformer
  h_SC   (128-d): 2nd hidden layer activation from sklearn MLP on scene_engineered

Trains logistic regression probes on train split, evaluates on val:
  - AO only (h_AO, 128-d)
  - Scene only (h_SC, 128-d)
  - Concat (h_AO || h_SC, 256-d)

Unique variance decomposition:
  Unique AO    = Concat - Scene only
  Unique Scene = Concat - AO only
  Shared       = AO + Scene - Concat
"""
import os, json, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import TRAIN_DIR, VAL_DIR, EPISODE_TEMPLATE, SCENE_OBS_KEY, MAX_SEQ_LEN
from utils import load_calvin_to_dataframe
from train_transformer import ActionToVerbTransformer, CalvinVerbDataset

# ═══════════════════════════════════════════════════════════════════════════
# 1. SETUP
# ═══════════════════════════════════════════════════════════════════════════
ao_ckpt_path = os.path.join(ROOT, "checkpoints", "r8_ao_native_best.pth")
if not os.path.exists(ao_ckpt_path):
    ao_ckpt_path = os.path.join(ROOT, "checkpoints", "r7_ao_native_j6464939_best.pth")
print("Loading AO checkpoint: {}".format(os.path.basename(ao_ckpt_path)))
ck = torch.load(ao_ckpt_path, map_location="cpu")

# Build 20-class mapping (collapse lift up → lift)
ao_verb_to_id = dict(ck["verb_to_id"])
ao_verb_to_id.pop("lift up", None)
sorted_verbs = sorted(ao_verb_to_id.keys())
verb_to_id = {v: i for i, v in enumerate(sorted_verbs)}
id_to_verb = {str(i): v for v, i in verb_to_id.items()}
num_classes = len(verb_to_id)

old_to_new = {}
for v, old_id in ck["verb_to_id"].items():
    collapsed = "lift" if v == "lift up" else v
    if collapsed in verb_to_id:
        old_to_new[old_id] = verb_to_id[collapsed]

# Load data
train_df = load_calvin_to_dataframe(TRAIN_DIR)
val_df = load_calvin_to_dataframe(VAL_DIR)
keep = set(verb_to_id.keys())
train_df = train_df[train_df['primary_verb'].isin(keep)].reset_index(drop=True)
val_df = val_df[val_df['primary_verb'].isin(keep)].reset_index(drop=True)
print("Train: {}, Val: {}, Classes: {}".format(len(train_df), len(val_df), num_classes))

# ═══════════════════════════════════════════════════════════════════════════
# 2. EXTRACT AO CLS TOKENS (128-d)
# ═══════════════════════════════════════════════════════════════════════════
ao_model = ActionToVerbTransformer(
    num_verbs=ck["num_verbs"], d_model=ck["d_model"], nhead=ck["nhead"],
    num_layers=ck["num_layers"], action_dim=ck["action_dim"],
    action_vocab_size=ck.get("action_vocab_size", 0), action_rep=ck["action_rep"],
    modality=ck["modality"], cross_layers=ck.get("cross_layers", 4),
    scene_dim=ck.get("scene_dim", 0),
)
ao_model.load_state_dict(ck["state_dict"])
ao_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ao_model.to(device)

max_seq = ck.get("max_action_len", MAX_SEQ_LEN)


def make_loader(df, data_dir, shuffle=False):
    ds = CalvinVerbDataset(df, data_dir, max_seq_len=max_seq,
                           modality=ck["modality"], num_frames=2)
    ds.verb_to_id = verb_to_id
    ds.id_to_verb = {i: v for v, i in verb_to_id.items()}
    return DataLoader(ds, batch_size=64, shuffle=shuffle, num_workers=4)


def extract_ao_cls(loader):
    """Extract CLS token (128-d) from frozen AO transformer."""
    cls_list, label_list = [], []
    with torch.no_grad():
        for batch in loader:
            frames, traj, scene_vec, label, seq_len = batch
            # Use _forward_core to get transformer hidden states before classification head
            x, _, _ = ao_model._forward_core(
                frames.to(device), traj.to(device),
                seq_lengths=seq_len.to(device),
                scene_vec=scene_vec.to(device))
            h_cls = x[:, 0, :]  # CLS token: (B, d_model=128)
            cls_list.append(h_cls.cpu().numpy())
            label_list.append(label.numpy())
    return np.concatenate(cls_list), np.concatenate(label_list)


print("\nExtracting AO CLS tokens (train)...")
train_loader = make_loader(train_df, TRAIN_DIR)
h_ao_tr, y_tr = extract_ao_cls(train_loader)
print("  Shape: {}".format(h_ao_tr.shape))

print("Extracting AO CLS tokens (val)...")
val_loader = make_loader(val_df, VAL_DIR)
h_ao_va, y_va = extract_ao_cls(val_loader)
print("  Shape: {}".format(h_ao_va.shape))

# ═══════════════════════════════════════════════════════════════════════════
# 3. TRAIN SCENE MLP + EXTRACT 128-d HIDDEN ACTIVATIONS
# ═══════════════════════════════════════════════════════════════════════════
def load_scene_obs(df, data_dir):
    feats, labels = [], []
    for _, row in df.iterrows():
        first_ep = np.load("{}/{}".format(data_dir, EPISODE_TEMPLATE.format(row['start_idx'])),
                           mmap_mode='r')
        last_ep = np.load("{}/{}".format(data_dir, EPISODE_TEMPLATE.format(row['end_idx'])),
                          mmap_mode='r')
        s0 = np.array(first_ep[SCENE_OBS_KEY], dtype=np.float32)
        s1 = np.array(last_ep[SCENE_OBS_KEY], dtype=np.float32)
        delta = s1 - s0
        feats.append(np.concatenate([
            delta, np.abs(delta), np.sign(delta),
            (np.abs(delta) > 0.01).astype(np.float32)
        ]))
        labels.append(verb_to_id[row['primary_verb']])
    return np.array(feats), np.array(labels)


print("\nLoading scene_obs features (train)...")
X_sc_tr, y_tr_sc = load_scene_obs(train_df, TRAIN_DIR)
print("Loading scene_obs features (val)...")
X_sc_va, y_va_sc = load_scene_obs(val_df, VAL_DIR)
assert np.array_equal(y_tr, y_tr_sc) and np.array_equal(y_va, y_va_sc), "Label mismatch!"

scaler = StandardScaler()
X_sc_tr_s = scaler.fit_transform(X_sc_tr)
X_sc_va_s = scaler.transform(X_sc_va)

print("Training scene MLP (256/128)...")
sc_mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=1000,
                       random_state=42, early_stopping=True,
                       validation_fraction=0.1)
sc_mlp.fit(X_sc_tr_s, y_tr)
sc_acc = accuracy_score(y_va, sc_mlp.predict(X_sc_va_s)) * 100
print("  Scene MLP val acc: {:.1f}%".format(sc_acc))

# Save scene MLP checkpoint
import joblib
sc_mlp_path = os.path.join(ROOT, "checkpoints", "scene_mlp_256_128.joblib")
joblib.dump({"mlp": sc_mlp, "scaler": scaler, "verb_to_id": verb_to_id}, sc_mlp_path)
print("  Saved scene MLP → {}".format(sc_mlp_path))


def extract_mlp_hidden(mlp, X):
    """Extract 2nd hidden layer activation (128-d) from trained sklearn MLP."""
    # Layer 1: ReLU(X @ W1 + b1) → 256-d
    h1 = np.maximum(0, X @ mlp.coefs_[0] + mlp.intercepts_[0])
    # Layer 2: ReLU(h1 @ W2 + b2) → 128-d
    h2 = np.maximum(0, h1 @ mlp.coefs_[1] + mlp.intercepts_[1])
    return h2


h_sc_tr = extract_mlp_hidden(sc_mlp, X_sc_tr_s)
h_sc_va = extract_mlp_hidden(sc_mlp, X_sc_va_s)
print("  Scene hidden activations: {}".format(h_sc_tr.shape))

# Save extracted features
feat_path = os.path.join(ROOT, "results", "probe_features.npz")
np.savez(feat_path,
         h_ao_tr=h_ao_tr, h_ao_va=h_ao_va,
         h_sc_tr=h_sc_tr, h_sc_va=h_sc_va,
         y_tr=y_tr, y_va=y_va)
print("Saved probe features → {}".format(feat_path))

# ═══════════════════════════════════════════════════════════════════════════
# 4. LINEAR PROBES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LINEAR PROBES ON LEARNED EMBEDDINGS")
print("=" * 70)


def probe(X_tr, X_va, y_tr, y_va, label=""):
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced',
                             multi_class='multinomial', solver='lbfgs')
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_va)
    acc = accuracy_score(y_va, preds) * 100
    mf1 = f1_score(y_va, preds, average='macro') * 100
    # NLL
    logits = clf.decision_function(X_va)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    nll = -np.mean(np.log(probs[np.arange(len(y_va)), y_va] + 1e-12))
    print("  {:<35s}  acc={:.1f}%  MacF1={:.1f}%  NLL={:.4f}".format(label, acc, mf1, nll))
    return {"acc": acc, "mf1": mf1, "nll": nll, "preds": preds}


ao_res = probe(h_ao_tr, h_ao_va, y_tr, y_va, "AO CLS (128-d)")
sc_res = probe(h_sc_tr, h_sc_va, y_tr, y_va, "Scene hidden (128-d)")

h_cat_tr = np.concatenate([h_ao_tr, h_sc_tr], axis=1)
h_cat_va = np.concatenate([h_ao_va, h_sc_va], axis=1)
cat_res = probe(h_cat_tr, h_cat_va, y_tr, y_va, "Concat AO+Scene (256-d)")

# ═══════════════════════════════════════════════════════════════════════════
# 5. UNIQUE VARIANCE DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("UNIQUE VARIANCE DECOMPOSITION")
print("=" * 70)

unique_ao_acc = cat_res["acc"] - sc_res["acc"]
unique_sc_acc = cat_res["acc"] - ao_res["acc"]
shared_acc = ao_res["acc"] + sc_res["acc"] - cat_res["acc"]

unique_ao_mf1 = cat_res["mf1"] - sc_res["mf1"]
unique_sc_mf1 = cat_res["mf1"] - ao_res["mf1"]
shared_mf1 = ao_res["mf1"] + sc_res["mf1"] - cat_res["mf1"]

print("\n  Accuracy decomposition:")
print("    AO probe:     {:.1f}%".format(ao_res["acc"]))
print("    Scene probe:  {:.1f}%".format(sc_res["acc"]))
print("    Concat probe: {:.1f}%".format(cat_res["acc"]))
print("    ─────────────────────────────")
print("    Unique AO:    {:+.1f}pp  (Concat − Scene)".format(unique_ao_acc))
print("    Unique Scene: {:+.1f}pp  (Concat − AO)".format(unique_sc_acc))
print("    Shared:       {:+.1f}pp  (AO + Scene − Concat)".format(shared_acc))

print("\n  Macro-F1 decomposition:")
print("    AO probe:     {:.1f}%".format(ao_res["mf1"]))
print("    Scene probe:  {:.1f}%".format(sc_res["mf1"]))
print("    Concat probe: {:.1f}%".format(cat_res["mf1"]))
print("    ─────────────────────────────")
print("    Unique AO:    {:+.1f}pp  (Concat − Scene)".format(unique_ao_mf1))
print("    Unique Scene: {:+.1f}pp  (Concat − AO)".format(unique_sc_mf1))
print("    Shared:       {:+.1f}pp  (AO + Scene − Concat)".format(shared_mf1))

print("\n  NLL (lower is better):")
print("    AO:     {:.4f}".format(ao_res["nll"]))
print("    Scene:  {:.4f}".format(sc_res["nll"]))
print("    Concat: {:.4f}".format(cat_res["nll"]))
nll_improve = ao_res["nll"] - cat_res["nll"]
print("    NLL reduction from adding Scene to AO: {:.4f} ({:.1f}%)".format(
    nll_improve, 100 * nll_improve / ao_res["nll"]))

# ═══════════════════════════════════════════════════════════════════════════
# 6. PER-CLASS PROBE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PER-CLASS RECALL (linear probe)")
print("=" * 70)
print("{:<14s} {:>5s} {:>7s} {:>7s} {:>7s}".format(
    "Verb", "Supp", "AO", "Scene", "Concat"))
print("-" * 45)
for cls_id in range(num_classes):
    v = id_to_verb[str(cls_id)]
    mask = (y_va == cls_id)
    supp = int(mask.sum())
    ao_r = 100.0 * (ao_res["preds"][mask] == cls_id).sum() / max(1, supp)
    sc_r = 100.0 * (sc_res["preds"][mask] == cls_id).sum() / max(1, supp)
    cat_r = 100.0 * (cat_res["preds"][mask] == cls_id).sum() / max(1, supp)
    print("{:<14s} {:>5d} {:>6.1f}% {:>6.1f}% {:>6.1f}%".format(
        v, supp, ao_r, sc_r, cat_r))

# ═══════════════════════════════════════════════════════════════════════════
# 6b. ADD PROBE CORRECTNESS TO episode_task_types.csv
# ═══════════════════════════════════════════════════════════════════════════
import pandas as pd

csv_path = os.path.join(ROOT, "data", "episode_task_types.csv")
ep_df = pd.read_csv(csv_path)

# Build lookup: (split, start_idx) → probe correctness
val_results = {}
for i in range(len(val_df)):
    key = int(val_df.iloc[i]["start_idx"])
    val_results[key] = {
        "ao_correct": int(ao_res["preds"][i] == y_va[i]),
        "scene_correct": int(sc_res["preds"][i] == y_va[i]),
        "concat_correct": int(cat_res["preds"][i] == y_va[i]),
    }

ep_df["ao_correct"] = ""
ep_df["scene_correct"] = ""
ep_df["concat_correct"] = ""
for idx, row in ep_df.iterrows():
    if row["split"] == "val" and row["start_idx"] in val_results:
        r = val_results[row["start_idx"]]
        ep_df.at[idx, "ao_correct"] = r["ao_correct"]
        ep_df.at[idx, "scene_correct"] = r["scene_correct"]
        ep_df.at[idx, "concat_correct"] = r["concat_correct"]

ep_df.to_csv(csv_path, index=False)
print("\nAdded ao_correct, scene_correct, concat_correct to {}".format(csv_path))
matched = sum(1 for k in val_results if k in ep_df[ep_df["split"] == "val"]["start_idx"].values)
print("  Matched {} / {} val episodes".format(matched, len(val_results)))

# ═══════════════════════════════════════════════════════════════════════════
# 7. SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════
out = os.path.join(ROOT, "results", "unique_variance.json")
with open(out, "w") as f:
    json.dump({
        "ao_probe":    {"acc": ao_res["acc"],  "mf1": ao_res["mf1"],  "nll": ao_res["nll"]},
        "scene_probe": {"acc": sc_res["acc"],  "mf1": sc_res["mf1"],  "nll": sc_res["nll"]},
        "concat_probe":{"acc": cat_res["acc"], "mf1": cat_res["mf1"], "nll": cat_res["nll"]},
        "decomposition": {
            "unique_ao_acc": unique_ao_acc, "unique_scene_acc": unique_sc_acc,
            "shared_acc": shared_acc,
            "unique_ao_mf1": unique_ao_mf1, "unique_scene_mf1": unique_sc_mf1,
            "shared_mf1": shared_mf1,
        },
    }, f, indent=2)
print("\nSaved {}".format(out))
