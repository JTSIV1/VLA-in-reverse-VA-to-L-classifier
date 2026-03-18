"""
Complementarity analysis: AO transformer vs scene_obs sklearn MLP.

Uses existing preds_ao.json (20 classes, from earlier AO inference).
Trains sklearn MLP on scene_obs features, saves preds_scene.json.
Produces per-episode CSV and instance-level error analysis.

Outputs:
  results/preds_scene.json           — scene_obs MLP val predictions
  results/scene_obs_mlp_metrics.json — per-class metrics
  results/complementarity.json       — 2x2 error analysis + summary
  results/episode_complementarity.csv — per-episode breakdown
"""
import os, json, sys, csv
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import TRAIN_DIR, VAL_DIR, EPISODE_TEMPLATE, SCENE_OBS_KEY
from utils import load_calvin_to_dataframe

SKIP_KEYS = {"accuracy", "macro avg", "weighted avg"}

# ═══════════════════════════════════════════════════════════════════════════
# 1. Load existing AO predictions (20 classes)
# ═══════════════════════════════════════════════════════════════════════════
ao_data = json.load(open(os.path.join(ROOT, "results", "preds_ao.json")))
id_to_verb = ao_data["id_to_verb"]
verb_to_id = {v: int(k) for k, v in id_to_verb.items()}
num_classes = len(verb_to_id)
ao_labels = np.array(ao_data["labels"])
ao_preds = np.array(ao_data["preds"])
print("Loaded preds_ao.json: {} samples, {} classes".format(len(ao_labels), num_classes))

# ═══════════════════════════════════════════════════════════════════════════
# 2. Load data + scene_obs features
# ═══════════════════════════════════════════════════════════════════════════
train_df = load_calvin_to_dataframe(TRAIN_DIR)
val_df = load_calvin_to_dataframe(VAL_DIR)
keep = set(verb_to_id.keys())
train_df = train_df[train_df['primary_verb'].isin(keep)].reset_index(drop=True)
val_df = val_df[val_df['primary_verb'].isin(keep)].reset_index(drop=True)
print("Train: {}, Val: {}".format(len(train_df), len(val_df)))
assert len(val_df) == len(ao_labels), "Val size mismatch!"


def load_scene_obs(df, data_dir):
    feats_eng, labels = [], []
    for _, row in df.iterrows():
        first_ep = np.load("{}/{}".format(data_dir, EPISODE_TEMPLATE.format(row['start_idx'])),
                           mmap_mode='r')
        last_ep = np.load("{}/{}".format(data_dir, EPISODE_TEMPLATE.format(row['end_idx'])),
                          mmap_mode='r')
        s0 = np.array(first_ep[SCENE_OBS_KEY], dtype=np.float32)
        s1 = np.array(last_ep[SCENE_OBS_KEY], dtype=np.float32)
        delta = s1 - s0
        # scene_engineered (96-d): delta + |delta| + sign(delta) + binary(|delta|>0.01)
        feats_eng.append(np.concatenate([
            delta, np.abs(delta), np.sign(delta),
            (np.abs(delta) > 0.01).astype(np.float32)
        ]))
        labels.append(verb_to_id[row['primary_verb']])
    return np.array(feats_eng), np.array(labels)


print("Loading train scene_obs...")
X_tr, y_tr = load_scene_obs(train_df, TRAIN_DIR)
print("Loading val scene_obs...")
X_va, y_va = load_scene_obs(val_df, VAL_DIR)
print("  scene_engineered: {}-d".format(X_tr.shape[1]))
assert np.array_equal(y_va, ao_labels), "Label order mismatch with preds_ao.json!"

# ═══════════════════════════════════════════════════════════════════════════
# 3. Train scene_obs MLP
# ═══════════════════════════════════════════════════════════════════════════
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_va_s = scaler.transform(X_va)

print("\nTraining scene_obs MLP (scene_engineered, 96-d)...")
sc_mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=1000,
                       random_state=42, early_stopping=True,
                       validation_fraction=0.1)
sc_mlp.fit(X_tr_s, y_tr)
sc_preds = sc_mlp.predict(X_va_s)
sc_acc = accuracy_score(y_va, sc_preds) * 100
sc_mf1 = f1_score(y_va, sc_preds, average='macro') * 100
print("Scene MLP: Acc={:.1f}%, MacF1={:.1f}%".format(sc_acc, sc_mf1))

# Save preds_scene.json
with open(os.path.join(ROOT, "results", "preds_scene.json"), "w") as f:
    json.dump({"labels": y_va.tolist(), "preds": sc_preds.tolist(),
               "id_to_verb": id_to_verb}, f)

# Save per-class metrics
target_names = [id_to_verb[str(i)] for i in range(num_classes)]
sc_report = classification_report(y_va, sc_preds, target_names=target_names,
                                  output_dict=True, zero_division=0)
sc_per_class = {k: v for k, v in sc_report.items() if k not in SKIP_KEYS}
with open(os.path.join(ROOT, "results", "scene_obs_mlp_metrics.json"), "w") as f:
    json.dump({"method": "MLP_scene_engineered", "accuracy": sc_acc,
               "macro_f1": sc_mf1, "per_class": sc_per_class}, f, indent=2)

ao_acc = accuracy_score(ao_labels, ao_preds) * 100
ao_mf1 = f1_score(ao_labels, ao_preds, average='macro') * 100
print("AO transformer: Acc={:.1f}%, MacF1={:.1f}%".format(ao_acc, ao_mf1))

# ═══════════════════════════════════════════════════════════════════════════
# 4. INSTANCE-LEVEL ERROR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
ao_correct = (ao_preds == ao_labels)
sc_correct = (sc_preds == y_va)

both    = ao_correct & sc_correct
ao_only = ao_correct & ~sc_correct
sc_only = ~ao_correct & sc_correct
neither = ~ao_correct & ~sc_correct

n = len(ao_labels)
print("\n" + "=" * 60)
print("INSTANCE-LEVEL ERROR ANALYSIS (n={})".format(n))
print("=" * 60)
print("  Both correct : {:4d}  ({:.1f}%)".format(int(both.sum()),    100*both.sum()/n))
print("  AO only      : {:4d}  ({:.1f}%)".format(int(ao_only.sum()), 100*ao_only.sum()/n))
print("  Scene only   : {:4d}  ({:.1f}%)".format(int(sc_only.sum()), 100*sc_only.sum()/n))
print("  Neither      : {:4d}  ({:.1f}%)".format(int(neither.sum()), 100*neither.sum()/n))
oracle = int((both | ao_only | sc_only).sum())
print("  Oracle union : {:4d}  ({:.1f}%)".format(oracle, 100*oracle/n))

# Per-class breakdown
ao_report = classification_report(ao_labels, ao_preds, target_names=target_names,
                                  output_dict=True, zero_division=0)
print("\n-- Per-class recall --")
print("{:<14s} {:>5s} {:>7s} {:>7s} {:>6s} {:>6s} {:>6s} {:>6s}".format(
    "Verb", "Supp", "AO", "Scene", "Both", "AO+", "SC+", "None"))
print("-" * 72)
for cls_id in range(num_classes):
    v = id_to_verb[str(cls_id)]
    mask = (ao_labels == cls_id)
    supp = int(mask.sum())
    ao_r = ao_report.get(v, {}).get("recall", 0) * 100
    sc_r = sc_per_class.get(v, {}).get("recall", 0) * 100
    b = int((both & mask).sum())
    a = int((ao_only & mask).sum())
    s = int((sc_only & mask).sum())
    ne = int((neither & mask).sum())
    print("{:<14s} {:>5d} {:>6.1f}% {:>6.1f}% {:>6d} {:>6d} {:>6d} {:>6d}".format(
        v, supp, ao_r, sc_r, b, a, s, ne))

# ═══════════════════════════════════════════════════════════════════════════
# 5. PER-EPISODE CSV
# ═══════════════════════════════════════════════════════════════════════════
csv_path = os.path.join(ROOT, "results", "episode_complementarity.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["idx", "start_ep", "end_ep", "instruction", "true_verb",
                     "ao_pred", "scene_pred", "ao_correct", "scene_correct", "category"])
    for i in range(n):
        row = val_df.iloc[i]
        true_v = id_to_verb[str(ao_labels[i])]
        ao_v   = id_to_verb[str(ao_preds[i])]
        sc_v   = id_to_verb[str(sc_preds[i])]
        ao_ok  = bool(ao_correct[i])
        sc_ok  = bool(sc_correct[i])
        if ao_ok and sc_ok:
            cat = "both_correct"
        elif ao_ok and not sc_ok:
            cat = "ao_only"
        elif not ao_ok and sc_ok:
            cat = "scene_only"
        else:
            cat = "neither"
        writer.writerow([i, row['start_idx'], row['end_idx'], row['instruction'],
                         true_v, ao_v, sc_v, ao_ok, sc_ok, cat])
print("\nSaved {}".format(csv_path))

# ═══════════════════════════════════════════════════════════════════════════
# 6. SAVE SUMMARY JSON
# ═══════════════════════════════════════════════════════════════════════════
comp_out = os.path.join(ROOT, "results", "complementarity.json")
with open(comp_out, "w") as f:
    json.dump({
        "ao": {"acc": float(ao_acc), "mf1": float(ao_mf1)},
        "scene_mlp": {"acc": float(sc_acc), "mf1": float(sc_mf1),
                      "method": "MLP_scene_engineered_96d"},
        "error_analysis": {
            "n": int(n),
            "both_correct": int(both.sum()),
            "ao_only": int(ao_only.sum()),
            "scene_only": int(sc_only.sum()),
            "neither": int(neither.sum()),
            "oracle_union": oracle,
            "both_correct_pct": round(100*both.sum()/n, 1),
            "ao_only_pct": round(100*ao_only.sum()/n, 1),
            "scene_only_pct": round(100*sc_only.sum()/n, 1),
            "neither_pct": round(100*neither.sum()/n, 1),
            "oracle_union_pct": round(100*oracle/n, 1),
        },
    }, f, indent=2)
print("Saved {}".format(comp_out))
print("\nDone.")
