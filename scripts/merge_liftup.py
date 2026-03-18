"""
Post-hoc collapse "lift up" → "lift" in R8 pred files and recompute metrics.
Overwrites the *_best_metrics.json for each model.
"""
import json, os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
MODELS  = ["r8_ao_native", "r8_scene_mlp", "r8_token_cl1", "r8_token_cl2", "r8_token_cl4"]

for name in MODELS:
    preds_path   = os.path.join(RESULTS, f"{name}_preds.json")
    metrics_path = os.path.join(RESULTS, f"{name}_best_metrics.json")

    if not os.path.exists(preds_path):
        print(f"SKIP {name}: preds not found")
        continue

    d = json.load(open(preds_path))
    id2verb = d["id_to_verb"]          # {str(id): verb}
    preds   = np.array(d["preds"])
    labels  = np.array(d["labels"])

    # Build remapped id2verb: merge "lift up" → "lift"
    # Find the int ids for "lift up" and "lift"
    verb2id = {v: int(k) for k, v in id2verb.items()}
    if "lift up" not in verb2id:
        print(f"{name}: no 'lift up' class, skipping merge")
        continue

    liftup_id = verb2id["lift up"]
    lift_id   = verb2id["lift"]

    # Remap both preds and labels
    preds_r  = np.where(preds  == liftup_id, lift_id, preds)
    labels_r = np.where(labels == liftup_id, lift_id, labels)

    # New id set (remove liftup_id, keep rest)
    kept_ids   = sorted(set(int(k) for k in id2verb) - {liftup_id})
    new_id2verb = {str(i): id2verb[str(kid)] for i, kid in enumerate(kept_ids)}
    # remap to contiguous ids
    old2new = {kid: i for i, kid in enumerate(kept_ids)}
    preds_f  = np.array([old2new[p] for p in preds_r])
    labels_f = np.array([old2new[l] for l in labels_r])

    acc   = accuracy_score(labels_f, preds_f) * 100
    mf1   = f1_score(labels_f, preds_f, average="macro", zero_division=0) * 100
    report = classification_report(labels_f, preds_f,
                                   target_names=[new_id2verb[str(i)] for i in range(len(new_id2verb))],
                                   output_dict=True, zero_division=0)

    metrics = {
        "modality": d.get("modality", name),
        "action_rep": d.get("action_rep", "native"),
        "accuracy": acc,
        "num_examples": len(labels_f),
        "per_class": report,
    }
    json.dump(metrics, open(metrics_path, "w"), indent=2)
    print(f"{name}: acc={acc:.1f}%  macroF1={mf1:.1f}%  n={len(labels_f)}  classes={len(kept_ids)}")

