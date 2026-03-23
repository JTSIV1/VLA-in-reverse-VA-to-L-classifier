"""
Unique variance explained: linear probe on frozen representations.

Extracts:
  h_AO   (128-d): CLS token from frozen AO transformer (r8_ao_native_best.pth)
  h_SC   (128-d): CLS token from frozen scene_obs allT transformer (scene_obs_allT_best.pth)

Trains logistic regression on train split, evaluates on val split:
  - AO only (h_AO)
  - Scene only (h_SC)
  - Concat (h_AO, h_SC)

Reports accuracy, macro-F1, and val NLL for each probe.
"""
import os, json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))

def load_model(ckpt_path):
    import sys; sys.path.insert(0, ROOT)
    from verb_probe.models import ActionToVerbTransformer
    ck = torch.load(ckpt_path, map_location="cpu")
    model = ActionToVerbTransformer(
        num_verbs=ck["num_verbs"],
        d_model=ck["d_model"],
        nhead=ck["nhead"],
        num_layers=ck["num_layers"],
        action_dim=ck["action_dim"],
        action_vocab_size=ck.get("action_vocab_size", 0),
        action_rep=ck["action_rep"],
        modality=ck["modality"],
        cross_layers=ck.get("cross_layers", 4),
        scene_dim=ck.get("scene_dim", 0),
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck

def extract_repr(model, loader, device):
    """Extract CLS token representations for all samples in loader."""
    reprs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            frames, traj, scene_vec, label, seq_len = batch
            frames    = frames.to(device)
            traj      = traj.to(device)
            scene_vec = scene_vec.to(device)
            seq_len   = seq_len.to(device)
            x, _, _ = model._forward_core(
                frames, traj, seq_lengths=seq_len, scene_vec=scene_vec)
            h = x[:, 0, :]  # CLS token (B, d_model)
            reprs.append(h.cpu().numpy())
            labels.append(label.numpy())
    return np.concatenate(reprs, axis=0), np.concatenate(labels, axis=0)

def probe(X_tr, y_tr, X_val, y_val, label=""):
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                             multi_class="multinomial", solver="lbfgs")
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds) * 100
    mf1 = f1_score(y_val, preds, average="macro") * 100
    # NLL: use decision_function -> softmax
    logits = clf.decision_function(X_val)
    probs = F.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
    nll = -np.mean(np.log(probs[np.arange(len(y_val)), y_val] + 1e-12))
    print(f"  {label:<25s}  acc={acc:.1f}%  macro-F1={mf1:.1f}%  NLL={nll:.4f}")
    # per-class recall
    per_class_recall = {}
    for cls_id in sorted(set(y_val)):
        mask = (y_val == cls_id)
        per_class_recall[int(cls_id)] = float((preds[mask] == cls_id).sum() / mask.sum() * 100)
    return acc, mf1, nll, preds, per_class_recall

def main():
    from config import TRAIN_DIR, VAL_DIR, MAX_SEQ_LEN
    from datasets.calvin_dataset import CalvinVerbProbeDataset as CalvinVerbDataset
    from utils import load_calvin_to_dataframe

    ao_ckpt   = os.path.join(ROOT, "checkpoints", "r8_ao_native_best.pth")
    sc_ckpt   = os.path.join(ROOT, "checkpoints", "scene_obs_allT_best.pth")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading AO model...")
    ao_model, ao_ck = load_model(ao_ckpt)
    ao_model.to(device)

    print("Loading Scene obs allT model...")
    sc_model, sc_ck = load_model(sc_ckpt)
    sc_model.to(device)

    # Use AO verb_to_id; collapse "lift up" -> "lift" for 20-class evaluation
    verb_to_id = dict(ao_ck["verb_to_id"])
    liftup_id = verb_to_id.pop("lift up", None)
    lift_id   = verb_to_id["lift"]
    # Remap remaining ids to contiguous range
    sorted_verbs = sorted(verb_to_id, key=lambda v: verb_to_id[v])
    verb_to_id   = {v: i for i, v in enumerate(sorted_verbs)}
    old_ao_id_to_new = {}
    for v, old_id in ao_ck["verb_to_id"].items():
        if v == "lift up":
            old_ao_id_to_new[old_id] = verb_to_id["lift"]
        else:
            old_ao_id_to_new[old_id] = verb_to_id[v]

    def make_loader(data_dir, ck, modality_override=None):
        df = load_calvin_to_dataframe(data_dir)
        df = df[df["primary_verb"].isin(verb_to_id)].reset_index(drop=True)
        max_seq = ck.get("max_action_len", MAX_SEQ_LEN)
        modality = modality_override or ck["modality"]
        # scene_obs allT: use num_frames=0 to load full sequence
        num_frames = 0 if modality == "scene_obs" else 2
        scene_rep = modality in ("scene_mlp", "scene_token", "scene_concat", "scene_film")
        ds = CalvinVerbDataset(df, data_dir,
                               max_seq_len=max_seq,
                               modality=modality,
                               num_frames=num_frames,
                               scene_rep=scene_rep)
        ds.verb_to_id = verb_to_id
        ds.id_to_verb = {i: v for v, i in verb_to_id.items()}
        return DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

    print("\nExtracting AO representations (train)...")
    ao_tr_loader  = make_loader(TRAIN_DIR, ao_ck)
    h_ao_tr, y_tr = extract_repr(ao_model, ao_tr_loader, device)

    print("Extracting AO representations (val)...")
    ao_val_loader  = make_loader(VAL_DIR, ao_ck)
    h_ao_val, y_val = extract_repr(ao_model, ao_val_loader, device)

    print("Extracting Scene obs allT representations (train)...")
    sc_tr_loader  = make_loader(TRAIN_DIR, sc_ck)
    h_sc_tr, y_tr2 = extract_repr(sc_model, sc_tr_loader, device)

    print("Extracting Scene obs allT representations (val)...")
    sc_val_loader  = make_loader(VAL_DIR, sc_ck)
    h_sc_val, y_val2 = extract_repr(sc_model, sc_val_loader, device)

    assert np.array_equal(y_tr, y_tr2),   "Train label mismatch between AO and Scene loaders"
    assert np.array_equal(y_val, y_val2), "Val label mismatch between AO and Scene loaders"

    print("\n── Linear probe results ──────────────────────────────────────────")
    ao_acc, ao_mf1, ao_nll, ao_preds, ao_pcr = probe(
        h_ao_tr, y_tr, h_ao_val, y_val, label="AO only (128-d)")
    sc_acc, sc_mf1, sc_nll, sc_preds, sc_pcr = probe(
        h_sc_tr, y_tr, h_sc_val, y_val, label="Scene allT only (128-d)")
    cat_acc, cat_mf1, cat_nll, cat_preds, cat_pcr = probe(
        np.concatenate([h_ao_tr, h_sc_tr], axis=1), y_tr,
        np.concatenate([h_ao_val, h_sc_val], axis=1), y_val,
        label="Concat AO+Scene (256-d)")
    print()

    # Per-class recall table
    id_to_verb = {i: v for v, i in verb_to_id.items()}
    all_cls = sorted(ao_pcr.keys())
    print("\n── Per-class recall (linear probe) ──────────────────────────────")
    print(f"  {'Verb':<12s}  {'Supp':>5s}  {'AO':>6s}  {'Scene':>6s}  {'Concat':>6s}")
    print("  " + "-" * 42)
    for cls_id in all_cls:
        verb = id_to_verb[cls_id]
        supp = int((y_val == cls_id).sum())
        print(f"  {verb:<12s}  {supp:>5d}  {ao_pcr.get(cls_id, 0.0):>6.1f}  "
              f"{sc_pcr.get(cls_id, 0.0):>6.1f}  {cat_pcr.get(cls_id, 0.0):>6.1f}")
    print(f"  {'Accuracy':<12s}  {'---':>5s}  {ao_acc:>6.1f}  {sc_acc:>6.1f}  {cat_acc:>6.1f}")
    print(f"  {'Macro F1':<12s}  {'---':>5s}  {ao_mf1:>6.1f}  {sc_mf1:>6.1f}  {cat_mf1:>6.1f}")

    # Save for further analysis
    out = os.path.join(ROOT, "results", "repr_complementarity.json")
    with open(out, "w") as f:
        json.dump({
            "ao":     {"acc": float(ao_acc),  "mf1": float(ao_mf1),  "nll": float(ao_nll),
                       "per_class_recall": {id_to_verb[k]: v for k, v in ao_pcr.items()}},
            "scene":  {"acc": float(sc_acc),  "mf1": float(sc_mf1),  "nll": float(sc_nll),
                       "per_class_recall": {id_to_verb[k]: v for k, v in sc_pcr.items()}},
            "concat": {"acc": float(cat_acc), "mf1": float(cat_mf1), "nll": float(cat_nll),
                       "per_class_recall": {id_to_verb[k]: v for k, v in cat_pcr.items()}},
        }, f, indent=2)
    print(f"\nSaved {out}")

if __name__ == "__main__":
    main()
