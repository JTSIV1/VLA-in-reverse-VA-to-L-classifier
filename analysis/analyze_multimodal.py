"""
analyze_multimodal.py — Five analyses in one script, two output figures.

Figure 1 (multimodal_analysis2.png) — model behavior:
  Panel 1: Prediction overlap (8 categories from AO × VO × MM correct/wrong)
  Panel 2: Per-class recall scatter (AO vs VO, colored by MM − best unimodal)
  Panel 3: CLS attention distribution in cross-modal layers (vision vs action vs self)

Figure 2 (multimodal_nll_decomp.png) — unique variance analysis:
  Panel 1: NLL decomposition Venn (unique AO, unique VO, shared, irreducible)
  Panel 2: Per-class NLL improvement (ΔI_action = NLL_VO−NLL_MM per class,
                                       ΔI_vision = NLL_AO−NLL_MM per class)
  Panel 3: Confusion matrix difference (AO_conf − VO_conf) on shared classes

Usage:
    python analyze_multimodal.py
"""

import os, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix

try:
    from scipy.stats import spearmanr
except ImportError:
    def spearmanr(a, b):
        from scipy.stats import rankdata
        n = len(a)
        ra, rb = rankdata(a), rankdata(b)
        d2 = np.sum((ra - rb) ** 2)
        rho = 1 - 6 * d2 / (n * (n**2 - 1))
        return type('R', (), {'correlation': rho, 'pvalue': None})()

from train_transformer import ActionToVerbTransformer, CalvinVerbDataset
from utils import load_calvin_to_dataframe
from config import (
    VAL_DIR, D_MODEL, NHEAD, NUM_LAYERS, ACTION_DIM, PATCH_SIZE,
    IMAGE_SIZE, IMG_MEAN, IMG_STD, BATCH_SIZE, MAX_SEQ_LEN, NUM_WORKERS,
    FAST_VOCAB_SIZE,
)

BASE   = os.path.dirname(os.path.abspath(__file__))
RDIR   = os.path.join(BASE, 'results')
FDIR   = os.path.join(BASE, 'figures')
CDIR   = os.path.join(BASE, 'checkpoints')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

CKPTS = {
    'AO': os.path.join(CDIR, 'ao_native_sparse_weighted_j6457852_best.pth'),
    'VO': os.path.join(CDIR, 'vision_vc1_delta16_sp_wt_j6459653_best.pth'),
    'MM': os.path.join(CDIR, 'full_vc1_d16_late2_sp_wt_j6459079_best.pth'),
}
SKIP = {'accuracy', 'macro avg', 'weighted avg'}

# ── helpers ───────────────────────────────────────────────────────────────────

def build(raw, val_df):
    """Build (model, dataset) from a checkpoint dict and val DataFrame."""
    sd              = raw['state_dict']
    num_verbs       = raw['num_verbs']
    verb_to_id      = raw['verb_to_id']
    id_to_verb      = raw['id_to_verb']
    d_model         = raw.get('d_model', D_MODEL)
    nhead           = raw.get('nhead', NHEAD)
    num_layers      = raw.get('num_layers', NUM_LAYERS)
    action_dim      = raw.get('action_dim', ACTION_DIM)
    patch_size      = raw.get('patch_size', PATCH_SIZE)
    img_size        = raw.get('img_size', IMAGE_SIZE[0])
    max_action_len  = raw.get('max_action_len', MAX_SEQ_LEN)
    modality        = raw.get('modality', 'action_only')
    action_rep      = raw.get('action_rep', 'native')
    fast_vocab_size = raw.get('fast_vocab_size', FAST_VOCAB_SIZE)
    cross_layers    = raw.get('cross_layers', num_layers)
    vision_encoder  = raw.get('vision_encoder', 'patch')
    freeze_vision   = raw.get('freeze_vision', True)
    num_frames      = raw.get('num_frames', 2)
    delta_patches   = raw.get('delta_patches', 0)

    eff = 224 if vision_encoder in ("r3m", "dinov2_s", "dinov2_b", "vc1") else img_size
    tf  = transforms.Compose([
        transforms.Resize((eff, eff)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])
    ds = CalvinVerbDataset(val_df, VAL_DIR, transform=tf,
                           max_seq_len=max_action_len, modality=modality,
                           vision_encoder=vision_encoder, img_size=eff,
                           num_frames=num_frames, delta_patches=delta_patches)
    ds.verb_to_id = verb_to_id
    ds.id_to_verb = id_to_verb
    mask = val_df['primary_verb'].isin(verb_to_id.keys())
    if (~mask).sum():
        ds.df = val_df[mask].reset_index(drop=True)

    m = ActionToVerbTransformer(
        num_verbs=num_verbs, d_model=d_model, nhead=nhead,
        num_layers=num_layers, action_dim=action_dim, img_size=eff,
        patch_size=patch_size, max_action_len=max_action_len,
        modality=modality, action_rep=action_rep, fast_vocab_size=fast_vocab_size,
        cross_layers=cross_layers, vision_encoder=vision_encoder,
        freeze_vision=freeze_vision, num_frames=num_frames, delta_patches=delta_patches)
    sd2 = {k.replace("transformer.layers.", "layers."): v for k, v in sd.items()}
    m.load_state_dict(sd2)
    m.to(device)
    m.eval()
    return m, ds


def run_inference(model, ds):
    """Returns (true_verbs, pred_verbs, log_probs_true) all as lists/arrays."""
    ldr  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS)
    lmap = ds.id_to_verb
    tv, pv, lp = [], [], []
    with torch.no_grad():
        for frames, acts, labels, seqlen in ldr:
            frames  = frames.to(device)
            acts    = acts.to(device)
            labels  = labels.to(device)
            seqlen  = seqlen.to(device)
            logits  = model(frames, acts, seq_lengths=seqlen)
            probs   = F.softmax(logits, dim=-1)           # (B, C)
            log_p   = F.log_softmax(logits, dim=-1)       # (B, C)
            preds   = torch.argmax(logits, 1)
            # log prob of true label
            lp_true = log_p[torch.arange(len(labels)), labels]
            tv.extend(lmap[l.item()] for l in labels)
            pv.extend(lmap[p.item()] for p in preds)
            lp.extend(lp_true.cpu().tolist())
    return tv, pv, np.array(lp)


def manual_cross_layer(layer, x, pmask):
    """TransformerEncoderLayer forward with need_weights=True."""
    try:
        attn_out, attn_w = layer.self_attn(
            x, x, x, attn_mask=None, key_padding_mask=pmask,
            need_weights=True, average_attn_weights=True)
    except TypeError:
        attn_out, attn_w = layer.self_attn(
            x, x, x, attn_mask=None, key_padding_mask=pmask, need_weights=True)
    x = layer.norm1(x + layer.dropout1(attn_out))
    x = layer.norm2(x + layer.dropout2(
        layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))))
    return x, attn_w  # (B, L, L)


def run_mm_with_attn(model, ds):
    """MM inference + CLS attention capture in cross-modal layers."""
    ldr  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS)
    lmap = ds.id_to_verb
    n_self   = model.num_layers - model.cross_layers
    cross_li = list(range(n_self, model.num_layers))
    nv = (max(model.num_frames - 1, 1) * model.num_patches
          if model.delta_patches > 0 else model.num_frames * model.num_patches)
    v_s, v_e = 1, 1 + nv

    tv, pv, lp = [], [], []
    attn_rows = {li: [] for li in cross_li}

    with torch.no_grad():
        for frames, acts, labels, seqlen in ldr:
            frames = frames.to(device);  acts = acts.to(device)
            labels = labels.to(device);  seqlen = seqlen.to(device)
            B = acts.size(0)

            # Build full_seq
            cls   = model.cls_token.expand(B,-1,-1) + model.cls_pos + model.type_cls
            parts = [cls]
            nf_   = frames.size(1)
            if model.delta_patches > 0:
                ap = [model.vision_enc(frames[:,fi]) for fi in range(nf_)]
                K, d = model.delta_patches, ap[0].size(-1)
                for pi in range(nf_ - 1):
                    diff = ap[pi+1] - ap[pi]
                    idx  = diff.norm(dim=-1).topk(K, dim=-1).indices
                    ie   = idx.unsqueeze(-1).expand(-1,-1,d)
                    sel  = torch.gather(diff, 1, ie)
                    pos  = torch.gather(model.patch_pos.expand(B,-1,-1), 1, ie)
                    parts.append(sel + pos + model.frame_pos[:,pi] + model.type_img)
            ae = (model.action_proj(acts)
                  + model.action_pos[:,:acts.size(1),:] + model.type_action)
            parts.append(ae)
            full_seq  = torch.cat(parts, 1)
            total_len = full_seq.size(1)
            pos_idx   = torch.arange(total_len, device=device).unsqueeze(0)
            pmask     = pos_idx >= seqlen.unsqueeze(1)

            self_mask = None
            if n_self > 0:
                self_mask = torch.full((total_len, total_len), float('-inf'),
                                       device=device)
                self_mask[0, 0] = 0.0
                self_mask[v_s:v_e, v_s:v_e] = 0.0
                self_mask[v_e:, v_e:] = 0.0

            x = full_seq
            for i, layer in enumerate(model.layers):
                if i < n_self:
                    x = layer(x, src_mask=self_mask, src_key_padding_mask=pmask)
                else:
                    x, aw = manual_cross_layer(layer, x, pmask)
                    attn_rows[i].append(aw[:, 0, :].cpu())  # CLS row

            logits  = model.classifier(x[:, 0, :])
            log_p   = F.log_softmax(logits, -1)
            preds   = torch.argmax(logits, 1)
            lp_true = log_p[torch.arange(len(labels)), labels]
            tv.extend(lmap[l.item()] for l in labels)
            pv.extend(lmap[p.item()] for p in preds)
            lp.extend(lp_true.cpu().tolist())

    for li in cross_li:
        attn_rows[li] = torch.cat(attn_rows[li], 0).numpy()  # (N, total_len)
    return tv, pv, np.array(lp), attn_rows, v_s, v_e, cross_li


def per_class_recall_from_json(tag):
    p = os.path.join(RDIR, f'{tag}_best_metrics.json')
    with open(p) as f:
        d = json.load(f)
    return {k: v['recall'] for k, v in d.get('per_class', {}).items()
            if k not in SKIP and isinstance(v, dict)}


# ── run inference ─────────────────────────────────────────────────────────────
print("Loading val data...")
val_df = load_calvin_to_dataframe(VAL_DIR)

print("Running AO model...")
ao_raw                  = torch.load(CKPTS['AO'], map_location=device, weights_only=False)
ao_model, ao_ds         = build(ao_raw, val_df)
ao_true, ao_pred, ao_lp = run_inference(ao_model, ao_ds)
del ao_model
torch.cuda.empty_cache()

print("Running VO model...")
vo_raw                  = torch.load(CKPTS['VO'], map_location=device, weights_only=False)
vo_model, vo_ds         = build(vo_raw, val_df)
vo_true, vo_pred, vo_lp = run_inference(vo_model, vo_ds)
del vo_model
torch.cuda.empty_cache()

print("Running MM model (with attention capture)...")
mm_raw = torch.load(CKPTS['MM'], map_location=device, weights_only=False)
mm_model, mm_ds = build(mm_raw, val_df)
mm_true, mm_pred, mm_lp, attn_rows, v_s, v_e, cross_li = run_mm_with_attn(mm_model, mm_ds)
del mm_model
torch.cuda.empty_cache()

# Sanity check
assert ao_true == mm_true == vo_true, "Val sample ordering mismatch!"
N = len(ao_true)
print(f"Val samples: {N}")

# ── 1. Prediction overlap ─────────────────────────────────────────────────────
ao_ok = np.array([t == p for t, p in zip(ao_true, ao_pred)])
vo_ok = np.array([t == p for t, p in zip(vo_true, vo_pred)])
mm_ok = np.array([t == p for t, p in zip(mm_true, mm_pred)])

print(f"AO acc={ao_ok.mean()*100:.1f}%  VO acc={vo_ok.mean()*100:.1f}%"
      f"  MM acc={mm_ok.mean()*100:.1f}%")

OVERLAP_CATS = [
    ('All correct  (AO∩VO∩MM)',     ao_ok  &  vo_ok  &  mm_ok,  '#5cb85c'),
    ('AO+MM, ~VO',                   ao_ok  & ~vo_ok  &  mm_ok,  '#a8d5a2'),
    ('VO+MM, ~AO',                  ~ao_ok  &  vo_ok  &  mm_ok,  '#6ab4f5'),
    ('MM only  (new from fusion)',   ~ao_ok  & ~vo_ok  &  mm_ok,  '#2196F3'),
    ('AO+VO, ~MM  (MM regresses)',   ao_ok  &  vo_ok  & ~mm_ok,  '#f0ad4e'),
    ('AO only',                      ao_ok  & ~vo_ok  & ~mm_ok,  '#d9534f'),
    ('VO only',                     ~ao_ok  &  vo_ok  & ~mm_ok,  '#e57373'),
    ('None correct',                ~ao_ok  & ~vo_ok  & ~mm_ok,  '#cccccc'),
]

# ── 2. CLS attention weights ──────────────────────────────────────────────────
vision_frac, action_frac, self_frac = {}, {}, {}
for li in cross_li:
    aw = attn_rows[li]            # (N, total_len)
    sf = aw[:, 0]                 # CLS → self
    vf = aw[:, v_s:v_e].sum(-1)  # CLS → vision
    af = aw[:, v_e:].sum(-1)     # CLS → action (includes ~0-weight padded positions)
    tot = sf + vf + af
    self_frac[li]   = sf / tot
    vision_frac[li] = vf / tot
    action_frac[li] = af / tot

# ── 3. Per-class recall correlation ──────────────────────────────────────────
ao_rc = per_class_recall_from_json('ao_native_sparse_weighted_j6457852')
vo_rc = per_class_recall_from_json('vision_vc1_delta16_sp_wt_j6459653')
mm_rc = per_class_recall_from_json('full_vc1_d16_late2_sp_wt_j6459079')
shared_cls = sorted(set(ao_rc) & set(vo_rc) & set(mm_rc))
ao_arr = np.array([ao_rc[c] for c in shared_cls])
vo_arr = np.array([vo_rc[c] for c in shared_cls])
mm_arr = np.array([mm_rc[c] for c in shared_cls])
mm_delta = mm_arr - np.maximum(ao_arr, vo_arr)
try:
    rho, pval = spearmanr(ao_arr, vo_arr)
except Exception:
    rho, pval = spearmanr(ao_arr, vo_arr).correlation, None

# ── 4. NLL decomposition (unique variance explained) ─────────────────────────
# H(Y) = empirical entropy of verb distribution in val set
verb_counts = {}
for v in ao_true:
    verb_counts[v] = verb_counts.get(v, 0) + 1
probs_y = np.array(list(verb_counts.values())) / N
H_Y = -np.sum(probs_y * np.log(probs_y + 1e-12))   # in nats

NLL_AO  = -ao_lp.mean()   # mean NLL under AO model ≈ H(Y|action)
NLL_VO  = -vo_lp.mean()   # ≈ H(Y|vision)
NLL_MM  = -mm_lp.mean()   # ≈ H(Y|action,vision)
NLL_RAND = H_Y             # uninformed model

# Unique info (nats)
I_action_given_vision = NLL_VO - NLL_MM    # unique: action given vision
I_vision_given_action = NLL_AO - NLL_MM    # unique: vision given action
I_AO_alone = NLL_RAND - NLL_AO             # total info in action alone
I_VO_alone = NLL_RAND - NLL_VO             # total info in vision alone
I_MM_total = NLL_RAND - NLL_MM             # total info in both
# Shared = I_AO_alone + I_VO_alone - I_MM_total (overlap)
I_shared   = I_AO_alone + I_VO_alone - I_MM_total
I_irreducible = NLL_MM                     # entropy not explained by either

print(f"\nNLL (nats): AO={NLL_AO:.3f}  VO={NLL_VO:.3f}  MM={NLL_MM:.3f}"
      f"  rand={NLL_RAND:.3f}")
print(f"Unique info from action (given vision): {I_action_given_vision:.4f} nats")
print(f"Unique info from vision (given action): {I_vision_given_action:.4f} nats")
print(f"Shared information: {I_shared:.4f} nats")

# Per-class NLL improvement
verb_list  = sorted(verb_counts.keys())
ao_lp_arr  = np.array(ao_lp)
vo_lp_arr  = np.array(vo_lp)
mm_lp_arr  = np.array(mm_lp)
true_arr   = np.array(ao_true)

delta_action = {}   # NLL_VO - NLL_MM per class = unique action info
delta_vision = {}   # NLL_AO - NLL_MM per class = unique vision info
for v in verb_list:
    mask = true_arr == v
    if mask.sum() == 0:
        continue
    delta_action[v] = (-vo_lp_arr[mask]).mean() - (-mm_lp_arr[mask]).mean()
    delta_vision[v] = (-ao_lp_arr[mask]).mean() - (-mm_lp_arr[mask]).mean()

# ── 5. Confusion matrix difference ───────────────────────────────────────────
verbs_in_ao = sorted(set(ao_pred + ao_true))
verbs_in_vo = sorted(set(vo_pred + vo_true))
shared_conf_verbs = sorted(set(verbs_in_ao) & set(verbs_in_vo))
v2i = {v: i for i, v in enumerate(shared_conf_verbs)}

ao_true_idx  = [v2i[v] for v in ao_true  if v in v2i]
ao_pred_idx  = [v2i[v] for v in ao_pred  if v in v2i]
vo_true_idx  = [v2i[v] for v in vo_true  if v in v2i]
vo_pred_idx  = [v2i[v] for v in vo_pred  if v in v2i]
n_cls = len(shared_conf_verbs)
cm_ao = confusion_matrix(ao_true_idx, ao_pred_idx, labels=list(range(n_cls)))
cm_vo = confusion_matrix(vo_true_idx, vo_pred_idx, labels=list(range(n_cls)))
# Normalize each confusion matrix row-wise (per true class) before subtracting
cm_ao_norm = cm_ao.astype(float) / (cm_ao.sum(1, keepdims=True) + 1e-9)
cm_vo_norm = cm_vo.astype(float) / (cm_vo.sum(1, keepdims=True) + 1e-9)
cm_diff = cm_ao_norm - cm_vo_norm   # positive = AO worse (confuses more); negative = VO worse

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Model behavior (overlap + per-class recall + attention)
# ═══════════════════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(20, 15))
fig1.suptitle('Multimodal Fusion: Prediction Behavior Analysis',
              fontsize=16, fontweight='bold', y=0.998)

outer1 = gridspec.GridSpec(2, 1, figure=fig1, hspace=0.50,
                            top=0.970, bottom=0.06)
top1   = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer1[0], wspace=0.35)
ax_over = fig1.add_subplot(top1[0])
ax_corr = fig1.add_subplot(top1[1])
ax_attn = fig1.add_subplot(outer1[1])

# Panel 1: prediction overlap
cat_vals   = [(lbl, mask.sum(), col) for lbl, mask, col in OVERLAP_CATS]
y_pos      = np.arange(len(cat_vals))
bars       = ax_over.barh(y_pos, [v for _,v,_ in cat_vals],
                          color=[c for _,_,c in cat_vals],
                          edgecolor='k', lw=0.5, height=0.65)
for bar, (lbl, val, _) in zip(bars, cat_vals):
    ax_over.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                 f'{val}  ({100*val/N:.1f}%)', va='center', fontsize=10)
ax_over.set_yticks(y_pos)
ax_over.set_yticklabels([lbl for lbl,_,_ in cat_vals], fontsize=10)
ax_over.set_xlabel('Number of val examples', fontsize=11)
ax_over.set_title(f'Panel 1 — Prediction Overlap  (N={N})\n'
                  f'MM={mm_ok.mean()*100:.1f}%  AO={ao_ok.mean()*100:.1f}%'
                  f'  VO={vo_ok.mean()*100:.1f}%', fontsize=11)
ax_over.set_xlim(0, max(v for _,v,_ in cat_vals) * 1.35)
ax_over.invert_yaxis()
ax_over.xaxis.grid(True, ls='--', alpha=0.4)
ax_over.set_axisbelow(True)

# Panel 2: per-class recall scatter
cmap_sc = plt.cm.RdYlGn
norm_sc = plt.Normalize(vmin=min(mm_delta.min(), -0.05), vmax=max(mm_delta.max(), 0.05))
sc = ax_corr.scatter(ao_arr * 100, vo_arr * 100, c=mm_delta * 100,
                     cmap=cmap_sc, norm=plt.Normalize(mm_delta.min()*100, mm_delta.max()*100),
                     s=90, edgecolors='k', lw=0.5, zorder=3)
ax_corr.plot([0, 100], [0, 100], ls='--', color='#aaaaaa', lw=1.0, zorder=1)
for i, cls in enumerate(shared_cls):
    ax_corr.annotate(cls, (ao_arr[i]*100, vo_arr[i]*100),
                     fontsize=7.5, xytext=(3, 2), textcoords='offset points', zorder=4)
plt.colorbar(sc, ax=ax_corr, label='MM recall − best unimodal  (pp)')
ax_corr.set_xlabel('AO recall (%)', fontsize=11)
ax_corr.set_ylabel('VO recall (%)', fontsize=11)
ax_corr.set_title(f'Panel 2 — Per-Class Recall: AO vs VO\n'
                  f'Spearman ρ={rho:.2f}  '
                  f'(color = MM improvement over best unimodal)', fontsize=11)
ax_corr.tick_params(labelsize=10)

# Panel 3: CLS attention violin
clr_v, clr_a, clr_s = '#4393c3', '#d6604d', '#888888'
bw = 0.22
for k, li in enumerate(cross_li):
    for offset, data, col in [(-bw, vision_frac[li]*100, clr_v),
                               (0.0, action_frac[li]*100, clr_a),
                               (+bw, self_frac[li]*100,   clr_s)]:
        vp = ax_attn.violinplot(data, positions=[k + offset],
                                widths=bw*1.7, showmedians=True)
        for pc in vp['bodies']:
            pc.set_facecolor(col); pc.set_alpha(0.7)
        for key in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if key in vp:
                vp[key].set_color(col)
ax_attn.set_xticks(range(len(cross_li)))
ax_attn.set_xticklabels([f'Cross-modal layer {li}' for li in cross_li], fontsize=11)
ax_attn.set_ylabel('Fraction of CLS attention (%)', fontsize=11)
ax_attn.set_title('Panel 3 — CLS Attention Distribution in Cross-Modal Layers\n'
                  'Token ratio: 16 vision / ~64 action / 1 self  '
                  '→ uniform baseline ≈ 19% / 79% / 1%', fontsize=11)
ax_attn.yaxis.grid(True, ls='--', alpha=0.4)
ax_attn.set_axisbelow(True)
leg_patches = [mpatches.Patch(color=clr_v, alpha=0.7,
                               label='Vision tokens  (positions 1–16)'),
               mpatches.Patch(color=clr_a, alpha=0.7,
                               label='Action tokens  (positions 17+)'),
               mpatches.Patch(color=clr_s, alpha=0.7,
                               label='CLS self-attention  (position 0)')]
ax_attn.legend(handles=leg_patches, fontsize=10, loc='upper right', framealpha=0.88)

out1 = os.path.join(FDIR, 'multimodal_analysis2.png')
fig1.savefig(out1, dpi=150, bbox_inches='tight')
print(f'Saved → {out1}')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Unique variance / NLL decomposition
# ═══════════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(20, 15))
fig2.suptitle('Unique Variance Explained: Action vs Vision',
              fontsize=16, fontweight='bold', y=0.998)
fig2.text(0.5, 0.971,
          'NLL decomposition: H(Y|action,vision) as lower bound of conditional entropy. '
          'Unique info = reduction in NLL when adding one modality given the other.',
          ha='center', va='top', fontsize=11, color='#444444', style='italic')

outer2 = gridspec.GridSpec(2, 1, figure=fig2, hspace=0.55,
                            top=0.950, bottom=0.06)
top2   = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer2[0], wspace=0.40)
ax_nll  = fig2.add_subplot(top2[0])   # NLL Venn bar
ax_pcnll = fig2.add_subplot(top2[1])  # per-class delta NLL
ax_cdiff = fig2.add_subplot(outer2[1])  # confusion matrix difference

# Panel 1: NLL decomposition stacked bar
nll_labels = ['Unique Action\n(given vision)',
              'Unique Vision\n(given action)',
              'Shared',
              'Irreducible\n(unexplained)']
nll_vals = [I_action_given_vision, I_vision_given_action,
            max(I_shared, 0),      I_irreducible]
nll_colors = ['#d6604d', '#4393c3', '#9e7bcc', '#cccccc']
x_pos = np.arange(len(nll_labels))
bars2 = ax_nll.bar(x_pos, nll_vals, color=nll_colors, edgecolor='k', lw=0.6, width=0.55)
for bar, val in zip(bars2, nll_vals):
    ax_nll.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                f'{val:.3f} nats', ha='center', va='bottom', fontsize=11)
ax_nll.set_xticks(x_pos)
ax_nll.set_xticklabels(nll_labels, fontsize=11)
ax_nll.set_ylabel('Information  (nats)', fontsize=11)
ax_nll.set_title(f'Panel 1 — NLL Decomposition  (H(Y) = {H_Y:.3f} nats)\n'
                 f'NLL: AO={NLL_AO:.3f}  VO={NLL_VO:.3f}  MM={NLL_MM:.3f}', fontsize=11)
ax_nll.yaxis.grid(True, ls='--', alpha=0.4)
ax_nll.set_axisbelow(True)
ax_nll.tick_params(axis='y', labelsize=10)

# Panel 2: per-class delta NLL (sorted by delta_action descending)
cls_sorted = sorted(delta_action.keys(),
                    key=lambda c: delta_action[c], reverse=True)
da = np.array([delta_action[c] for c in cls_sorted])
dv = np.array([delta_vision[c] for c in cls_sorted])
xc = np.arange(len(cls_sorted))
bwc = 0.30
ax_pcnll.bar(xc - bwc/2, da, bwc, color='#d6604d', edgecolor='k', lw=0.4,
             label='Unique action info  (NLL_VO − NLL_MM per class)')
ax_pcnll.bar(xc + bwc/2, dv, bwc, color='#4393c3', edgecolor='k', lw=0.4, alpha=0.85,
             label='Unique vision info  (NLL_AO − NLL_MM per class)')
ax_pcnll.axhline(0, color='k', lw=1.0, zorder=3)
ax_pcnll.set_xticks(xc)
ax_pcnll.set_xticklabels(cls_sorted, rotation=45, ha='right', fontsize=10)
ax_pcnll.set_ylabel('ΔNLL  (nats, positive = modality helps)', fontsize=11)
ax_pcnll.set_title('Panel 2 — Per-Class Unique Information\n'
                   'Sorted by unique action info  (red > 0 = action helps this class)',
                   fontsize=11)
ax_pcnll.legend(fontsize=10, loc='upper right', framealpha=0.88)
ax_pcnll.yaxis.grid(True, ls='--', alpha=0.4)
ax_pcnll.tick_params(axis='y', labelsize=10)
ax_pcnll.set_axisbelow(True)

# Panel 3: confusion matrix difference  (AO_norm − VO_norm)
# Zero diagonal so we focus on off-diagonal confusions
np.fill_diagonal(cm_diff, 0)
vmax = np.abs(cm_diff).max()
im = ax_cdiff.imshow(cm_diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
plt.colorbar(im, ax=ax_cdiff,
             label='AO confusion rate − VO confusion rate\n'
                   '(+: action confuses more;  −: vision confuses more)')
ax_cdiff.set_xticks(range(n_cls))
ax_cdiff.set_yticks(range(n_cls))
ax_cdiff.set_xticklabels(shared_conf_verbs, rotation=45, ha='right', fontsize=9)
ax_cdiff.set_yticklabels(shared_conf_verbs, fontsize=9)
ax_cdiff.set_xlabel('Predicted verb', fontsize=11)
ax_cdiff.set_ylabel('True verb', fontsize=11)
ax_cdiff.set_title('Panel 3 — Confusion Matrix Difference  (AO_norm − VO_norm)\n'
                   'Red cell (i→j): action confuses i as j more than vision does  '
                   '→ vision disambiguates i vs j\n'
                   'Blue cell (i→j): vision confuses i as j more than action does  '
                   '→ action disambiguates i vs j', fontsize=11)

out2 = os.path.join(FDIR, 'multimodal_nll_decomp.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f'Saved → {out2}')

# ── Print summary ─────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('UNIQUE VARIANCE SUMMARY')
print('='*60)
frac_A = I_action_given_vision / H_Y
frac_V = I_vision_given_action / H_Y
print(f'Unique info from action (given vision): {I_action_given_vision:.4f} nats'
      f'  ({frac_A*100:.1f}% of H(Y))')
print(f'Unique info from vision (given action): {I_vision_given_action:.4f} nats'
      f'  ({frac_V*100:.1f}% of H(Y))')
print(f'Shared information:                     {I_shared:.4f} nats'
      f'  ({max(I_shared,0)/H_Y*100:.1f}% of H(Y))')
print(f'Irreducible entropy:                    {I_irreducible:.4f} nats'
      f'  ({I_irreducible/H_Y*100:.1f}% of H(Y))')
print('\nPer-class unique info (sorted by action contribution):')
print(f'  {"Verb":<15} {"Unique action":>15} {"Unique vision":>15}')
for c in cls_sorted:
    print(f'  {c:<15} {delta_action[c]:>14.4f}n {delta_vision[c]:>14.4f}n')

# Save JSON summary
summary = {
    'NLL': {'AO': float(NLL_AO), 'VO': float(NLL_VO), 'MM': float(NLL_MM),
            'H_Y': float(H_Y)},
    'unique_info': {
        'action_given_vision_nats': float(I_action_given_vision),
        'vision_given_action_nats': float(I_vision_given_action),
        'shared_nats': float(I_shared),
        'irreducible_nats': float(I_irreducible),
    },
    'per_class_unique': {c: {'action': float(delta_action[c]),
                             'vision': float(delta_vision[c])}
                         for c in verb_list if c in delta_action},
}
jout = os.path.join(RDIR, 'multimodal_unique_variance.json')
with open(jout, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\nSaved JSON → {jout}')
