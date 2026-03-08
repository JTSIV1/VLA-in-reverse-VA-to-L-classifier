"""Compare sklearn classifiers on action trajectory features."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from config import TRAIN_DIR, VAL_DIR, EPISODE_TEMPLATE
from utils import load_calvin_to_dataframe

train_df = load_calvin_to_dataframe(TRAIN_DIR)
val_df = load_calvin_to_dataframe(VAL_DIR)
vc = train_df['primary_verb'].value_counts()
keep = vc[vc >= 30].index
train_df = train_df[train_df['primary_verb'].isin(keep)].reset_index(drop=True)
val_df = val_df[val_df['primary_verb'].isin(keep)].reset_index(drop=True)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Classes: {train_df['primary_verb'].nunique()}")


def load_actions(df, data_dir):
    raw_trajs, verbs = [], []
    for _, row in df.iterrows():
        actions = []
        for i in range(row['start_idx'], row['end_idx'] + 1):
            ep = np.load(f"{data_dir}/{EPISODE_TEMPLATE.format(i)}", mmap_mode='r')
            actions.append(np.array(ep['rel_actions']))
        raw_trajs.append(np.array(actions))
        verbs.append(row['primary_verb'])
    return raw_trajs, np.array(verbs)


print("Loading train actions...")
train_trajs, y_tr = load_actions(train_df, TRAIN_DIR)
print("Loading val actions...")
val_trajs, y_va = load_actions(val_df, VAL_DIR)


def traj_to_features(trajs, mode):
    feats = []
    for traj in trajs:
        T = len(traj)
        if mode == 'delta':
            feats.append(traj[-1] - traj[0])
        elif mode == 'concat_2f':
            feats.append(np.concatenate([traj[0], traj[-1]]))
        elif mode == 'stats':
            feats.append(np.concatenate([
                traj.mean(0), traj.std(0), traj.min(0), traj.max(0),
                traj.sum(0), traj.max(0) - traj.min(0),
            ]))
        elif mode == 'stats_extended':
            cumsum = np.cumsum(traj, axis=0)
            feats.append(np.concatenate([
                traj.mean(0), traj.std(0), traj.min(0), traj.max(0),
                traj.sum(0), traj.max(0) - traj.min(0),
                traj[0], traj[-1], cumsum[-1], np.abs(traj).mean(0),
            ]))
        elif mode == 'uniform_8f':
            idx = np.linspace(0, T - 1, 8, dtype=int)
            feats.append(traj[idx].flatten())
        elif mode == 'padded_full':
            if T < 64:
                padded = np.pad(traj, ((0, 64 - T), (0, 0)))
            else:
                padded = traj[:64]
            feats.append(padded.flatten())
    return np.array(feats)


features = {}
for mode in ['delta', 'concat_2f', 'stats', 'stats_extended', 'uniform_8f', 'padded_full']:
    X_tr = traj_to_features(train_trajs, mode)
    X_va = traj_to_features(val_trajs, mode)
    features[mode] = (X_tr, X_va, X_tr.shape[1])
    print(f"  {mode}: {X_tr.shape[1]}-d")

classifiers = {
    'LogReg': LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced'),
    'RF': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'GBM': GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
}

print("\n" + "=" * 100)
print(f"{'Feature':<20s} {'dim':>4s} | {'LogReg':>12s} {'RF':>12s} {'GBM':>12s} {'MLP':>12s} | {'Best':>14s}")
print("=" * 100)

for feat_name, (X_tr_f, X_va_f, dim) in features.items():
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_f)
    X_va_s = scaler.transform(X_va_f)
    accs, mf1s = {}, {}
    for clf_name, clf in classifiers.items():
        clf_copy = type(clf)(**clf.get_params())
        clf_copy.fit(X_tr_s, y_tr)
        preds = clf_copy.predict(X_va_s)
        accs[clf_name] = accuracy_score(y_va, preds) * 100
        mf1s[clf_name] = f1_score(y_va, preds, average='macro') * 100
    best_clf = max(accs, key=accs.get)
    print(f"{feat_name:<20s} {dim:>4d} | " +
          " ".join(f"{accs[c]:5.1f}/{mf1s[c]:4.1f}" for c in classifiers) +
          f" | {accs[best_clf]:5.1f}/{mf1s[best_clf]:4.1f} ({best_clf})")

print("\n--- Comparison ---")
print("AO native transformer (sp+wt):  39.5% acc / 38.7% MacF1")
print("scene_delta RF:                 48.4% acc / 38.9% MacF1")
