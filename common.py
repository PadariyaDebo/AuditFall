"""
Shared helpers for the AuditFall experiment scripts.

* features are extracted once and cached in results/cache/features.npz
* all splits are participant-disjoint (StratifiedGroupKFold on subject IDs)
* sensitivity = recall on falls (label 1), specificity = recall on ADLs (label 0)
"""

import os
import random

import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from config import BASE_DIR, RESULTS_DIR, RANDOM_STATE, FEATURE_NAMES
from data_loading import find_sisfall_base_dir, load_sisfall

SEED = RANDOM_STATE
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
FS = 200  # SisFall sampling rate (Hz)


def out_dir(name):
    d = os.path.join(RESULTS_DIR, name)
    os.makedirs(d, exist_ok=True)
    return d


def set_seeds(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_raw():
    base_dir = find_sisfall_base_dir() or BASE_DIR
    data, labels, subjects, filenames = load_sisfall(base_dir, return_filenames=True)
    return data, np.array(labels), np.array(subjects), np.array(filenames)


def load_features(rebuild=False):
    """Returns X (n x 76), y, subjects, filenames, durations (s)."""
    from features import extract_features
    path = os.path.join(CACHE_DIR, "features.npz")
    if os.path.exists(path) and not rebuild:
        z = np.load(path, allow_pickle=True)
        return z["X"], z["y"], z["subjects"], z["filenames"], z["durations"]
    data, y, subjects, filenames = load_raw()
    print("Extracting features (cached after the first run)...")
    X = np.nan_to_num(extract_features(data))
    durations = np.array([len(d) / FS for d in data])
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(path, X=X, y=y, subjects=subjects, filenames=filenames, durations=durations)
    return X, y, subjects, filenames, durations


def grouped_folds(y, subjects, n_splits=5, seed=SEED):
    """Participant-disjoint, class-stratified folds. Returns a list of
    (train_idx, test_idx)."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups=subjects))


def grouped_holdout(y, subjects, seed=SEED):
    """Participant-disjoint ~75:25 split (first fold of a 4-fold grouped CV).
    Used for the explainability figures and the privacy experiments."""
    return grouped_folds(y, subjects, n_splits=4, seed=seed)[0]


def saved_folds(y, subjects):
    """The 5 participant-disjoint folds saved by grouped_evaluation.py (so every script uses
    identical folds, whatever the scikit-learn version); computed if absent."""
    path = os.path.join(RESULTS_DIR, "grouped_cv", "fold_assignments.csv")
    legacy_path = os.path.join(RESULTS_DIR, "day2", "fold_assignments.csv")  # older output folder name
    if not os.path.exists(path) and os.path.exists(legacy_path):
        path = legacy_path
    if os.path.exists(path):
        fa = pd.read_csv(path)
        if len(fa) == len(y) and (fa["subject"].astype(str).values == np.asarray(subjects).astype(str)).all():
            fold = fa["fold"].values
            return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in sorted(set(fold))]
    return grouped_folds(y, subjects, n_splits=5)


def saved_holdout(y, subjects):
    """The participant-disjoint holdout saved by explainability_feature_selection.py; computed if absent."""
    path = os.path.join(CACHE_DIR, "holdout_top15.npz")
    if os.path.exists(path):
        z = np.load(path)
        if max(z["train_idx"].max(), z["test_idx"].max()) < len(y):
            return z["train_idx"], z["test_idx"]
    return grouped_holdout(y, subjects)


def subject_group(subjects):
    """SA = young adults (falls + ADLs), SE = older adults (ADLs only, except SE06)."""
    return np.array(["Young (SA)" if str(s).upper().startswith("SA") else "Older (SE)" for s in subjects])


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

def classical_models():
    return {
        "SVM": LinearSVC(C=10, max_iter=5000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=10),
        "Random Forest": RandomForestClassifier(n_estimators=10, min_samples_split=3,
                                                bootstrap=True, n_jobs=-1, random_state=0),
        "KNN": KNeighborsClassifier(n_neighbors=15, algorithm="ball_tree", n_jobs=-1),
        "Gaussian NB": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(random_state=0),
    }


def make_rf():
    return RandomForestClassifier(n_estimators=10, min_samples_split=3,
                                  bootstrap=True, n_jobs=-1, random_state=0)


def build_mlp(input_dim, regularised, use_l2=True):
    from tensorflow.keras import Sequential, layers, regularizers
    reg = regularizers.l2(0.01) if (regularised and use_l2) else None
    model = Sequential([layers.Input(shape=(input_dim,)),
                        layers.Dense(32, activation="relu", kernel_regularizer=reg)])
    if regularised:
        model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation="relu", kernel_regularizer=reg))
    if regularised:
        model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


MLP_CONFIGS = {"MLP (SGD)": dict(regularised=False, opt="sgd", patience=10),
               "MLP (Regularised)": dict(regularised=True, opt="adam", patience=15)}


def fit_predict_mlp(name, X_tr, y_tr, groups_tr, X_te):
    """Trains an MLP with early stopping on a participant-disjoint validation
    split taken from the training participants only."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    cfg = MLP_CONFIGS[name]
    set_seeds()
    tr, va = grouped_folds(y_tr, groups_tr, n_splits=5)[0]
    model = build_mlp(X_tr.shape[1], cfg["regularised"])
    opt = tf.keras.optimizers.SGD() if cfg["opt"] == "sgd" else tf.keras.optimizers.Adam()
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    es = EarlyStopping(monitor="val_loss", patience=cfg["patience"], restore_best_weights=True)
    model.fit(X_tr[tr], y_tr[tr], validation_data=(X_tr[va], y_tr[va]),
              epochs=100, batch_size=32, shuffle=True, callbacks=[es], verbose=0)
    return (model.predict(X_te, verbose=0).ravel() > 0.5).astype(int)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": 100 * accuracy_score(y_true, y_pred),
        "Sensitivity": 100 * tp / (tp + fn) if tp + fn else np.nan,
        "Specificity": 100 * tn / (tn + fp) if tn + fp else np.nan,
        "Macro F1": 100 * f1_score(y_true, y_pred, average="macro", zero_division=0),
        "TP": int(tp), "FN": int(fn), "TN": int(tn), "FP": int(fp),
    }


def mean_std_table(rows, by, cols=("Accuracy", "Sensitivity", "Specificity", "Macro F1")):
    """rows: list of dicts with key `by` plus metric columns (one row per fold).
    Returns a table with 'mean ± std' strings and numeric mean columns."""
    df = pd.DataFrame(rows)
    g = df.groupby(by, sort=False)[list(cols)]
    mean, std = g.mean(), g.std(ddof=1)
    out = pd.DataFrame(index=mean.index)
    for c in cols:
        out[c] = [f"{m:.2f} ± {s:.2f}" for m, s in zip(mean[c], std[c])]
    for c in cols:
        out[f"{c} (mean)"] = mean[c].round(2)
    return out.reset_index()


def top_k_names(idx, k=15):
    return [FEATURE_NAMES[i] for i in idx[:k]]
