"""
Recording-level random-split baseline (stratified 80:20), for comparison with
earlier studies that use this protocol, with McNemar tests and their 2x2
contingency tables.

  * labels follow SisFall (F = fall, D = activity of daily living); example
    filename -> label mappings and class counts are printed
  * every model is evaluated on the same test set; McNemar uses the exact
    two-sided test when there are fewer than 25 discordant pairs
  * accuracy, sensitivity and specificity are reported for every model
  * the feedforward networks are multilayer perceptrons (no convolutions)

Corrections relative to the original pipeline (see legacy/README.md):
  * labels: FALL_PREFIX = "F", ADL_PREFIX = ("D",)
  * split is stratified (the paper says it is; the old script was not)
  * StandardScaler is fitted on the training set only
  * 5-fold CV uses correctly aligned labels and a Pipeline (scaler inside CV)
  * TensorFlow, NumPy and Python are seeded so the MLP results are repeatable
  * the MLP predictions used for McNemar are the same ones that are reported

Usage (Colab):
    !python random_split_baseline.py                 # everything
    !python random_split_baseline.py --no-dl         # classical models only (fast)
Outputs go to results/random_split/.
"""

import argparse
import json
import os
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score

from config import BASE_DIR, RANDOM_STATE, TEST_SIZE, CV_FOLDS, RESULTS_DIR
from data_loading import find_sisfall_base_dir, load_sisfall
from features import extract_features
from significance_testing import mcnemar_test

SEED = RANDOM_STATE
OUT_DIR = os.path.join(RESULTS_DIR, "random_split")


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


def metrics_row(name, y_true, y_pred, cv_acc=np.nan):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Model": name,
        "Test Acc. (%)": 100 * accuracy_score(y_true, y_pred),
        "CV 5-Fold (%)": 100 * cv_acc if not np.isnan(cv_acc) else np.nan,
        "Sensitivity (%)": 100 * tp / (tp + fn),          # recall on falls
        "Specificity (%)": 100 * tn / (tn + fp),          # recall on ADLs
        "Precision Fall (%)": 100 * precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1 Fall": f1_score(y_true, y_pred, pos_label=1),
        "F1 ADL": f1_score(y_true, y_pred, pos_label=0),
        "Macro F1": f1_score(y_true, y_pred, average="macro"),
        "TP": tp, "FN": fn, "TN": tn, "FP": fp,
    }


def train_mlps(X_train, y_train, X_test):
    """Same architectures as the paper's 'CNN (SGD)' and 'CNN (Regularised)'.
    They contain only fully connected layers, so they are reported as MLPs."""
    import tensorflow as tf
    from tensorflow.keras import Sequential, layers, regularizers
    from tensorflow.keras.callbacks import EarlyStopping

    def build(regularised):
        reg = regularizers.l2(0.01) if regularised else None
        model = Sequential([layers.Input(shape=(X_train.shape[1],)),
                            layers.Dense(32, activation="relu", kernel_regularizer=reg)])
        if regularised:
            model.add(layers.Dropout(0.3))
        model.add(layers.Dense(32, activation="relu", kernel_regularizer=reg))
        if regularised:
            model.add(layers.Dropout(0.3))
        model.add(layers.Dense(1, activation="sigmoid"))
        return model

    configs = {
        "MLP (SGD)": dict(regularised=False, opt=tf.keras.optimizers.SGD(), patience=10),
        "MLP (Regularised)": dict(regularised=True, opt=tf.keras.optimizers.Adam(), patience=15),
    }
    preds, archs = {}, {}
    for name, cfg in configs.items():
        set_seeds()
        model = build(cfg["regularised"])
        model.compile(optimizer=cfg["opt"], loss="binary_crossentropy", metrics=["accuracy"])
        es = EarlyStopping(monitor="val_loss", patience=cfg["patience"], restore_best_weights=True)
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
                  shuffle=True, callbacks=[es], verbose=0)
        preds[name] = (model.predict(X_test, verbose=0).ravel() > 0.5).astype(int)
        lines = []
        model.summary(print_fn=lambda s, **k: lines.append(s))
        archs[name] = "\n".join(lines)
        print(f"{name}: {100 * accuracy_score(y_test_global, preds[name]):.2f}%")
    return preds, archs


def main(run_dl=True):
    global y_test_global
    set_seeds()
    os.makedirs(OUT_DIR, exist_ok=True)

    base_dir = find_sisfall_base_dir() or BASE_DIR
    data, labels, subjects, filenames = load_sisfall(base_dir, return_filenames=True)
    y = np.array(labels)
    subjects = np.array(subjects)

    # ---- filename -> label mapping examples and class counts ----------------
    mapping = pd.DataFrame({"Subject": subjects, "Filename": filenames,
                            "Activity code": [f[:3] for f in filenames],
                            "Label": np.where(y == 1, "Fall (1)", "ADL (0)")})
    examples = (mapping.groupby(mapping["Activity code"].str[0])
                .head(3).sort_values("Filename"))
    print("\nFilename -> label examples:")
    print(examples.to_string(index=False))
    counts = mapping.groupby("Label").size()
    codes = mapping.groupby("Label")["Activity code"].agg(lambda s: f"{s.min()}-{s.max()} ({s.nunique()} codes)")
    print("\nClass counts:\n", pd.concat([counts.rename("Recordings"), codes.rename("Codes")], axis=1))
    examples.to_csv(os.path.join(OUT_DIR, "label_mapping_examples.csv"), index=False)

    # ---- features and leakage-free split ----------------------------------
    print("\nExtracting features...")
    X = np.nan_to_num(extract_features(data))
    X_tr_raw, X_te_raw, y_train, y_test, s_train, s_test = train_test_split(
        X, y, subjects, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    y_test_global = y_test
    scaler = StandardScaler().fit(X_tr_raw)
    X_train, X_test = scaler.transform(X_tr_raw), scaler.transform(X_te_raw)
    shared = sorted(set(s_train) & set(s_test))
    print(f"Train: {len(y_train)} | Test: {len(y_test)} | "
          f"subjects appearing in both: {len(shared)} of {len(set(subjects))}")
    np.savez(os.path.join(OUT_DIR, "split.npz"), X_train=X_train, X_test=X_test,
             y_train=y_train, y_test=y_test, s_train=s_train, s_test=s_test)

    # ---- classical models ---------------------------------------------------
    rows, predictions = [], {}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    for name, model in classical_models().items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)
        cv_acc = cross_val_score(make_pipeline(StandardScaler(), classical_models()[name]),
                                 X, y, cv=cv, n_jobs=-1).mean()
        rows.append(metrics_row(name, y_test, predictions[name], cv_acc))
        print(f"{name}: {rows[-1]['Test Acc. (%)']:.2f}%")

    # ---- MLPs ---------------------------------------------------------------
    if run_dl:
        mlp_preds, archs = train_mlps(X_train, y_train, X_test)
        predictions.update(mlp_preds)
        for name, pred in mlp_preds.items():
            rows.append(metrics_row(name, y_test, pred))
        with open(os.path.join(OUT_DIR, "mlp_architectures.txt"), "w") as f:
            for name, a in archs.items():
                f.write(f"{name}\n{a}\n\n")

    table2 = pd.DataFrame(rows).sort_values("Test Acc. (%)", ascending=False)
    table2.to_csv(os.path.join(OUT_DIR, "table2_baseline.csv"), index=False)
    pd.DataFrame(predictions).assign(y_true=y_test, subject=s_test).to_csv(
        os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

    # ---- McNemar with 2x2 tables ------------------------------------------
    pairs = [("Random Forest", m) for m in predictions if m != "Random Forest"]
    pairs += [("AdaBoost", "SVM")]
    if "MLP (Regularised)" in predictions:
        pairs += [("AdaBoost", "MLP (Regularised)"), ("SVM", "MLP (Regularised)")]
    mc_rows = []
    for a, b in pairs:
        p, sig, t = mcnemar_test(y_test, predictions[a], predictions[b], return_table=True)
        mc_rows.append({"Model A": a, "Model B": b,
                        "Both correct": int(t[0][0]), "A correct, B wrong": int(t[0][1]),
                        "A wrong, B correct": int(t[1][0]), "Both wrong": int(t[1][1]),
                        "Test": "exact binomial" if t[0][1] + t[1][0] < 25 else "chi-squared (cc)",
                        "p-value": round(p, 4),
                        "Result": "significant difference" if sig else "no significant difference detected"})
    table3 = pd.DataFrame(mc_rows)
    table3.to_csv(os.path.join(OUT_DIR, "table3_mcnemar.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 70 + "\nTABLE 2 (corrected baseline)\n" + "=" * 70)
    print(table2.round(2).to_string(index=False))
    print("\n" + "=" * 70 + "\nTABLE 3 (McNemar, two-sided, alpha = 0.05)\n" + "=" * 70)
    print(table3.to_string(index=False))

    summary = {"n_recordings": int(len(y)), "n_falls": int(y.sum()), "n_adl": int((1 - y).sum()),
               "majority_baseline_%": float(round(100 * max(y.mean(), 1 - y.mean()), 2)),
               "n_train": int(len(y_train)), "n_test": int(len(y_test)),
               "subjects_in_both_train_and_test": len(shared)}
    json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
    print("\nSummary:", summary)
    print(f"\nAll outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dl", action="store_true", help="skip the MLPs")
    main(run_dl=not ap.parse_args().no_dl)
