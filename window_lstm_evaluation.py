"""
Fixed-length window features and LSTM evaluation at the recording level.

Part A - fixed-length window instead of the whole recording
  SisFall recordings differ in length by activity (e.g. falls ~15 s, some ADLs
  ~100 s), so features computed over a whole recording can partly encode its
  duration. Here the same 76 features are computed on a fixed 4 s window
  (800 samples at 200 Hz) centred on the peak acceleration magnitude
  (ADXL345), i.e. the information a detector would have shortly after an
  impact. Evaluated with 5-fold participant-grouped CV for the six classical
  models, alongside whole-recording features for comparison.

Part B - LSTM evaluated per recording
  The LSTM uses 200 x 9 raw windows (1 s, 50% overlap). It is trained and tested
  on the participant-disjoint holdout from explainability_feature_selection.py (windows never cross the
  split), and its window predictions are aggregated per recording (mean
  probability, majority vote, maximum probability) so it can be compared
  with the feature-based models on the same recordings.
  A GPU runtime is recommended (Runtime > Change runtime type > T4 GPU).

Usage:  !python window_lstm_evaluation.py                 (both parts)
        !python window_lstm_evaluation.py --part A        (fast, no LSTM)
        !python window_lstm_evaluation.py --part B --epochs 30
Outputs: results/window_lstm/
"""

import argparse
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from features import extract_features
from common import (load_raw, grouped_folds, grouped_holdout, saved_folds, saved_holdout, classical_models, make_rf,
                             metrics, mean_std_table, out_dir, set_seeds, FS)

WIN_S = 4.0
LSTM_WIN, LSTM_STEP, N_CH = 200, 100, 9


def peak_window(rec, win=int(WIN_S * FS)):
    mag = np.sqrt((rec[:, :3] ** 2).sum(axis=1))
    c = int(np.argmax(mag))
    start = min(max(c - win // 2, 0), max(len(rec) - win, 0))
    w = rec[start:start + win]
    if len(w) < win:  # pad very short recordings by edge repetition
        w = np.pad(w, ((0, win - len(w)), (0, 0)), mode="edge")
    return w


def part_a(data, y, subjects, OUT):
    durations = np.array([len(d) / FS for d in data])
    dur = pd.DataFrame({"Class": np.where(y == 1, "Fall", "ADL"), "Duration (s)": durations})
    dur_tbl = dur.groupby("Class")["Duration (s)"].describe()[["count", "mean", "std", "min", "max"]].round(1)
    dur_tbl.to_csv(f"{OUT}/recording_durations.csv")
    print("Recording durations by class:\n", dur_tbl)

    print(f"\nExtracting features on {WIN_S:.0f} s peak-centred windows...")
    X_win = np.nan_to_num(extract_features([peak_window(d) for d in data]))
    X_full = np.nan_to_num(extract_features(data))
    np.savez(f"{OUT}/window_features.npz", X_win=X_win)

    rows = []
    for k, (tr, te) in enumerate(saved_folds(y, subjects)):   # same folds as grouped_evaluation.py
        for setting, X in [("Whole recording", X_full), (f"{WIN_S:.0f} s window", X_win)]:
            sc = StandardScaler().fit(X[tr])
            for name, model in classical_models().items():
                model.fit(sc.transform(X[tr]), y[tr])
                rows.append({"Input": setting, "Model": name, "Fold": k,
                             **metrics(y[te], model.predict(sc.transform(X[te])))})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/window_vs_whole_per_fold.csv", index=False)
    tbl = mean_std_table(rows, ["Input", "Model"])
    tbl.to_csv(f"{OUT}/window_vs_whole_grouped_cv.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nWhole recording vs fixed window (participant-grouped 5-fold CV, mean ± std):")
    print(tbl[["Input", "Model", "Accuracy", "Sensitivity", "Specificity", "Macro F1"]].to_string(index=False))
    return X_full, X_win


def segment(data, idx, y):
    Xw, yw, rec = [], [], []
    for i in idx:
        s = data[i][:, :N_CH]
        for start in range(0, len(s) - LSTM_WIN + 1, LSTM_STEP):
            Xw.append(s[start:start + LSTM_WIN])
            yw.append(y[i])
            rec.append(i)
    return np.asarray(Xw, dtype="float32"), np.asarray(yw), np.asarray(rec)


def part_b(data, y, subjects, OUT, epochs, X_full=None, X_win=None):
    import tensorflow as tf
    from tensorflow.keras import Sequential, layers
    from tensorflow.keras.callbacks import EarlyStopping
    set_seeds()

    tr, te = saved_holdout(y, subjects)   # same holdout as the explainability and privacy experiments
    assert not set(subjects[tr]) & set(subjects[te])
    # participant-disjoint validation split from the training participants
    vtr_rel, vva_rel = grouped_folds(y[tr], subjects[tr], n_splits=5)[0]
    fit_idx, val_idx = tr[vtr_rel], tr[vva_rel]

    Xf, yf, _ = segment(data, fit_idx, y)
    Xv, yv, _ = segment(data, val_idx, y)
    Xt, yt, rec_t = segment(data, te, y)
    mu = Xf.reshape(-1, N_CH).mean(axis=0)
    sd = Xf.reshape(-1, N_CH).std(axis=0) + 1e-8
    Xf, Xv, Xt = (Xf - mu) / sd, (Xv - mu) / sd, (Xt - mu) / sd
    print(f"Windows - fit: {len(yf)}, validation: {len(yv)}, test: {len(yt)} "
          f"(from {len(te)} test recordings, {len(set(subjects[te]))} participants)")

    model = Sequential([layers.Input(shape=(LSTM_WIN, N_CH)), layers.LSTM(64), layers.Dropout(0.3),
                        layers.Dense(32, activation="relu"), layers.Dropout(0.2),
                        layers.Dense(1, activation="sigmoid")])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(Xf, yf, validation_data=(Xv, yv), epochs=epochs, batch_size=64,
              callbacks=[es], verbose=2)
    p = model.predict(Xt, batch_size=512, verbose=0).ravel()

    res = [{"Model": "LSTM", "Unit": "per window", "n": len(yt), **metrics(yt, (p > 0.5).astype(int))}]
    agg = pd.DataFrame({"rec": rec_t, "p": p}).groupby("rec")["p"]
    recs = np.array(sorted(set(rec_t)))
    y_rec = y[recs]
    for rule, vals in [("per recording (mean probability > 0.5)", agg.mean()),
                       ("per recording (majority vote)", agg.apply(lambda s: (s > 0.5).mean())),
                       ("per recording (max probability > 0.5)", agg.max())]:
        res.append({"Model": "LSTM", "Unit": rule, "n": len(recs),
                    **metrics(y_rec, (vals.loc[recs].values > 0.5).astype(int))})

    # feature-based models on exactly the same held-out recordings
    if X_full is None:
        X_full = np.nan_to_num(extract_features(data))
    for setting, X in [("whole recording", X_full)] + ([(f"{WIN_S:.0f} s window", X_win)] if X_win is not None else []):
        sc = StandardScaler().fit(X[tr])
        rf = make_rf().fit(sc.transform(X[tr]), y[tr])
        pred = rf.predict(sc.transform(X[recs]))
        res.append({"Model": f"Random Forest ({setting} features)", "Unit": "per recording",
                    "n": len(recs), **metrics(y_rec, pred)})

    res = pd.DataFrame(res).round(2)
    res.to_csv(f"{OUT}/lstm_per_window_vs_per_recording.csv", index=False)
    json.dump({"epochs_max": epochs, "window": LSTM_WIN, "step": LSTM_STEP,
               "test_participants": sorted(map(str, set(subjects[te])))},
              open(f"{OUT}/lstm_settings.json", "w"), indent=2)
    print("\nLSTM vs Random Forest on the same participant-disjoint test recordings:")
    print(res[["Model", "Unit", "n", "Accuracy", "Sensitivity", "Specificity", "Macro F1"]].to_string(index=False))


def main(part, epochs):
    set_seeds()
    OUT = out_dir("window_lstm")
    data, y, subjects, _ = load_raw()
    X_full = X_win = None
    if part in ("A", "AB"):
        X_full, X_win = part_a(data, y, subjects, OUT)
    if part in ("B", "AB"):
        part_b(data, y, subjects, OUT, epochs, X_full, X_win)
    print(f"\nAll outputs saved to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="AB", choices=["A", "B", "AB"])
    ap.add_argument("--epochs", type=int, default=50)
    a = ap.parse_args()
    main(a.part, a.epochs)
