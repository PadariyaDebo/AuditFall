
import argparse
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

from common import (load_features, grouped_folds, classical_models, fit_predict_mlp,
                             MLP_CONFIGS, metrics, mean_std_table, subject_group, make_rf,
                             out_dir, set_seeds)
from significance_testing import mcnemar_test


def main(run_dl=True, loso=False):
    set_seeds()
    OUT = out_dir("grouped_cv")
    X, y, subjects, filenames, _ = load_features()
    folds = grouped_folds(y, subjects, n_splits=5)

    # fold assignments (reused by later scripts and for the paper)
    fold_id = np.empty(len(y), dtype=int)
    for k, (_, te) in enumerate(folds):
        fold_id[te] = k
    fa = pd.DataFrame({"subject": subjects, "filename": filenames, "label": y, "fold": fold_id})
    fa.to_csv(f"{OUT}/fold_assignments.csv", index=False)
    fold_summary = fa.groupby("fold").agg(
        participants=("subject", lambda s: ", ".join(sorted(set(s)))),
        n_participants=("subject", "nunique"), recordings=("label", "size"),
        falls=("label", "sum"))
    print("Participant-disjoint folds:\n", fold_summary[["n_participants", "recordings", "falls"]])
    fold_summary.to_csv(f"{OUT}/fold_summary.csv")
    for k, (tr, te) in enumerate(folds):
        assert not set(subjects[tr]) & set(subjects[te]), "participant leakage!"

    model_names = list(classical_models()) + (list(MLP_CONFIGS) if run_dl else [])
    oof = {m: np.full(len(y), -1) for m in model_names}
    rows = []
    for k, (tr, te) in enumerate(folds):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        for name, model in classical_models().items():
            model.fit(Xtr, y[tr])
            oof[name][te] = model.predict(Xte)
        if run_dl:
            for name in MLP_CONFIGS:
                oof[name][te] = fit_predict_mlp(name, Xtr, y[tr], subjects[tr], Xte)
        for name in model_names:
            rows.append({"Model": name, "Fold": k, **metrics(y[te], oof[name][te])})
        print(f"Fold {k + 1}/5 done: " + ", ".join(
            f"{n} {r['Accuracy']:.2f}%" for n, r in
            [(r["Model"], r) for r in rows if r["Fold"] == k]))

    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(f"{OUT}/per_fold_metrics.csv", index=False)
    summary = mean_std_table(rows, "Model").sort_values("Accuracy (mean)", ascending=False)
    pooled = pd.DataFrame([{"Model": m, **metrics(y, oof[m])} for m in model_names]).round(2)
    summary.to_csv(f"{OUT}/table_grouped_cv.csv", index=False)
    pooled.to_csv(f"{OUT}/table_grouped_cv_pooled.csv", index=False)
    pd.DataFrame(oof).assign(y_true=y, subject=subjects, fold=fold_id).to_csv(
        f"{OUT}/oof_predictions.csv", index=False)

    # young vs older participants (older adults recorded ADLs only, except SE06)
    grp = subject_group(subjects)
    by_group = []
    for m in model_names:
        for g in np.unique(grp):
            mask = grp == g
            by_group.append({"Model": m, "Participants": g, "Recordings": int(mask.sum()),
                             **metrics(y[mask], oof[m][mask])})
    by_group = pd.DataFrame(by_group).round(2)
    by_group.to_csv(f"{OUT}/table_by_age_group.csv", index=False)

    # McNemar on pooled out-of-fold predictions (n = all recordings)
    best = summary["Model"].iloc[0]
    mc = []
    for m in model_names:
        if m == best:
            continue
        p, sig, t = mcnemar_test(y, oof[best], oof[m], return_table=True)
        mc.append({"Model A": best, "Model B": m, "Both correct": int(t[0][0]),
                   "A correct, B wrong": int(t[0][1]), "A wrong, B correct": int(t[1][0]),
                   "Both wrong": int(t[1][1]), "p-value": round(p, 4),
                   "Result": "significant difference" if sig else "no significant difference detected"})
    mc = pd.DataFrame(mc)
    mc.to_csv(f"{OUT}/table_mcnemar_grouped.csv", index=False)

    pd.set_option("display.width", 220)
    cols = ["Model", "Accuracy", "Sensitivity", "Specificity", "Macro F1"]
    print("\n" + "=" * 80 + "\nPARTICIPANT-GROUPED 5-FOLD CV (mean ± std over folds)\n" + "=" * 80)
    print(summary[cols].to_string(index=False))
    print("\nPooled out-of-fold:\n" + pooled.to_string(index=False))
    print("\nBy participant group:\n" + by_group[["Model", "Participants", "Recordings", "Accuracy",
                                                   "Sensitivity", "Specificity"]].to_string(index=False))
    print(f"\nMcNemar vs best model ({best}), pooled out-of-fold predictions:\n" + mc.to_string(index=False))

    # comparison with the recording-level random split, if available
    try:
        baseline = pd.read_csv(f"{out_dir('random_split')}/table2_baseline.csv")
        cmp = baseline[["Model", "Test Acc. (%)", "Sensitivity (%)", "Specificity (%)"]].merge(
            summary[["Model", "Accuracy (mean)", "Sensitivity (mean)", "Specificity (mean)"]], on="Model")
        cmp.columns = ["Model", "Random split Acc", "Random split Sens", "Random split Spec",
                       "Grouped CV Acc", "Grouped CV Sens", "Grouped CV Spec"]
        cmp["Acc drop (pp)"] = (cmp["Random split Acc"] - cmp["Grouped CV Acc"]).round(2)
        cmp.to_csv(f"{OUT}/random_vs_grouped.csv", index=False)
        print("\nRandom split vs participant-grouped CV:\n" + cmp.round(2).to_string(index=False))
    except FileNotFoundError:
        pass

    if loso:
        logo = LeaveOneGroupOut()
        pred = np.full(len(y), -1)
        per_subj = []
        for tr, te in logo.split(X, y, groups=subjects):
            sc = StandardScaler().fit(X[tr])
            rf = make_rf().fit(sc.transform(X[tr]), y[tr])
            pred[te] = rf.predict(sc.transform(X[te]))
            per_subj.append({"Participant": subjects[te][0], "Recordings": len(te),
                             "Falls": int(y[te].sum()), **metrics(y[te], pred[te])})
        per_subj = pd.DataFrame(per_subj).round(2)
        per_subj.to_csv(f"{OUT}/loso_random_forest_per_participant.csv", index=False)
        print("\nLOSO Random Forest, pooled:", {k: round(float(v), 2) for k, v in metrics(y, pred).items()})
        print("Per-participant accuracy: mean {:.2f}, min {:.2f} ({})".format(
            per_subj["Accuracy"].mean(), per_subj["Accuracy"].min(),
            per_subj.loc[per_subj["Accuracy"].idxmin(), "Participant"]))

    json.dump({"n_folds": 5, "n_participants": int(len(set(subjects))),
               "best_model": best}, open(f"{OUT}/summary.json", "w"), indent=2)
    print(f"\nAll outputs saved to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dl", action="store_true")
    ap.add_argument("--loso", action="store_true")
    a = ap.parse_args()
    main(run_dl=not a.no_dl, loso=a.loso)
