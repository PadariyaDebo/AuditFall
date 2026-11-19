
import json
import os
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import FEATURE_NAMES, FIGURES_DIR
from common import (load_features, grouped_folds, grouped_holdout, saved_folds, make_rf, metrics,
                             mean_std_table, out_dir, set_seeds, CACHE_DIR, SEED)

K_GRID = (3, 5, 8, 10, 15, 20, 30, 50, 76)
SHAP_N = 300
TOL_PP = 0.5


def shap_ranking(rf, X_train, n=SHAP_N, seed=SEED):
    """Mean |SHAP| for the fall class on a seeded sample of TRAINING rows."""
    import shap
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train), min(n, len(X_train)), replace=False)
    sv = shap.TreeExplainer(rf).shap_values(X_train[idx])
    sv_fall = sv[1] if isinstance(sv, list) else sv[:, :, 1]
    imp = np.abs(sv_fall).mean(axis=0)
    return np.argsort(imp)[::-1], imp


def choose_k(X_tr, y_tr, groups_tr, order):
    inner = grouped_folds(y_tr, groups_tr, n_splits=3)
    acc = {}
    for k in K_GRID:
        a = []
        for itr, ite in inner:
            rf = make_rf().fit(X_tr[itr][:, order[:k]], y_tr[itr])
            a.append(metrics(y_tr[ite], rf.predict(X_tr[ite][:, order[:k]]))["Accuracy"])
        acc[k] = np.mean(a)
    best = max(acc.values())
    k_star = min(k for k in K_GRID if acc[k] >= best - TOL_PP)
    return k_star, acc


def part_a(X, y, subjects, OUT):
    folds = saved_folds(y, subjects)   # same folds as grouped_evaluation.py
    curve, chosen, top15_sets = [], [], []
    for f, (tr, te) in enumerate(folds):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        rf_full = make_rf().fit(Xtr, y[tr])
        order, _ = shap_ranking(rf_full, Xtr)
        top15_sets.append(order[:15])
        for k in K_GRID:
            rf = make_rf().fit(Xtr[:, order[:k]], y[tr])
            curve.append({"k": k, "Fold": f, **metrics(y[te], rf.predict(Xte[:, order[:k]]))})
        k_star, inner_acc = choose_k(Xtr, y[tr], subjects[tr], order)
        rf = make_rf().fit(Xtr[:, order[:k_star]], y[tr])
        chosen.append({"Fold": f, "Selected k": k_star,
                       "Inner CV acc at k": round(inner_acc[k_star], 2),
                       **metrics(y[te], rf.predict(Xte[:, order[:k_star]]))})
        print(f"Fold {f + 1}/5: selected k = {k_star}, "
              f"test acc = {chosen[-1]['Accuracy']:.2f}% (top-15: "
              f"{[c for c in curve if c['Fold'] == f and c['k'] == 15][0]['Accuracy']:.2f}%)")

    curve_tbl = mean_std_table(curve, "k")
    chosen_df = pd.DataFrame(chosen).round(2)
    freq = pd.Series(np.concatenate(top15_sets)).value_counts()
    stability = pd.DataFrame({"Feature": [FEATURE_NAMES[i] for i in freq.index],
                              "Folds in top-15 (of 5)": freq.values})
    curve_tbl.to_csv(f"{OUT}/selection_curve_grouped_cv.csv", index=False)
    chosen_df.to_csv(f"{OUT}/selected_k_per_fold.csv", index=False)
    stability.to_csv(f"{OUT}/top15_stability.csv", index=False)

    # figure: accuracy vs number of features (outer test folds, train-only ranking)
    import matplotlib.pyplot as plt
    c = pd.DataFrame(curve).groupby("k")["Accuracy"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(c.index, c["mean"], yerr=c["std"], fmt="o-", color="#1F3864", capsize=4, lw=2)
    ax.axvline(15, color="#c0392b", ls="--", alpha=0.7, label="15 features")
    ax.set_xlabel("Number of SHAP-selected features (ranked on training participants only)")
    ax.set_ylabel("Accuracy on unseen participants (%)")
    ax.legend()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, "feature_selection_curve_grouped.pdf"), bbox_inches="tight")
    plt.close(fig)

    print("\nAccuracy vs number of features (mean ± std over 5 participant-disjoint folds):")
    print(curve_tbl[["k", "Accuracy", "Sensitivity", "Specificity"]].to_string(index=False))
    print("\nk selected on training data in each fold:\n" +
          chosen_df[["Fold", "Selected k", "Accuracy", "Sensitivity", "Specificity"]].to_string(index=False))
    print(f"\nSelected-k model: accuracy {chosen_df['Accuracy'].mean():.2f} ± {chosen_df['Accuracy'].std():.2f}")
    print("\nFeatures most often in the top 15:\n" + stability.head(20).to_string(index=False))
    return curve_tbl, chosen_df, stability


def part_b(X, y, subjects, filenames, OUT):
    import explainability as ex
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: None

    tr, te = grouped_holdout(y, subjects)
    assert not set(subjects[tr]) & set(subjects[te])
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    rf = make_rf().fit(Xtr, y[tr])
    order, imp = shap_ranking(rf, Xtr)
    top15 = order[:15]

    ranking = pd.DataFrame({"Rank": np.arange(1, 77), "Feature": [FEATURE_NAMES[i] for i in order],
                            "Mean |SHAP| (fall class, training sample)": imp[order].round(5)})
    ranking.to_csv(f"{OUT}/shap_ranking_holdout_train.csv", index=False)

    rf15 = make_rf().fit(Xtr[:, top15], y[tr])
    res = pd.DataFrame([{"Model": "RF, 76 features", **metrics(y[te], rf.predict(Xte))},
                        {"Model": "RF, top-15 SHAP features", **metrics(y[te], rf15.predict(Xte[:, top15]))}]).round(2)
    res.to_csv(f"{OUT}/holdout_rf_76_vs_15.csv", index=False)

    # global figures from the TRAINING sample (same sample used for the ranking)
    # each plot starts on a fresh figure (shap draws onto the current figure)
    g = ex.compute_shap_values(rf, Xtr, sample_size=SHAP_N, seed=SEED)
    plt.close("all")
    top_idx, top_names, _ = ex.plot_shap_global_importance(g)
    plt.close("all")
    ex.plot_shap_beeswarm(g, top_idx, top_names)
    # local figures for held-out participants
    l = ex.compute_shap_values(rf, Xte, sample_size=SHAP_N, seed=SEED)
    y_l = y[te][l["sample_idx"]]
    plt.close("all")
    ex.plot_shap_waterfall(l, rf, y_l, which="fall")
    plt.close("all")
    ex.plot_shap_waterfall(l, rf, y_l, which="nonfall")
    plt.close("all")
    ex.run_lime_analysis(rf, Xtr, Xte, y[te])
    plt.close("all")

    np.savez(os.path.join(CACHE_DIR, "holdout_top15.npz"), train_idx=tr, test_idx=te,
             top15=top15, order=order)
    info = {"holdout_train_participants": sorted(map(str, set(subjects[tr]))),
            "holdout_test_participants": sorted(map(str, set(subjects[te]))),
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "top15": [FEATURE_NAMES[i] for i in top15]}
    json.dump(info, open(f"{OUT}/holdout_info.json", "w"), indent=2)

    print("\n" + "=" * 70 + "\nPARTICIPANT-DISJOINT HOLDOUT (figures + privacy experiments)\n" + "=" * 70)
    print(f"Train: {len(tr)} recordings / {len(set(subjects[tr]))} participants | "
          f"Test: {len(te)} recordings / {len(set(subjects[te]))} participants")
    print("Test participants:", ", ".join(info["holdout_test_participants"]))
    print(res[["Model", "Accuracy", "Sensitivity", "Specificity", "Macro F1"]].to_string(index=False))
    print("\nTop 15 features (ranked on training participants):")
    print(ranking.head(15).to_string(index=False))
    print(f"\nFigures saved to {FIGURES_DIR}")


def main():
    set_seeds()
    OUT = out_dir("explainability")
    X, y, subjects, filenames, _ = load_features()
    print("PART A: nested participant-grouped evaluation of SHAP selection\n")
    part_a(X, y, subjects, OUT)
    print("\nPART B: holdout ranking, figures and privacy-experiment inputs")
    part_b(X, y, subjects, filenames, OUT)
    print(f"\nAll outputs saved to {OUT}")


if __name__ == "__main__":
    main()
