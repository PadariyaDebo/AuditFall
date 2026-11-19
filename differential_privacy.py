"""
Differential privacy experiments: DP-GNB, DP-LR, DP-SGD-LR and DP-SGD-MLP.
"""

import argparse
import json
import math
import os
import time
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from config import FEATURE_NAMES, FIGURES_DIR
from common import (load_features, make_rf, build_mlp, metrics, out_dir, set_seeds,
                             CACHE_DIR, SEED)

EPSILONS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
B = 3.0                 
DELTA = 1e-5            
EPOCHS = 15
EXPECTED_BATCH = 64
CLIP_NORM = 1.0
LR = 0.005


# --------------------------------------------------------------------------
# DP-SGD with Poisson sampling (matches the RDP accountant's assumptions)
# --------------------------------------------------------------------------

def rdp_epsilon(noise_multiplier, q, steps, delta=DELTA):
    from dp_accounting import dp_event
    from dp_accounting.rdp import rdp_privacy_accountant
    orders = [1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] + list(range(5, 64)) + [128, 256, 512]
    acc = rdp_privacy_accountant.RdpAccountant(orders)
    acc.compose(dp_event.SelfComposedDpEvent(
        dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(noise_multiplier)), steps))
    return acc.get_epsilon(delta)


def noise_for_epsilon(target, q, steps, lo=0.3, hi=200.0, iters=60):
    for _ in range(iters):
        mid = (lo + hi) / 2
        if rdp_epsilon(mid, q, steps) > target:
            lo = mid
        else:
            hi = mid
    return hi, rdp_epsilon(hi, q, steps)   # hi always satisfies eps <= target


def build_logreg(input_dim):
    from tensorflow.keras import Sequential, layers
    return Sequential([layers.Input(shape=(input_dim,)), layers.Dense(1, activation="sigmoid")])


def train_dpsgd(arch, X_tr, y_tr, X_te, noise_multiplier, seed):
    import tensorflow as tf
    set_seeds(seed)
    rng = np.random.default_rng(seed)
    n, d = X_tr.shape
    q = EXPECTED_BATCH / n
    steps = int(math.ceil(EPOCHS / q))
    # MLP under DP-SGD: same 32-32-1 layers with Dropout(0.3); no L2 term, since the
    # DP-SGD loss below is the per-example cross-entropy only
    model = build_logreg(d) if arch == "lr" else build_mlp(d, regularised=True, use_l2=False)
    opt = tf.keras.optimizers.Adam(learning_rate=LR)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    variables = model.trainable_variables
    Xt, yt = X_tr.astype("float32"), y_tr.astype("float32").reshape(-1, 1)

    @tf.function(input_signature=[tf.TensorSpec([None, d], tf.float32),
                                  tf.TensorSpec([None, 1], tf.float32)])
    def step(xb, yb):
        def per_example(elems):
            x, t = elems
            with tf.GradientTape() as tape:
                loss = loss_fn(tf.expand_dims(t, 0), model(tf.expand_dims(x, 0), training=True))
            grads = tape.gradient(loss, variables)
            norm = tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(g)) for g in grads]))
            return [g * tf.minimum(1.0, CLIP_NORM / (norm + 1e-12)) for g in grads]
        g = tf.vectorized_map(per_example, (xb, yb))
        summed = [tf.reduce_sum(gi, axis=0) for gi in g]
        noised = [(s + tf.random.normal(tf.shape(s), stddev=CLIP_NORM * noise_multiplier)) / EXPECTED_BATCH
                  for s in summed]
        opt.apply_gradients(zip(noised, variables))

    for _ in range(steps):
        mask = rng.random(n) < q            # Poisson sampling
        if mask.sum() == 0:
            # empty batch: still a valid DP step (noise only)
            xb, yb = np.zeros((1, d), "float32"), np.zeros((1, 1), "float32")
            xb_t, yb_t = tf.constant(xb[:0]), tf.constant(yb[:0])
            noise_only = [tf.random.normal(tf.shape(v), stddev=CLIP_NORM * noise_multiplier) / EXPECTED_BATCH
                          for v in variables]
            opt.apply_gradients(zip(noise_only, variables))
            continue
        step(tf.constant(Xt[mask]), tf.constant(yt[mask]))
    return (model(X_te.astype("float32"), training=False).numpy().ravel() > 0.5).astype(int)


# --------------------------------------------------------------------------

def _diffprivlib_compat():
    """diffprivlib 0.6.x predates recent scikit-learn/SciPy releases. These shims
    only restore removed names/arguments; they do not change any computation."""
    import scipy.optimize as so
    import sklearn.tree._tree as t
    if not hasattr(t, "DOUBLE"):
        t.DOUBLE = np.float64
    if not hasattr(t, "DTYPE"):
        t.DTYPE = np.float32
    orig = so.fmin_l_bfgs_b
    if not getattr(orig, "_dp_compat", False):
        def patched(*a, **k):
            try:
                return orig(*a, **k)
            except TypeError as e:
                if "iprint" not in str(e):
                    raise
                k.pop("iprint", None)
                return orig(*a, **k)
        patched._dp_compat = True
        so.fmin_l_bfgs_b = patched


def main(quick=False):
    import sklearn
    major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):
        raise SystemExit(
            f"scikit-learn {sklearn.__version__} is too new for diffprivlib 0.6.x.\n"
            'Run:  !pip install -q "scikit-learn==1.6.1" diffprivlib dp-accounting\n'
            "then run this script again.")
    _diffprivlib_compat()
    import diffprivlib
    import diffprivlib.models as dpm
    set_seeds()
    OUT = out_dir("privacy")
    seeds_dpl = range(3 if quick else 10)
    seeds_sgd = range(2 if quick else 5)

    X, y, subjects, _, _ = load_features()
    z = np.load(os.path.join(CACHE_DIR, "holdout_top15.npz"))
    tr, te, top15 = z["train_idx"], z["test_idx"], z["top15"]
    assert not set(subjects[tr]) & set(subjects[te])
    sc = StandardScaler().fit(X[tr][:, top15])
    Xtr = np.clip(sc.transform(X[tr][:, top15]), -B, B)
    Xte = np.clip(sc.transform(X[te][:, top15]), -B, B)
    ytr, yte = y[tr], y[te]
    d = Xtr.shape[1]
    bounds = (np.full(d, -B), np.full(d, B))
    data_norm = B * math.sqrt(d)
    q = EXPECTED_BATCH / len(ytr)
    steps = int(math.ceil(EPOCHS / q))

    # ---- non-private references on the same 15 clipped features -----------
    ref = [{"Model": "RF (non-private)", **metrics(yte, make_rf().fit(Xtr, ytr).predict(Xte))},
           {"Model": "GNB (non-private)", **metrics(yte, GaussianNB().fit(Xtr, ytr).predict(Xte))},
           {"Model": "LR (non-private)", **metrics(yte, LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict(Xte))},
           {"Model": "LR, SGD (non-private, same optimiser as DP-SGD)",
            **metrics(yte, train_dpsgd("lr", Xtr, ytr, Xte, 0.0, SEED))},
           {"Model": "MLP, SGD (non-private, same optimiser as DP-SGD)",
            **metrics(yte, train_dpsgd("mlp", Xtr, ytr, Xte, 0.0, SEED))}]
    ref = pd.DataFrame(ref).round(2)
    ref.to_csv(f"{OUT}/non_private_references.csv", index=False)
    print("Non-private references (15 features, participant-disjoint holdout):")
    print(ref[["Model", "Accuracy", "Sensitivity", "Specificity"]].to_string(index=False))

    rows = []
    for eps in EPSILONS:
        t0 = time.time()
        for name, make in [("DP-GNB", lambda s: dpm.GaussianNB(epsilon=eps, bounds=bounds, random_state=s)),
                           ("DP-LR (objective perturbation)",
                            lambda s: dpm.LogisticRegression(epsilon=eps, data_norm=data_norm,
                                                             random_state=s, max_iter=1000))]:
            for s in seeds_dpl:
                clf = make(s).fit(Xtr, ytr)
                rows.append({"Model": name, "Target eps": eps, "Achieved eps": eps, "delta": 0,
                             "Noise multiplier": np.nan, "Seed": s, **metrics(yte, clf.predict(Xte))})
        nm, eps_ach = noise_for_epsilon(eps, q, steps)
        for name, arch in [("DP-SGD-LR", "lr"), ("DP-SGD-MLP", "mlp")]:
            for s in seeds_sgd:
                pred = train_dpsgd(arch, Xtr, ytr, Xte, nm, s)
                rows.append({"Model": name, "Target eps": eps, "Achieved eps": round(eps_ach, 3),
                             "delta": DELTA, "Noise multiplier": round(nm, 3), "Seed": s,
                             **metrics(yte, pred)})
        print(f"eps = {eps:>4}: noise multiplier {nm:.3f} (achieved eps {eps_ach:.3f}) "
              f"-- {time.time() - t0:.0f}s")

    raw = pd.DataFrame(rows)
    raw.to_csv(f"{OUT}/dp_all_runs.csv", index=False)
    agg = (raw.groupby(["Model", "Target eps"], sort=False)
           .agg(**{"Achieved eps": ("Achieved eps", "first"), "Noise multiplier": ("Noise multiplier", "first"),
                   "Seeds": ("Seed", "nunique"),
                   "Acc mean": ("Accuracy", "mean"), "Acc std": ("Accuracy", "std"),
                   "Sens mean": ("Sensitivity", "mean"), "Sens std": ("Sensitivity", "std"),
                   "Spec mean": ("Specificity", "mean"), "Spec std": ("Specificity", "std")})
           .reset_index().round(2))
    rf_ref = ref.loc[0, "Accuracy"]
    agg["Drop vs RF (pp)"] = (rf_ref - agg["Acc mean"]).round(2)
    agg.to_csv(f"{OUT}/dp_summary.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n" + "=" * 90 + "\nDIFFERENTIAL PRIVACY RESULTS (mean ± std over seeds)\n" + "=" * 90)
    for m in agg["Model"].unique():
        print(f"\n{m}")
        print(agg[agg["Model"] == m].drop(columns="Model").to_string(index=False))

    # capacity vs mechanism at eps = 1
    e1 = agg[agg["Target eps"] == 1.0].set_index("Model")["Acc mean"]
    print("\nAt eps = 1:")
    print(f"  capacity  (DP-SGD-MLP - DP-SGD-LR, same mechanism):   {e1['DP-SGD-MLP'] - e1['DP-SGD-LR']:+.2f} pp")
    print(f"  mechanism (DP-SGD-LR - DP-LR objective perturbation): {e1['DP-SGD-LR'] - e1['DP-LR (objective perturbation)']:+.2f} pp")

    # figure
    import matplotlib.pyplot as plt
    style = {"DP-GNB": ("o-", "#c0392b"), "DP-LR (objective perturbation)": ("^-", "#27ae60"),
             "DP-SGD-LR": ("v--", "#8e44ad"), "DP-SGD-MLP": ("s-", "#1F3864")}
    fig, ax = plt.subplots(figsize=(9, 5))
    for m, (fmt, col) in style.items():
        a = agg[agg["Model"] == m]
        ax.errorbar(a["Target eps"], a["Acc mean"], yerr=a["Acc std"], fmt=fmt, color=col,
                    lw=2, ms=8, capsize=4, label=m)
    ax.axhline(rf_ref, color="grey", ls="--", label=f"Non-private RF, 15 features ({rf_ref:.2f}%)")
    ax.axvline(1.0, color="grey", ls=":", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"Privacy budget $\varepsilon$ (log scale)")
    ax.set_ylabel("Accuracy on unseen participants (%)")
    ax.legend(fontsize=9, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, "privacy_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)

    import tensorflow as tf
    settings = {"diffprivlib_version": diffprivlib.__version__, "tensorflow_version": tf.__version__,
                "epsilons": EPSILONS, "feature_clip_B": B, "dp_lr_data_norm": round(data_norm, 3),
                "dpsgd": {"delta": DELTA, "epochs": EPOCHS, "expected_batch": EXPECTED_BATCH,
                          "sampling": "Poisson", "sampling_rate_q": round(q, 5), "steps": steps,
                          "clip_norm": CLIP_NORM, "optimizer": f"Adam(lr={LR})",
                          "mlp_under_dpsgd": "Dense(32,ReLU)-Dropout(0.3)-Dense(32,ReLU)-Dropout(0.3)-Dense(1,sigmoid), no L2",
                          "lr_under_dpsgd": "Dense(1,sigmoid)",
                          "accountant": "RDP (dp_accounting), PoissonSampledDpEvent(GaussianDpEvent)"},
                "n_train": int(len(ytr)), "n_test": int(len(yte)),
                "features": [FEATURE_NAMES[i] for i in top15],
                "not_covered_by_eps": ["standardisation statistics (training set)",
                                       "SHAP-based feature selection (training set)"]}
    json.dump(settings, open(f"{OUT}/dp_settings.json", "w"), indent=2)
    print("\nSettings for the methods section:\n" + json.dumps(settings, indent=2))
    print(f"\nAll outputs saved to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(quick=ap.parse_args().quick)
