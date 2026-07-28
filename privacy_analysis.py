"""
SHAP-guided feature selection (5.1, Figure 12)
and the three-way differential privacy comparison (5.2, Tables 5-7,
Figure 13) -- DP-GaussianNB, DP-SGD logistic regression, and DP-SGD on
the regularised CNN.
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

EPSILONS = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


# ---------------------------------------------------------------------------
# 5.1 SHAP-guided feature selection sweep
# ---------------------------------------------------------------------------

def run_feature_selection_sweep(top_idx, X_train, X_test, y_train, y_test,
                                 feature_counts=(3, 5, 8, 10, 15, 20, 30, 50, 76)):
    """Retrains RF on the top-k SHAP-ranked features for each k, to find
    the smallest subset that doesn't cost meaningful accuracy (Figure 12).
    top_idx should already be sorted by SHAP importance, most important first."""
    results = []
    for n_feats in feature_counts:
        idx_n = top_idx[:n_feats]
        rf_n = RandomForestClassifier(n_estimators=10, min_samples_split=3,
                                       bootstrap=True, n_jobs=-1, random_state=0)
        rf_n.fit(X_train[:, idx_n], y_train)
        acc_n = accuracy_score(y_test, rf_n.predict(X_test[:, idx_n]))
        results.append((n_feats, round(acc_n * 100, 2)))
        print(f"Top {n_feats:2d} SHAP features: {acc_n * 100:.2f}%")
    return results


def plot_feature_selection_curve(results, full_feature_acc=99.82, privacy_threshold=15, save_path=None):
    feats = [r[0] for r in results]
    accs = [r[1] for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(feats, accs, "o-", color="#1F3864", linewidth=2, markersize=8)
    ax.axhline(full_feature_acc, color="grey", linestyle="--", alpha=0.7,
               label=f"Full 76 features ({full_feature_acc}%)")
    ax.axvline(privacy_threshold, color="#c0392b", linestyle="--", alpha=0.7,
               label=f"Privacy-preserving threshold ({privacy_threshold} features)")
    ax.set_xlabel("Number of SHAP-Selected Features")
    ax.set_ylabel("Test Accuracy (%)")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 5.2.1 DP-Gaussian Naive Bayes
# ---------------------------------------------------------------------------

def run_dp_gnb_sweep(X_train_15, y_train, X_test_15, y_test, epsilons=EPSILONS, n_seeds=10):
    """diffprivlib's analytic Gaussian mechanism applied to GaussianNB's
    class-conditional means/variances. Averaged over n_seeds because a
    single run at low epsilon is extremely noisy (the injected noise
    dominates the signal -- see the std columns in Table 5)."""
    import diffprivlib as dp

    X_min = X_train_15.min(axis=0)
    X_max = X_train_15.max(axis=0)

    results = []
    for epsilon in epsilons:
        accs = []
        for seed in range(n_seeds):
            try:
                clf = dp.models.GaussianNB(epsilon=epsilon, bounds=(X_min, X_max), random_state=seed)
                clf.fit(X_train_15, y_train)
                accs.append(accuracy_score(y_test, clf.predict(X_test_15)))
            except Exception as e:
                print(f"epsilon={epsilon} seed={seed}: {e}")

        mean_acc, std_acc = np.mean(accs) * 100, np.std(accs) * 100
        results.append((epsilon, mean_acc, std_acc))
        print(f"epsilon={epsilon:5.2f} | acc={mean_acc:.2f}% +/- {std_acc:.2f}% (n={len(accs)} seeds)")

    return results


# ---------------------------------------------------------------------------
# 5.2.3 DP-SGD logistic regression (capacity-matched control for DP-GNB)
# ---------------------------------------------------------------------------

def run_dp_lr_sweep(X_train_15, y_train, X_test_15, y_test, epsilons=EPSILONS, n_seeds=10):
    """Same low capacity as GNB, but the iterative DP-SGD mechanism instead
    of GNB's single-shot perturbation -- lets us tell whether DP-CNN's
    advantage over DP-GNB (below) comes from model capacity or from the
    privacy mechanism itself."""
    import diffprivlib as dp

    data_norm = np.linalg.norm(X_train_15, axis=1).max()

    results = []
    for epsilon in epsilons:
        accs = []
        for seed in range(n_seeds):
            try:
                clf = dp.models.LogisticRegression(epsilon=epsilon, data_norm=data_norm,
                                                    random_state=seed, max_iter=200)
                clf.fit(X_train_15, y_train)
                accs.append(accuracy_score(y_test, clf.predict(X_test_15)))
            except Exception as e:
                print(f"epsilon={epsilon} seed={seed}: {e}")

        mean_acc, std_acc = np.mean(accs) * 100, np.std(accs) * 100
        results.append((epsilon, mean_acc, std_acc))
        print(f"epsilon={epsilon:5.2f} | acc={mean_acc:.2f}% +/- {std_acc:.2f}% (n={len(accs)} seeds)")

    return results


# ---------------------------------------------------------------------------
# 5.2.2 DP-SGD training of the regularised CNN
# ---------------------------------------------------------------------------

class DPCNNTrainer:
    """Wraps the DP-SGD training loop for the 15-feature regularised CNN.
    Kept as a class mostly so achieved_epsilon() and the training loop
    share N_TRAIN/BATCH_SIZE without passing them around everywhere."""

    def __init__(self, X_train_15, y_train, X_test_15, y_test, batch_size=32):
        import tensorflow as tf
        self.tf = tf
        self.X_train = X_train_15.astype("float32")
        self.y_train = y_train.astype("float32")
        self.X_test = X_test_15.astype("float32")
        self.y_test = y_test.astype("float32")
        self.batch_size = batch_size
        self.n_train = self.X_train.shape[0]
        self.delta = 1.0 / self.n_train

    def achieved_epsilon(self, noise_multiplier, epochs):
        """Mirrors tensorflow_privacy's own accounting logic (same order
        grid and event construction as compute_dp_sgd_privacy_lib.py)."""
        from dp_accounting import dp_event
        from dp_accounting.rdp import rdp_privacy_accountant

        q = self.batch_size / self.n_train
        orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5]
                  + list(range(5, 64)) + [128, 256, 512])
        steps = int(math.ceil(epochs * self.n_train / self.batch_size))
        accountant = rdp_privacy_accountant.RdpAccountant(orders)
        accountant.compose(dp_event.SelfComposedDpEvent(
            dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(noise_multiplier)), steps,
        ))
        return accountant.get_epsilon(self.delta)

    def _build_model(self):
        from tensorflow.keras import layers, regularizers
        return self.tf.keras.Sequential([
            layers.Input(shape=(self.X_train.shape[1],)),
            layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ])

    def _make_train_step(self, model, optimizer):
        loss_fn = self.tf.keras.losses.BinaryCrossentropy(from_logits=False)
        tf = self.tf

        @tf.function
        def dp_train_step(xb, yb, l2_norm_clip, noise_multiplier):
            trainable_vars = model.trainable_variables

            def single_example_grad(elems):
                x_i, y_i = elems
                x_i, y_i = tf.expand_dims(x_i, 0), tf.expand_dims(y_i, 0)
                with tf.GradientTape() as tape:
                    pred = model(x_i, training=True)
                    loss = loss_fn(y_i, pred)
                grads = tape.gradient(loss, trainable_vars)
                flat = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
                norm = tf.norm(flat)
                clip_factor = tf.minimum(1.0, l2_norm_clip / (norm + 1e-12))
                return [g * clip_factor for g in grads]

            per_ex_grads = tf.vectorized_map(single_example_grad, (xb, yb))
            summed = [tf.reduce_sum(g, axis=0) for g in per_ex_grads]
            bs = tf.cast(tf.shape(xb)[0], tf.float32)
            noised = [(s + tf.random.normal(tf.shape(s), stddev=l2_norm_clip * noise_multiplier)) / bs
                      for s in summed]
            optimizer.apply_gradients(zip(noised, trainable_vars))

        return dp_train_step

    def train_once(self, noise_multiplier, epochs, seed, l2_norm_clip=1.0):
        self.tf.random.set_seed(seed)
        np.random.seed(seed)

        model = self._build_model()
        optimizer = self.tf.keras.optimizers.Adam(learning_rate=0.001)
        train_step = self._make_train_step(model, optimizer)

        steps_per_epoch = self.n_train // self.batch_size
        clip_t = self.tf.constant(l2_norm_clip, dtype="float32")
        noise_t = self.tf.constant(noise_multiplier, dtype="float32")

        for _ in range(epochs):
            perm = np.random.permutation(self.n_train)
            Xs, ys = self.X_train[perm], self.y_train[perm]
            for step in range(steps_per_epoch):
                xb = self.tf.constant(Xs[step * self.batch_size:(step + 1) * self.batch_size])
                yb = self.tf.constant(ys[step * self.batch_size:(step + 1) * self.batch_size])
                train_step(xb, yb, clip_t, noise_t)

        preds = model(self.X_test, training=False).numpy().flatten()
        return np.mean((preds > 0.5).astype("float32") == self.y_test)

    def find_noise_multiplier_for_epsilon(self, target_epsilon, epochs, tol=0.02, lo=0.05, hi=80.0, max_iter=60):
        """Bisection search -- achieved_epsilon is monotonically decreasing
        in noise_multiplier, so this converges reliably."""
        lo_nm, hi_nm = lo, hi
        mid, eps_mid = (lo + hi) / 2, None
        for _ in range(max_iter):
            mid = (lo_nm + hi_nm) / 2
            eps_mid = self.achieved_epsilon(mid, epochs)
            if eps_mid > target_epsilon:
                lo_nm = mid
            else:
                hi_nm = mid
            if target_epsilon > 0 and abs(eps_mid - target_epsilon) / target_epsilon < tol:
                break
        return mid, eps_mid


def run_dp_cnn_sweep(trainer, configs=None, n_seeds=5):
    """The original sweep: noise multiplier and epochs both vary together
    to hit each target epsilon. Kept for comparison against the
    deconfounded version below -- see the module docstring."""
    if configs is None:
        configs = [(25.0, 5), (10.0, 5), (4.0, 8), (2.0, 10), (1.2, 10), (0.8, 15), (0.5, 15)]

    results = []
    for noise_multiplier, epochs in configs:
        eps = trainer.achieved_epsilon(noise_multiplier, epochs)
        accs = [trainer.train_once(noise_multiplier, epochs, seed) for seed in range(n_seeds)]
        mean_acc, std_acc = np.mean(accs) * 100, np.std(accs) * 100
        results.append((eps, mean_acc, std_acc, noise_multiplier, epochs))
        print(f"nm={noise_multiplier:5.2f} epochs={epochs:2d} | eps={eps:.3f} | "
              f"acc={mean_acc:.2f}% +/- {std_acc:.2f}%")
    return results
# ---------------------------------------------------------------------------
# Figure 13: three-way comparison plot
# ---------------------------------------------------------------------------

def plot_privacy_comparison(results_gnb, results_lr, results_cnn, non_private_baseline,
                             save_path=None):
    eps_gnb, acc_gnb, std_gnb = zip(*[(r[0], r[1], r[2]) for r in results_gnb])
    eps_lr, acc_lr, std_lr = zip(*[(r[0], r[1], r[2]) for r in results_lr])
    eps_cnn, acc_cnn, std_cnn = zip(*[(r[0], r[1], r[2]) for r in results_cnn])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(eps_gnb, acc_gnb, yerr=std_gnb, fmt="o-", color="#c0392b",
                linewidth=2.2, markersize=9, capsize=4, label="DP-GNB (15 SHAP features)")
    ax.errorbar(eps_lr, acc_lr, yerr=std_lr, fmt="^-", color="#27ae60",
                linewidth=2.2, markersize=9, capsize=4, label="DP-LR (15 SHAP features)")
    ax.errorbar(eps_cnn, acc_cnn, yerr=std_cnn, fmt="s-", color="#1F3864",
                linewidth=2.2, markersize=9, capsize=4, label="DP-SGD Regularised CNN (15 SHAP features)")
    ax.axhline(non_private_baseline, color="grey", linestyle="--", linewidth=1.5, alpha=0.8,
               label=f"Non-private RF - 15 features ({non_private_baseline:.2f}%)")
    ax.axvline(1.0, color="grey", linestyle=":", alpha=0.6, linewidth=1.2)

    ax.set_xscale("log")
    ax.set_xlabel(r"Privacy Budget ($\varepsilon$) - log scale")
    ax.set_ylabel("Test Accuracy (%)")
    ax.legend(fontsize=9.5, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    print("This module is meant to be imported -- see")
    print("notebooks/AuditFall_full_pipeline.ipynb for the full run order")
    print("(needs X_train_15/X_test_15 from the SHAP feature ranking step first).")
