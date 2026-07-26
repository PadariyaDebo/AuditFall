"""
SHAP + LIME explainability for the Random Forest model (Section 3.5,
Section 4.4-4.6, Figures 6-11).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

import shap
import lime
import lime.lime_tabular

from config import FEATURE_NAMES, FIGURES_DIR

TOP_N = 15


def compute_shap_values(rf_model, X_test, sample_size=300, seed=0):
    """TreeSHAP on a random sample of the test set (300 instances, matching
    the paper -- using the full test set isn't necessary for stable
    rankings and is a lot slower)."""
    explainer = shap.TreeExplainer(rf_model)

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(X_test), min(sample_size, len(X_test)), replace=False)
    X_sample = X_test[sample_idx]

    shap_values = explainer.shap_values(X_sample)

    # shap changed its output shape across versions -- sometimes a list of
    # per-class arrays, sometimes one 3D array. Handle both.
    if isinstance(shap_values, list):
        shap_fall, shap_nonfall = shap_values[1], shap_values[0]
        base_fall, base_nonfall = explainer.expected_value[1], explainer.expected_value[0]
    else:
        shap_fall, shap_nonfall = shap_values[:, :, 1], shap_values[:, :, 0]
        base_fall, base_nonfall = explainer.expected_value[1], explainer.expected_value[0]

    return {
        "explainer": explainer,
        "X_sample": X_sample,
        "sample_idx": sample_idx,
        "shap_fall": shap_fall,
        "shap_nonfall": shap_nonfall,
        "base_fall": base_fall,
        "base_nonfall": base_nonfall,
    }


def plot_shap_global_importance(shap_result, top_n=TOP_N, save=True):
    """Bar chart of mean |SHAP value| -- Figure 6."""
    shap_fall = shap_result["shap_fall"]
    mean_shap = np.abs(shap_fall).mean(axis=0)
    top_idx = np.argsort(mean_shap)[::-1][:top_n]
    top_names = [FEATURE_NAMES[i] for i in top_idx]
    top_vals = mean_shap[top_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#c0392b" if v > top_vals.mean() else "#2980b9" for v in top_vals]
    ax.barh(range(top_n), top_vals[::-1], color=colors[::-1], edgecolor="white", height=0.65)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=11)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.axvline(top_vals.mean(), color="grey", linestyle="--", linewidth=1.2, alpha=0.8, label="Mean importance")
    for i, val in enumerate(top_vals[::-1]):
        ax.text(val + top_vals.max() * 0.01, i, f"{val:.4f}", va="center", fontsize=9)

    ax.legend(handles=[
        mpatches.Patch(color="#c0392b", label="Above average importance"),
        mpatches.Patch(color="#2980b9", label="Below average importance"),
    ], fontsize=10, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")

    plt.tight_layout()
    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, "shap_bar.pdf"), bbox_inches="tight")
    plt.show()

    return top_idx, top_names, top_vals


def plot_shap_beeswarm(shap_result, top_idx, top_names, save=True):
    """Impact-direction beeswarm -- Figure 7."""
    plt.figure(figsize=(11, 8))
    shap.summary_plot(
        shap_result["shap_fall"][:, top_idx],
        shap_result["X_sample"][:, top_idx],
        feature_names=top_names,
        plot_type="dot",
        max_display=len(top_idx),
        show=False,
    )
    plt.tight_layout()
    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, "shap_beeswarm.pdf"), bbox_inches="tight")
    plt.show()


def plot_shap_waterfall(shap_result, rf_model, y_sample, which="fall", save=True):
    """Local instance-level explanation for one correctly classified
    example -- Figures 8/9. which='fall' or 'nonfall'."""
    if which == "fall":
        shap_vals, base_val = shap_result["shap_fall"], shap_result["base_fall"]
        target_label = 1
        fname = "shap_waterfall_fall.pdf"
        title_label = "FALL"
    else:
        shap_vals, base_val = shap_result["shap_nonfall"], shap_result["base_nonfall"]
        target_label = 0
        fname = "shap_waterfall_nonfall.pdf"
        title_label = "NON-FALL"

    preds = rf_model.predict(shap_result["X_sample"])
    candidates = np.where((y_sample == target_label) & (preds == target_label))[0]
    if len(candidates) == 0:
        print(f"No correctly classified {title_label} instance in this sample -- try a bigger sample_size.")
        return
    idx = candidates[0]

    exp = shap.Explanation(
        values=shap_vals[idx], base_values=base_val,
        data=shap_result["X_sample"][idx], feature_names=FEATURE_NAMES,
    )
    shap.plots.waterfall(exp, max_display=14, show=False)
    fig = plt.gcf()
    fig.set_size_inches(11, 8)
    plt.title(f"SHAP Waterfall - {title_label} Prediction Explained", fontsize=14)
    plt.tight_layout()
    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, fname), bbox_inches="tight")
    plt.show()


def make_lime_explainer(X_train):
    return lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=FEATURE_NAMES,
        class_names=["Non-Fall", "Fall"],
        mode="classification",
        random_state=42,
    )


def plot_lime_explanation(exp, title, filename, save=True):
    lime_list = exp.as_list()
    feats = [x[0] for x in lime_list]
    vals = [x[1] for x in lime_list]
    colors = ["#c0392b" if v > 0 else "#2980b9" for v in vals]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(feats)), vals[::-1], color=colors[::-1], edgecolor="white", height=0.65)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats[::-1], fontsize=11)
    ax.set_xlabel("LIME Feature Weight", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.legend(handles=[
        mpatches.Patch(color="#c0392b", label="Supports Fall"),
        mpatches.Patch(color="#2980b9", label="Supports Non-Fall"),
    ], fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")

    plt.tight_layout()
    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, filename), bbox_inches="tight")
    plt.show()


def run_lime_analysis(rf_model, X_train, X_test, y_test, save=True):
    """LIME on one representative fall + one non-fall test instance --
    Figures 10/11, num_features/num_samples matching Section 3.5.2."""
    explainer_lime = make_lime_explainer(X_train)

    fall_idx = np.where(y_test == 1)[0][0]
    proba_fall = rf_model.predict_proba([X_test[fall_idx]])[0]
    exp_fall = explainer_lime.explain_instance(
        data_row=X_test[fall_idx], predict_fn=rf_model.predict_proba,
        num_features=12, num_samples=2000,
    )
    plot_lime_explanation(
        exp_fall, f"P(Fall)={proba_fall[1]:.3f} | P(Non-Fall)={proba_fall[0]:.3f}",
        "lime_fall.pdf", save=save,
    )

    nonfall_idx = np.where(y_test == 0)[0][0]
    proba_nonfall = rf_model.predict_proba([X_test[nonfall_idx]])[0]
    exp_nonfall = explainer_lime.explain_instance(
        data_row=X_test[nonfall_idx], predict_fn=rf_model.predict_proba,
        num_features=12, num_samples=2000,
    )
    plot_lime_explanation(
        exp_nonfall, f"P(Fall)={proba_nonfall[1]:.3f} | P(Non-Fall)={proba_nonfall[0]:.3f}",
        "lime_nonfall.pdf", save=save,
    )

    return exp_fall, exp_nonfall


if __name__ == "__main__":
    print("Needs a fitted `rf` model plus X_train/X_test/y_test in scope --")
    print("run this via notebooks/AuditFall_full_pipeline.ipynb rather than standalone.")
