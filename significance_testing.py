"""
Pairwise McNemar's test between classifiers (Section 4.2 / Table 3).

Raw accuracy differences between the top few models turn out to be small
enough that they could just be noise from the specific test split -- this
is what actually checks that rather than eyeballing the numbers.
"""

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


def mcnemar_test(y_true, pred_a, pred_b):
    """Returns (p_value, is_significant) for two classifiers evaluated on
    the same test set. Uses the exact binomial test when the discordant
    cell counts are small (b+c < 25), otherwise the chi-squared version --
    same convention statsmodels/most papers use."""
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    b = np.sum(correct_a & ~correct_b)   # A right, B wrong
    c = np.sum(~correct_a & correct_b)   # A wrong, B right

    table = [[np.sum(correct_a & correct_b), b],
             [c, np.sum(~correct_a & ~correct_b)]]

    result = mcnemar(table, exact=(b + c) < 25)
    return result.pvalue, result.pvalue < 0.05


def run_significance_tests(y_test, predictions):
    """predictions: dict of {model_name: y_pred array}. Prints the same
    two comparisons the paper reports -- RF vs everything, then the
    top-cluster pairwise comparisons -- and returns a results dataframe."""
    import pandas as pd

    rows = []

    rf_pred = predictions["Random Forest"]
    print("McNemar Pairwise Significance Tests (alpha = 0.05)")
    print("=" * 65)
    for name, pred in predictions.items():
        if name == "Random Forest":
            continue
        p, sig = mcnemar_test(y_test, rf_pred, pred)
        interp = "Equivalent" if not sig else "RF superior"
        print(f"RF vs {name:<30} p={p:.4f}  significant={sig}")
        rows.append({"Comparison": f"RF vs {name}", "p-value": p,
                      "Significant": sig, "Interpretation": interp})

    print("\nTop-cluster pairwise comparisons:")
    top_pairs = [
        ("Random Forest", "AdaBoost"),
        ("Random Forest", "SVM"),
        ("Random Forest", "CNN (Regularised)"),
        ("AdaBoost", "SVM"),
    ]
    for a, b_name in top_pairs:
        if a not in predictions or b_name not in predictions:
            continue
        p, sig = mcnemar_test(y_test, predictions[a], predictions[b_name])
        interp = "Equivalent" if not sig else f"{a} superior"
        print(f"{a} vs {b_name:<25} p={p:.4f}  significant={sig}")
        rows.append({"Comparison": f"{a} vs {b_name}", "p-value": p,
                      "Significant": sig, "Interpretation": interp})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # tiny sanity check with made-up predictions, just to make sure the
    # test statistic doesn't blow up on edge cases
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    pred_a = y_true.copy()
    pred_a[:10] = 1 - pred_a[:10]
    pred_b = y_true.copy()
    pred_b[:30] = 1 - pred_b[:30]
    p, sig = mcnemar_test(y_true, pred_a, pred_b)
    print(f"sanity check: p={p:.4f}, significant={sig}")
