"""
Pairwise McNemar's tests between classifiers.
Used to compare classifier predictions on the same test set.
"""

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


def mcnemar_test(y_true, pred_a, pred_b, return_table=False):
   
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    assert y_true.shape == pred_a.shape == pred_b.shape, "predictions must be on the same test set"
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    b = np.sum(correct_a & ~correct_b)   # A right, B wrong
    c = np.sum(~correct_a & correct_b)   # A wrong, B right

    table = [[np.sum(correct_a & correct_b), b],
             [c, np.sum(~correct_a & ~correct_b)]]

    if b + c == 0:
        p = 1.0
    else:
        p = float(mcnemar(table, exact=(b + c) < 25, correction=True).pvalue)
    if return_table:
        return p, p < 0.05, table
    return p, p < 0.05


def run_significance_tests(y_test, predictions):
    
    import pandas as pd

    rows = []

    rf_pred = predictions["Random Forest"]
    print("McNemar Pairwise Significance Tests (alpha = 0.05)")
    print("=" * 65)
    for name, pred in predictions.items():
        if name == "Random Forest":
            continue
        p, sig = mcnemar_test(y_test, rf_pred, pred)
        interp = "No significant difference detected" if not sig else "Significant difference"
        print(f"RF vs {name:<30} p={p:.4f}  significant={sig}")
        rows.append({"Comparison": f"RF vs {name}", "p-value": p,
                      "Significant": sig, "Interpretation": interp})

    print("\nTop-cluster pairwise comparisons:")
    top_pairs = [
        ("Random Forest", "AdaBoost"),
        ("Random Forest", "SVM"),
        ("Random Forest", "MLP (Regularised)"),
        ("AdaBoost", "SVM"),
    ]
    for a, b_name in top_pairs:
        if a not in predictions or b_name not in predictions:
            continue
        p, sig = mcnemar_test(y_test, predictions[a], predictions[b_name])
        interp = "No significant difference detected" if not sig else "Significant difference"
        print(f"{a} vs {b_name:<25} p={p:.4f}  significant={sig}")
        rows.append({"Comparison": f"{a} vs {b_name}", "p-value": p,
                      "Significant": sig, "Interpretation": interp})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    pred_a = y_true.copy()
    pred_a[:10] = 1 - pred_a[:10]
    pred_b = y_true.copy()
    pred_b[:30] = 1 - pred_b[:30]
    p, sig = mcnemar_test(y_true, pred_a, pred_b)
    print(f"sanity check: p={p:.4f}, significant={sig}")

    
