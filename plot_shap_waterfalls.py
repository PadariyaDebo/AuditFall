
import os
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

import explainability as ex
from config import FIGURES_DIR
from common import load_features, make_rf, set_seeds, CACHE_DIR, SEED

SHAP_N = 300


def main():
    set_seeds()
    plt.show = lambda *a, **k: None
    X, y, subjects, _, _ = load_features()
    z = np.load(os.path.join(CACHE_DIR, "holdout_top15.npz"))
    tr, te = z["train_idx"], z["test_idx"]
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    rf = make_rf().fit(Xtr, y[tr])

    l = ex.compute_shap_values(rf, Xte, sample_size=SHAP_N, seed=SEED)
    y_l = y[te][l["sample_idx"]]
    for which in ("fall", "nonfall"):
        plt.close("all")
        ex.plot_shap_waterfall(l, rf, y_l, which=which)
        plt.close("all")
        print("saved", os.path.join(FIGURES_DIR, f"shap_waterfall_{which}.pdf"))


if __name__ == "__main__":
    main()
