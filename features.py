"""
Turns a raw recording (Nx9 matrix, one column per sensor axis) into the
76-dim statistical feature vector used by every classical ML model and
the CNN variants in the paper.

Per channel: mean, median, std, max, min, variance, skewness, kurtosis
(8 stats x 9 channels = 72), plus 4 more from the resultant acceleration
magnitude of the ADXL345 (mean/std/max/min) = 76 total.

Falls tend to show up as a short, high-amplitude, heavy-tailed spike in
these stats -- that's basically the whole premise of the feature set.
"""

import numpy as np
import pandas as pd

from config import FEATURE_NAMES


def extract_features(data):
    features = []
    for window in data:
        w = np.array(window, dtype=float)
        feat = []
        for col in range(w.shape[1]):
            s = w[:, col]
            feat += [
                np.mean(s), np.median(s), np.std(s), np.max(s),
                np.min(s), np.var(s),
                pd.Series(s).skew(), pd.Series(s).kurtosis(),
            ]
        # resultant acceleration magnitude, first 3 columns = ADXL345 x/y/z
        mag = np.sqrt(w[:, 0] ** 2 + w[:, 1] ** 2 + w[:, 2] ** 2)
        feat += [np.mean(mag), np.std(mag), np.max(mag), np.min(mag)]
        features.append(feat)
    return np.array(features)


if __name__ == "__main__":
    # quick smoke test on a fake recording, mostly to check nothing above
    # blows up on edge cases (constant signal -> skew/kurtosis = nan, etc.)
    fake = np.random.randn(500, 9)
    X = extract_features([fake])
    assert X.shape == (1, 76), f"expected (1, 76), got {X.shape}"
    assert len(FEATURE_NAMES) == 76
    print("features.py smoke test passed:", X.shape)
