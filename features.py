"""
Feature extraction for the AuditFall pipeline.

Computes 76 time-domain statistical features from each recording.
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
