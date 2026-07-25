"""
Not part of the paper's reported results -- this is a sanity check we ran
during review to make sure the random train/test split (Section 3.2.2)
wasn't letting the same participant's recordings show up on both sides,
which would inflate accuracy relative to a real deployment where the
system meets someone it's never seen before.

Short version of what we found: participant-disjoint splitting drops RF
accuracy by about half a point (99.82% -> ~99.3%), so the random-split
numbers in the paper aren't meaningfully inflated. Included here in case
anyone wants to verify that themselves or build on it.

Needs `subjects` from data_loading.load_sisfall(track_subjects=True).
"""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def participant_disjoint_split(X, y, subjects, test_size=0.25, random_state=0):
    """GroupShuffleSplit keyed on subject ID, with a hard check that
    nobody ends up on both sides."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=subjects))

    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    subj_train, subj_test = np.array(subjects)[train_idx], np.array(subjects)[test_idx]

    overlap = set(subj_train.tolist()) & set(subj_test.tolist())
    assert not overlap, f"participant leakage: {overlap}"

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"Train: {len(X_train)} recordings, {len(set(subj_train.tolist()))} subjects")
    print(f"Test:  {len(X_test)} recordings, {len(set(subj_test.tolist()))} subjects")
    print(f"Subject overlap: {len(overlap)} (should always be 0)")

    return X_train, X_test, y_train, y_test


def compare_random_vs_disjoint(X, y, subjects, random_state=0):
    """Trains the identical RF config both ways and prints the gap. This
    is the actual leakage check -- run it, don't just eyeball the split
    sizes."""
    X_disjoint_train, X_disjoint_test, y_disjoint_train, y_disjoint_test = \
        participant_disjoint_split(X, y, subjects, random_state=random_state)

    rf_disjoint = RandomForestClassifier(n_estimators=10, min_samples_split=3,
                                          bootstrap=True, n_jobs=-1, random_state=0)
    rf_disjoint.fit(X_disjoint_train, y_disjoint_train)
    acc_disjoint = accuracy_score(y_disjoint_test, rf_disjoint.predict(X_disjoint_test))

    X_scaled = StandardScaler().fit_transform(X)
    X_rand_train, X_rand_test, y_rand_train, y_rand_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=random_state
    )
    rf_random = RandomForestClassifier(n_estimators=10, min_samples_split=3,
                                        bootstrap=True, n_jobs=-1, random_state=0)
    rf_random.fit(X_rand_train, y_rand_train)
    acc_random = accuracy_score(y_rand_test, rf_random.predict(X_rand_test))

    gap = (acc_random - acc_disjoint) * 100
    print(f"\nParticipant-disjoint split: {acc_disjoint * 100:.2f}%")
    print(f"Random split (paper's protocol): {acc_random * 100:.2f}%")
    print(f"Gap: {gap:+.2f} pp")

    return {"disjoint": acc_disjoint, "random": acc_random, "gap_pp": gap}


if __name__ == "__main__":
    print("Needs X, y, subjects already loaded -- see data_loading.py")
    print("and features.py, then call compare_random_vs_disjoint(X, y, subjects).")
