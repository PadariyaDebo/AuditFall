"""
Loads the raw SisFall recordings off disk.

SisFall ships as one folder per subject (SA01, SA02, ..., SE01, ...),
each containing .txt files named like D01_SA01_R01.txt (activity_subject_trial).
We just walk the tree, read every file, and tag it Fall / Non-Fall based on
the activity prefix (D = fall, F/R = ADL).

A couple of Kaggle mirrors of SisFall nest the actual subject folders one
level deeper than you'd expect (SisFall_dataset/SisFall_dataset/SA01/...),
so find_sisfall_base_dir() below just searches for it instead of hardcoding
a path -- saved a lot of manual poking around in Colab.
"""

import os
import re
import numpy as np
import pandas as pd

from config import BASE_DIR, FALL_PREFIX, ADL_PREFIX

SUBJECT_PATTERN = re.compile(r"^(SA|SE)\d+", re.IGNORECASE)


def find_sisfall_base_dir(search_root=BASE_DIR):
    """Walk the extracted archive and return the folder whose direct
    children are the subject folders (SA01, SE01, ...). Returns None if
    nothing matching was found, in which case check your extraction path."""
    for dirpath, dirnames, filenames in os.walk(search_root):
        subject_dirs = [d for d in dirnames if SUBJECT_PATTERN.match(d)]
        if not subject_dirs:
            continue
        sample = os.path.join(dirpath, subject_dirs[0])
        if any(f.endswith(".txt") for f in os.listdir(sample)):
            return dirpath
    return None


def load_sisfall(base_dir, track_subjects=True):
    """Read every recording under base_dir.

    Returns (data, labels) as lists, or (data, labels, subjects) if
    track_subjects=True -- the subject IDs aren't used by the main
    pipeline (which matches the paper's random split) but are handy if
    you want to run the participant-disjoint sanity check in
    participant_split_check.py.
    """
    data, labels, subjects = [], [], []
    skipped = 0

    for subject in sorted(os.listdir(base_dir)):
        subject_path = os.path.join(base_dir, subject)
        if not os.path.isdir(subject_path):
            continue

        for fname in sorted(os.listdir(subject_path)):
            if not fname.endswith(".txt"):
                continue

            fpath = os.path.join(subject_path, fname)
            activity_code = fname[:3]  # e.g. D01, F02, R01

            if activity_code[0] == FALL_PREFIX:
                label = 1
            elif activity_code[0] in ADL_PREFIX:
                label = 0
            else:
                skipped += 1
                continue

            try:
                df = pd.read_csv(fpath, header=None)
                df[8] = df[8].astype(str).str.replace(";", "").astype(float)
                data.append(df.values.astype(float))
                labels.append(label)
                subjects.append(subject)
            except Exception:
                # a handful of SisFall files have malformed trailing rows --
                # not worth crashing the whole load over, just skip them
                skipped += 1

    print(f"Loaded: {len(data)} files | Skipped: {skipped}")
    print(f"Falls: {sum(l == 1 for l in labels)} | Non-falls: {sum(l == 0 for l in labels)}")

    if track_subjects:
        print(f"Unique subjects: {len(set(subjects))}")
        return data, labels, subjects
    return data, labels


if __name__ == "__main__":
    resolved_dir = find_sisfall_base_dir() or BASE_DIR
    print(f"Using base dir: {resolved_dir}")
    data, labels, subjects = load_sisfall(resolved_dir)
