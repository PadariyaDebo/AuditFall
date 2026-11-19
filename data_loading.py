"""
Loads the raw SisFall recordings from disk

"""

import os
import re
import numpy as np
import pandas as pd

from config import BASE_DIR, FALL_PREFIX, ADL_PREFIX

SUBJECT_PATTERN = re.compile(r"^(SA|SE)\d+", re.IGNORECASE)


def find_sisfall_base_dir(search_root=BASE_DIR):
    
    for dirpath, dirnames, filenames in os.walk(search_root):
        subject_dirs = [d for d in dirnames if SUBJECT_PATTERN.match(d)]
        if not subject_dirs:
            continue
        sample = os.path.join(dirpath, subject_dirs[0])
        if any(f.endswith(".txt") for f in os.listdir(sample)):
            return dirpath
    return None


def load_sisfall(base_dir, track_subjects=True, return_filenames=False):
    
    data, labels, subjects, filenames = [], [], [], []
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
                filenames.append(fname)
            except Exception:
                # a handful of SisFall files have malformed trailing rows --
                # not worth crashing the whole load over, just skip them
                skipped += 1

    print(f"Loaded: {len(data)} files | Skipped: {skipped}")
    print(f"Falls: {sum(l == 1 for l in labels)} | Non-falls: {sum(l == 0 for l in labels)}")

    if return_filenames:
        print(f"Unique subjects: {len(set(subjects))}")
        return data, labels, subjects, filenames
    if track_subjects:
        print(f"Unique subjects: {len(set(subjects))}")
        return data, labels, subjects
    return data, labels


if __name__ == "__main__":
    resolved_dir = find_sisfall_base_dir() or BASE_DIR
    print(f"Using base dir: {resolved_dir}")
    data, labels, subjects = load_sisfall(resolved_dir)



if __name__ == "__main__":
    resolved_dir = find_sisfall_base_dir() or BASE_DIR
    print(f"Using base dir: {resolved_dir}")
    data, labels, subjects = load_sisfall(resolved_dir)
