# AuditFall: Explainable and Privacy-Preserving Fall Detection Using Wearable Sensors

# AuditFall

This repository contains the implementation of the code for the paper "AuditFall: Explainable and Privacy-Preserving Fall Detection Using Wearable Sensors."

## About

Falls are a leading cause of injury-related mortality among adults over 75. Although wearable sensor-based systems can detect falls with high accuracy, clinical deployment also requires interpretable predictions to support clinician trust and regulatory approval, alongside privacy-preserving methods to protect sensitive movement data.

This paper benchmarks nine fall detection models on the SisFall wearable sensor dataset, comprising six classical ML classifiers, two CNN variants, and one LSTM. We use SHAP and LIME to explain the best-performing model's decisions, then use SHAP's feature importance rankings to cut the input down from 76 features to 15 with no meaningful loss in accuracy, and measure how much of that accuracy survives under differential privacy across three different DP mechanisms (DP-GNB, DP-LR, DP-CNN).


## What's in here

```
src/
  config.py                  shared constants (paths, feature names, etc.)
  data_loading.py             reads the raw SisFall .txt recordings
  features.py                  76-feature time-domain extraction
  train_ml_models.py          SVM / Decision Tree / RF / KNN / GNB / AdaBoost   -> Table 2
  train_deep_models.py        CNN (SGD/Adam/regularised) + LSTM                -> Table 4
  significance_testing.py     McNemar pairwise tests                          -> Table 3
  explainability.py           SHAP (global + local) and LIME                  -> Figures 6-11
  privacy_analysis.py         SHAP feature selection + DP-GNB/DP-LR/DP-CNN    -> Tables 5-7, Figures 12-13
```

## Getting the data

Dataset is [SisFall](http://sistemic.udea.edu.co/en/investigacion/proyectos/english-falls/)
(Sucerquia et al., *Sensors* 2017). We used a Kaggle mirror during development:

```
kaggle datasets download -d kushajm/sisfall-dataset-fall-detection --unzip -p ./SisFall_dataset/
```

Already have it locally? Point `SISFALL_DIR` at wherever the subject folders
live (`SA01`, `SA02`, ..., `SE01`, ...):

```bash
export SISFALL_DIR=/path/to/SisFall_dataset
```

## Running it

Each `src/*.py` file is importable on its own, and most have a `__main__`
block for a quick check:

```bash
cd src
python data_loading.py        # loads the dataset, prints counts
python features.py            # sanity-checks feature extraction on fake data
python train_ml_models.py     # trains all 6 classifiers, prints Table 2
```

To reproduce the full set of results (Tables 2-7, Figures 6-13), run the
scripts in order: `data_loading.py` -> `features.py` -> `train_ml_models.py`
-> `train_deep_models.py` -> `significance_testing.py` -> `explainability.py` -> `privacy_analysis.py`

## Requirements

Python 3.10 or 3.11.

## Setup

```bash
pip install -r requirements.txt
```

TensorFlow and the DP libraries (`diffprivlib`, `dp-accounting`) are the heavy
deps. 

## Reproducibility notes

- Fixed random seeds throughout (mostly `random_state=0`), matching the paper.
- Hyperparameters (RF n_estimators=10, KNN k=15, SVM C=10, etc.) weren't
  grid-searched — documented choices from Section 3.4.
- LSTM runs on raw windowed signal (200 samples/window, 50% overlap) rather
  than the 76-d feature vector everything else uses. Windows inherit their
  source recording's label, which introduces some label noise near fall
  boundaries — see limitations in the paper.

## Citation

```
Padariya, D. and Yu, M. "AuditFall: Explainable and Privacy-Preserving
Fall Detection Using Wearable Sensors."
```

## License

MIT — see `LICENSE`.


