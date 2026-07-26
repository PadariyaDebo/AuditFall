# AuditFall: Explainable and Privacy-Preserving Fall Detection Using Wearable Sensors

Ten fall detection models (six classical ML classifiers, three feedforward
neural networks, one LSTM) trained on the SisFall wearable sensor dataset. Includes
SHAP/LIME explainability on the best model and a SHAP-guided privacy analysis
comparing DP under three mechanisms.

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
  participant_split_check.py  optional leakage sanity check (see below)

results/
  figures/     saved plots (populated when you run the pipeline)
  tables/      saved CSVs matching the paper's tables
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
-> `train_deep_models.py` -> `significance_testing.py` -> `explainability.py`
-> `privacy_analysis.py`. Each writes its outputs to `results/tables/` or
`results/figures/` as it goes.

## Requirements

Python 3.10 or 3.11.

## Setup

```bash
pip install -r requirements.txt
```

TensorFlow and the DP libraries (`diffprivlib`, `dp-accounting`) are the heavy
deps. Skip `tensorflow`, `tensorflow-privacy`, `dp-accounting` if you only
want the classical models + explainability.

## Privacy analysis — a note on the DP-CNN sweep

`privacy_analysis.py` has two versions:

- `run_dp_cnn_sweep` — noise multiplier and epochs both change to hit each
  target epsilon.
- `run_dp_cnn_sweep_fixed_epochs` — epochs fixed, only noise multiplier
  varies. This is what's actually used for Table 6. The first version
  confounds "less noise" with "more training" so you can't really tell
  which one is pushing accuracy up as epsilon increases.

Use the fixed-epoch version unless you've got a reason not to.

## Participant-disjoint split check

Not part of the paper's main results — the paper uses the standard SisFall
protocol, a random 75:25 split (Section 3.2.2). `participant_split_check.py`
checks what happens if no participant's recordings appear on both sides of
the split instead (random splits can technically let near-duplicate
recordings leak across train/test). Effect was small, well under a point on
RF. Kept the script in anyway in case anyone wants to dig further.

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


