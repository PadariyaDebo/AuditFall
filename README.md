# AuditFall

Code for *AuditFall: Explainable and Privacy-Preserving Fall Detection Using Wearable Sensors*.

Ten fall detection models (six classical ML classifiers, three feedforward
neural nets, one LSTM) trained on the SisFall wearable sensor dataset,
with SHAP/LIME explainability on the best-performing model and a
SHAP-guided privacy analysis comparing differential privacy under three
different mechanisms.

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

The dataset is [SisFall](http://sistemic.udea.edu.co/en/investigacion/proyectos/english-falls/)
(Sucerquia et al., *Sensors* 2017). We used a Kaggle mirror for convenience
during development:

```
kaggle datasets download -d kushajm/sisfall-dataset-fall-detection --unzip -p ./SisFall_dataset/
```

If you've already got it locally, just point `SISFALL_DIR` at wherever the
subject folders (`SA01`, `SA02`, ..., `SE01`, ...) live:

```bash
export SISFALL_DIR=/path/to/SisFall_dataset
```

## Running it

The easiest way is the notebook -- open `notebooks/AuditFall_full_pipeline.ipynb`
in Colab (or locally with Jupyter) and run top to bottom. It downloads the
data, extracts features, trains all ten models, runs the significance
tests, generates all SHAP/LIME figures, and does the privacy analysis,
saving everything to `results/`.

If you'd rather script it or import pieces individually, each `src/*.py`
file is importable on its own and most have a `__main__` block for a quick
smoke test:

```bash
cd src
python data_loading.py        # loads the dataset, prints counts
python features.py            # sanity-checks feature extraction on fake data
python train_ml_models.py     # trains all 6 classifiers, prints Table 2
```

## Setup

```bash
pip install -r requirements.txt
```

TensorFlow and the DP libraries (`diffprivlib`, `dp-accounting`) are the
heavy dependencies -- if you only care about the classical ML models and
explainability (no CNN/LSTM/DP), you can skip `tensorflow`,
`tensorflow-privacy`, and `dp-accounting`.

## A note on the privacy analysis

`privacy_analysis.py` includes two versions of the DP-CNN sweep:

- `run_dp_cnn_sweep` -- the original approach, where noise multiplier and
  training epochs both change together to reach each target epsilon.
- `run_dp_cnn_sweep_fixed_epochs` -- holds the epoch count (and therefore
  the total number of gradient steps) constant, and only varies the noise
  multiplier. This is the version actually used in the paper's Table 6,
  because the first approach confounds "less noise" with "more training",
  making it hard to say which one is actually driving accuracy up as
  epsilon increases.

Use the fixed-epoch version unless you have a specific reason not to.

## Participant-disjoint split check

`participant_split_check.py` isn't part of the paper's main results --
the paper follows the standard SisFall evaluation protocol (a random
75:25 split, Section 3.2.2). This script checks what happens if you
instead make sure no participant's recordings appear on both sides of the
split, since a purely random split can in principle let near-duplicate
recordings from the same person leak across the train/test boundary. We
found the effect is small (well under a percentage point on Random
Forest), which is reassuring, but we've kept the script in so it's easy
to verify or dig into further.

## Reproducibility notes

- All models use fixed random seeds (mostly `random_state=0`), matching
  what's reported in the paper.
- Hyperparameters (RF n_estimators=10, KNN k=15, SVM C=10, etc.) were not
  tuned via grid search -- they're documented choices explained in the
  paper's Section 3.4, not the result of an automated search.
- The LSTM operates on raw windowed signal (200 samples/window, 50%
  overlap) rather than the 76-d feature vector the other models use, and
  windows inherit their source recording's label -- this introduces some
  label noise near fall boundaries, discussed in the paper's limitations.

## Citation

If you use this code, please cite the paper:

```
Padariya, D. and Yu, M. "AuditFall: Explainable and Privacy-Preserving
Fall Detection Using Wearable Sensors."
```

## License

MIT -- see `LICENSE`.
