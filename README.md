# AuditFall: Explainable and Privacy-Preserving Fall Detection Using Wearable Sensors

Code for the paper "AuditFall: Explainable and Privacy-Preserving Fall Detection Using Wearable Sensors" (D. Padariya and M. Yu).

## About

Falls are a leading cause of injury-related death among older adults. Wearable sensor-based systems can detect falls with high accuracy, but clinical deployment also requires interpretable predictions and privacy-preserving training.

This repository benchmarks fall detection models on the SisFall wearable sensor dataset (six classical classifiers, two multilayer perceptrons and an LSTM), evaluates them on **previously unseen participants**, explains the selected model with SHAP and LIME, selects a compact feature subset using **training participants only**, and measures the privacy–utility trade-off of four differentially private models (DP-GNB, DP-LR with objective perturbation, and logistic regression and an MLP trained with DP-SGD).

## Methodology

- **Labels follow the SisFall definition:** F01–F15 = falls (1,798 recordings), D01–D19 = activities of daily living (2,707 recordings).
- **Participant-disjoint evaluation:** 5-fold `StratifiedGroupKFold` on participant IDs and a participant-disjoint holdout split, plus leave-one-subject-out for Random Forest. No participant appears in both training and test data.
- **Leakage-free feature selection:** SHAP rankings and the number of features are chosen on training participants only (nested cross-validation).
- **Common unit of evaluation:** LSTM window predictions are aggregated per recording; a fixed 4 s peak-centred window is also evaluated.
- **Differential privacy:** data-independent feature bounds; the same ε grid for all models; DP-SGD with Poisson sampling and a Rényi DP accountant; logistic regression trained with both objective perturbation and DP-SGD to separate the effect of the privacy mechanism from model capacity.
- **Reproducibility:** fixed seeds (Python, NumPy, TensorFlow) and deterministic TensorFlow operations; McNemar tests report their 2×2 tables.


## Repository structure

```
config.py                              shared constants (paths, labels, feature names)
data_loading.py                        reads the raw SisFall .txt recordings
features.py                            76-feature time-domain extraction
significance_testing.py                McNemar tests (with 2x2 contingency tables)
explainability.py                      SHAP (global and local) and LIME plotting
common.py                              shared helpers: cached features, participant-disjoint splits, models, metrics

random_split_baseline.py               recording-level random split (comparison with earlier studies) and McNemar tests
grouped_evaluation.py                  participant-grouped 5-fold CV (optional --loso), McNemar on pooled predictions
explainability_feature_selection.py    training-only SHAP feature selection (nested CV) and SHAP/LIME figures
differential_privacy.py                DP-GNB, DP-LR (objective perturbation), DP-SGD-LR, DP-SGD-MLP
window_lstm_evaluation.py              4 s window features; LSTM evaluated per recording
plot_shap_waterfalls.py                redraws the SHAP waterfall figures from the cached results
```

## Getting the data

Dataset: [SisFall](http://sistemic.udea.edu.co/en/investigacion/proyectos/english-falls/) (Sucerquia et al., *Sensors* 2017, 17(1), 198, doi:10.3390/s17010198). 
```bash
kaggle datasets download -d kushajm/sisfall-dataset-fall-detection --unzip -p ./SisFall_dataset/
```

Point `SISFALL_DIR` at the folder containing the subject folders (`SA01`, ..., `SE15`); nested folders are found automatically:

```bash
export SISFALL_DIR=/path/to/SisFall_dataset
```

## Running it

```bash
pip install -r requirements.txt

python random_split_baseline.py
python grouped_evaluation.py --loso
python explainability_feature_selection.py   
python differential_privacy.py               
python window_lstm_evaluation.py             
python plot_shap_waterfalls.py              
```


## Requirements

Python 3.10+. 
Versions used for the reported results: scikit-learn 1.6.1, TensorFlow 2.21.0, diffprivlib 0.6.6.

## Reproducibility notes

- All splits are seeded; later scripts reuse the folds (`results/grouped_cv/fold_assignments.csv`) and the holdout (`results/cache/holdout_top15.npz`).
- Hyperparameters (RF n_estimators=10, KNN k=15, SVM C=10, etc.) 
- ε covers model training only; the training-set standardisation statistics and SHAP feature selection are computed non-privately.
- The LSTM uses raw 200-sample windows (1 s at 200 Hz, 50% overlap); windows inherit their recording's label.

## Citation

```
Padariya, D. and Yu, M. "AuditFall: Explainable and Privacy-Preserving
Fall Detection Using Wearable Sensors."
```

## License

MIT; see `LICENSE`.
