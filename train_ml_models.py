"""
Training functions for the classical ML models used in AuditFall.

Includes SVM, Decision Tree, Random Forest, KNN, Gaussian NB, and AdaBoost.

"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from config import CV_FOLDS, TEST_SIZE, RANDOM_STATE


def show_results(y_true, y_pred, name, plot=True):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'=' * 55}")
    print(f"{name}  -  Accuracy: {acc * 100:.2f}%")
    print(f"{'=' * 55}")
    print(classification_report(y_true, y_pred, target_names=["Non-Fall", "Fall"]))

    if plot:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Non-Fall", "Fall"], yticklabels=["Non-Fall", "Fall"])
        plt.title(f"Confusion Matrix - {name}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.show()

    return acc


def train_all_ml_models(X_scaled, X_train, X_test, y_train, y_test, plot=True):
    """Trains all six classifiers and returns everything downstream code
    needs: fitted models, predictions, and a results dataframe matching
    Table 2's layout (test acc, CV acc, precision, recall, per-class F1,
    macro F1)."""

    models = {}
    predictions = {}
    rows = []

    def macro_f1(y_true, y_pred):
        return f1_score(y_true, y_pred, average="macro")

    # 1. SVM
    print("Running SVM...")
    svm = LinearSVC(C=10, max_iter=5000, random_state=42)
    svm.fit(X_train, y_train)
    pred_svm = svm.predict(X_test)
    acc_svm = show_results(y_test, pred_svm, "SVM", plot=plot)
    models["SVM"] = svm
    predictions["SVM"] = pred_svm

    # 2. Decision Tree
    print("\nRunning Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=6, random_state=10)
    dt.fit(X_train, y_train)
    pred_dt = dt.predict(X_test)
    acc_dt = show_results(y_test, pred_dt, "Decision Tree", plot=plot)
    models["Decision Tree"] = dt
    predictions["Decision Tree"] = pred_dt

    # 3. Random Forest
    print("\nRunning Random Forest...")
    rf = RandomForestClassifier(n_estimators=10, min_samples_split=3,
                                 bootstrap=True, n_jobs=-1, random_state=0)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    acc_rf = show_results(y_test, pred_rf, "Random Forest", plot=plot)
    models["Random Forest"] = rf
    predictions["Random Forest"] = pred_rf

    # 4. KNN
    print("\nRunning KNN...")
    knn = KNeighborsClassifier(n_neighbors=15, algorithm="ball_tree", n_jobs=-1)
    knn.fit(X_train, y_train)
    pred_knn = knn.predict(X_test)
    acc_knn = show_results(y_test, pred_knn, "KNN", plot=plot)
    models["KNN"] = knn
    predictions["KNN"] = pred_knn

    # 5. Gaussian Naive Bayes
    print("\nRunning Gaussian Naive Bayes...")
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    pred_gnb = gnb.predict(X_test)
    acc_gnb = show_results(y_test, pred_gnb, "Gaussian NB", plot=plot)
    models["Gaussian NB"] = gnb
    predictions["Gaussian NB"] = pred_gnb

    # 6. AdaBoost
    print("\nRunning AdaBoost...")
    ada = AdaBoostClassifier(random_state=0)
    ada.fit(X_train, y_train)
    pred_ada = ada.predict(X_test)
    acc_ada = show_results(y_test, pred_ada, "AdaBoost", plot=plot)
    models["AdaBoost"] = ada
    predictions["AdaBoost"] = pred_ada

    for name in models:
        pred = predictions[name]
        cv_acc = cross_val_score(models[name], X_scaled,
                                  np.concatenate([y_train, y_test]),
                                  cv=CV_FOLDS, n_jobs=-1).mean()
        rows.append({
            "Model": name,
            "Test Acc.": accuracy_score(y_test, pred),
            "CV 5-Fold": cv_acc,
            "Precision": classification_report(y_test, pred, output_dict=True)["weighted avg"]["precision"],
            "Recall": classification_report(y_test, pred, output_dict=True)["weighted avg"]["recall"],
            "F1 Fall": classification_report(y_test, pred, target_names=["Non-Fall", "Fall"], output_dict=True)["Fall"]["f1-score"],
            "F1 ADL": classification_report(y_test, pred, target_names=["Non-Fall", "Fall"], output_dict=True)["Non-Fall"]["f1-score"],
            "Macro F1": macro_f1(y_test, pred),
        })

    results_df = pd.DataFrame(rows).sort_values("Test Acc.", ascending=False).reset_index(drop=True)
    return models, predictions, results_df


if __name__ == "__main__":
    from data_loading import find_sisfall_base_dir, load_sisfall
    from features import extract_features
    from config import BASE_DIR

    base_dir = find_sisfall_base_dir() or BASE_DIR
    data, labels, _ = load_sisfall(base_dir)
    y = np.array(labels)

    print("Extracting features...")
    X = extract_features(data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    models, predictions, results_df = train_all_ml_models(
        X_scaled, X_train, X_test, y_train, y_test, plot=False
    )
    print("\n" + "=" * 55)
    print("ML RESULTS SUMMARY (Table 2)")
    print("=" * 55)
    print(results_df.to_string(index=False))
