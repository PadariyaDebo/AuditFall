"""
Deep learning models from Section 3.4.2 / Table 4:
  - CNN (SGD)          -- plain feedforward net, SGD optimiser, no regularisation
  - CNN (Adam)          -- same architecture, Adam optimiser
  - CNN (Regularised)   -- Adam + L2 + Dropout, the one we actually recommend
  - LSTM                -- operates on raw windowed signal, not the 76-d features

Calling these "CNN" is a bit of a misnomer carried over from the paper --
they're plain dense feedforward networks, not convolutional. Kept the
naming consistent with the paper so results line up with Table 4.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def plot_history(history, title, filename=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#888888")
        ax.spines["bottom"].set_color("#888888")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax1.plot(history.history["loss"], label="Train", color="#2E5C8A", linewidth=2)
    ax1.plot(history.history["val_loss"], label="Validation", color="#D9534F", linewidth=2)
    ax1.set_title(f"Loss - {title}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(frameon=False)

    ax2.plot(history.history["accuracy"], label="Train", color="#2E5C8A", linewidth=2)
    ax2.plot(history.history["val_accuracy"], label="Validation", color="#D9534F", linewidth=2)
    ax2.set_title(f"Accuracy - {title}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend(frameon=False)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)


def build_feedforward(input_dim, regularised=False):
    if regularised:
        return Sequential([
            Dense(32, activation="relu", kernel_regularizer=l2(0.01), input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ])
    return Sequential([
        Dense(32, activation="relu", input_shape=(input_dim,)),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])


def train_cnn_variants(X_train, y_train, X_test, y_test, save_figures=True):
    """Trains all three feedforward variants and returns a dict of
    {name: (model, accuracy, history)}."""

    input_dim = X_train.shape[1]
    results = {}

    print("Training CNN with SGD...")
    model_sgd = build_feedforward(input_dim)
    model_sgd.compile(optimizer=SGD(), loss="binary_crossentropy", metrics=["accuracy"])
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    h_sgd = model_sgd.fit(X_train, y_train, epochs=100, batch_size=32,
                           validation_split=0.2, callbacks=[es], verbose=0)
    _, acc_sgd = model_sgd.evaluate(X_test, y_test, verbose=0)
    print(f"CNN (SGD) accuracy: {acc_sgd * 100:.2f}%")
    results["CNN (SGD)"] = (model_sgd, acc_sgd, h_sgd)

    print("\nTraining CNN with Adam...")
    model_adam = build_feedforward(input_dim)
    model_adam.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    h_adam = model_adam.fit(X_train, y_train, epochs=100, batch_size=32,
                             validation_split=0.2, callbacks=[es], verbose=0)
    _, acc_adam = model_adam.evaluate(X_test, y_test, verbose=0)
    print(f"CNN (Adam) accuracy: {acc_adam * 100:.2f}%")
    print("(perfect or near-perfect accuracy here is expected but should be")
    print(" read cautiously -- see the paper's discussion of overfitting risk)")
    results["CNN (Adam)"] = (model_adam, acc_adam, h_adam)

    print("\nTraining CNN with regularisation + dropout...")
    model_reg = build_feedforward(input_dim, regularised=True)
    model_reg.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    es = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    h_reg = model_reg.fit(X_train, y_train, epochs=100, batch_size=32,
                           validation_split=0.2, callbacks=[es], verbose=0)
    _, acc_reg = model_reg.evaluate(X_test, y_test, verbose=0)
    y_pred_reg = (model_reg.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    print(f"CNN (Regularised) accuracy: {acc_reg * 100:.2f}%")
    print(classification_report(y_test, y_pred_reg, target_names=["Non-Fall", "Fall"]))
    results["CNN (Regularised)"] = (model_reg, acc_reg, h_reg)

    if save_figures:
        import os
        from config import FIGURES_DIR
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plot_history(h_sgd, "CNN with SGD", os.path.join(FIGURES_DIR, "cnn_sgd_history.pdf"))
        plot_history(h_adam, "CNN with Adam", os.path.join(FIGURES_DIR, "cnn_adam_history.pdf"))
        plot_history(h_reg, "CNN with Regularisation + Dropout",
                     os.path.join(FIGURES_DIR, "cnn_regularised_history.pdf"))

    return results


def segment_signals(data, labels, window_size=200, overlap=100, n_channels=9):
    """Slices each raw recording into overlapping windows for the LSTM.
    Default matches the paper: 200 samples (1s @ 200Hz) with 50% overlap."""
    X_windows, y_windows = [], []
    step = window_size - overlap
    for signal, label in zip(data, labels):
        signal = np.nan_to_num(np.array(signal, dtype=float))
        for start in range(0, signal.shape[0] - window_size + 1, step):
            window = signal[start:start + window_size, :n_channels]
            if window.shape == (window_size, n_channels):
                X_windows.append(window)
                y_windows.append(label)
    return np.array(X_windows), np.array(y_windows)


def train_lstm(data, labels, window_size=200, overlap=100, n_channels=9, save_figures=True):
    """Segments the raw recordings into windows and trains the LSTM.
    Note this uses recording-level labels inherited by every window from
    that recording -- see the paper's limitations section (7.4) on the
    label-noise this introduces for windows near the pre/post-fall boundary."""

    print("Segmenting into windows...")
    X_seq, y_seq = segment_signals(data, labels, window_size, overlap, n_channels)
    print(f"Windows shape: {X_seq.shape}")
    print(f"Falls: {sum(y_seq == 1)} | Non-falls: {sum(y_seq == 0)}")

    # per-channel normalisation
    X_seq_norm = X_seq.copy().astype(float)
    for ch in range(n_channels):
        mu = X_seq[:, :, ch].mean()
        std = X_seq[:, :, ch].std()
        X_seq_norm[:, :, ch] = (X_seq[:, :, ch] - mu) / (std + 1e-8)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_seq_norm, y_seq, test_size=0.25, random_state=0, stratify=y_seq
    )
    print(f"Train: {len(X_tr)} | Test: {len(X_te)}")

    model_lstm = Sequential([
        LSTM(64, input_shape=(window_size, n_channels)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model_lstm.compile(optimizer=Adam(learning_rate=0.001),
                        loss="binary_crossentropy", metrics=["accuracy"])
    model_lstm.summary()

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
    print("\nTraining LSTM (this takes a while -- raw 200x9 windows, not the 76-d features)...")
    history = model_lstm.fit(X_tr, y_tr, epochs=50, batch_size=64,
                              validation_split=0.2, callbacks=[es], verbose=1)

    _, acc_lstm = model_lstm.evaluate(X_te, y_te, verbose=0)
    y_pred_lstm = (model_lstm.predict(X_te, verbose=0) > 0.5).astype(int).flatten()
    print(f"\nLSTM accuracy: {acc_lstm * 100:.2f}%")
    print(classification_report(y_te, y_pred_lstm, target_names=["Non-Fall", "Fall"]))

    if save_figures:
        import os
        from config import FIGURES_DIR
        os.makedirs(FIGURES_DIR, exist_ok=True)

        cm = confusion_matrix(y_te, y_pred_lstm)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                    xticklabels=["Non-Fall", "Fall"], yticklabels=["Non-Fall", "Fall"])
        plt.title("Confusion Matrix - LSTM")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "lstm_confusion_matrix.pdf"), bbox_inches="tight")
        plt.show()

        plot_history(history, "LSTM", os.path.join(FIGURES_DIR, "lstm_training_curves.pdf"))

    return model_lstm, acc_lstm, history


if __name__ == "__main__":
    print("This script expects X_train/y_train/X_test/y_test (for the CNNs)")
    print("and raw data/labels (for the LSTM) to already be in memory --")
    print("see notebooks/AuditFall_full_pipeline.ipynb for the full run order.")
