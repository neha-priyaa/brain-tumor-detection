"""Evaluate trained models on the test set. Usage:
python src/evaluate.py --models custom efficientnet
Writes outputs/{model}_confusion_matrix.png, {model}_roc_curves.png,
outputs/comparison.csv and prints a per-class report + markdown table.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

from src.config import CLASS_NAMES, MODELS_DIR, OUTPUTS_DIR
from src.preprocessing import load_datasets


def collect_predictions(model, test_ds):
    y_true, y_prob = [], []
    for x, y in test_ds:
        y_prob.append(model.predict(x, verbose=0))
        y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(y_prob)


def evaluate_model(name, test_ds, rows):
    path = os.path.join(MODELS_DIR, f"{name}.keras")
    if not os.path.exists(path):
        print(f"SKIP {name}: {path} not found (train it first)")
        return
    model = tf.keras.models.load_model(path)
    y_true, y_prob = collect_predictions(model, test_ds)
    y_pred = y_prob.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    y_onehot = np.eye(len(CLASS_NAMES))[y_true]
    auc = roc_auc_score(y_onehot, y_prob, multi_class="ovr", average="macro")
    rows.append(dict(model=name, accuracy=acc, precision_macro=prec,
                     recall_macro=rec, f1_macro=f1, roc_auc_macro=auc))

    per_cls = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {cls:12s} precision={per_cls[0][i]:.3f} "
              f"recall={per_cls[1][i]:.3f} f1={per_cls[2][i]:.3f}")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"{name} confusion matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, f"{name}_confusion_matrix.png"), dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_onehot[:, i], y_prob[:, i])
        cls_auc = roc_auc_score(y_true == i, y_prob[:, i])
        ax.plot(fpr, tpr, label=f"{cls} (AUC={cls_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"{name} ROC (one-vs-rest)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, f"{name}_roc_curves.png"), dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["custom", "efficientnet"])
    args = ap.parse_args()

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    _, _, test_ds = load_datasets()
    rows = []
    for name in args.models:
        print(f"\n=== {name} ===")
        evaluate_model(name, test_ds, rows)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUTS_DIR, "comparison.csv"), index=False)
    if not df.empty:
        print("\n" + df.to_markdown(index=False))


if __name__ == "__main__":
    main()
