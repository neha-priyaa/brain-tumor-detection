"""Grid search on EfficientNetB0: lr x dropout x frozen-base (8 runs, reduced epochs).

Usage: python src/tune.py [--epochs 8]
Writes outputs/tuning_results.csv and prints the best config as a train.py command.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd
import tensorflow as tf

from src.config import MODELS_DIR, OUTPUTS_DIR, TRAIN_DIR, set_seeds
from src.model import build_efficientnet
from src.preprocessing import class_weights_from_dirs, load_datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()

    set_seeds()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    train_ds, val_ds, _ = load_datasets()
    class_weights = class_weights_from_dirs(TRAIN_DIR)

    rows = []
    for lr in (1e-3, 1e-4):
        for dropout in (0.2, 0.5):
            for freeze in (True, False):
                tag = f"lr{lr}_do{dropout}_frozen{int(freeze)}"
                print(f"\n=== {tag} ===")
                set_seeds()
                model = build_efficientnet(dropout=dropout, freeze_base=freeze, lr=lr)
                hist = model.fit(
                    train_ds, validation_data=val_ds, epochs=args.epochs,
                    class_weight=class_weights, verbose=2,
                    callbacks=[tf.keras.callbacks.EarlyStopping(
                        monitor="val_accuracy", patience=3,
                        restore_best_weights=True)])
                val_acc = max(hist.history["val_accuracy"])
                rows.append(dict(config=tag, lr=lr, dropout=dropout,
                                 freeze_base=freeze, val_accuracy=float(val_acc)))
                del model
                tf.keras.backend.clear_session()

    df = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False)
    df.to_csv(os.path.join(OUTPUTS_DIR, "tuning_results.csv"), index=False)
    best = df.iloc[0]
    print("\nBest config:")
    cmd = (f"python src/train.py --model efficientnet --lr {best.lr} "
           f"--dropout {best.dropout}")
    if not best.freeze_base:
        cmd += " --no-freeze-base"
    print(cmd)


if __name__ == "__main__":
    main()
