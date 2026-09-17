"""Train a model. Usage:
python src/train.py --model custom|efficientnet [--epochs N] [--dropout F] [--lr F] [--no-freeze-base]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import EPOCHS, MODELS_DIR, OUTPUTS_DIR, TRAIN_DIR, set_seeds
from src.model import get_model
from src.preprocessing import class_weights_from_dirs, load_datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["custom", "efficientnet"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-freeze-base", action="store_true")
    args = ap.parse_args()

    set_seeds()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    train_ds, val_ds, _ = load_datasets()
    class_weights = class_weights_from_dirs(TRAIN_DIR)

    kwargs = {}
    if args.model == "efficientnet":
        kwargs = dict(dropout=args.dropout, lr=args.lr,
                      freeze_base=not args.no_freeze_base)
    model = get_model(args.model, **kwargs)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, f"{args.model}.keras"),
            monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=args.epochs, class_weight=class_weights,
                        callbacks=callbacks)

    hist = {k: [float(v) for v in vs] for k, vs in history.history.items()}
    with open(os.path.join(MODELS_DIR, f"{args.model}_history.json"), "w") as f:
        json.dump(hist, f, indent=2)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist["accuracy"], label="train")
    plt.plot(hist["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(hist["loss"], label="train")
    plt.plot(hist["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, f"{args.model}_training_history.png"),
                dpi=120)
    plt.close()

    print("\nFinal training summary:")
    print(f"  best val_accuracy: {max(hist['val_accuracy']):.4f}")
    print(f"  model saved: models/{args.model}.keras")


if __name__ == "__main__":
    main()
