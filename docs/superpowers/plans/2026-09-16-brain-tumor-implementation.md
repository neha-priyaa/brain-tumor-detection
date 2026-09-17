# Brain Tumor MRI Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify brain MRI images into 4 classes (glioma, meningioma, notumor, pituitary) with a custom CNN and EfficientNetB0 transfer learning, evaluate both, and serve a Streamlit demo — per the approved design spec.

**Architecture:** Single config-driven pipeline under `src/`, model factory pattern, tf.data pipelines, scripted data prep, pytest test suite. Trained `.keras` models are committed so the repo ships with a working model.

**Tech Stack:** Python 3.12, TensorFlow (tensorflow-macos + tensorflow-metal on Apple Silicon), scikit-learn (metrics), Streamlit, matplotlib, pytest.

**Reference spec:** `docs/superpowers/specs/2026-09-15-brain-tumor-detection-design.md`

---

### Task 1: Project scaffolding + environment

**Files:**
- Create: `.gitignore`, `requirements.txt`, `README.md`, `src/__init__.py`

- [ ] **Step 1: Create .gitignore**

```gitignore
__pycache__/
*.pyc
.venv/
venv/
data/raw/
.ipynb_checkpoints/
.DS_Store
outputs/*.png
```

Note: `models/*.keras` is intentionally NOT ignored — the trained model is committed to git (user requirement).

- [ ] **Step 2: Create requirements.txt**

```
tensorflow==2.16.2
scikit-learn==1.5.2
matplotlib==3.9.2
streamlit==1.39.0
pillow==10.4.0
pandas==2.2.3
pytest==8.3.3
```

(If tensorflow 2.16.2 is unavailable for the platform, fall back to latest TF ≥2.16 available via pip; record actual version in requirements.txt.)

- [ ] **Step 3: Create venv and install**

```bash
cd ~/Desktop/college-projects/brain-tumor-detection
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Expected: all packages install successfully; `python -c "import tensorflow"` works.

- [ ] **Step 4: Create README.md with disclaimer header**

```markdown
# Brain Tumor MRI Classification

> **DISCLAIMER:** This is an educational machine-learning project and is NOT a
> medical diagnostic system. It must never be presented or used as one.

Classifies brain MRI images into 4 classes: **glioma, meningioma, notumor, pituitary**
using a custom CNN and EfficientNetB0 transfer learning.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data setup (one-time)

1. Download the ZIP from https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
2. `python scripts/prepare_data.py --zip ~/Downloads/archive.zip`

## Train

```bash
python src/train.py --model efficientnet   # best model
python src/train.py --model custom         # baseline
```

## Evaluate

```bash
python src/evaluate.py --models custom efficientnet
```

## Predict a single image

```bash
python src/predict.py --model efficientnet --image path/to/mri.jpg
```

## Demo app

```bash
streamlit run app/streamlit_app.py
```

## Tests

```bash
pytest -q
```
```

- [ ] **Step 5: Commit**

```bash
git add .gitignore requirements.txt README.md
git commit -m "chore: scaffold project, requirements, README with disclaimer"
```

---

### Task 2: Config module

**Files:**
- Create: `src/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_config.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import CLASS_NAMES, SEED, IMAGE_SIZE, BATCH_SIZE, set_seeds


def test_class_names_exact_order():
    assert CLASS_NAMES == ["glioma", "meningioma", "notumor", "pituitary"]


def test_constants():
    assert SEED == 42
    assert IMAGE_SIZE == (224, 224)
    assert BATCH_SIZE == 32


def test_set_seeds_deterministic():
    import numpy as np
    set_seeds()
    a = np.random.rand(10)
    set_seeds()
    b = np.random.rand(10)
    assert (a == b).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 3: Write implementation**

```python
# src/config.py
"""Central configuration: paths, constants, and seed control."""
import os
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR = os.path.join(DATA_DIR, "Testing")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 25

DISCLAIMER = (
    "EDUCATIONAL PROJECT ONLY - NOT A MEDICAL DIAGNOSTIC SYSTEM. "
    "Never use this output for medical decisions."
)


def set_seeds():
    random.seed(SEED)
    import numpy as np
    np.random.seed(SEED)
    try:
        import tensorflow as tf
        tf.random.set_seed(SEED)
    except ImportError:
        pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
mkdir -p models outputs tests
touch src/__init__.py tests/__init__.py
git add src/config.py src/__init__.py tests/ models/.gitkeep outputs/.gitkeep
git commit -m "feat: config module with class names, paths, seed control"
```

---

### Task 3: Preprocessing module

**Files:**
- Create: `src/preprocessing.py`
- Test: `tests/test_preprocessing.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_preprocessing.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def test_label_mapping():
    from src.preprocessing import label_from_dirname
    assert label_from_dirname("glioma") == 0
    assert label_from_dirname("pituitary") == 3


def test_preprocess_single_image():
    from PIL import Image
    from src.preprocessing import preprocess_image
    import tempfile

    # grayscale 100x100 image -> 224x224x3 float in [0,1]
    img = Image.new("L", (100, 100), 128)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name)
        arr = preprocess_image(f.name)
        os.unlink(f.name)
    assert arr.shape == (224, 224, 3)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0 and arr.max() <= 1.0


def test_build_augmenter_output_range():
    import tensorflow as tf
    from src.preprocessing import build_augmenter
    aug = build_augmenter()
    x = tf.random.uniform((1, 224, 224, 3))
    out = aug(x, training=True)
    assert out.shape == (1, 224, 224, 3)
    assert float(tf.reduce_min(out)) >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_preprocessing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.preprocessing'`

- [ ] **Step 3: Write implementation**

```python
# src/preprocessing.py
"""Data loading: tf.data pipelines and single-image preprocessing."""
import os

import numpy as np
import tensorflow as tf

from src.config import CLASS_NAMES, IMAGE_SIZE, BATCH_SIZE, TRAIN_DIR, TEST_DIR, SEED


def label_from_dirname(dirname):
    return CLASS_NAMES.index(dirname)


def preprocess_image(path):
    """Load an image file -> (224,224,3) float32 array scaled to [0,1]."""
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img.numpy()


def build_augmenter():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="augmenter")


def _make_dataset(directory, training=False):
    """ImageFolder-style dataset from directory with one subdir per class."""
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=training,
    )
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
                num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(lambda x, y: (build_augmenter()(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
    else:
        ds = ds.cache().prefetch(tf.data.AUTOTUNE)
    return ds


def load_datasets():
    """Return (train, val, test) datasets. Val is carved from Training 90/10."""
    # split files 90/10 within each class subdir via image_dataset_from_directory validation_split
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, labels="inferred", label_mode="int", class_names=CLASS_NAMES,
        color_mode="rgb", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        seed=SEED, validation_split=0.1, subset="training", shuffle=True)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, labels="inferred", label_mode="int", class_names=CLASS_NAMES,
        color_mode="rgb", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        seed=SEED, validation_split=0.1, subset="validation", shuffle=True)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, labels="inferred", label_mode="int", class_names=CLASS_NAMES,
        color_mode="rgb", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        shuffle=False)

    def normalize(ds):
        return ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
                      num_parallel_calls=tf.data.AUTOTUNE)

    augmenter = build_augmenter()
    train_ds = (normalize(train_ds)
                .map(lambda x, y: (augmenter(x, training=True), y),
                     num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))
    val_ds = normalize(val_ds).cache().prefetch(tf.data.AUTOTUNE)
    test_ds = normalize(test_ds).cache().prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, test_ds


def class_weights_from_dirs(directory):
    """Inverse-frequency class weights dict {0: w, ...} from subdir file counts."""
    counts = []
    for name in CLASS_NAMES:
        d = os.path.join(directory, name)
        n = len([f for f in os.listdir(d)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        counts.append(n)
    total = sum(counts)
    weights = {i: total / (len(counts) * c) for i, c in enumerate(counts)}
    return weights
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_preprocessing.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: preprocessing with tf.data pipelines, augmenter, single-image prep"
```

---

### Task 4: Model factory

**Files:**
- Create: `src/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_model.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_custom_cnn_shape():
    from src.model import build_custom_cnn
    m = build_custom_cnn()
    assert m.input_shape == (None, 224, 224, 3)
    assert m.output_shape == (None, 4)
    assert m.optimizer is not None


def test_efficientnet_shape_and_freeze():
    from src.model import build_efficientnet
    m = build_efficientnet(dropout=0.2, freeze_base=True)
    assert m.input_shape == (None, 224, 224, 3)
    assert m.output_shape == (None, 4)
    base = [l for l in m.layers if l.name.startswith("efficientnetb0")]
    assert base and base[0].trainable is False
    m2 = build_efficientnet(dropout=0.2, freeze_base=False)
    base2 = [l for l in m2.layers if l.name.startswith("efficientnetb0")]
    assert base2 and base2[0].trainable is True


def test_factory_unknown_raises():
    from src.model import get_model
    try:
        get_model("resnet")
        assert False, "should raise"
    except ValueError:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.model'`

- [ ] **Step 3: Write implementation**

```python
# src/model.py
"""Model factory: custom CNN baseline and EfficientNetB0 transfer learning."""
import tensorflow as tf

from src.config import CLASS_NAMES


def build_custom_cnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ], name="custom_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_efficientnet(dropout=0.2, freeze_base=True, lr=1e-3):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base.trainable = not freeze_base
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="efficientnetb0_head")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def get_model(name, **kwargs):
    if name == "custom":
        return build_custom_cnn()
    if name == "efficientnet":
        return build_efficientnet(**kwargs)
    raise ValueError(f"Unknown model name: {name!r} (use 'custom' or 'efficientnet')")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_model.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat: model factory with custom CNN and EfficientNetB0"
```

---

### Task 5: Data preparation script

**Files:**
- Create: `scripts/prepare_data.py`

- [ ] **Step 1: Write implementation**

```python
# scripts/prepare_data.py
"""Extract Kaggle ZIP into data/raw/ and verify structure. Idempotent.

Usage: python scripts/prepare_data.py --zip ~/Downloads/archive.zip
"""
import argparse
import os
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import CLASS_NAMES, DATA_DIR

EXPECTED_DIRS = {"Training", "Testing"}


def verify(data_dir):
    ok = True
    for split in ("Training", "Testing"):
        for cls in CLASS_NAMES:
            d = os.path.join(data_dir, split, cls)
            if not os.path.isdir(d):
                print(f"MISSING: {d}")
                ok = False
                continue
            n = len([f for f in os.listdir(d)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            print(f"{split}/{cls}: {n} images")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to Kaggle archive.zip")
    args = ap.parse_args()

    if not os.path.isfile(args.zip):
        sys.exit(f"ERROR: ZIP not found: {args.zip}")

    os.makedirs(DATA_DIR, exist_ok=True)

    corrupted = 0
    with zipfile.ZipFile(args.zip) as z:
        names = z.namelist()
        # sanity: expect Training/ and Testing/ at some level
        if not any(EXPECTED_DIRS.issubset(set(n.split("/")[:2])) for n in names):
            sys.exit("ERROR: ZIP does not contain expected Training/Testing layout")
        for info in z.infolist():
            if info.is_dir():
                continue
            target = os.path.join(DATA_DIR, info.filename)
            if os.path.isfile(target):
                continue  # idempotent: already extracted
            os.makedirs(os.path.dirname(target), exist_ok=True)
            try:
                with z.open(info) as src, open(target, "wb") as dst:
                    dst.write(src.read())
            except Exception:
                corrupted += 1

    if corrupted:
        print(f"WARNING: {corrupted} files failed to extract")

    if not verify(DATA_DIR):
        sys.exit("ERROR: extracted data failed verification")
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify on the real ZIP**

Run: `.venv/bin/python scripts/prepare_data.py --zip ~/Downloads/archive.zip`
Expected: per-class image counts printed (~5,712 train / 1,311 test), "Data preparation complete."

- [ ] **Step 3: Commit**

```bash
git add scripts/prepare_data.py
git commit -m "feat: idempotent data preparation script with verification"
```

---

### Task 6: EDA notebook

**Files:**
- Create: `notebooks/01_eda.ipynb`

- [ ] **Step 1: Write notebook** (cells)

Cell 1 (markdown): `# Brain MRI EDA — class distribution and samples` + disclaimer.
Cell 2 (code):

```python
import os
import matplotlib.pyplot as plt
from PIL import Image

from src.config import CLASS_NAMES, TRAIN_DIR

counts = {}
for cls in CLASS_NAMES:
    d = os.path.join(TRAIN_DIR, cls)
    counts[cls] = len([f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
plt.figure(figsize=(6, 4))
plt.bar(counts.keys(), counts.values())
plt.title("Training set class distribution")
plt.ylabel("images")
plt.tight_layout()
plt.savefig("../outputs/eda_class_distribution.png")
plt.show()
counts
```

Cell 3 (code):

```python
import random
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, cls in zip(axes, CLASS_NAMES):
    d = os.path.join(TRAIN_DIR, cls)
    f = random.choice(os.listdir(d))
    ax.imshow(Image.open(os.path.join(d, f)))
    ax.set_title(cls)
    ax.axis("off")
plt.suptitle("Sample MRI per class")
plt.tight_layout()
plt.savefig("../outputs/eda_samples.png")
plt.show()
```

- [ ] **Step 2: Run notebook end to end**

Run: `.venv/bin/jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --inplace`
Expected: executes without error, `outputs/eda_class_distribution.png` and `outputs/eda_samples.png` created.

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_eda.ipynb
git commit -m "feat: EDA notebook with class distribution and sample images"
```

---

### Task 7: Train script

**Files:**
- Create: `src/train.py`

- [ ] **Step 1: Write implementation**

```python
# src/train.py
"""Train a model. Usage:
python src/train.py --model custom|efficientnet [--epochs N] [--dropout F] [--lr F] [--no-freeze-base]
"""
import argparse
import json
import os

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

    name = args.model
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, f"{name}.keras"),
            monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=args.epochs, class_weight=class_weights,
                        callbacks=callbacks)

    hist = {k: [float(v) for v in vs] for k, vs in history.history.items()}
    with open(os.path.join(MODELS_DIR, f"{name}_history.json"), "w") as f:
        json.dump(hist, f, indent=2)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist["accuracy"], label="train")
    plt.plot(hist["val_accuracy"], label="val")
    plt.title("Accuracy"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(hist["loss"], label="train")
    plt.plot(hist["val_loss"], label="val")
    plt.title("Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, f"{name}_training_history.png"))

    print("\nFinal training summary:")
    print(f"  best val_accuracy: {max(hist['val_accuracy']):.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test on 1 epoch**

Run: `.venv/bin/python src/train.py --model custom --epochs 1`
Expected: completes; `models/custom.keras` and `models/custom_history.json` created.

- [ ] **Step 3: Commit**

```bash
git add src/train.py
git commit -m "feat: train script with early stopping, checkpointing, history plots"
```

---

### Task 8: Tuning script

**Files:**
- Create: `src/tune.py`

- [ ] **Step 1: Write implementation**

```python
# src/tune.py
"""Grid search on EfficientNetB0: lr x dropout x frozen-base (8 runs, reduced epochs).

Usage: python src/tune.py [--epochs 8]
Writes outputs/tuning_results.csv and prints the best config as a train.py command.
"""
import argparse
import os

import pandas as pd

from src.config import MODELS_DIR, OUTPUTS_DIR, set_seeds
from src.model import build_efficientnet
from src.preprocessing import class_weights_from_dirs, load_datasets
from src.config import TRAIN_DIR

import tensorflow as tf


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
```

- [ ] **Step 2: Run the grid**

Run: `.venv/bin/python src/tune.py --epochs 8` (long — run in background)
Expected: `outputs/tuning_results.csv` with 8 rows, best-config train.py command printed.

- [ ] **Step 3: Commit**

```bash
git add src/tune.py
git commit -m "feat: EfficientNet grid-search tuning script"
```

---

### Task 9: Evaluate script

**Files:**
- Create: `src/evaluate.py`

- [ ] **Step 1: Write implementation**

```python
# src/evaluate.py
"""Evaluate trained models on the test set.

Usage: python src/evaluate.py --models custom efficientnet
Writes outputs/{model}_confusion_matrix.png, {model}_roc_curves.png,
outputs/comparison.csv and prints a markdown table.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)

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
        print(f"SKIP {name}: {path} not found (train first)")
        return
    model = tf.keras.models.load_model(path)
    y_true, y_prob = collect_predictions(model, test_ds)
    y_pred = y_prob.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(np.eye(len(CLASS_NAMES))[y_true], y_prob,
                            multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    rows.append(dict(model=name, accuracy=acc, precision_macro=prec,
                     recall_macro=rec, f1_macro=f1, roc_auc_macro=auc))

    # per-class table
    per_cls = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0)
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {cls:12s} precision={per_cls[0][i]:.3f} recall={per_cls[1][i]:.3f} f1={per_cls[2][i]:.3f}")

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues")
    plt.title(f"{name} confusion matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, f"{name}_confusion_matrix.png"))
    plt.close()

    # ROC curves (one-vs-rest)
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
        ax.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc_score(y_true == i, y_prob[:, i]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.5)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"{name} ROC (one-vs-rest)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, f"{name}_roc_curves.png"))
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
    print("\n" + df.to_markdown(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run against trained models**

Run: `.venv/bin/python src/evaluate.py --models custom efficientnet`
Expected: outputs/comparison.csv + per-model PNGs; markdown table printed.

- [ ] **Step 3: Commit**

```bash
git add src/evaluate.py
git commit -m "feat: evaluation with confusion matrices, ROC curves, comparison table"
```

---

### Task 10: Predict module

**Files:**
- Create: `src/predict.py`
- Test: `tests/test_predict.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_predict.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """Tiny compiled model saved as models dir stand-in."""
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    d = tmp_path_factory.mktemp("models")
    m.save(os.path.join(str(d), "tiny.keras"))
    return str(d)


def test_predict_image(tiny_model, monkeypatch, tmp_path):
    from src import predict as predict_mod

    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    from PIL import Image
    p = tmp_path / "mri.png"
    Image.fromarray(img).save(p)

    result = predict_mod.predict_image(str(p), "tiny", models_dir=tiny_model)
    assert result["class"] in ["glioma", "meningioma", "notumor", "pituitary"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["probabilities"]) == 4
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5


def test_predict_missing_model(tiny_model, tmp_path):
    from src import predict as predict_mod
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    from PIL import Image
    p = tmp_path / "mri.png"
    Image.fromarray(img).save(p)
    with pytest.raises(FileNotFoundError):
        predict_mod.predict_image(str(p), "nonexistent", models_dir=tiny_model)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_predict.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.predict'`

- [ ] **Step 3: Write implementation**

```python
# src/predict.py
"""Single-image prediction. Usage:
python src/predict.py --model efficientnet --image path/to/mri.jpg
"""
import argparse
import os

import numpy as np

from src.config import CLASS_NAMES, DISCLAIMER, MODELS_DIR
from src.preprocessing import preprocess_image


def predict_image(image_path, model_name, models_dir=None):
    """Returns {class, confidence, probabilities}. Raises FileNotFoundError
    if the model file does not exist."""
    models_dir = models_dir or MODELS_DIR
    model_path = os.path.join(models_dir, f"{model_name}.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train it first with src/train.py")
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    arr = preprocess_image(image_path)
    probs = model.predict(arr[np.newaxis, ...], verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "class": CLASS_NAMES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    print(DISCLAIMER)
    result = predict_image(args.image, args.model)
    print(f"\nPrediction: {result['class']} ({result['confidence']:.1%})")
    for c, p in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {c:12s} {p:.1%}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_predict.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/predict.py tests/test_predict.py
git commit -m "feat: predict module with graceful missing-model handling"
```

---

### Task 11: Streamlit app

**Files:**
- Create: `app/streamlit_app.py`

- [ ] **Step 1: Write implementation**

```python
# app/streamlit_app.py
"""Streamlit demo. Run: streamlit run app/streamlit_app.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from src.config import DISCLAIMER, MODELS_DIR
from src.predict import predict_image

st.set_page_config(page_title="Brain MRI Classifier (Educational)", page_icon="🧠")

st.warning(f"**{DISCLAIMER}**", icon="⚠️")

st.title("Brain Tumor MRI Classification")
st.sidebar.header("Settings")
disclaimer_sidebar = st.sidebar.markdown(f"> ⚠️ {DISCLAIMER}")

available = sorted(
    f[:-6] for f in os.listdir(MODELS_DIR) if f.endswith(".keras")
) if os.path.isdir(MODELS_DIR) else []

if not available:
    st.info("No trained models found in models/. Train one first: "
            "`python src/train.py --model efficientnet`")
    st.stop()

model_name = st.sidebar.selectbox("Model", available)

uploaded = st.file_uploader("Upload a brain MRI image",
                            type=["jpg", "jpeg", "png", "bmp"])
if uploaded is not None:
    st.image(uploaded, caption=uploaded.name, width=300)
    if st.button("Predict"):
        tmp_path = os.path.join("/tmp", uploaded.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        try:
            with st.spinner("Predicting..."):
                result = predict_image(tmp_path, model_name)
            st.success(f"**{result['class']}** — confidence {result['confidence']:.1%}")
            probs = result["probabilities"]
            st.bar_chart(probs)
            st.caption(DISCLAIMER)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
```

- [ ] **Step 2: Smoke-test import**

Run: `.venv/bin/python -c "import ast; ast.parse(open('app/streamlit_app.py').read())"`
Expected: no output (parses).

- [ ] **Step 3: Commit**

```bash
mkdir -p app
git add app/streamlit_app.py
git commit -m "feat: Streamlit demo app with disclaimer and model picker"
```

---

### Task 12: Full test run + final training runs

- [ ] **Step 1: Run full test suite**

Run: `.venv/bin/pytest -q`
Expected: all tests PASS.

- [ ] **Step 2: Run tuning grid** (background, long)

Run: `.venv/bin/python src/tune.py --epochs 8`
Expected: outputs/tuning_results.csv; note best config command.

- [ ] **Step 3: Train EfficientNet with best config** (background, long)

Run the command printed by tune.py, e.g.:
`.venv/bin/python src/train.py --model efficientnet --lr <best_lr> --dropout <best_do>`
Expected: models/efficientnet.keras with best val accuracy.

- [ ] **Step 4: Train custom baseline**

Run: `.venv/bin/python src/train.py --model custom`
Expected: models/custom.keras.

- [ ] **Step 5: Evaluate both models**

Run: `.venv/bin/python src/evaluate.py --models custom efficientnet`
Expected: comparison.csv + PNGs in outputs/.

- [ ] **Step 6: Commit artifacts**

```bash
git add models/ outputs/comparison.csv outputs/tuning_results.csv \
        outputs/*.png src/ requirements.txt
git commit -m "feat: trained models (custom CNN + EfficientNetB0) and evaluation artifacts"
```

Note: if `models/*.keras` exceeds ~90 MB, use Git LFS or commit only the efficientnet model. EfficientNetB0 (~20 MB) is fine for plain git.

---

### Task 13: Final verification + push

- [ ] **Step 1: Full test suite green**

Run: `.venv/bin/pytest -q`
Expected: all PASS.

- [ ] **Step 2: Verify model loads and predicts**

Run: `.venv/bin/python src/predict.py --model efficientnet --image <any test image from data/raw/Testing/>`
Expected: prints disclaimer + class probabilities.

- [ ] **Step 3: Push to remote**

```bash
git remote -v   # check remote exists
git push origin main
```

If no remote exists, ask the user for the repository URL and add it.

---

## Self-Review Notes

- Spec coverage: config (T2), preprocessing+augmenter (T3), model factory (T4), data prep (T5), EDA notebook (T6), train (T7), tune (T8), evaluate (T9), predict (T10), Streamlit app (T11), trained committed models (T12), push (T13). Error handling and pytest per spec. Disclaimers in README, app (sidebar + banner), predict output per spec. ✓
- Placeholders: none — all code complete. ✓
- Type consistency: `predict_image(image_path, model_name, models_dir=None)` used consistently in T10/T11 tests. `get_model(name, **kwargs)` consistent with train.py kwargs. ✓
