# Brain Tumor MRI Classification

> **DISCLAIMER:** This is an educational machine-learning project and is NOT a
> medical diagnostic system. It must never be presented or used as one.

Classifies brain MRI images into 4 classes — **glioma, meningioma, notumor, pituitary** —
using a custom CNN baseline and EfficientNetB0 transfer learning.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data setup (one-time)

1. Download the ZIP from https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
2. `python scripts/prepare_data.py --zip ~/Downloads/archive.zip`
3. `data/raw/{Training,Testing}` are populated (never committed).

## Train

```bash
python src/train.py --model efficientnet   # transfer learning model
python src/train.py --model custom         # CNN baseline
```

## Tune (grid search)

```bash
python src/tune.py --epochs 8
```

## Evaluate

```bash
python src/evaluate.py --models custom efficientnet
```

Outputs confusion matrices, ROC curves, and `outputs/comparison.csv`.

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

## Project layout

- `src/config.py` — paths, constants, seeds
- `src/preprocessing.py` — tf.data pipelines + augmenter
- `src/model.py` — model factory (custom CNN, EfficientNetB0)
- `src/train.py` — training CLI (early stopping + checkpointing)
- `src/tune.py` — 8-config grid search
- `src/evaluate.py` — test-set metrics + plots
- `src/predict.py` — single-image prediction
- `app/streamlit_app.py` — demo UI
- `models/` — trained `.keras` models (committed)
- `outputs/` — evaluation artifacts
