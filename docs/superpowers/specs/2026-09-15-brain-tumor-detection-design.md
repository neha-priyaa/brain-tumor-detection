# Brain Tumor MRI Classification — Design (Educational Project)

> **DISCLAIMER:** This is an educational machine-learning project and is NOT a medical
> diagnostic system. It must never be presented or used as one. The disclaimer appears in
> the README header, the Streamlit app (sidebar + top banner), and is printed by predict code.

## Goal

Classify brain MRI images into 4 classes — glioma, meningioma, notumor, pituitary — with a
custom CNN baseline and an EfficientNetB0 transfer-learning model, compare them, and serve a
Streamlit demo. Reproducible (fixed seed 42, pinned requirements, scripted data prep).

## Decisions (user-approved)

- **Dataset:** Kaggle `masoudnickparvar/brain-tumor-mri-dataset` (~5,712 train / 1,311 test,
  pre-split `Training/` + `Testing/` dirs; validation carved from train 90/10, stratified, seed 42).
- **Training budget:** full quality — 224×224 RGB, ~25 epochs max, EarlyStopping(patience=5),
  EfficientNetB0 frozen base + trainable head (fine-tune of base is a grid-search option).
- **Tuning:** small scripted grid on EfficientNet: lr {1e-3, 1e-4} × dropout {0.2, 0.5} ×
  frozen-base {yes, no} → `outputs/tuning_results.csv` (8 short runs, reduced epochs, best
  config re-trained in full by `train.py`).
- **Structure:** Approach A — single config-driven pipeline, model factory, one EDA notebook.
- **Runtime:** macOS, CPU/Metal (no discrete GPU); project venv, own git repo on `main`.

## Modules

| File | Responsibility |
|---|---|
| `src/config.py` | Paths, CLASS_NAMES = [glioma, meningioma, notumor, pituitary], IMAGE_SIZE=224, BATCH_SIZE=32, SEED=42, EPOCHS; `set_seeds()` |
| `src/preprocessing.py` | `load_datasets()` → tf.data pipelines from directories (train/val carved stratified, test as-is); grayscale→RGB, resize 224, rescale 1/255; `build_augmenter()` (flip, rot 10%, zoom 10%) applied to train only |
| `src/model.py` | `build_custom_cnn()` (3× [Conv32/64/128 + BN + MaxPool] + Dropout + Dense head), `build_efficientnet(dropout, freeze_base)` (EfficientNetB0, ImageNet weights, GAP + Dense(128) + Dropout + Softmax4), `get_model(name, **kw)` factory; class-weight helper |
| `src/train.py` | CLI `--model custom\|efficientnet --epochs N --dropout F --lr F --no-freeze-base`; compiles (Adam(lr), sparse-CCE), trains with EarlyStopping + checkpoint to `models/{name}.keras`, saves history plot + `models/{name}_history.json`; final test evaluation printed |
| `src/tune.py` | Runs the 8-config grid at reduced epochs (8) on train+val; writes `outputs/tuning_results.csv` sorted by val accuracy; prints best config as ready-to-run train.py command |
| `src/evaluate.py` | CLI `--models custom efficientnet`; test-set predictions → accuracy, per-class + macro precision/recall/F1, one-vs-rest ROC-AUC; confusion matrix PNG + ROC curves PNG per model; `outputs/comparison.csv` + markdown table |
| `src/predict.py` | `predict_image(path, model_name) -> {class, confidence, probabilities}`; loads model, preprocesses single image; prints educational disclaimer |
| `app/streamlit_app.py` | Sidebar: model picker (from `models/*.keras`), disclaimer; main: file upload → show image → Predict button → class + confidence + probability bar chart |
| `scripts/prepare_data.py` | CLI `--zip path`: extracts `Training/`+`Testing/` into `data/raw/`, verifies class counts, prints summary; idempotent |
| `notebooks/01_eda.ipynb` | Class distribution bar charts + sample images per class from `data/raw/Training` |

## Outputs

`models/{custom,efficientnet}.keras` + history artifacts; `outputs/` → confusion matrices,
ROC curves, comparison table, tuning results; training-history plots.

## Error handling

- `prepare_data.py`: clear errors for missing ZIP, unexpected folder layout, corrupt images (skips + logs count).
- `predict.py` / app: graceful handling of non-image uploads; missing model file → message to train first.
- App never crashes on bad input; all exceptions surfaced as friendly text.

## Testing (pytest, offline, small)

- `tests/test_preprocessing.py`: label mapping, image decode/resize/normalize values, augmenter output shape/range.
- `tests/test_model.py`: both factories output expected input/output shapes; frozen base flag honored.
- `tests/test_predict.py`: end-to-end predict on a synthetic random image with a tiny compiled model (mock path), disclaimer present.
- Config test: seed reproducibility, class names order.

## Data setup (README documents)

1. Download ZIP from Kaggle (manual, one-time).
2. `python scripts/prepare_data.py --zip ~/Downloads/archive.zip`
3. `data/raw/{Training,Testing}` populated; never committed (.gitignore).

## Out of scope

Medical validation, DICOM support, model explainability (Grad-CAM etc.), deployment beyond local Streamlit, Kaggle API automation.
