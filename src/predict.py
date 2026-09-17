"""Single-image prediction. Usage:
python src/predict.py --model efficientnet --image path/to/mri.jpg
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from src.config import CLASS_NAMES, DISCLAIMER, MODELS_DIR
from src.preprocessing import preprocess_image


def predict_image(image_path, model_name, models_dir=None):
    """Return {class, confidence, probabilities}.

    Raises FileNotFoundError when the model file does not exist.
    """
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
    ap.add_argument("--model", required=True,
                    choices=["custom", "efficientnet"])
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    print(DISCLAIMER)
    try:
        result = predict_image(args.image, args.model)
    except FileNotFoundError as e:
        sys.exit(str(e))
    print(f"\nPrediction: {result['class']} ({result['confidence']:.1%})")
    for c, p in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {c:12s} {p:.1%}")


if __name__ == "__main__":
    main()
