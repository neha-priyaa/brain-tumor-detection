import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """Tiny compiled model saved as <dir>/tiny.keras to stand in for models/."""
    import tensorflow as tf

    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    d = tmp_path_factory.mktemp("models")
    m.save(os.path.join(str(d), "tiny.keras"))
    return str(d)


def test_predict_image(tiny_model_dir, tmp_path):
    from PIL import Image
    from src.predict import predict_image

    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    p = tmp_path / "mri.png"
    Image.fromarray(img).save(p)

    result = predict_image(str(p), "tiny", models_dir=tiny_model_dir)
    assert result["class"] in ["glioma", "meningioma", "notumor", "pituitary"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["probabilities"]) == 4
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5


def test_predict_missing_model(tiny_model_dir, tmp_path):
    from PIL import Image
    from src.predict import predict_image

    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    p = tmp_path / "mri.png"
    Image.fromarray(img).save(p)
    with pytest.raises(FileNotFoundError):
        predict_image(str(p), "nonexistent", models_dir=tiny_model_dir)
