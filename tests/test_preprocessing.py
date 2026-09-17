import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def test_label_mapping():
    from src.preprocessing import label_from_dirname

    assert label_from_dirname("glioma") == 0
    assert label_from_dirname("pituitary") == 3


def test_preprocess_image():
    from PIL import Image
    from src.preprocessing import preprocess_image

    img = Image.new("L", (100, 100), 128)
    import tempfile
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
