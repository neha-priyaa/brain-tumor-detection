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
