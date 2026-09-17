"""Data loading: tf.data pipelines and single-image preprocessing."""
import os

import numpy as np
import tensorflow as tf

from src.config import CLASS_NAMES, IMAGE_SIZE, BATCH_SIZE, TRAIN_DIR, TEST_DIR, SEED


def label_from_dirname(dirname):
    return CLASS_NAMES.index(dirname)


def preprocess_image(path):
    """Load an image file -> (224, 224, 3) float32 numpy array scaled to [0, 1]."""
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img.numpy()


def build_augmenter():
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmenter",
    )


def load_datasets():
    """Return (train, val, test) tf.data datasets.

    Validation set is carved from Training/ 90/10 stratified-ish via
    keras validation_split (per-class file shuffling, seed fixed).
    """
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
        return ds.map(
            lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    augmenter = build_augmenter()
    train_ds = (
        normalize(train_ds)
        .map(lambda x, y: (augmenter(x, training=True), y),
             num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
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
    return {i: total / (len(counts) * c) for i, c in enumerate(counts)}
