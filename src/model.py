"""Model factory: custom CNN baseline and EfficientNetB0 transfer learning."""
import tensorflow as tf

from src.config import CLASS_NAMES


def build_custom_cnn(lr=1e-3):
    model = tf.keras.Sequential(
        [
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
        ],
        name="custom_cnn",
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
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
        return build_custom_cnn(lr=kwargs.get("lr", 1e-3))
    if name == "efficientnet":
        return build_efficientnet(
            dropout=kwargs.get("dropout", 0.2),
            freeze_base=kwargs.get("freeze_base", True),
            lr=kwargs.get("lr", 1e-3),
        )
    raise ValueError(f"Unknown model name: {name!r} (use 'custom' or 'efficientnet')")
