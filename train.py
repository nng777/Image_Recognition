import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS_HEAD = 5       # train classifier head first
EPOCHS_FINE = 5       # then fine-tune some base layers
DATA_DIR_TRAIN = "data/train"
DATA_DIR_VAL = "data/val"
OUT_PATH = "glasses_detector.keras"  # SavedModel in .keras format

def build_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR_TRAIN, label_mode="binary", image_size=IMG_SIZE, batch_size=BATCH, shuffle=True
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR_VAL, label_mode="binary", image_size=IMG_SIZE, batch_size=BATCH, shuffle=False
    )
    # Cache+prefetch for speed
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    # Preprocess to match MobileNetV2
    def pp(x, y): return preprocess_input(tf.cast(x, tf.float32)), y
    return train_ds.map(pp), val_ds.map(pp)

def build_model():
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
    base.trainable = False  # start frozen
    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ])
    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = aug(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model, base

def main():
    train_ds, val_ds = build_datasets()
    model, base = build_model()

    # Train classifier head
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD)

    # Fine-tune last ~40% of base
    base.trainable = True
    for layer in base.layers[:-60]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE)

    model.save(OUT_PATH)
    print(f"Saved model to {OUT_PATH}")

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
