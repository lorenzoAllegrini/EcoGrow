import os
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf
from roboflow import Roboflow
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input


AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 5
FINE_TUNE_UNFREEZE_FRACTION = 0.3
SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpu_memory_growth() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # Memory growth must be set before GPUs have been initialized
            pass


def collect_split_samples(dataset_root: str, split: str) -> Dict[str, Dict[str, List[str]]]:
    """Return mapping plant -> condition -> [image paths] for a given split."""
    split_dir = os.path.join(dataset_root, split)
    if not os.path.isdir(split_dir):
        return {}

    per_plant: Dict[str, Dict[str, List[str]]] = {}

    for folder in sorted(os.listdir(split_dir)):
        folder_path = os.path.join(split_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        parts = folder.split("_")
        if len(parts) > 2:
            plant_name = "_".join(parts[:2])
            condition = "_".join(parts[2:])
        else:
            plant_name = parts[0]
            condition = parts[1] if len(parts) > 1 else "Unknown"

        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            img_path = os.path.join(folder_path, fname)
            per_plant.setdefault(plant_name, {}).setdefault(condition, []).append(img_path)

    return per_plant


def build_dataset_index(dataset_root: str, splits: Tuple[str, ...]) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    return {split: collect_split_samples(dataset_root, split) for split in splits}


def prepare_file_label_pairs(
    per_condition: Dict[str, List[str]],
    label2idx: Dict[str, int],
) -> List[Tuple[str, int]]:
    pairs: List[Tuple[str, int]] = []
    for condition, paths in per_condition.items():
        if condition not in label2idx:
            continue
        label = label2idx[condition]
        for path in paths:
            pairs.append((path, label))
    return pairs


def preprocess_image(
    path: tf.Tensor,
    label: tf.Tensor,
    augment: bool = False,
) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32)

    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    image = preprocess_input(image)
    return image, label


def build_tf_dataset(
    file_label_pairs: List[Tuple[str, int]],
    batch_size: int,
    augment: bool = False,
) -> Optional[tf.data.Dataset]:
    if not file_label_pairs:
        return None

    paths, labels = zip(*file_label_pairs)
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    if augment:
        ds = ds.shuffle(buffer_size=len(file_label_pairs), reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, l: preprocess_image(p, l, augment=augment),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def build_model(num_classes: int) -> tf.keras.Model:
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.base_model = base_model  # type: ignore[attr-defined]
    return model


def run_experiment_for_plant(
    plant_name: str,
    split_index: Dict[str, Dict[str, Dict[str, List[str]]]],
) -> Dict:
    train_samples = split_index.get("train", {}).get(plant_name)
    if not train_samples:
        raise ValueError(f"No training samples found for plant '{plant_name}'")

    label_names = sorted(train_samples.keys())
    label2idx = {name: idx for idx, name in enumerate(label_names)}

    train_pairs = prepare_file_label_pairs(train_samples, label2idx)
    val_samples = split_index.get("valid", {}).get(plant_name) or split_index.get("val", {}).get(plant_name)
    test_samples = split_index.get("test", {}).get(plant_name)

    val_pairs = prepare_file_label_pairs(val_samples, label2idx) if val_samples else []
    test_pairs = prepare_file_label_pairs(test_samples, label2idx) if test_samples else []

    train_ds = build_tf_dataset(train_pairs, BATCH_SIZE, augment=True)
    val_ds = build_tf_dataset(val_pairs, BATCH_SIZE, augment=False) if val_pairs else None
    test_ds = build_tf_dataset(test_pairs, BATCH_SIZE, augment=False) if test_pairs else None

    if val_ds is None:
        print(f"[WARN] Validation set missing for {plant_name}. Using training set for validation.")
        val_ds = build_tf_dataset(train_pairs, BATCH_SIZE, augment=False)

    model = build_model(num_classes=len(label_names))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        )
    ]

    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

    base_model = model.base_model
    base_model.trainable = True
    fine_tune_from = int((1.0 - FINE_TUNE_UNFREEZE_FRACTION) * len(base_model.layers))
    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=INITIAL_EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

    test_metrics = None
    if test_ds is not None:
        test_metrics = model.evaluate(test_ds, verbose=0, return_dict=True)
        print(
            f"[{plant_name}] test loss {test_metrics['loss']:.4f} "
            f"acc {test_metrics['accuracy']:.3f}"
        )

    return {
        "label2idx": label2idx,
        "history_stage1": history_stage1.history,
        "history_stage2": history_stage2.history,
        "test_metrics": test_metrics,
    }


def main():
    set_global_seed(SEED)
    configure_gpu_memory_growth()

    rf = Roboflow(api_key="B5qYFzGDrNSzMkOluc7f")
    project = rf.workspace("lorenzo-dgkm4").project("indoor-plant-disease-dataset-odg74-ystag")
    dataset = project.version(1).download("folder")

    dataset_root = dataset.location
    print(f"Dataset downloaded to: {dataset_root}")

    splits = ("train", "valid", "test")
    split_index = build_dataset_index(dataset_root, splits)

    plants = sorted(split_index.get("train", {}).keys())
    results: Dict[str, Dict] = {}

    for plant_name in plants:
        print(f"\n===== Training EfficientNetB0 for {plant_name} =====")
        results[plant_name] = run_experiment_for_plant(plant_name, split_index)

    return results


if __name__ == "__main__":
    main()
