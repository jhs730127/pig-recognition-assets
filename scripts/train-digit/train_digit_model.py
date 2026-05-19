"""
Train Stage 1 broad-dataset digit classifier for pig-math worksheet OCR.

Datasets (auto-downloaded via tfds / HF, no manual steps):
  - MNIST: 60k train + 10k test
  - EMNIST digits: 240k train + 40k test (NIST SD-19 source,含 hsf_4 高中生)
  - USPS: 7.3k train + 2k test (郵編，自然手寫)

Total ~360k samples, ~6x more diverse than mnist_transfer_cnn_v1 alone.

Architecture: 保留 mnist_transfer_cnn_v1 經典 Keras MNIST CNN，方便直接
換掉 worksheet 的 MODEL_URL（架構相容，loadLayersModel 不用改）。

Output: ./tfjs-model/ 內含 model.json + group1-shard*.bin（TF.js layers
format），直接 upload Supabase Storage 後改 MODEL_URL 即可。

Usage (Colab):
    !git clone https://github.com/jhs730127/Math.git
    %cd Math
    !pip install -q -r scripts/training/requirements.txt
    !python scripts/training/train_digit_model.py --output ./tfjs-model
    # 然後 zip + download：
    from google.colab import files
    import shutil; shutil.make_archive("model", "zip", "tfjs-model")
    files.download("model.zip")

Usage (local Mac):
    python -m venv .venv && source .venv/bin/activate
    pip install -r scripts/training/requirements.txt
    python scripts/training/train_digit_model.py --output ./tfjs-model
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

# tfjs 4.22+ import chain 強行載入 tensorflow_decision_forests，但 tfdf 會把 TF 降版
# 跟 tensorflow-text / ydf-tf 二進位衝突。我們是 plain Keras CNN 不用 decision forests，
# stub 一個假 tfdf module 騙過 tfjs 的 import chain。必須在 import tensorflowjs 之前 stub。
sys.modules.setdefault(
    "tensorflow_decision_forests",
    types.ModuleType("tensorflow_decision_forests"),
)

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import classification_report, confusion_matrix


# ---------------------------------------------------------------------------
# Dataset loaders — 統一輸出（black-bg white-ink uint8 [N,28,28]）
#
# 為何全部統一成 black-bg white-ink？worksheet inference 時
# `preprocessDigit` 內部會做 `sub(255, x)` 把 PadView 產的白底黑字翻成
# 黑底白字送 model。訓練資料必須跟那一刻一致，否則 model 學到的特徵全反。
# MNIST/EMNIST raw 本來就是黑底白字（直接用），USPS 動態偵測。
# ---------------------------------------------------------------------------


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    assert x_train.mean() < 128, "MNIST raw 不是黑底白字"
    return (
        x_train.astype(np.uint8),
        y_train.astype(np.int64),
        x_test.astype(np.uint8),
        y_test.astype(np.int64),
    )


def load_emnist_digits() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """EMNIST digits split (240k train + 40k test)。tfds 不會 un-transpose，要手動翻。"""
    ds_train, ds_test = tfds.load(
        "emnist/digits",
        split=["train", "test"],
        as_supervised=True,
        batch_size=-1,
    )
    x_train, y_train = tfds.as_numpy(ds_train)
    x_test, y_test = tfds.as_numpy(ds_test)
    # EMNIST images are TRANSPOSED relative to MNIST — 要翻回 row-major
    # https://www.tensorflow.org/datasets/catalog/emnist
    x_train = np.transpose(x_train.squeeze(-1), (0, 2, 1))
    x_test = np.transpose(x_test.squeeze(-1), (0, 2, 1))
    assert x_train.mean() < 128, "EMNIST 不是黑底白字"
    return (
        x_train.astype(np.uint8),
        y_train.astype(np.int64),
        x_test.astype(np.uint8),
        y_test.astype(np.int64),
    )


def load_usps() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """USPS: 16×16 灰階郵編 → resize 28×28。polarity 動態偵測（HF 版本可能不同）。"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[warn] huggingface datasets not installed, skipping USPS")
        empty = np.array([]).reshape(0, 28, 28).astype(np.uint8)
        return empty, np.array([], dtype=np.int64), empty, np.array([], dtype=np.int64)

    ds = load_dataset("flwrlabs/usps")

    def _stack(split):
        images = []
        labels = []
        for row in ds[split]:
            img = np.array(row["image"])
            if img.ndim == 3:
                img = img[..., 0]
            images.append(img)
            labels.append(row["label"])
        return np.stack(images), np.array(labels, dtype=np.int64)

    x_train, y_train = _stack("train")
    x_test, y_test = _stack("test")

    # Polarity 動態偵測：mean > 128 = 白底黑字 → 反色成黑底白字
    sample_mean = x_train[:200].mean()
    if sample_mean > 128:
        print(f"  USPS raw mean={sample_mean:.1f} > 128 → 偵測為白底黑字，反色")
        x_train = 255 - x_train
        x_test = 255 - x_test
    else:
        print(f"  USPS raw mean={sample_mean:.1f} < 128 → 已是黑底白字")

    def _resize_batch(imgs: np.ndarray) -> np.ndarray:
        t = tf.image.resize(
            imgs[..., None].astype(np.float32), [28, 28], method="bilinear"
        )
        return tf.cast(t[..., 0], tf.uint8).numpy()

    x_train = _resize_batch(x_train)
    x_test = _resize_batch(x_test)
    assert x_train.mean() < 128, "USPS 反色後仍非黑底白字"
    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# Model — 對齊 mnist_transfer_cnn_v1（Google 公開的 Keras MNIST CNN）
# ---------------------------------------------------------------------------


def build_model(input_shape=(28, 28, 1), num_classes=10) -> tf.keras.Model:
    """Classic Keras MNIST CNN（小架構 ~600k params，bundle ~4MB）。
    `mnist_transfer_cnn_v1` 大致就是這個架構（feature_layers + classification_layers
    分兩段 fine-tune；這裡合併重訓）。"""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="pig_math_digit_v1",
    )
    return model


# ---------------------------------------------------------------------------
# Augmentation — 訓練時動態加干擾，逼近兒童手寫多樣性
# ---------------------------------------------------------------------------


def make_augmenter() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            # input 是 black-bg white-ink [0,1]，rotation/translation 用 0.0 fill（黑色）
            tf.keras.layers.RandomRotation(
                factor=10 / 360, fill_mode="constant", fill_value=0.0
            ),
            tf.keras.layers.RandomTranslation(
                height_factor=0.08, width_factor=0.08, fill_mode="constant", fill_value=0.0
            ),
            tf.keras.layers.RandomZoom(0.08, fill_mode="constant", fill_value=0.0),
        ],
        name="augmenter",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("./tfjs-model"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-usps", action="store_true", help="跳過 USPS（HF 拉失敗時）")
    args = parser.parse_args(argv)

    tf.keras.utils.set_random_seed(args.seed)

    # ---- Load ----
    print("[1/6] Loading datasets...")
    print("  - MNIST...")
    mx_tr, my_tr, mx_te, my_te = load_mnist()
    print(f"    train={mx_tr.shape} test={mx_te.shape}")
    print("  - EMNIST digits...")
    ex_tr, ey_tr, ex_te, ey_te = load_emnist_digits()
    print(f"    train={ex_tr.shape} test={ex_te.shape}")
    if args.no_usps:
        print("  - USPS skipped")
        ux_tr = np.array([]).reshape(0, 28, 28).astype(np.uint8)
        uy_tr = np.array([], dtype=np.int64)
        ux_te = ux_tr
        uy_te = uy_tr
    else:
        print("  - USPS...")
        ux_tr, uy_tr, ux_te, uy_te = load_usps()
        print(f"    train={ux_tr.shape} test={ux_te.shape}")

    # ---- Concatenate ----
    x_train = np.concatenate([mx_tr, ex_tr, ux_tr], axis=0)
    y_train = np.concatenate([my_tr, ey_tr, uy_tr], axis=0)
    x_test = np.concatenate([mx_te, ex_te, ux_te], axis=0)
    y_test = np.concatenate([my_te, ey_te, uy_te], axis=0)
    print(f"[2/6] Combined: train={x_train.shape} test={x_test.shape}")

    # uint8 → float32 [0, 1]
    x_train = x_train.astype(np.float32)[..., None] / 255.0
    x_test = x_test.astype(np.float32)[..., None] / 255.0

    # ---- Build model + augmenter ----
    print("[3/6] Building model...")
    model = build_model()
    model.summary()
    augmenter = make_augmenter()

    def _augment(x, y):
        return augmenter(x, training=True), y

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(20000, seed=args.seed)
        .batch(args.batch_size)
        .map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # ---- Train ----
    print(f"[4/6] Training {args.epochs} epochs...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
        ),
    ]
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=2
    )

    # ---- Evaluate ----
    print("[5/6] Evaluating...")
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"  Overall val_acc = {val_acc:.4f} (loss={val_loss:.4f})")

    # Confusion matrix on 6/8/9（檢查是否仍誤辨成 3）
    y_pred = model.predict(x_test, batch_size=256, verbose=0).argmax(-1)
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("     " + "  ".join(f"{i:>5d}" for i in range(10)))
    for i, row in enumerate(cm):
        print(f"{i:>3d}: " + "  ".join(f"{v:>5d}" for v in row))

    print("\nPer-class report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Specifically check 6/8/9 → 3 误判（pig-math 主訴）
    for src in [6, 8, 9, 0]:
        mis_3 = cm[src, 3]
        total = cm[src].sum()
        print(f"  Class {src} → predicted 3: {mis_3}/{total} ({100 * mis_3 / total:.1f}%)")

    # ---- Export to TF.js ----
    print(f"[6/6] Exporting to TF.js format at {args.output}...")
    try:
        import tensorflowjs as tfjs
    except ImportError:
        print("[error] tensorflowjs not installed. pip install tensorflowjs")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    # Save .h5 first then convert (tfjs.converters.save_keras_model 直接寫 layers format)
    tfjs.converters.save_keras_model(model, str(args.output))
    print(f"  Done. Files in {args.output}/:")
    for f in sorted(args.output.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name}  ({size_kb:.1f} KB)")

    # Also write a manifest with training info
    manifest = {
        "version": "v1-broad-stage1",
        "val_acc": float(val_acc),
        "val_loss": float(val_loss),
        "training_samples": int(len(x_train)),
        "validation_samples": int(len(x_test)),
        "datasets": ["mnist", "emnist_digits"] + ([] if args.no_usps else ["usps"]),
        "epochs_trained": len(history.history["loss"]),
        "confusion_6_to_3_pct": float(100 * cm[6, 3] / cm[6].sum()),
        "confusion_8_to_3_pct": float(100 * cm[8, 3] / cm[8].sum()),
        "confusion_9_to_3_pct": float(100 * cm[9, 3] / cm[9].sum()),
    }
    import json

    (args.output / "training-manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nDone. Zip the output folder and download.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
