"""
Train single-letter (A-Z + a-z, 52 classes) classifier from EMNIST ByClass.

Dataset: EMNIST ByClass via tensorflow_datasets — 62 classes (0-9 + A-Z + a-z),
filtered to letters only (class 10-61), label remapped to 0-51.

Approx 580k train + 110k test letter samples.

Architecture: 4-Conv + Dense(256) + Dense(52). Same backbone as pig-math digit
model, head expanded from 128->256 and 10->52. Bundle ~4.5MB.

Output: ./tfjs-model/ 內含
  - model.json + group1-shard*.bin (TF.js layers format)
  - classes.json (52 label mapping + case_pairs + confusable_pairs)
  - confusion_pairs.json (high-confusion mutual-confusion stats)
  - training-manifest.json (val_acc top-1/2/3 + per-class samples)

Usage (Colab):
    !git clone https://github.com/jhs730127/Math.git
    %cd Math
    !pip install -q -r scripts/letter-training/requirements.txt
    !python scripts/letter-training/train_letter_model.py --output ./tfjs-model
    # 然後 zip + download：見 train_letter_model.ipynb

Usage (local Mac):
    python -m venv .venv-train && source .venv-train/bin/activate
    pip install -r scripts/letter-training/requirements.txt
    python scripts/letter-training/train_letter_model.py --output ./tfjs-model
"""

from __future__ import annotations

import argparse
import json
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
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight


# ---------------------------------------------------------------------------
# Constants — 52 letter class mapping & known confusable pairs
#
# label 0-25  = A-Z (uppercase)
# label 26-51 = a-z (lowercase)
# ---------------------------------------------------------------------------

LABELS: list[str] = [chr(ord("A") + i) for i in range(26)] + [
    chr(ord("a") + i) for i in range(26)
]

# Upper/lower pairs，給應用層做大小寫無關 fallback 用
CASE_PAIRS: list[list[int]] = [[i, i + 26] for i in range(26)]

# 同形混淆對：手寫大小寫長相幾乎一樣的字母。Top-1 在這幾對上會卡 ~70%，
# Top-2 通常 ~95%。前端拿 top-K 後處理是必要的。
CONFUSABLE_PAIRS_LETTERS: list[tuple[str, str]] = [
    ("C", "c"),
    ("K", "k"),
    ("M", "m"),
    ("O", "o"),
    ("P", "p"),
    ("S", "s"),
    ("U", "u"),
    ("V", "v"),
    ("W", "w"),
    ("X", "x"),
    ("Z", "z"),
]


def _label_idx(letter: str) -> int:
    return LABELS.index(letter)


CONFUSABLE_PAIRS: list[list[int]] = [
    [_label_idx(u), _label_idx(l)] for (u, l) in CONFUSABLE_PAIRS_LETTERS
]


# ---------------------------------------------------------------------------
# Dataset loader — EMNIST ByClass, filter letters, remap labels
#
# Polarity: EMNIST raw 是黑底白字 → 跟前端 inference 流程（preprocessDigit
# 內部 sub(255, x)）對齊，不需翻。
# Transpose: EMNIST 是 column-major，要 transpose (0, 2, 1) 翻回 row-major。
# ---------------------------------------------------------------------------


def load_emnist_letters() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """EMNIST ByClass filter to class 10-61 (letters)，label remap subtract 10。"""
    ds_train, ds_test = tfds.load(
        "emnist/byclass",
        split=["train", "test"],
        as_supervised=True,
        batch_size=-1,
    )
    x_train_all, y_train_all = tfds.as_numpy(ds_train)
    x_test_all, y_test_all = tfds.as_numpy(ds_test)

    # un-transpose（EMNIST raw 是 column-major）
    x_train_all = np.transpose(x_train_all.squeeze(-1), (0, 2, 1))
    x_test_all = np.transpose(x_test_all.squeeze(-1), (0, 2, 1))

    # Filter to letters (class 10-61) and remap to 0-51
    train_mask = y_train_all >= 10
    test_mask = y_test_all >= 10
    x_train = x_train_all[train_mask].astype(np.uint8)
    y_train = (y_train_all[train_mask] - 10).astype(np.int64)
    x_test = x_test_all[test_mask].astype(np.uint8)
    y_test = (y_test_all[test_mask] - 10).astype(np.int64)

    assert x_train.mean() < 128, "EMNIST raw 不是黑底白字（preprocessing pipeline 出包）"
    assert y_train.max() == 51 and y_train.min() == 0, "label remap 失敗"

    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# Model — 4-Conv + Dense + Dense(52)，可選 BatchNormalization
# ---------------------------------------------------------------------------


def build_model(
    input_shape=(28, 28, 1),
    num_classes=52,
    dense_units: int = 256,
    use_bn: bool = False,
) -> tf.keras.Model:
    """同 pig-math digit model backbone。
    - dense_units=256 (v2) / 512 (v3) 控 head capacity
    - use_bn=True 在每個 Conv2D 後加 BatchNormalization（Conv→BN→ReLU）。
      ⚠️ BN + Keras 3 → TF.js 4.22 有 known schema risk，patch 後務必跑反向 load 驗證。
    """
    if use_bn:
        # Conv → BN → ReLU 順序，需要把 activation 拆出來
        def conv_bn(filters: int, padding: str):
            return [
                tf.keras.layers.Conv2D(filters, (3, 3), padding=padding, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
            ]

        layers: list = [
            tf.keras.layers.Input(shape=input_shape),
            *conv_bn(32, "same"),
            *conv_bn(32, "valid"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            *conv_bn(64, "same"),
            *conv_bn(64, "valid"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(dense_units, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    else:
        layers = [
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
            tf.keras.layers.Dense(dense_units, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]

    return tf.keras.Sequential(layers, name="pig_math_letter_v1")


# ---------------------------------------------------------------------------
# Augmentation — 強度可調，過強會 under-fit common class（v1/v2 lesson）
# 絕對不要 horizontal flip（b/d, p/q 鏡像就是別字）
# ---------------------------------------------------------------------------


def make_augmenter(
    rotation_deg: float = 7.0,
    translation_pct: float = 0.08,
    zoom_pct: float = 0.08,
    brightness_pct: float = 0.10,
) -> tf.keras.Sequential:
    layers = [
        tf.keras.layers.RandomRotation(
            factor=rotation_deg / 360, fill_mode="constant", fill_value=0.0
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=translation_pct,
            width_factor=translation_pct,
            fill_mode="constant",
            fill_value=0.0,
        ),
        tf.keras.layers.RandomZoom(zoom_pct, fill_mode="constant", fill_value=0.0),
    ]
    if brightness_pct > 0:
        layers.append(
            tf.keras.layers.RandomBrightness(factor=brightness_pct, value_range=(0.0, 1.0))
        )
    return tf.keras.Sequential(layers, name="augmenter")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def top_k_accuracy(y_true: np.ndarray, probs: np.ndarray, k: int) -> float:
    """Top-K accuracy from softmax probs."""
    top_k = np.argsort(-probs, axis=1)[:, :k]
    hits = (top_k == y_true[:, None]).any(axis=1)
    return float(hits.mean())


def confusable_pair_stats(
    y_true: np.ndarray,
    probs: np.ndarray,
    pairs: list[list[int]],
) -> list[dict]:
    """對每一對 (upper_idx, lower_idx)，算 mutual confusion + top-2 accuracy。"""
    top_2 = np.argsort(-probs, axis=1)[:, :2]
    top_1 = top_2[:, 0]

    stats = []
    for upper_idx, lower_idx in pairs:
        # Samples whose true label is in this pair
        mask = (y_true == upper_idx) | (y_true == lower_idx)
        if mask.sum() == 0:
            continue
        true_pair = y_true[mask]
        pred_top1_pair = top_1[mask]
        pred_top2_pair = top_2[mask]

        # Mutual confusion: pred 跑到 pair 內另一個的比例
        cross = (
            (true_pair == upper_idx) & (pred_top1_pair == lower_idx)
        ).sum() + ((true_pair == lower_idx) & (pred_top1_pair == upper_idx)).sum()
        mutual_pct = float(100 * cross / mask.sum())

        # Top-1 / Top-2 acc on this pair
        top1_acc = float((pred_top1_pair == true_pair).mean())
        top2_hits = (pred_top2_pair == true_pair[:, None]).any(axis=1)
        top2_acc = float(top2_hits.mean())

        stats.append(
            {
                "pair": [LABELS[upper_idx], LABELS[lower_idx]],
                "mutual_confusion_pct": round(mutual_pct, 2),
                "top1_acc": round(top1_acc, 4),
                "top2_acc": round(top2_acc, 4),
                "samples": int(mask.sum()),
            }
        )
    return stats


def non_confusable_top1_acc(
    y_true: np.ndarray,
    probs: np.ndarray,
    confusable_pairs: list[list[int]],
) -> float:
    """剔除混淆對的字母只算 top-1 acc，看「正常字母」表現。"""
    confusable_set: set[int] = set()
    for u, l in confusable_pairs:
        confusable_set.add(u)
        confusable_set.add(l)
    mask = np.array([y not in confusable_set for y in y_true])
    if mask.sum() == 0:
        return 0.0
    top_1 = probs[mask].argmax(axis=1)
    return float((top_1 == y_true[mask]).mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("./tfjs-model"))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-class-weight",
        action="store_true",
        help="開 class_weight balanced（rare class 加權）。v2 預設關閉因為跟 label_smoothing 雙重 regularization 反而傷 top-1。",
    )
    parser.add_argument(
        "--no-label-smoothing",
        action="store_true",
        help="關掉 label_smoothing。v2 預設開 0.1 緩和 over-confidence + 提升 top-2。",
    )
    parser.add_argument(
        "--dense-units",
        type=int,
        default=256,
        help="Dense head units。v2=256, v3=512（為 non-confusable letters 加 capacity）",
    )
    parser.add_argument(
        "--use-bn",
        action="store_true",
        help="加 BatchNormalization（Conv→BN→ReLU）。v3 拼準確率用。⚠️ Keras 3 → TF.js 4.22 有 known schema risk，patch 後務必 reload 驗證",
    )
    parser.add_argument(
        "--augment-preset",
        choices=["v2", "v3-light"],
        default="v2",
        help="v2: rot7/trans8/zoom8/bright10（v1/v2 用）。v3-light: rot5/trans5/zoom5/bright0（拼 non-confusable top-1）",
    )
    args = parser.parse_args(argv)

    tf.keras.utils.set_random_seed(args.seed)

    # ---- Load ----
    print("[1/7] Loading EMNIST ByClass (letters subset)...")
    x_train, y_train, x_test, y_test = load_emnist_letters()
    print(f"  train={x_train.shape} test={x_test.shape}")
    print(f"  label range: {y_train.min()}-{y_train.max()} (expect 0-51)")

    # Per-class sample count（用於 manifest 跟 class_weight）
    per_class_train = {
        LABELS[i]: int((y_train == i).sum()) for i in range(52)
    }
    rare_count = sorted(per_class_train.values())[:5]
    common_count = sorted(per_class_train.values(), reverse=True)[:5]
    print(f"  rare classes (5 smallest): {rare_count}")
    print(f"  common classes (5 largest): {common_count}")

    # ---- Filter + normalize ----
    print("[2/7] Normalize uint8 → float32 [0, 1]...")
    x_train = x_train.astype(np.float32)[..., None] / 255.0
    x_test = x_test.astype(np.float32)[..., None] / 255.0

    # ---- Class weight ----
    # v1 開 class_weight balanced 結果 top-1 0.826、non-confusable 只 0.89（預期 0.97）。
    # 推測：class_weight + augmentation 雙重 regularization 太強，model 對 common class
    # 也學不到底。v2 預設關掉 class_weight，rare class top-1 會掉但 overall 應該升。
    if args.use_class_weight:
        print("[3/7] Computing class weights (balanced)...")
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(52),
            y=y_train,
        )
        class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
        print(f"  weight range: {min(class_weights):.3f} - {max(class_weights):.3f}")
    else:
        class_weight_dict = None
        print("[3/7] class_weight disabled (v2 default)")

    # ---- Build model + augmenter ----
    print(
        f"[4/7] Building model (dense_units={args.dense_units}, use_bn={args.use_bn})..."
    )
    model = build_model(dense_units=args.dense_units, use_bn=args.use_bn)
    model.summary()

    if args.augment_preset == "v3-light":
        print("  augmenter: v3-light (rot5/trans5/zoom5/bright0)")
        augmenter = make_augmenter(
            rotation_deg=5.0,
            translation_pct=0.05,
            zoom_pct=0.05,
            brightness_pct=0.0,
        )
    else:
        print("  augmenter: v2 (rot7/trans8/zoom8/bright10)")
        augmenter = make_augmenter()

    use_label_smoothing = not args.no_label_smoothing

    def _augment_sparse(x, y):
        return augmenter(x, training=True), y

    def _augment_categorical(x, y):
        return augmenter(x, training=True), tf.one_hot(y, 52)

    if use_label_smoothing:
        # label_smoothing 0.1：CategoricalCrossentropy 接 one-hot，緩和 over-confidence，
        # 對 confusable pair 的 top-2 有顯著幫助。需要 one-hot y。
        print("  loss: CategoricalCrossentropy(label_smoothing=0.1) + one-hot")
        train_ds = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(20000, seed=args.seed)
            .batch(args.batch_size)
            .map(_augment_categorical, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((x_test, y_test))
            .map(lambda x, y: (x, tf.one_hot(y, 52)), num_parallel_calls=tf.data.AUTOTUNE)
            .batch(args.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        metrics = [
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_acc"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_acc"),
        ]
    else:
        print("  loss: SparseCategoricalCrossentropy (no label smoothing)")
        train_ds = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(20000, seed=args.seed)
            .batch(args.batch_size)
            .map(_augment_sparse, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((x_test, y_test))
            .batch(args.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = [
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top_2_acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top_3_acc"),
        ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=loss,
        metrics=metrics,
    )

    # ---- Train ----
    # v1 patience=4 在 epoch 11 就停（25 epoch 沒跑滿），val_acc 應該還有上升空間。
    # v2 patience 10 + epochs 35 給 model 時間突破 plateau。
    print(f"[5/7] Training {args.epochs} epochs (early stop patience=10)...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
    ]
    fit_kwargs = dict(
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    if class_weight_dict is not None:
        fit_kwargs["class_weight"] = class_weight_dict
    history = model.fit(train_ds, **fit_kwargs)

    # ---- Evaluate ----
    print("[6/7] Evaluating...")
    probs = model.predict(x_test, batch_size=256, verbose=0)
    y_pred = probs.argmax(-1)

    val_acc_top1 = top_k_accuracy(y_test, probs, 1)
    val_acc_top2 = top_k_accuracy(y_test, probs, 2)
    val_acc_top3 = top_k_accuracy(y_test, probs, 3)
    print(f"  val_acc top-1 = {val_acc_top1:.4f}")
    print(f"  val_acc top-2 = {val_acc_top2:.4f}")
    print(f"  val_acc top-3 = {val_acc_top3:.4f}")

    # Per-class report (52 lines)
    print("\nPer-class report:")
    print(
        classification_report(
            y_test, y_pred, target_names=LABELS, digits=4, zero_division=0
        )
    )

    # Confusable pair focus
    print("\nConfusable pair stats:")
    pair_stats = confusable_pair_stats(y_test, probs, CONFUSABLE_PAIRS)
    for s in pair_stats:
        print(
            f"  {s['pair'][0]}/{s['pair'][1]}: "
            f"top1={s['top1_acc']:.3f} top2={s['top2_acc']:.3f} "
            f"mutual={s['mutual_confusion_pct']:.1f}% "
            f"(n={s['samples']})"
        )

    non_conf_top1 = non_confusable_top1_acc(y_test, probs, CONFUSABLE_PAIRS)
    print(f"\nNon-confusable letters top-1 acc = {non_conf_top1:.4f}")

    confusable_top1 = float(
        np.mean([s["top1_acc"] for s in pair_stats]) if pair_stats else 0.0
    )
    confusable_top2 = float(
        np.mean([s["top2_acc"] for s in pair_stats]) if pair_stats else 0.0
    )

    # ---- Export to TF.js ----
    print(f"\n[7/7] Exporting to TF.js format at {args.output}...")
    try:
        import tensorflowjs as tfjs
    except ImportError:
        print("[error] tensorflowjs not installed. pip install tensorflowjs")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    tfjs.converters.save_keras_model(model, str(args.output))

    # Decide version 標籤：v3 = 任何 non-default (use_bn / dense != 256 / augment v3-light)
    is_v3 = args.use_bn or args.dense_units != 256 or args.augment_preset == "v3-light"
    version = f"v{3 if is_v3 else 2}-emnist-byclass-52"

    # Architecture string
    bn_suffix = "-BN" if args.use_bn else ""
    arch_str = (
        f"Conv32-Conv32-Pool-Conv64-Conv64-Pool-Dense{args.dense_units}-Dense52{bn_suffix}"
    )

    # Augmentation string
    if args.augment_preset == "v3-light":
        aug_str = "rot5_trans5_zoom5_brightness0"
    else:
        aug_str = "rot7_trans8_zoom8_brightness10"

    # ---- Write metadata JSONs ----
    classes_json = {
        "version": version,
        "num_classes": 52,
        "expected_polarity": "black_bg_white_ink",
        "input_shape": [28, 28, 1],
        "labels": LABELS,
        "case_pairs": CASE_PAIRS,
        "confusable_pairs": CONFUSABLE_PAIRS,
    }
    (args.output / "classes.json").write_text(json.dumps(classes_json, indent=2))

    confusion_json = {
        "high_confusion_pairs": pair_stats,
        "low_confusion_letters": [
            LABELS[i]
            for i in range(52)
            if i not in {idx for pair in CONFUSABLE_PAIRS for idx in pair}
        ],
    }
    (args.output / "confusion_pairs.json").write_text(
        json.dumps(confusion_json, indent=2)
    )

    manifest = {
        "version": version,
        "val_acc_top1": round(val_acc_top1, 4),
        "val_acc_top2": round(val_acc_top2, 4),
        "val_acc_top3": round(val_acc_top3, 4),
        "non_confusable_letters_top1": round(non_conf_top1, 4),
        "confusable_pairs_top1_mean": round(confusable_top1, 4),
        "confusable_pairs_top2_mean": round(confusable_top2, 4),
        "training_samples": int(len(x_train)),
        "validation_samples": int(len(x_test)),
        "per_class_samples": per_class_train,
        "datasets": ["emnist_byclass"],
        "epochs_trained": len(history.history["loss"]),
        "architecture": arch_str,
        "training_config": {
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "early_stop_patience": 10,
            "use_class_weight": args.use_class_weight,
            "use_bn": args.use_bn,
            "dense_units": args.dense_units,
            "label_smoothing": 0.0 if args.no_label_smoothing else 0.1,
            "loss": "SparseCategoricalCrossentropy" if args.no_label_smoothing else "CategoricalCrossentropy",
            "augmentation": aug_str,
        },
    }
    (args.output / "training-manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"  Done. Files in {args.output}/:")
    for f in sorted(args.output.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name}  ({size_kb:.1f} KB)")

    print(
        "\n下一步：跑 patch_tfjs_model_json.py 把 Keras 3 schema patch 給 TF.js 4.22 吃："
    )
    print(f"  python scripts/letter-training/patch_tfjs_model_json.py {args.output}/model.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
