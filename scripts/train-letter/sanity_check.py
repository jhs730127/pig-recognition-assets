"""
Letter recognition model sanity check — verify downloaded model files are valid.

Usage:
    python scripts/letter-training/sanity_check.py
    python scripts/letter-training/sanity_check.py --run-inference  # 要先 pip install tensorflowjs

Checks:
1. model.json schema valid (all layer classes supported by TF.js 4.22)
2. classes.json schema (52 labels, expected_polarity, case_pairs, confusable_pairs)
3. training-manifest.json fields complete
4. group1-shard*.bin size matches weights metadata
5. (optional) inference on EMNIST test sample matches manifest val_acc within 5pp

跑這個確認 model 沒被 patch_tfjs_model_json.py 弄壞、TF.js 4.22 可以 load。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# TF.js 4.22 公認支援的 core layer（從 @tensorflow/tfjs-layers built-in registry）
TFJS_SUPPORTED_LAYERS = {
    "InputLayer",
    "Conv2D",
    "BatchNormalization",
    "Activation",
    "MaxPooling2D",
    "AveragePooling2D",
    "GlobalAveragePooling2D",
    "Dropout",
    "Flatten",
    "Dense",
    "Softmax",
    "Reshape",
}

EXPECTED_LABELS = [chr(ord("A") + i) for i in range(26)] + [
    chr(ord("a") + i) for i in range(26)
]


def red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


def green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m"


class CheckResult:
    def __init__(self) -> None:
        self.passes: list[str] = []
        self.fails: list[str] = []
        self.warns: list[str] = []

    def passed(self, msg: str) -> None:
        self.passes.append(msg)
        print(f"  {green('✅')} {msg}")

    def failed(self, msg: str) -> None:
        self.fails.append(msg)
        print(f"  {red('❌')} {msg}")

    def warn(self, msg: str) -> None:
        self.warns.append(msg)
        print(f"  {yellow('⚠️')} {msg}")


def check_model_json(model_dir: Path, result: CheckResult) -> dict | None:
    print("\n[1/4] Checking model.json schema...")
    model_path = model_dir / "model.json"
    if not model_path.exists():
        result.failed(f"{model_path} not found")
        return None

    try:
        data = json.loads(model_path.read_text())
    except json.JSONDecodeError as e:
        result.failed(f"model.json invalid JSON: {e}")
        return None

    if data.get("format") != "layers-model":
        result.failed(f"format != 'layers-model' (got '{data.get('format')}')")
        return None
    result.passed("format = layers-model")

    keras_version = data.get("modelTopology", {}).get("keras_version", "")
    if not keras_version:
        result.warn("no keras_version field")
    else:
        result.passed(f"keras_version = {keras_version}")

    config = data.get("modelTopology", {}).get("model_config", {}).get("config", {})
    layers = config.get("layers", [])
    if not layers:
        result.failed("no layers in model_config")
        return data

    layer_classes = [l["class_name"] for l in layers]
    unique_classes = set(layer_classes)
    print(f"  Layer classes: {sorted(unique_classes)}")
    unsupported = unique_classes - TFJS_SUPPORTED_LAYERS
    if unsupported:
        result.failed(f"unsupported by TF.js 4.22: {unsupported}")
    else:
        result.passed(f"all {len(unique_classes)} layer types supported by TF.js 4.22")

    # Check InputLayer has batch_input_shape (patch worked)
    input_layer = next((l for l in layers if l["class_name"] == "InputLayer"), None)
    if input_layer:
        cfg = input_layer["config"]
        if "batch_input_shape" in cfg:
            result.passed(f"InputLayer.batch_input_shape = {cfg['batch_input_shape']}")
        elif "batch_shape" in cfg:
            result.failed(
                "InputLayer still has 'batch_shape' (Keras 3 format) — "
                "patch_tfjs_model_json.py 沒跑或失敗"
            )
        else:
            result.warn("InputLayer 沒 batch_input_shape 也沒 batch_shape")

    # Check dtype is string not dict (patch worked)
    dict_dtypes = sum(1 for l in layers if isinstance(l["config"].get("dtype"), dict))
    if dict_dtypes > 0:
        result.failed(f"{dict_dtypes} layers still have DTypePolicy dict — patch 沒跑")
    else:
        result.passed("all layer dtypes are string (DTypePolicy dict already patched)")

    # Check weight name prefix (patch worked)
    model_name = config.get("name", "")
    if model_name:
        prefix = f"{model_name}/"
        prefix_remaining = 0
        for group in data.get("weightsManifest", []):
            for w in group.get("weights", []):
                if w.get("name", "").startswith(prefix):
                    prefix_remaining += 1
        if prefix_remaining > 0:
            result.failed(
                f"{prefix_remaining} weights still have '{prefix}' prefix — patch 沒跑"
            )
        else:
            result.passed(f"weight names already stripped of '{prefix}' prefix")

    return data


def check_classes_json(model_dir: Path, result: CheckResult) -> dict | None:
    print("\n[2/4] Checking classes.json...")
    classes_path = model_dir / "classes.json"
    if not classes_path.exists():
        result.failed(f"{classes_path} not found")
        return None

    try:
        data = json.loads(classes_path.read_text())
    except json.JSONDecodeError as e:
        result.failed(f"classes.json invalid: {e}")
        return None

    if data.get("num_classes") != 52:
        result.failed(f"num_classes != 52 (got {data.get('num_classes')})")
    else:
        result.passed("num_classes = 52")

    if data.get("expected_polarity") != "black_bg_white_ink":
        result.failed(
            f"expected_polarity != 'black_bg_white_ink' "
            f"(got '{data.get('expected_polarity')}')"
        )
    else:
        result.passed("expected_polarity = black_bg_white_ink")

    labels = data.get("labels", [])
    if labels != EXPECTED_LABELS:
        # 顯示前 5 個差異
        diffs = [
            f"[{i}] expected '{e}' got '{a}'"
            for i, (e, a) in enumerate(zip(EXPECTED_LABELS, labels))
            if e != a
        ][:5]
        result.failed(
            f"labels 順序錯誤（A-Z + a-z 52 個）。前 5 個差: {diffs}"
        )
    else:
        result.passed(f"labels = ['A', ..., 'Z', 'a', ..., 'z'] (52 letters)")

    case_pairs = data.get("case_pairs", [])
    if len(case_pairs) != 26:
        result.failed(f"case_pairs 數量 != 26 (got {len(case_pairs)})")
    else:
        result.passed(f"case_pairs has 26 entries")

    confusable = data.get("confusable_pairs", [])
    if len(confusable) < 5:
        result.warn(f"confusable_pairs only {len(confusable)} entries (expected ~11)")
    else:
        result.passed(f"confusable_pairs has {len(confusable)} entries")

    input_shape = data.get("input_shape")
    if input_shape != [28, 28, 1]:
        result.failed(f"input_shape != [28, 28, 1] (got {input_shape})")
    else:
        result.passed("input_shape = [28, 28, 1]")

    return data


def check_manifest(model_dir: Path, result: CheckResult) -> dict | None:
    print("\n[3/4] Checking training-manifest.json...")
    manifest_path = model_dir / "training-manifest.json"
    if not manifest_path.exists():
        result.failed(f"{manifest_path} not found")
        return None

    try:
        data = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        result.failed(f"training-manifest.json invalid: {e}")
        return None

    required_fields = [
        "version",
        "val_acc_top1",
        "val_acc_top2",
        "val_acc_top3",
        "non_confusable_letters_top1",
        "confusable_pairs_top1_mean",
        "confusable_pairs_top2_mean",
        "training_samples",
        "validation_samples",
        "epochs_trained",
        "architecture",
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        result.failed(f"missing fields: {missing}")
    else:
        result.passed("all required fields present")

    val_acc_top1 = data.get("val_acc_top1", 0)
    val_acc_top2 = data.get("val_acc_top2", 0)
    print(f"  Version: {data.get('version')}")
    print(f"  Architecture: {data.get('architecture')}")
    print(f"  val_acc top-1 = {val_acc_top1:.4f}")
    print(f"  val_acc top-2 = {val_acc_top2:.4f}")

    if val_acc_top1 < 0.80:
        result.warn(f"val_acc_top1 = {val_acc_top1:.4f} < 0.80（偏低）")
    elif val_acc_top1 < 0.85:
        result.warn(f"val_acc_top1 = {val_acc_top1:.4f} < 0.85（在 plan 預期下限）")
    else:
        result.passed(f"val_acc_top1 = {val_acc_top1:.4f} ≥ 0.85")

    if val_acc_top2 < 0.95:
        result.warn(f"val_acc_top2 = {val_acc_top2:.4f} < 0.95")
    else:
        result.passed(f"val_acc_top2 = {val_acc_top2:.4f} ≥ 0.95")

    return data


def check_weights_shard(model_dir: Path, model_data: dict, result: CheckResult) -> None:
    print("\n[4/4] Checking weight shards (.bin)...")
    if not model_data:
        result.failed("model.json 沒讀到，skip")
        return

    manifest = model_data.get("weightsManifest", [])
    if not manifest:
        result.failed("model.json 沒 weightsManifest")
        return

    # 預期 dtype size mapping
    dtype_sizes = {"float32": 4, "int32": 4, "uint8": 1, "bool": 1}

    for group_idx, group in enumerate(manifest):
        paths = group.get("paths", [])
        weights = group.get("weights", [])

        expected_bytes = 0
        for w in weights:
            shape = w["shape"]
            dtype = w.get("dtype", "float32")
            count = 1
            for d in shape:
                count *= d
            expected_bytes += count * dtype_sizes.get(dtype, 4)

        actual_bytes = 0
        for p in paths:
            shard_path = model_dir / p
            if not shard_path.exists():
                result.failed(f"shard {p} not found")
                continue
            actual_bytes += shard_path.stat().st_size

        if actual_bytes == expected_bytes:
            result.passed(
                f"group {group_idx}: {len(weights)} weights = "
                f"{expected_bytes:,} bytes ({expected_bytes/1024:.1f} KB) matches"
            )
        else:
            result.failed(
                f"group {group_idx}: expected {expected_bytes:,} bytes, "
                f"got {actual_bytes:,} bytes (diff {actual_bytes - expected_bytes:+,})"
            )


def run_inference_check(model_dir: Path, manifest: dict, result: CheckResult) -> None:
    """Optional: 用 tensorflowjs python 反向 load model + 跑 EMNIST 100 張驗 accuracy。
    需要 pip install tensorflowjs tensorflow-datasets。"""
    print("\n[5/5] Optional inference check (requires tensorflowjs + tfds)...")
    try:
        import tensorflowjs as tfjs
        import tensorflow_datasets as tfds
        import numpy as np
    except ImportError as e:
        result.warn(
            f"skip inference check: {e}. "
            f"裝法: pip install tensorflowjs tensorflow-datasets"
        )
        return

    try:
        # tfjs 反向 load
        model = tfjs.converters.load_keras_model(str(model_dir / "model.json"))
    except (AttributeError, TypeError) as e:
        result.warn(f"tfjs.converters.load_keras_model failed: {e}")
        return

    print("  Loading EMNIST byclass test split (filter letters)...")
    ds = tfds.load("emnist/byclass", split="test", as_supervised=True, batch_size=-1)
    x, y = tfds.as_numpy(ds)
    x = np.transpose(x.squeeze(-1), (0, 2, 1))
    mask = y >= 10
    x_letters = x[mask][..., None].astype(np.float32) / 255.0
    y_letters = (y[mask] - 10).astype(np.int64)

    np.random.seed(42)
    idx = np.random.choice(len(x_letters), 100, replace=False)
    x_sample = x_letters[idx]
    y_sample = y_letters[idx]

    probs = model.predict(x_sample, verbose=0)
    top_1 = probs.argmax(axis=1)
    top_2 = np.argsort(-probs, axis=1)[:, :2]
    top_1_acc = float((top_1 == y_sample).mean())
    top_2_acc = float((top_2 == y_sample[:, None]).any(axis=1).mean())

    manifest_top1 = manifest.get("val_acc_top1", 0)
    diff = abs(top_1_acc - manifest_top1)
    print(f"  Sanity 100 random EMNIST samples:")
    print(f"    top-1: {top_1_acc:.4f} (manifest: {manifest_top1:.4f}, diff {diff:.4f})")
    print(f"    top-2: {top_2_acc:.4f} (manifest: {manifest.get('val_acc_top2', 0):.4f})")
    if diff < 0.05:
        result.passed(f"inference top-1 跟 manifest 差 {diff:.4f} < 5pp")
    elif diff < 0.10:
        result.warn(
            f"inference top-1 跟 manifest 差 {diff:.4f}（可接受但偏高，"
            f"可能 polarity/preprocess 微差或 100 張 sample 抖動）"
        )
    else:
        result.failed(
            f"inference top-1 跟 manifest 差 {diff:.4f} > 10pp — "
            f"polarity 沒對齊？patch script 改錯 weight？"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).parent / "letter-tfjs-model",
        help="Directory containing model.json / classes.json / etc.",
    )
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="跑 EMNIST 100 張 sample inference 驗 accuracy（需 tensorflowjs + tfds）",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    print(f"Sanity check model_dir: {model_dir}")
    if not model_dir.exists():
        print(red(f"❌ {model_dir} 不存在"))
        return 1

    result = CheckResult()
    model_data = check_model_json(model_dir, result)
    check_classes_json(model_dir, result)
    manifest = check_manifest(model_dir, result)
    if model_data:
        check_weights_shard(model_dir, model_data, result)

    if args.run_inference and manifest:
        run_inference_check(model_dir, manifest, result)

    print(f"\n{'=' * 50}")
    print(f"Summary: {green(str(len(result.passes)) + ' pass')}, "
          f"{red(str(len(result.fails)) + ' fail')}, "
          f"{yellow(str(len(result.warns)) + ' warn')}")
    print(f"{'=' * 50}")

    return 0 if not result.fails else 1


if __name__ == "__main__":
    sys.exit(main())
