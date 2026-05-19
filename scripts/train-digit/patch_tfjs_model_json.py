"""
Patch tfjs model.json from Keras 3 schema to TF.js 4.x loader-compatible schema.

Keras 3 (Python TF 2.20) export 的 model.json 有 schema 差異：
- InputLayer: `batch_shape` field（Keras 3 新名）→ TF.js 4.x 期望 `batch_input_shape`
- 所有 layer 的 `dtype: {DTypePolicy dict}` → TF.js 期望 plain string 'float32'

跑法：python scripts/training/patch_tfjs_model_json.py public/models/v1/model.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def patch(model_json_path: Path) -> None:
    data = json.loads(model_json_path.read_text())
    mc = data["modelTopology"]["model_config"]
    model_name = mc.get("config", {}).get("name", "")
    config = mc["config"]
    n_patched_input = 0
    n_patched_dtype = 0
    for layer in config.get("layers", []):
        cfg = layer.setdefault("config", {})
        # 1. InputLayer: batch_shape → batch_input_shape
        if layer.get("class_name") == "InputLayer" and "batch_shape" in cfg:
            cfg["batch_input_shape"] = cfg.pop("batch_shape")
            n_patched_input += 1
        # 2. dtype: DTypePolicy dict → plain string
        if isinstance(cfg.get("dtype"), dict):
            try:
                cfg["dtype"] = cfg["dtype"]["config"]["name"]
                n_patched_dtype += 1
            except (KeyError, TypeError):
                pass
    # 3. Strip "{model_name}/" prefix from weights names（Keras 3 在 weight name 加 model prefix，
    #    TF.js 4.22 loader 不認）
    n_patched_weight = 0
    if model_name:
        prefix = f"{model_name}/"
        for group in data.get("weightsManifest", []):
            for w in group.get("weights", []):
                if w.get("name", "").startswith(prefix):
                    w["name"] = w["name"][len(prefix) :]
                    n_patched_weight += 1
    model_json_path.write_text(json.dumps(data, indent=2))
    print(
        f"Patched: InputLayer batch_shape × {n_patched_input}, "
        f"dtype dict → string × {n_patched_dtype}, "
        f"weight name prefix '{model_name}/' stripped × {n_patched_weight}"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path/to/model.json>", file=sys.stderr)
        sys.exit(1)
    patch(Path(sys.argv[1]))
