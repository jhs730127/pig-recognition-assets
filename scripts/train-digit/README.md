# Stage 1: Broad-dataset Fine-tune Training

訓練 pig-math worksheet 本機 OCR 的 base model — 用 MNIST + EMNIST + USPS 約 360k 樣本重訓 weights，預期 50% → 65-75% 兒童手寫辨識率。詳見 `docs/plans/2026-05-11-handwriting-finetune-broad-dataset.md`。

## Quick Start — VS Code + Colab extension（**推薦**）

最省事的路線：本機編輯 + Colab GPU。

1. VS Code 擴充功能搜「Google Colab」安裝
2. VS Code 開 `scripts/training/train_digit_model.ipynb`
3. 右上「選取核心」→ **Colab** → 選 **T4 GPU**
4. 登入 Google 帳號
5. **Run All**，等 30-60 分鐘
6. 最後一格自動下載 `pig-math-tfjs-model.zip` 到本機
7. 把 zip 拖給 Claude 接後續

中間若想 sanity check（看 training curve、confusion matrix），可以一格一格跑。

## Alternative — Colab 網頁版（一條龍 CLI）

開 https://colab.research.google.com → 新增筆記本 → T4 GPU → 4 個 cell：

```python
!git clone https://github.com/jhs730127/Math.git
%cd Math
```
```python
!pip install -q -r scripts/training/requirements.txt
```
```python
!python scripts/training/train_digit_model.py --output ./tfjs-model
```
```python
from google.colab import files
import shutil
shutil.make_archive("pig-math-tfjs-model", "zip", "tfjs-model")
files.download("pig-math-tfjs-model.zip")
```

## Alternative — 本機 Mac

```bash
cd /path/to/pig-recognition-assets
python3 -m venv .venv-train
source .venv-train/bin/activate
pip install -r scripts/train-digit/requirements.txt
python scripts/train-digit/train_digit_model.py --output ./scripts/train-digit/tfjs-model
```

M1+ Mac 會自動用 Metal GPU，約 1-2 小時。CPU 約 4-6 小時。

## 訓練資料

| Dataset | 樣本數 | Source |
|---|---|---|
| MNIST | 70k | `tf.keras.datasets.mnist`（自動下載） |
| EMNIST digits | 280k | `tensorflow_datasets`（自動下載） |
| USPS | 9.3k | HuggingFace `flwrlabs/usps`（自動下載） |
| **合計** | **~360k** | 不需手動 download |

USPS 因為要靠 HuggingFace 拉，網路若斷可 fallback：

```bash
python scripts/training/train_digit_model.py --output ./tfjs-model --no-usps
```

## 模型架構

對齊 Google 提供的 `mnist_transfer_cnn_v1`（Keras MNIST CNN）：

```
Conv2D(32) → Conv2D(32) → MaxPool → Dropout
Conv2D(64) → Conv2D(64) → MaxPool → Dropout
Flatten → Dense(128) → Dropout → Dense(10, softmax)
```

約 600k params，輸出 TF.js layers format（model.json + group1-shard*.bin），**可直接換掉 worksheet 的 MODEL_URL，前端 `loadLayersModel` 不用改 code**。

## 驗收指標

訓完 script 會輸出：
- Overall val_acc（**目標 ≥ 99%**，這部分不難）
- 完整 confusion matrix
- **6/8/9/0 → 預測 3 的誤判率**（pig-math 主訴的 domain gap，**目標 < 5%**）
- `training-manifest.json` 含上述數字

真正驗收要等 model 部署到 pig-math 後實機測（你切「離線辨識」寫 worksheet 看是否改善 50% → 65%+）。

## 輸出檔案

`./tfjs-model/` 內：

- `model.json`（model architecture + weight manifest）
- `group1-shard1of2.bin`、`group1-shard2of2.bin`（weights，總共 ~2-3MB）
- `training-manifest.json`（訓練統計，含 6/8/9 confusion）

## 下一步

把 `pig-math-tfjs-model.zip` 給我（drag 進對話 / `~/Downloads` 路徑），我會：

1. 上傳到 Supabase Storage `pig-math-models/v1/`
2. 改 `src/lib/recognition/tensorflow-fallback.ts:MODEL_URL`
3. 升 `MODEL_CACHE_KEY` 強制 user IndexedDB 重抓
4. 部署 prod
5. 你再切「離線辨識」實測驗收

如果第一版 accuracy 不夠（< 65% 在 pig-math 真實樣本），會考慮加 DIDA / HWD+ / kensanata（手動 download）重訓 v2。
