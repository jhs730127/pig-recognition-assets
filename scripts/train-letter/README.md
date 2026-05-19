# Letter Recognition Training（A-Z + a-z，52 類）

訓練 single-letter handwriting classifier，給**之後的新專案**用。**獨立於 pig-math 既有 digit model**（不會動 `tensorflow-fallback.ts` 或 `public/models/v1/`）。

> 訓練架構、dataset、augmentation 等決策見 `~/.claude-wf/plans/magical-scribbling-sun.md`。

## Quick Start — Colab 網頁版（**推薦**）

Notebook 是 self-contained 設計（`%%writefile` 把 .py 內容 inline 進 cell），Colab 網頁版直接上傳即可，不需 git clone 或 GitHub OAuth。

1. 打開 https://colab.research.google.com
2. **File → Upload notebook** → 選本機 `scripts/letter-training/train_letter_model.ipynb`
3. **Runtime → Change runtime type → T4 GPU**（免費）
4. **Runtime → Run all**
5. Cell 2 跑 2-5 分鐘下載 ~700MB deps
6. Cell 8-9 pre-flight schema check ~5 分鐘
7. Cell 11 真正訓練 60-90 分鐘
8. 最後 zip 下載 cell 點按鈕拿 `letter-tfjs-model.zip`
9. unzip 拿去新專案 `public/models/letter-v1/`

中間想 sanity check（看 val_acc、confusion pair stats），cell 14 印整套指標。

## Alternative — VS Code + Colab extension（**可能不穩**）

VS Code 開本機 ipynb，kernel 選 Google Colab。理論上方便（本機編輯 + 雲端 GPU），實際上 `google.colab` VS Code extension v0.8.1 跟新版 VS Code (1.120+) / Jupyter extension (2025.9+) 有 **WebSocket subprotocol 協商失敗** bug，kernel 起來後 0.15 秒就斷線、cell 看起來「卡住」但其實沒執行。

如果遇到 log 內出現以下任一：

```
Error in websocket [Error: Server sent no subprotocol
Canceled future for execute_request message before replies were done
```

**立刻改用 Colab 網頁版**（上面那條路），不要 debug VS Code extension。

如果想試（VS Code → 右上「選取核心」→ Google Colab → T4 GPU）跑得通就跑，跑不通直接撤退。

## Alternative — 本機 Mac

## Alternative — 本機 Mac

```bash
cd /Users/jerry.wu/Project/Math
python3 -m venv .venv-train && source .venv-train/bin/activate
pip install -r scripts/letter-training/requirements.txt
python scripts/letter-training/train_letter_model.py --output ./scripts/letter-training/tfjs-model
python scripts/letter-training/patch_tfjs_model_json.py ./scripts/letter-training/tfjs-model/model.json
```

M1+ Mac 自動用 Metal GPU，約 1.5-2 小時。CPU 約 4-6 小時。

## 訓練資料

| Dataset | 樣本數 | Source |
|---|---|---|
| EMNIST ByClass（filter letters 10-61） | ~580k train + ~110k test | `tensorflow_datasets` 自動下載 |
| **合計** | **~690k letter samples** | 不需手動 download |

EMNIST ByClass 是 NIST SD-19 的 superset 已 preprocessed 28×28，不再加 IAM / QuickDraw 等其他 dataset（domain shift 大易降 val_acc）。

## 模型架構

同 pig-math digit model backbone，head 從 Dense(10) 換成 Dense(52)，中間 Dense 從 128 升 256（52 類需要 capacity）：

```
Conv2D(32) → Conv2D(32) → MaxPool → Dropout
Conv2D(64) → Conv2D(64) → MaxPool → Dropout
Flatten → Dense(256) → Dropout → Dense(52, softmax)
```

約 650k params，輸出 TF.js layers format，**前端 `loadLayersModel` 不用改 code**（跟 digit model 同 backbone）。

Bundle size 估計約 **4.5MB**。

## 驗收指標

訓完 script 會輸出（基於 EMNIST byclass SOTA range）：

| Metric | 預期 | 解釋 |
|---|---|---|
| `val_acc_top1` | **0.86-0.89** | EMNIST byclass top-1 上限（同形混淆 ceiling）|
| `val_acc_top2` | **0.95-0.97** | 大多數錯案 top-2 內找得回來 |
| `val_acc_top3` | **0.97-0.98** | 應用層 fallback 用 |
| `non_confusable_letters_top1` | **0.96-0.98** | A/B/D/E/F/G/H 等「正常字母」表現 |
| `confusable_pairs_top1` | ~0.70 | C/c, O/o, S/s, P/p, X/x, Z/z, K/k, W/w, V/v, U/u, M/m |
| `confusable_pairs_top2` | ~0.95 | 上面這些對的 top-2 acc |

`confusion_pairs.json` 詳列每對的 mutual confusion percentage，給應用層做 case-insensitive 後處理用。

## 輸出檔案格式

`./letter-tfjs-model/` 內：

```
model.json                 # TF.js layers format model architecture
group1-shard1of1.bin       # weights (~4MB)
classes.json               # 52 label mapping + expected_polarity + confusable_pairs
confusion_pairs.json       # per-pair confusion stats from validation
training-manifest.json     # val_acc top-1/2/3 + per-class samples + architecture
```

### `classes.json` 範例（inference 必讀）

```json
{
  "version": "v1-emnist-byclass-52",
  "num_classes": 52,
  "expected_polarity": "black_bg_white_ink",
  "input_shape": [28, 28, 1],
  "labels": ["A","B","C",...,"Z","a","b","c",...,"z"],
  "case_pairs": [[0,26],[1,27],...,[25,51]],
  "confusable_pairs": [[2,28],[14,40],[18,44],...]
}
```

`labels[i]` = 顯示字元，順序 0-25 = A-Z, 26-51 = a-z。Inference 時 `label = classes.labels[argmax(probs)]`。

### Inference 範例（給未來新專案）

完整可 copy-paste 的 helper：見 **`inference_example.ts`**（`LetterRecognizer` class 含 top-K 跟 case-insensitive 預測模式）。

簡化版：

```ts
import * as tf from '@tensorflow/tfjs';

const model = await tf.loadLayersModel('/models/letter-v1/model.json');
const classes = await fetch('/models/letter-v1/classes.json').then(r => r.json());

// classes.expected_polarity === 'black_bg_white_ink'，前端拿到白底黑字 → 翻黑底白字
const grayPixels = preprocessTo28x28(canvas);  // 白底黑字 [0, 255]
const input = tf.tidy(() => {
  const tensor = tf.tensor3d(grayPixels, [28, 28, 1]);
  const inverted = tf.sub(255, tensor);              // 翻成黑底白字
  return tf.div(inverted, 255).expandDims(0);        // → [1, 28, 28, 1] in [0,1]
});

const probs = (model.predict(input) as tf.Tensor).dataSync();
const topK = Array.from(probs)
  .map((p, i) => ({ label: classes.labels[i], confidence: p }))
  .sort((a, b) => b.confidence - a.confidence)
  .slice(0, 3);

console.log(topK);  // [{label:'A', confidence:0.91}, {label:'a', confidence:0.07}, ...]
```

## Sanity check（下載 model 後第一件事）

訓完 zip 解壓到 `letter-tfjs-model/` 後，本機跑：

```bash
python scripts/letter-training/sanity_check.py
```

驗 4 件事：
1. **model.json schema**：所有 layer class 都是 TF.js 4.22 supported；InputLayer batch_input_shape、layer dtypes、weight names 都已被 `patch_tfjs_model_json.py` 處理
2. **classes.json**：52 labels 順序 = `[A...Z, a...z]`、expected_polarity = `black_bg_white_ink`、case_pairs/confusable_pairs 數量對
3. **training-manifest.json**：val_acc_top1 ≥ 0.85、required 欄位齊
4. **weights .bin**：實際 byte size 跟 model.json metadata 一致

`16 pass / 0 fail / 0 warn` 才算過。任何 ❌ 都代表 model **無法 deploy 到 TF.js 前端**，回頭 debug。

進階 inference 驗證（需 `pip install tensorflowjs tensorflow-datasets`）：

```bash
python scripts/letter-training/sanity_check.py --run-inference
```

跑 100 張 EMNIST test sample，inference top-1 跟 manifest val_acc_top1 差 < 5pp 才過。差太大 = polarity 沒對齊 / patch 改錯 weight。

## Known gotchas（從 pig-math digit pipeline 繼承）

1. **Polarity 一致**：訓練黑底白字 → inference 前端必須 `tf.sub(255, x)` 翻過來。`classes.json:expected_polarity` 寫死，前端 load 時建議 assert
2. **`patch_tfjs_model_json.py` 必跑**：Keras 3 → TF.js 4.22 schema patch（InputLayer batch_shape、DTypePolicy dict、weight name prefix 三項）。notebook 已自動跑
3. **Stub `tensorflow_decision_forests`**：tfjs 4.22 強行 import tfdf 但裝 tfdf 會把 TF 降版衝突。train_letter_model.py 主檔已內建 stub
4. **EMNIST transpose**：raw 是 column-major 要 `np.transpose(x, (0, 2, 1))` 翻回 row-major。已在 `load_emnist_letters()` 處理
5. **~~不要加 BatchNorm~~**（**已破除**）：v3 model 帶 BN，sanity_check.py 確認 TF.js 4.22 支援。pig-math digit pipeline 一直避開是過時擔心。未來新專案放心用 BN
6. **`google.colab.files.download` 在 VS Code Colab extension 不 work**：用 IPython HTML data URI link 解掉
7. **VS Code + Google Colab extension v0.8.1 WebSocket 不穩**：直接走 Colab 網頁版（見 Quick Start）

## Cross-person sanity check（**很重要**）

EMNIST tfds 預設 val split 是隨機抽 sample，**不是按書寫者**抽。對「跨人泛化」測不準。

**Colab 報 87% 不代表 iPad 實機 87%。** pig-math 一開始踩過的坑：colab val_acc 99% 但實機掉到 50%（domain gap）。

**訓完強烈建議**：
1. 自己用 iPad 手寫 20-50 個 letter（每個 unique class 1-2 個）
2. 一張一張跑 inference 看 top-1 / top-2 accuracy
3. 如果跟 val_acc 差 > 15pp，可能 polarity / 預處理沒對齊，回看 inference 範例的 `tf.sub(255, x)`

## 跟 pig-math digit pipeline 的關係

| 項目 | Digit (pig-math) | Letter (本目錄 v3) |
|---|---|---|
| Output classes | 10 (0-9) | 52 (A-Z + a-z) |
| Dataset | MNIST + EMNIST digits + USPS (360k) | EMNIST ByClass filter letters (350k train + 60k test) |
| Architecture | Conv32-Conv32-Conv64-Conv64-Dense128-Dense10 | Conv32-Conv32-Conv64-Conv64-Dense512-Dense52 + **BN** |
| Bundle | ~2MB | ~3.65MB |
| Final val_acc top-1 | 0.997 | 0.871 |
| Production target | pig-math `public/models/v1/` | 未來新專案（model 不放 repo） |
| `requirements.txt` | 共用 | symlink → `../training/` |
| `patch_tfjs_model_json.py` | 共用 | symlink → `../training/` |

兩條 pipeline 完全獨立。改 letter 不會影響 digit，反之亦然。

## v1/v2/v3 訓練實驗總結

| Metric | v1 | v2 | v3 (current) |
|---|---|---|---|
| val_acc top-1 | 0.826 | 0.865 | **0.871** |
| val_acc top-2 | 0.978 | 0.984 | **0.986** |
| non-confusable letters top-1 | 0.890 | 0.903 | 0.908 |
| confusable pairs top-1 mean | 0.701 | 0.763 | 0.773 |
| Architecture 差異 | Dense 256, class_weight | Dense 256, label_smoothing | Dense 512, BN, light augment |
| Bundle | 1.95MB | 1.95MB | 3.65MB |

**結論**：non-confusable letters top-1 三版都打不上 0.94（plan 預期）。EMNIST byclass + CNN 架構的天花板就在這。再上去要走 Stage 2 fine-tune（收真實兒童手寫樣本）或換 dataset。

## 下一步（給 Claude / 自己）

v3 model 已 ready。**接下來做的順序**：

1. **跑 sanity_check.py**（已通過 16/16 ✅）
2. **Cross-person sanity check**：自己用 iPad 手寫 20-50 個 letter（每 unique class 1-2 個），用 `inference_example.ts` 跑 inference 看實機 top-1 / top-2 accuracy。如果跟 val_acc 差 > 15pp，polarity 或 preprocess 沒對齊
3. **部署到新專案**：把 `letter-tfjs-model/` 整個 copy 到新專案 `public/models/letter-v1/`
4. **整合 `LetterRecognizer`**：copy `inference_example.ts` 進新專案，按 README 範例用

未來「相機印刷字 word/line OCR」走另一條路（**Tesseract.js 6.0** 推薦 POC），plan 內有完整評估。

## 相關文件

- Plan: `~/.claude-wf/plans/magical-scribbling-sun.md`
- TS Inference SDK: `inference_example.ts`（LetterRecognizer class）
- Python Sanity Check: `sanity_check.py`
- 上層 digit pipeline: `scripts/training/README.md`
- pig-math repo: https://github.com/jhs730127/Math
