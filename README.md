# recognition-assets

跨專案共用的文字辨識 model + zh-TW TTS 音檔，透過 **jsdelivr CDN** 直接 serve，免部署。

訓練 / 生成所有資產的 script 也都在這，方便日後升版。

## Quick Start：jsdelivr CDN

URL pattern：

```
https://cdn.jsdelivr.net/gh/jhs730127/recognition-assets@<version>/<path>
```

例（鎖 v1.0.0）：

```ts
import * as tf from "@tensorflow/tfjs";

const BASE = "https://cdn.jsdelivr.net/gh/jhs730127/recognition-assets@v1.0.0";

// Letter (A-Z + a-z, 52 類)
const letterModel = await tf.loadLayersModel(`${BASE}/models/letter-v3/model.json`);
const letterClasses = await fetch(`${BASE}/models/letter-v3/classes.json`).then(r => r.json());

// Digit (0-9)
const digitModel = await tf.loadLayersModel(`${BASE}/models/digit-v1/model.json`);

// TTS MP3
const correctAudio = new Audio(`${BASE}/audio/tts-zh-tw/feedback_correct_1.mp3`);
correctAudio.play();
```

`@v1.0.0` 改成 `@main` 拿最新（無版本鎖，會自動跟著 main 更新），或 `@^1` 拿 major 1 的最新 minor。

完整 inference helper 見 `sdk/ts/letter-recognizer.ts`，直接 copy 到新專案即可用。

## 內容清單

### Models（TF.js layers format）

| 路徑 | 內容 | 類別 | Bundle | val_acc top-1 |
|---|---|---|---|---|
| `models/digit-v1/` | MNIST + EMNIST digits + USPS 360k samples | 10 (0-9) | ~1.1MB | 0.997 |
| `models/letter-v3/` | EMNIST ByClass letters subset 410k samples | 52 (A-Z + a-z) | ~3.5MB | 0.871 |

每個 model folder 含：
- `model.json` — TF.js layers format
- `group1-shard1of1.bin` — weights
- `training-manifest.json` — val_acc / per-class / training config
- `classes.json`（letter 才有）— label mapping + case_pairs + confusable_pairs
- `confusion_pairs.json`（letter 才有）— 同形混淆對 stats

### Audio（zh-TW edge-tts pre-generated MP3）

`audio/tts-zh-tw/` — 188 個 MP3，voice = Microsoft Edge TTS `zh-TW-YunJheNeural`（男聲）

| 分類 | 數量 | Naming pattern |
|---|---|---|
| 中文數字 0-100 | 101 | `num_zh_{0..100}.mp3` |
| 英文數字 0-9 | 10 | `num_en_{0..9}.mp3` |
| 運算符 | 9 | `op_{add,sub,mul,div,equals,and,or,is,which}.mp3` |
| 正確/錯誤 feedback | 22 | `feedback_{correct,wrong}{,_en}_{1..6}.mp3` |
| 中場休息 | 6 | `break_{1..6}.mp3` |
| 題號（第 N 題） | 20 | `problem_n_{1..20}.mp3` |
| 描紅引導 | 12 | `trace_{zh_{0..9},intro,great}.mp3` |
| 連勝里程碑 | 5 | `streak_{3,5,10,15,20}.mp3` |
| 問句結尾 | 2 | `q_equals_{how,many}.mp3` |

`manifest.json` 提供 key → 文本對應表。

### SDK

| 檔 | 用途 |
|---|---|
| `sdk/ts/letter-recognizer.ts` | LetterRecognizer class — top-K + case-insensitive 預測 + React hook 範例 |

### 訓練 / 生成 scripts

| 路徑 | 用途 |
|---|---|
| `scripts/train-digit/` | Digit model Colab 訓練 pipeline（MNIST + EMNIST + USPS, 14 epoch on T4） |
| `scripts/train-letter/` | Letter model Colab 訓練 pipeline（EMNIST ByClass, 35 epoch on T4） |
| `scripts/gen-tts/` | edge-tts 預生成 MP3 Colab notebook（phase 1a sample → phase 1b 量產） |

每個 folder 內有 `README.md` 跑法說明。

## 怎麼新版加進去

新訓 model / 新生 TTS 後：

```bash
cd ~/Project/recognition-assets

# 1. 替換 asset 檔
cp -r 新訓的 model 進 models/letter-v4/

# 2. commit + push + tag
git add models/letter-v4/
git commit -m "feat: add letter v4 (val_acc top-1 0.91)"
git push origin main
git tag v1.1.0
git push origin v1.1.0

# 3. 開 GitHub Release
gh release create v1.1.0 --title "v1.1.0 letter v4" --notes "..."

# 4. 等 jsdelivr propagate (1-5 min)
curl -sI https://cdn.jsdelivr.net/gh/jhs730127/recognition-assets@v1.1.0/models/letter-v4/model.json
```

新專案用 `@v1.1.0` 或 `@^1` 自動升級。

## 版本對應

| Tag | 日期 | 內容 |
|---|---|---|
| v1.0.0 | 2026-05-19 | Initial release — digit v1, letter v3, TTS zh-TW YunJhe 188 個 |

## 已知限制

- **letter model non-confusable top-1 0.91** vs EMNIST byclass SOTA ~0.97 — 屬於 EMNIST dataset 天花板（label noise + 隨機 val split 不是按書寫者）。實機跨人 accuracy 可能 65-80%
- **letter model 11 個同形混淆對**（C/c, K/k, M/m, O/o, P/p, S/s, U/u, V/v, W/w, X/x, Z/z）top-1 mean 0.77、top-2 mean 0.99。應用層走 top-K + 後處理體感較好
- **digit model 是 pig-math 專案訓**，跟 letter model 用同 backbone 但不同 dataset，**不可共 weight**
- **TTS voice 固定 zh-TW-YunJheNeural**，要不同 voice 重跑 `scripts/gen-tts/phase1b_full_generation.ipynb` 換 voice 參數

## LICENSE

MIT — 見 LICENSE 檔。Model 是 EMNIST / MNIST / USPS 公開 dataset trained 的 vanilla CNN，無 IP claim。TTS MP3 由 Microsoft Edge TTS API 生成（自由使用）。

## Source

由個人 pig-math (math-game) 專案抽出共用，維持兩邊各自獨立。pig-math 不依賴此 repo（model + TTS 各自有一份本機 copy 跑 production）。
