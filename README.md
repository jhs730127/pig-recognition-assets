# pig-recognition-assets

跨專案共用的文字辨識 model + zh-TW TTS 音檔，透過 **jsdelivr CDN** 直接 serve，免部署。

訓練 / 生成所有資產的 script 也都在這，方便日後升版。

## 📖 文件導覽

| 文件 | 給誰 |
|---|---|
| **[INTEGRATION.md](./INTEGRATION.md)** | **其他專案** — 完整整合指南（Next.js / Vite / HTML / CSP / 效能優化 / 錯誤處理） |
| **[CONTRIBUTING.md](./CONTRIBUTING.md)** | **Repo maintainer** — 上架新 model / TTS 的 step-by-step 流程、版本規則、OPSEC checklist |
| 本檔（README）| 快速概覽 + 內容清單 |

## Quick Start：jsdelivr CDN

URL pattern：

```
https://cdn.jsdelivr.net/gh/jhs730127/pig-recognition-assets@<version>/<path>
```

例（鎖 v1.2.0）：

```ts
import { DigitRecognizer } from "./digit-recognizer";
import { LetterRecognizer } from "./letter-recognizer";
import { PrerenderedTtsPlayer } from "./prerendered-tts-player";

const BASE = "https://cdn.jsdelivr.net/gh/jhs730127/pig-recognition-assets@v1.2.0";

// Digit (0-9) — 含重心置中 / TTA / 多位數切割
const digit = new DigitRecognizer({
  modelUrl: `${BASE}/models/digit-v1/model.json`,
  cacheKey: "indexeddb://my-app-digit-v1",
});
await digit.load();
const { digit: n, confidence } = await digit.predictFromDataUrl(dataUrl, { useTta: true });

// Letter (A-Z + a-z, 52 類)
const letter = new LetterRecognizer();
await letter.load(`${BASE}/models/letter-v3/model.json`, `${BASE}/models/letter-v3/classes.json`);
const top = letter.predict(imageData, 3);

// TTS — Web Audio API 精準排程組合句（5 減 1 等於多少）
const tts = new PrerenderedTtsPlayer({ baseUrl: `${BASE}/audio/tts-zh-tw` });
tts.setupAudioUnlock();
await tts.playSentence(["num_zh_5", "op_sub", "num_zh_1", "q_equals_what"]);
```

`@v1.2.0` 改成 `@main` 拿最新（無版本鎖，會自動跟著 main 更新），或 `@^1` 拿 major 1 的最新 minor。

完整 inference / TTS / voice helper 見 `sdk/ts/`，直接 copy 到新專案即可用。

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

### Audio

**`audio/tts-zh-tw/`** — 188 個 zh-TW MP3，voice = Microsoft Edge TTS `zh-TW-YunJheNeural`（男聲，沉穩雲哲）

**`audio/tts-zh-tw-hsiaochen/`**（v1.3.0+）— 188 個 zh-TW MP3，voice = `zh-TW-HsiaoChenNeural`（女聲，溫柔曉臻）— 同 phrase 集合與 manifest schema，純 voice 替換

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

**`audio/tts-en/`** — 558 個 en-US MP3（3 voice × 186 utterance），由 [pig-english](https://github.com/jhs730127/pig-english) 用 Microsoft Edge TTS 生成

| Slug | Voice | 風格 |
|---|---|---|
| `aria/` | `en-US-AriaNeural` rate +0% pitch +5Hz | 年輕活潑（姐姐/老師） |
| `jenny/` | `en-US-JennyNeural` rate -8% pitch +0Hz | 成熟溫暖（媽媽/阿姨） |
| `guy/` | `en-US-GuyNeural` rate -5% pitch -2Hz | 沉穩男聲（爸爸/哥哥） |

每 voice 子目錄含 `letters/A-Z`、`numbers/one..six`、`praise/great_job 等 8 個`、`words/cat 等 ~146 個`。詳見 `audio/tts-en/README.md`。

### Fonts / 字型資料

**`fonts/letter-stroke-order/`** — 52 個英文字母（A-Z + a-z）SVG path stroke order，含四線格規格。給字母筆順教學 app 用。

| 檔 | 內容 |
|---|---|
| `strokes.json` | 52 字母 stroke paths + `start`/`length`（length 可直接做 stroke-dasharray 動畫） |
| `grid-spec.json` | viewBox 320×320，四線格 cap-line=60 / midline=140 / baseline=260 / descender=300 |

詳見 `fonts/letter-stroke-order/README.md`。

### Data / 詞庫資料

**`data/vocab-en-preschool/`** — 152 個學齡前英文單字（~4-6 歲）+ emoji + 中文 + 音節 + tag。生活化詞庫，不含罕見抽象字。

| 檔 | 內容 |
|---|---|
| `words.json` | 152 字完整 metadata（id/en/zh/emoji/syllables/tags） |

詳見 `data/vocab-en-preschool/README.md`。

### SDK

| 檔 | 用途 |
|---|---|
| `sdk/ts/digit-recognizer.ts` | DigitRecognizer class — 0-9 預測，內建 MNIST 重心置中 + TTA (±4°/±8°) + 多位數切割 + IndexedDB cache |
| `sdk/ts/letter-recognizer.ts` | LetterRecognizer class — top-K + case-insensitive 預測 |
| `sdk/ts/prerendered-tts-player.ts` | PrerenderedTtsPlayer class — Web Audio API + trim silence + iOS unlock + 組合句精準排程（取代 `new Audio().play()` 卡頓） |
| `sdk/ts/voice-input-parser.ts` | VoiceInputRecognizer class + `parseChineseNumber` util — 麥克風答題（「十二」→12） |

### 訓練 / 生成 scripts

| 路徑 | 用途 |
|---|---|
| `scripts/train-digit/` | Digit model Colab 訓練 pipeline（MNIST + EMNIST + USPS, 14 epoch on T4） |
| `scripts/train-letter/` | Letter model Colab 訓練 pipeline（EMNIST ByClass, 35 epoch on T4） |
| `scripts/gen-tts/` | zh-TW edge-tts 預生成 MP3 Colab notebook（phase 1a sample → phase 1b 量產） |
| `scripts/gen-tts-en/` | en-US edge-tts 本機 CPU pipeline（3 voice × 186 utterance），含 voice sampler |

每個 folder 內有 `README.md` 跑法說明。

## 怎麼新版加進去

新訓 model / 新生 TTS 後：

```bash
cd ~/Project/pig-recognition-assets

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
curl -sI https://cdn.jsdelivr.net/gh/jhs730127/pig-recognition-assets@v1.1.0/models/letter-v4/model.json
```

新專案用 `@v1.1.0` 或 `@^1` 自動升級。

## 版本對應

| Tag | 日期 | 內容 |
|---|---|---|
| v1.0.0 | 2026-05-19 | Initial release — digit v1, letter v3, TTS zh-TW YunJhe 188 個 |
| v1.1.0 | 2026-05-19 | 加 en-US TTS 558 mp3（aria/jenny/guy 3 voice）、字母筆順 52 字、學齡前詞庫 152 字、en TTS pipeline script — 由 pig-english 貢獻 |
| v1.2.0 | 2026-05-20 | 加 3 個 SDK — `digit-recognizer`（含 MNIST 重心置中 + TTA + 多位數切割）、`prerendered-tts-player`（Web Audio API + trim silence + iOS unlock 精準組合句）、`voice-input-parser`（中文數字 + Web Speech wrapper） — 由 pig-math 貢獻 |
| v1.3.0 | 2026-05-21 | 加 `audio/tts-zh-tw-hsiaochen/` 188 MP3（zh-TW HsiaoChenNeural 女聲）— 與 v1.0.0 的 YunJhe 男聲同 phrase 集合與 manifest schema，純 voice 替換給選擇 |

## 已知限制

- **letter model non-confusable top-1 0.91** vs EMNIST byclass SOTA ~0.97 — 屬於 EMNIST dataset 天花板（label noise + 隨機 val split 不是按書寫者）。實機跨人 accuracy 可能 65-80%
- **letter model 11 個同形混淆對**（C/c, K/k, M/m, O/o, P/p, S/s, U/u, V/v, W/w, X/x, Z/z）top-1 mean 0.77、top-2 mean 0.99。應用層走 top-K + 後處理體感較好
- **digit model 是 pig-math 專案訓**，跟 letter model 用同 backbone 但不同 dataset，**不可共 weight**
- **zh-TW TTS 目前 2 個 voice**：YunJhe（男聲）/ HsiaoChen（女聲）— 要加新 voice 重跑 `scripts/gen-tts/phase1b_full_generation.ipynb` 改 `ZH_VOICE` 參數

## LICENSE

MIT — 見 LICENSE 檔。Model 是 EMNIST / MNIST / USPS 公開 dataset trained 的 vanilla CNN，無 IP claim。TTS MP3 由 Microsoft Edge TTS API 生成（自由使用）。

## Source

由個人 pig-math (math-game) 專案抽出共用，維持兩邊各自獨立。pig-math 不依賴此 repo（model + TTS 各自有一份本機 copy 跑 production）。
