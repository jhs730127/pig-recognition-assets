# gen-tts-en-v2 — 用開源模型生成英文預錄語音

產出 `audio/tts-en-v2/`。在 Apple Silicon Mac 上本機跑，不需要 GPU 或 Colab；1,245 個檔約 5 分鐘。

## 環境

```bash
brew install espeak-ng            # Kokoro 的讀音推測（misaki）遇到詞典沒有的字要用
cd scripts/gen-tts-en-v2
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python mlx-audio "misaki[en]" piper-tts soundfile faster-whisper
```

第一次執行會從 Hugging Face 下載 Kokoro 模型（`prince-canuma/Kokoro-82M`，約 8 分鐘）。

## 錄音清單

清單由 pig-english 產生，來源是 app 裡所有會被唸的固定英文：

```bash
cd <pig-english repo>
npx --yes tsx scripts/export-utterances.ts <本 repo>/audio/tts-en-v2/utterances.json
```

## 生成

```bash
.venv/bin/python generate.py --utterances ../../audio/tts-en-v2/utterances.json \
    --out ../../audio/tts-en-v2 --voices kokoro:af_heart,kokoro:af_bella,kokoro:am_michael
```

- `--only A,sofa,tv`：只重生指定的 slug
- `--skip-existing`：已存在的檔不重生
- 清單裡如果有字母（category 為 `letter`），會輸出到 `{voice}/letters/`，A、I 用行內音標指定讀音（見 `generate.py` 的 `LETTER_PHONEMES`）。目前清單不含字母，原因見下方「字母為什麼不用 Kokoro」。

## 自動檢查

```bash
.venv/bin/python check.py --utterances ../../audio/tts-en-v2/utterances.json \
    --dir ../../audio/tts-en-v2 --report-dir tmp/check --voices af_heart,af_bella,am_michael --sample 20
```

檢查內容：

- 用 faster-whisper small 轉錄每個檔，跟原文比對
- 標出長度異常的檔：超過 2.5 秒或短於 0.15 秒
- 輸出 `report.json` 和試聽頁 `listen.html`：被標記的句子，以及每種聲音隨機抽 20 句

**音質最後要人工聽過**，自動檢查只能抓出明顯的錯誤。

## 修正沒過的句子

Kokoro 依音素長度挑風格向量，單字太短時字首常含糊，例如 fish 聽成 wish、key 聽成 I see。初版約 11% 的句子被自動檢查標記。

```bash
.venv/bin/python fix_flagged.py --utterances ../../audio/tts-en-v2/utterances.json \
    --dir ../../audio/tts-en-v2 --report tmp/check/report.json --voices af_heart,af_bella,am_michael
```

- 對每個被標記的句子依序試變體：改用較長輸入的風格向量（第 20、40、60、100 個）、放慢到 0.85、加驚嘆號，留第一個 ASR 聽對的。
- 選中的變體寫進 `audio/tts-en-v2/overrides.json`，`generate.py` 重產時照用。
- 全部變體都沒過的句子保留原檔，要人工聽。

試過但沒用的做法：

- **載體句**：先念「The word is fish.」，再依音素時長切出 fish。切點會偏後，字首子音被切掉。
- **加大去靜音的保留範圍**：沒效果。字首在去靜音之前就已經含糊了。

## 字母為什麼不用 Kokoro

v1.8.0 前試聽時，26 個字母三種聲音都被判定「聽起來沒唸完或唸錯」，所以不發布字母。

- **原因**：Kokoro 依輸入的音素數量挑說話風格（`pipeline.py` 的 `pack[len(ps)-1]`）。單一字母只有 1–3 個音素，挑到的是很趕的風格。A、I、O 這種雙母音只有 0.2–0.3 秒，edge-tts 舊檔約 0.4 秒。讀音（音素）本身是對的，結尾也不是後製剪掉的。
- **試過的變體**（faster-whisper 聽錯的字母數，三種聲音分別列出）：

  | 做法 | af_heart | af_bella | am_michael |
  |---|---|---|---|
  | 原本 | 3 | 3 | 6 |
  | 加句號 | 長度幾乎不變，沒效果 | | |
  | 句號＋改用第 40 個風格向量 | 15 | 15 | 15 |
  | 句號＋第 40 個風格向量＋放慢 0.85 | 23 | 20 | 21 |

  改用長句的風格向量後，長度變成 2 倍，但多出來的是雜音，例如 E 被聽成 "I'll leave you"。
- **pig-english 的做法**：字母改用裝置內建語音，送出前轉小寫。Apple 的聲音遇到單獨的大寫字母會唸成 "capital A"。

## 試過但沒採用

- **Piper `en_US-norman-medium`**（公有領域資料）：單個字母或很短的字會亂唸一長串，例如 `Z` 唸成 "So, yeah, yeah…"。
- **Piper `en_US-john-medium`**：sofa、giraffe、penguin 被自動檢查聽錯。
- 試聽時沒選上的 Kokoro 聲音：`af_nicole`、`am_fenrir`、`am_puck`。
