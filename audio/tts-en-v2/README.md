# tts-en-v2 — 英文預錄語音（開源模型）

3 種聲音，各 415 句，共 **1,245 個 MP3**（約 10 MB）。取代 `audio/tts-en/`（edge-tts 生成、沒有授權依據）。

由 [pig-english](https://github.com/jhs730127/pig-english) 的錄音清單生成：app 裡所有會被唸的固定英文，包括詞庫 311 字、字母關鍵字、世界名稱、讚美語、月份星期、顏色形狀、故事單字。

**不含單一字母（A–Z）**：Kokoro 遇到這麼短的輸入，會挑一種講很快的說話風格，A、I、O 只有 0.2–0.3 秒，試聽時聽起來沒唸完或唸錯。pig-english 的字母改用裝置內建聲音唸。細節見 `scripts/gen-tts-en-v2/README.md`。

## 聲音

| 資料夾 | 模型 / 聲音 | pig-english 設定 |
|---|---|---|
| `af_heart/` | Kokoro-82M `af_heart` | 姐姐／老師（年輕活潑） |
| `af_bella/` | Kokoro-82M `af_bella` | 媽媽／阿姨（成熟溫暖） |
| `am_michael/` | Kokoro-82M `am_michael` | 爸爸／哥哥（沉穩） |

## 目錄結構與命名

```
audio/tts-en-v2/
├── utterances.json          ← 錄音清單：[{ category, slug, text, spoken? }]
├── overrides.json           ← 個別句子改用的生成參數（見重產）
└── {voice}/{slug}.mp3       ← 每一句一個檔
```

`slug` 規則：轉小寫、非英數字元換成 `-`，例如 `Mother's Day` → `mother-s-day.mp3`、`Great job!` → `great-job.mp3`。

## 格式

單聲道、24 kHz、MP3 64 kbps。頭尾去靜音（保留 50ms），峰值正規化到 −3 dBFS。

## 重產

見 `scripts/gen-tts-en-v2/README.md`。

## 授權

用 Kokoro-82M 生成（hexgrad，程式與權重都是 Apache-2.0，對生成的音檔沒有限制）。這批音檔跟本 repo 其他內容一樣用 MIT 授權，見根目錄 LICENSE。

注意：Kokoro 模型卡寫明，它的訓練資料包含大廠閉源 TTS 生成的合成語音。
