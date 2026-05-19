# pig-math TTS 預生成 pipeline

把網站常用語音(數字、運算詞、固定回饋、休息提示)用 **edge-tts** 一次性離線生成為 MP3，
跑生產時直接播放靜態檔，取代 Web Speech API 的不穩定音色。

## 為什麼用 edge-tts

- 免費、不需 API Key、無額度限制
- 使用 Microsoft Neural TTS 雲端模型，**生成在 Colab 端**，**生產環境播放本地 MP3**
- zh-TW 三個高品質 voice：
  - `zh-TW-HsiaoChenNeural` — 成熟女聲，溫暖
  - `zh-TW-HsiaoYuNeural` — 年輕女聲，活潑
  - `zh-TW-YunJheNeural` — 男聲，沉穩

## 流程

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Phase 1A (試聽)  │ →  │ Phase 1B (全量)  │ →  │ Phase 2 (整合)   │
│ 20 句 × 4 voice  │    │ ~300 句 × 1 voice│    │ prerendered-tts  │
│ Colab → 試聽 zip │    │ Colab → MP3 zip  │    │ + speech.ts 改   │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

## Phase 1A：試聽 ✓ 完成

開 Colab 跑 `phase1a_voice_samples.ipynb`。User 試聽後決定 **zh-TW-YunJheNeural**（沉穩男聲）。

## Phase 1B：全量生成（**現在這裡**）

開 Colab 跑 `phase1b_full_generation.ipynb`，輸出 `audio-tts.zip`（~188 個 MP3 + manifest.json）。

清單（從 codebase 抽出，與 src/lib/audio/speech.ts、FeedbackOverlay.tsx、BreakReminder.tsx 對齊）：
- 數字 zh 0-100（101 檔）
- 數字 en 0-9（10 檔，practice page 用）
- 運算詞（10 檔：加/減/乘以/除以/等於/和/哪個比較大/還是/是/不是）
- 題號 第一題～第二十題（20 檔）
- 答對/答錯鼓勵 zh+en（22 檔）
- 連勝里程碑（5 檔）
- 中場休息（6 檔，沿用 BreakReminder.BREAK_VOICE_LINES）
- 描紅引導（12 檔，0-9 + intro + great）
- 問句結尾（2 檔：等於多少？/等於幾？）

合計 ~188 檔，預期 < 8 MB。

## Phase 2：整合到 webapp

- 加 `src/lib/audio/prerendered-tts.ts`，提供 `playPhrase(key)` / `playNumber(n)` / `playSentence(keys[])`
- `manifest.json` 對 phrase key → MP3 url
- 重寫 `speech.ts` 走「預生成優先 → Web Speech 回退」
