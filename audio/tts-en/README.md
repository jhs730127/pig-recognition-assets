# tts-en — 英文 TTS 預錄音檔

3 個風格，各 186 個 utterance，共 **558 個 MP3**（~6.5 MB）。

由 [pig-english](https://github.com/jhs730127/pig-english) 用 Microsoft Edge TTS（`edge-tts`）預錄產生。

## Voice 對照

| Slug | Voice | 風格 | Rate | Pitch | 對應角色 |
|---|---|---|---|---|---|
| `aria/` | `en-US-AriaNeural` | 年輕活潑 | `+0%` | `+5Hz` | 姐姐 / 幼教老師 |
| `jenny/` | `en-US-JennyNeural` | 成熟溫暖 | `-8%` | `+0Hz` | 媽媽 / 阿姨 |
| `guy/` | `en-US-GuyNeural` | 沉穩男聲 | `-5%` | `-2Hz` | 爸爸 / 哥哥 |

## 目錄結構

```
audio/tts-en/{voice}/
├── letters/   ← A-Z 26 個（大寫，因 letter name /eɪ/ 大小寫共用）
├── numbers/   ← one, two, three, four, five, six (6 個)
├── praise/    ← great_job, awesome, well_done, perfect, yay, amazing, good_try, try_again (8 個)
└── words/     ← cat, dog, apple, ...（~146 個學齡前單字 + tv 特殊處理為 "T.V."）
```

## 快速接入

```ts
const BASE = "https://cdn.jsdelivr.net/gh/jhs730127/pig-recognition-assets@v1.1.0";

// 一般單字
new Audio(`${BASE}/audio/tts-en/aria/words/cat.mp3`).play();

// 字母（letter name 不分大小寫，統一查 uppercase）
new Audio(`${BASE}/audio/tts-en/jenny/letters/A.mp3`).play();

// Praise 鼓勵
new Audio(`${BASE}/audio/tts-en/guy/praise/great_job.mp3`).play();
```

## 詞庫清單

完整 utterance 清單與對應原字串見 `scripts/gen-tts-en/generate_tts_local.py` 的 `LETTERS / NUMBERS / PRAISE / WORDS / SPECIAL` 常數。

主要分類：
- 動物（cat, dog, fish, ...）
- 食物 / 飲料（apple, milk, ...）
- 家具 / 家裡（chair, sofa, ...）
- 衣物 / 玩具 / 學校用品
- 自然 / 天氣
- 顏色 / 形狀 / 數字 / 家人 / 身體
- 動作 / 問候 / 形容詞

## 重產

```bash
cd scripts/gen-tts-en/
pip install edge-tts
python3 generate_tts_local.py
# 已存在的 mp3 會 skip，可安全重跑補生失敗項目
```

## 授權

MP3 內容透過 Microsoft Edge TTS 公開服務生成（[ToS](https://learn.microsoft.com/en-us/legal/cognitive-services/speech-service/translator/speech-service-improvement-additional-terms)）。重新生成不限。
