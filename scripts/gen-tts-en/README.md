# gen-tts-en — 英文 TTS 生成 pipeline

用 [Microsoft Edge TTS](https://github.com/rany2/edge-tts)（免費，本機 CPU 即可）生 `audio/tts-en/` 預錄 MP3。

## 快速重產

```bash
pip install edge-tts
cd scripts/gen-tts-en/
python3 generate_tts_local.py
```

- 已存在的 mp3 會 skip — 重跑可補生失敗項
- ~2 分鐘跑完 3 voice × 186 utterance = 558 個 mp3
- Output：`../../audio/tts-en/{aria,jenny,guy}/{letters,numbers,praise,words}/*.mp3`

## 換 voice 試聽

挑新 voice 前可先用 `tts_voice_samples.py` 試聽：

```bash
python3 tts_voice_samples.py
# Output: _voice-samples/{slug}/*.mp3
# 試聽後改 generate_tts_local.py 的 STYLES list 重跑
```

## 加新單字

1. 編 `generate_tts_local.py` 的 `WORDS / NUMBERS / PRAISE / SPECIAL`
2. 重跑（已存在自動 skip）
3. commit + tag PATCH bump (`vX.Y.Z+1`) 或 MINOR（若新增足夠多）

## Voice 設計考量

| Voice | 為什麼這樣設 |
|---|---|
| `en-US-AriaNeural` rate +0% pitch +5Hz | Aria 本來就年輕活潑，pitch 加一點更童趣，rate 不變保持清晰 |
| `en-US-JennyNeural` rate -8% pitch +0Hz | Jenny 溫暖但語速正常偏快，給小孩用要慢一點 |
| `en-US-GuyNeural` rate -5% pitch -2Hz | Guy 沉穩，pitch 降一點更像爸爸，rate 微慢給小孩聽得清楚 |

## 失敗處理

Azure anonymous tier 偶爾 throttle。重跑可補生失敗項：

```bash
# 已成功的自動 skip，失敗的會再試
python3 generate_tts_local.py
```

如果某些 utterance 重跑多次仍失敗，檢查文字內容（特殊字元、太長等）。
