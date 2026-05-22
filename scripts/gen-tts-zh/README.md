# gen-tts-zh — Pig-English 中文 TTS 生成器

用 Microsoft Edge TTS（Azure neural voice）產 pig-english 的中文 mp3。

## 產什麼

3 種 voice preset × (152 vocab + 13 world names) = **495 mp3 / 全部**（~15 MB）

| Slug（en + zh 共用） | en voice | zh-TW voice | 風格 |
|---|---|---|---|
| `young-aria` | en-US-AriaNeural | zh-TW-HsiaoChenNeural | 年輕女、姐姐 |
| `warm-jenny` | en-US-JennyNeural | zh-TW-HsiaoYuNeural | 溫暖女、媽媽 |
| `calm-guy` | en-US-GuyNeural | zh-TW-YunJheNeural | 低沉男、爸爸 |

## 跑法 A · Google Colab（推薦）

1. 在 pig-english repo 確保詞庫是最新的：
   ```bash
   cd ~/Project/Pig-English
   npx tsx scripts/export_to_shared_assets.ts
   ```
   產出：`~/Project/pig-recognition-assets/data/vocab-en-preschool/words.json`

2. 把 `generate_tts_zh.ipynb` 開到 Colab（直接 `File > Upload notebook`）

3. 依序跑 cell：
   - cell 1：`!pip install -q edge-tts`
   - cell 3：會 prompt 上傳 `words.json`
   - cell 5：批次生成 ~495 mp3（~3-5 分鐘）
   - cell 7：內嵌 audio player 試聽 3 voice 的 `cat.mp3`
   - cell 9：打包 `tts-zh-tw.zip` 並下載

4. 把下載的 zip 解到 `~/Project/pig-recognition-assets/audio/tts-zh-tw/`（會新增 3 個 voice 子資料夾，原本的 break/feedback 等系統音檔不動）

5. Commit + tag：
   ```bash
   cd ~/Project/pig-recognition-assets
   git add audio/tts-zh-tw/{young-aria,warm-jenny,calm-guy} scripts/gen-tts-zh
   git commit -m "Add zh-TW TTS: 3 voice × (152 vocab + 13 worlds)"
   git tag v1.3.0
   git push --tags
   ```

## 跑法 B · 本機 Jupyter

```bash
cd ~/Project/Pig-English
npx tsx scripts/export_to_shared_assets.ts

pip install edge-tts jupyter
cd ~/Project/pig-recognition-assets/scripts/gen-tts-zh
jupyter notebook generate_tts_zh.ipynb
```

Notebook 會自動偵測非 Colab 環境，直接讀 repo 的 `words.json` + 寫到 `audio/tts-zh-tw/`。
跑完直接 `cd ~/Project/pig-recognition-assets && git add ... && git tag v1.3.0`。

## 輸出結構

```
audio/tts-zh-tw/
├── young-aria/
│   ├── words/  (152 mp3, ex: cat.mp3 = "貓")
│   └── worlds/ (13 mp3, ex: zoo.mp3 = "動物園")
├── warm-jenny/
└── calm-guy/
```

pig-english 之後升 `manifest.ts` 的 `AUDIO_CDN_BASE` 到 `@v1.3.0/audio/tts-zh-tw` 接這套。

## 擴詞 / 改名字怎麼辦

- **詞庫加字**：改 pig-english `vocabulary.ts` → 跑 `export_to_shared_assets.ts` → 重跑 notebook（會自動 skip 既有，只補新字）。
- **改世界名字**：直接改 notebook cell 3 的 `WORLDS` list、刪對應 mp3 重跑。
- **加新 voice preset**：在 `STYLES` 加 entry，跑 notebook。

## Microsoft 限速注意

Edge TTS 對 anonymous 並發敏感。notebook 預設 `BATCH=4 + 0.8s sleep`，跟 Math 專案的 phase1b 一樣。如果跑到一半多筆 error，**直接重跑 cell 5**——已生成的會 skip。
