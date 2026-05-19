# 貢獻 / 上架新 asset 流程（給 maintainer）

> 這份文件給 repo owner 自己看 — 怎麼把新訓的 model / 新生的 TTS / 新版 SDK push 上來。

## 總體流程

```
訓 model / 生 TTS （在 pig-math 或別處）
    ↓
sanity check + sha256 + 跑 inference 驗證
    ↓
cp 進 pig-recognition-assets/
    ↓
更新 README 的版本表
    ↓
commit + push main
    ↓
tag vX.Y.Z + push tag
    ↓
gh release create vX.Y.Z
    ↓
驗證 jsdelivr URL 200 + sha256 一致
```

整個流程 ~10 分鐘（不含訓練時間）。

---

## Semver 版本規則

格式：`vMAJOR.MINOR.PATCH`

| Bump | 觸發條件 | 例子 |
|---|---|---|
| **MAJOR** (`v1.x.y` → `v2.0.0`) | 不相容 API change | classes.json schema 改、移除某 model、TTS naming convention 改 |
| **MINOR** (`v1.0.x` → `v1.1.0`) | 新增 asset（不破壞既有） | 加 letter v4 model、加新 TTS voice、加新 SDK class |
| **PATCH** (`v1.0.0` → `v1.0.1`) | bug fix / 純 metadata 修正 | README typo、training-manifest.json 補欄位、model 沒變但重 patch |

**現有專案怎麼接 minor / major？**
- `@^1` 自動跟 minor（不會跳到 v2）
- `@~1.0` 只跟 patch（不會跳到 v1.1）
- `@v1.0.0` 永遠精準鎖

---

## Case 1: 上架新 model（letter v4 範例）

### Step 1：訓出 model 並驗證

在 pig-math（或其他訓練專案）跑 Colab notebook 訓完，下載 zip 到本機解壓。然後：

```bash
# 確保 model 完整
cd /path/to/letter-tfjs-model-v4
ls
# 預期：model.json + group1-shard*.bin + classes.json + confusion_pairs.json + training-manifest.json

# 跑 sanity check（用 pig-recognition-assets 已有的 script）
python /path/to/pig-recognition-assets/scripts/train-letter/sanity_check.py \
  --model-dir .

# 預期：16 pass / 0 fail / 0 warn
```

如果有 fail，**先 debug 再繼續**。常見問題：
- `patch_tfjs_model_json.py` 沒跑（InputLayer batch_shape 沒改）
- Polarity 沒對齊（manifest 內 expected_polarity 不對）
- classes.json labels 順序錯

### Step 2：copy 進 pig-recognition-assets

```bash
cd ~/Project/pig-recognition-assets

# 新版開新 folder（不覆蓋舊版 — 讓既有專案用舊版仍然 work）
mkdir -p models/letter-v4
cp /path/to/letter-tfjs-model-v4/* models/letter-v4/

ls models/
# 應該看到 digit-v1/、letter-v3/、letter-v4/ 並存
```

### Step 3：更新 README 版本表

編輯 [README.md](./README.md) `## 內容清單` → Models 表格：

```diff
| 路徑 | 內容 | 類別 | Bundle | val_acc top-1 |
|---|---|---|---|---|
| `models/digit-v1/` | ... | 10 | ~1.1MB | 0.997 |
| `models/letter-v3/` | ... | 52 | ~3.5MB | 0.871 |
+| `models/letter-v4/` | ResNet18-lite + augmented | 52 | ~12MB | 0.91 |
```

`## 版本對應` 加新版：

```diff
| Tag | 日期 | 內容 |
|---|---|---|
| v1.0.0 | 2026-05-19 | Initial — digit v1, letter v3, TTS 188 個 |
+| v1.1.0 | 2026-MM-DD | 加 letter v4 (ResNet18-lite, val_acc 0.91) |
```

### Step 4：commit + push + tag + release

```bash
cd ~/Project/pig-recognition-assets

git add models/letter-v4/ README.md
git commit -m "feat: add letter v4 (ResNet18-lite, val_acc top-1 0.91)

Architecture upgrade from CNN(4-conv + Dense512) to ResNet18-lite.
Bundle increases 3.5MB → 12MB but top-1 accuracy 0.871 → 0.91.

letter v3 remains available at models/letter-v3/ for projects
using lighter model."

git push origin main

git tag v1.1.0
git push origin v1.1.0

gh release create v1.1.0 \
  --title "v1.1.0 — letter v4" \
  --notes "## 新增
- \`models/letter-v4/\` — ResNet18-lite + augmented training, val_acc top-1 0.91 (vs v3 0.871)
- Bundle 12MB（v3 是 3.5MB），交換準確率提升 4pp
- letter v3 仍可用 \`@v1.1.0/models/letter-v3/\`

## 升版指引
- 想留 v3：用 \`@v1.0.0\` 或 \`@~1.0\`
- 想用 v4：用 \`@v1.1.0\` + 路徑改 \`models/letter-v4/\`

## 不相容變更
無 — v3 model 路徑不變。"
```

### Step 5：驗證 jsdelivr 抓到

```bash
# 等 1-5 分鐘 jsdelivr propagate
sleep 60

# 驗證新 URL 200
curl -sI "https://cdn.jsdelivr.net/gh/jhs730127/pig-recognition-assets@v1.1.0/models/letter-v4/model.json" | head -3
# 預期：HTTP/2 200

# 驗證 sha256 一致
echo "local:"
shasum -a 256 models/letter-v4/model.json | awk '{print $1}'
echo "jsdelivr:"
curl -sL "https://cdn.jsdelivr.net/gh/jhs730127/pig-recognition-assets@v1.1.0/models/letter-v4/model.json" | shasum -a 256 | awk '{print $1}'
```

如果 sha256 不一致 = jsdelivr cache 還沒 update，等更久或用 `?purge` API：

```bash
curl "https://purge.jsdelivr.net/gh/jhs730127/pig-recognition-assets@v1.1.0/models/letter-v4/model.json"
```

---

## Case 2: 加新 TTS voice（例如 zh-TW 女聲 HsiaoChen）

### Step 1: 重跑 phase1b 生成 notebook

```bash
cd scripts/gen-tts/

# 開 phase1b_full_generation.ipynb，在 voice 設定改：
# VOICE = "zh-TW-HsiaoChenNeural"  (本來是 zh-TW-YunJheNeural)

# 上 Colab Run All
```

### Step 2: 放新目錄

```bash
mkdir -p audio/tts-zh-tw-hsiaochen
unzip ~/Downloads/tts-mp3.zip -d audio/tts-zh-tw-hsiaochen/

ls audio/
# digit-v1 路徑：tts-zh-tw/（YunJhe，default）
# 新增：tts-zh-tw-hsiaochen/（HsiaoChen）
```

### Step 3: README 更新

```diff
### Audio

| 路徑 | Voice | 檔數 |
|---|---|---|
| `audio/tts-zh-tw/` | zh-TW-YunJheNeural (男聲, default) | 188 |
+| `audio/tts-zh-tw-hsiaochen/` | zh-TW-HsiaoChenNeural (女聲) | 188 |
```

### Step 4: commit + tag + release

minor bump → `v1.1.0`（如果還沒 v1.1.0）或 `v1.2.0`（如果已有 v1.1.0）。

---

## Case 3: 純 metadata 修正（README typo / manifest 補欄位）

PATCH bump：`v1.0.1`。

```bash
git add README.md
git commit -m "docs: fix README typo in letter-v3 val_acc"
git push
git tag v1.0.1 && git push origin v1.0.1
gh release create v1.0.1 --title "v1.0.1 — docs fix" --notes "..."
```

---

## Release notes 模板

```markdown
## 新增 (MINOR / MAJOR)
- 簡述新功能 / model / asset

## 修正 (PATCH)
- bug fix 列點

## 不相容變更 (MAJOR only)
- 列出 breaking changes + migration guide

## 升版指引
- 從 v0.x → v1.y 的升級步驟
- 注意事項

## 已知問題
- 暫時 workaround
```

---

## 回滾流程（出問題快速 revert）

如果新 release（e.g. v1.1.0）發現重大問題，要讓既有專案能立刻退回穩定版：

### 短期應對：在 README 大字警告

```markdown
> ⚠️ **v1.1.0 有問題**：letter v4 在 Safari 偶爾載不進。請暫時鎖 `@v1.0.0`。
```

### 長期方案 A：標 v1.1.0 為 prerelease

```bash
gh release edit v1.1.0 --prerelease
# 不滿意這版本的人會看到 prerelease 標籤，自然會避開
```

### 長期方案 B：發 v1.1.1 修掉再讓他取代

```bash
# 修 model / 修 code
git tag v1.1.1
git push origin v1.1.1
gh release create v1.1.1 ...
# 用 @^1 的專案自動拿到 v1.1.1
```

### 不要刪 tag

刪 tag 會讓鎖 `@v1.1.0` 的專案 404，比有 bug 還慘。永遠用「再發新版本」而不是「刪舊版本」。

---

## OPSEC 提醒（commit 前必跑）

每次 push 前 grep 確認沒洩漏：

```bash
cd ~/Project/pig-recognition-assets

# 1. 內部 path
grep -rnE "/Users/|/home/|jerry\.wu|vici\.corp" . | grep -v ".git/"

# 2. API key / token  
grep -rnE "AIza[0-9A-Za-z_-]{20,}|sk-[0-9A-Za-z]{20,}|gh[pousr]_[0-9A-Za-z]{20,}|Bearer\s+[A-Za-z0-9_.-]{20,}" . | grep -v ".git/"

# 3. 確認 commit author email 是 noreply
git log -1 --pretty=format:"%ae"
# 預期：46418763+jhs730127@users.noreply.github.com
```

3 項全 0 matches + email 是 noreply 才 push。

如果發現過去 commit 有 leak，**force push history rewrite**：

```bash
git filter-branch -f --env-filter '
export GIT_AUTHOR_EMAIL="46418763+jhs730127@users.noreply.github.com"
export GIT_AUTHOR_NAME="jhs730127"
export GIT_COMMITTER_EMAIL="46418763+jhs730127@users.noreply.github.com"
export GIT_COMMITTER_NAME="jhs730127"
' -- --all

git push origin main --force
git push origin --tags --force
```

---

## 訓練 pipeline 維護

`scripts/train-digit/`、`scripts/train-letter/`、`scripts/gen-tts/` 是訓 model 的 source code，**這些檔案改動算 PATCH bump**（model 本身沒變）。

如果改 training script 而且要重訓 model（model 變了），則是 MINOR/MAJOR bump（看 schema 變沒）。

### 更新 sanity_check.py

`scripts/train-letter/sanity_check.py` 是給未來 maintainer 跑的驗證工具。如果加新 model architecture（例如 ResNet），記得更新內建的 `TFJS_SUPPORTED_LAYERS` set 確保 layer 都被驗證。

---

## 緊急聯絡 / 問題

repo 是 single maintainer（jhs730127）。問題開 GitHub issue 或在 commit message 留 note。

未來如果想多人維護，把這份 CONTRIBUTING.md 內 OPSEC 部分留意（user.email 設置）+ 加 GPG 簽 commit。
