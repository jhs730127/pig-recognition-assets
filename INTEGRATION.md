# 整合指南（給其他專案）

如何把 pig-recognition-assets 的 model / TTS / SDK 整合進你的新專案。

零安裝、零後端 — 全部走 jsdelivr CDN 直接 fetch。

## 30 秒上手

Copy 你需要的 SDK 到專案（從 [`sdk/ts/`](./sdk/ts/)），然後：

```ts
import { DigitRecognizer } from "./digit-recognizer";
import { LetterRecognizer } from "./letter-recognizer";
import { PrerenderedTtsPlayer } from "./prerendered-tts-player";

const BASE = "https://cdn.jsdelivr.net/gh/jhs730127/pig-recognition-assets@v1.2.0";

// Digit 0-9（含 MNIST 重心置中 + TTA + 多位數切割）
const digit = new DigitRecognizer({ modelUrl: `${BASE}/models/digit-v1/model.json` });
await digit.load();
const r = await digit.predictFromDataUrl(dataUrl, { useTta: true });  // { digit, confidence }

// Letter A-Z + a-z
const letter = new LetterRecognizer();
await letter.load(`${BASE}/models/letter-v3/model.json`, `${BASE}/models/letter-v3/classes.json`);

// TTS — Web Audio 精準排程組合句
const tts = new PrerenderedTtsPlayer({ baseUrl: `${BASE}/audio/tts-zh-tw` });
tts.setupAudioUnlock();
await tts.playSentence(["num_zh_5", "op_sub", "num_zh_1", "q_equals_what"]);
```

你的專案 `package.json` 只需要一個依賴：

```bash
npm install @tensorflow/tfjs
```

> 不想用 SDK 也可以直接 `tf.loadLayersModel(URL)` 自己組裝；但 SDK 已包好 MNIST 重心置中、TTA、TTS Web Audio 精準排程等踩過的細節，建議直接用。

---

## 各框架整合範例

### Next.js 16 + App Router (Vercel 部署)

`src/lib/letter-recognizer.ts`：直接 copy 自 [`sdk/ts/letter-recognizer.ts`](./sdk/ts/letter-recognizer.ts)（已就緒，含 LetterRecognizer class）。

`src/app/letter-demo/page.tsx`：

```tsx
"use client";

import { useEffect, useRef, useState } from "react";
import { LetterRecognizer } from "@/lib/letter-recognizer";

const BASE = "https://cdn.jsdelivr.net/gh/jhs730127/pig-recognition-assets@v1.0.0";

export default function LetterDemo() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const recognizerRef = useRef<LetterRecognizer | null>(null);
  const [ready, setReady] = useState(false);
  const [result, setResult] = useState<string>("");

  useEffect(() => {
    const r = new LetterRecognizer();
    r.load(`${BASE}/models/letter-v3/model.json`, `${BASE}/models/letter-v3/classes.json`)
      .then(() => {
        recognizerRef.current = r;
        setReady(true);
      });
    return () => { recognizerRef.current?.dispose(); };
  }, []);

  function predict() {
    const ctx = canvasRef.current!.getContext("2d")!;
    const imageData = ctx.getImageData(0, 0, canvasRef.current!.width, canvasRef.current!.height);
    const top = recognizerRef.current!.predict(imageData, 3);
    setResult(top.map(t => `${t.label}: ${(t.confidence * 100).toFixed(1)}%`).join(" / "));
  }

  return (
    <div>
      <canvas ref={canvasRef} width={280} height={280} style={{ border: "1px solid" }} />
      <button onClick={predict} disabled={!ready}>
        {ready ? "辨識" : "載入中..."}
      </button>
      <p>{result}</p>
    </div>
  );
}
```

### Vite + React

API 完全相同，只是 import path 隨你 alias 設定。

### 純 HTML + CDN（不用 build tool）

```html
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>
</head>
<body>
  <script>
    const BASE = "https://cdn.jsdelivr.net/gh/jhs730127/pig-recognition-assets@v1.0.0";
    
    (async () => {
      const model = await tf.loadLayersModel(`${BASE}/models/letter-v3/model.json`);
      const classes = await fetch(`${BASE}/models/letter-v3/classes.json`).then(r => r.json());
      console.log(`Loaded ${classes.num_classes} classes`);
    })();
  </script>
</body>
</html>
```

### Node.js (TF.js Node, server-side inference)

```bash
npm install @tensorflow/tfjs-node node-fetch
```

```js
import * as tf from "@tensorflow/tfjs-node";
import fetch from "node-fetch";

const BASE = "https://cdn.jsdelivr.net/gh/jhs730127/pig-recognition-assets@v1.0.0";

// TF.js Node 不能直接 fetch URL，要先下載再 load file
const handler = tf.io.browserHTTPRequest(`${BASE}/models/letter-v3/model.json`, { fetch });
const model = await tf.loadLayersModel(handler);
```

---

## CSP 設定（重要 — 不設會被擋）

如果你的專案有 Content Security Policy，必須把 jsdelivr 加進 `connect-src`：

### Next.js (next.config.ts)

```ts
const cspHeader = `
  default-src 'self';
  connect-src 'self' https://cdn.jsdelivr.net;  
  ...
`;
```

### Vercel (vercel.json)

```json
{
  "headers": [{
    "source": "/(.*)",
    "headers": [{
      "key": "Content-Security-Policy",
      "value": "connect-src 'self' https://cdn.jsdelivr.net"
    }]
  }]
}
```

### 純 HTML

```html
<meta http-equiv="Content-Security-Policy" 
      content="connect-src 'self' https://cdn.jsdelivr.net">
```

**沒設 CSP 完全不用管這節**（瀏覽器預設允許 cross-origin fetch）。

---

## 版本鎖策略

| URL 模式 | 行為 | 適用 |
|---|---|---|
| `@v1.0.0` | 精準鎖到 v1.0.0 | **production 推薦** — 升 model 不會炸 |
| `@^1` | 跟著 v1.x 最新 minor | 自動拿 bug fix + 新增功能 |
| `@~1.0` | 跟 v1.0.x 最新 patch | 保守，只拿 patch fix |
| `@main` | 跟 main 分支最新 commit | 開發階段、不穩定 |
| `@latest` | 跟最新 release | 最新但版本不確定 |

```ts
// 推薦：production 鎖精準版本
const BASE = "https://cdn.jsdelivr.net/gh/jhs730127/pig-recognition-assets@v1.0.0";

// 開發：跟 main，每次拉到最新
const BASE = "https://cdn.jsdelivr.net/gh/jhs730127/pig-recognition-assets@main";
```

升版時改 `@v1.0.0` 到 `@v1.1.0`，jsdelivr 會自動切換。

---

## 效能優化

### 1. Lazy load model（不要進首頁就 load）

Letter model 3.5MB，全頁載入會慢。dynamic import + 首次需要時才 load：

```tsx
const [recognizer, setRecognizer] = useState<LetterRecognizer | null>(null);

async function ensureLoaded() {
  if (recognizer) return recognizer;
  const { LetterRecognizer } = await import("@/lib/letter-recognizer");
  const r = new LetterRecognizer();
  await r.load(`${BASE}/models/letter-v3/model.json`, `${BASE}/models/letter-v3/classes.json`);
  setRecognizer(r);
  return r;
}

// User click "開始辨識" 才觸發
async function handleClick() {
  const r = await ensureLoaded();
  // ... predict
}
```

### 2. IndexedDB 快取（避免每次 reload 都重 fetch）

TF.js 內建 IndexedDB cache：

```ts
const MODEL_CACHE_KEY = "indexeddb://my-app-letter-v3";

async function loadCached() {
  try {
    return await tf.loadLayersModel(MODEL_CACHE_KEY);  // 命中 cache
  } catch {
    // miss → fetch + save
    const model = await tf.loadLayersModel(`${BASE}/models/letter-v3/model.json`);
    await model.save(MODEL_CACHE_KEY);
    return model;
  }
}
```

第一次 ~3 秒下載，之後每次開頁 < 100ms 從 IndexedDB load。

### 3. 預先 preload audio（TTS 點即播）

```ts
const preloadList = ["feedback_correct_1", "feedback_wrong_1", "num_zh_0", "num_zh_1"];
const audioCache = new Map<string, HTMLAudioElement>();

for (const key of preloadList) {
  const a = new Audio(`${BASE}/audio/tts-zh-tw/${key}.mp3`);
  a.preload = "auto";
  audioCache.set(key, a);
}

// Play
audioCache.get("feedback_correct_1")?.play();
```

### 4. 在 idle time 預載 model

```ts
if ('requestIdleCallback' in window) {
  requestIdleCallback(() => ensureLoaded());
}
```

---

## 錯誤處理

### jsdelivr 暫時 down（極少發生）

```ts
const FALLBACK_URLS = [
  `https://cdn.jsdelivr.net/gh/jhs730127/pig-recognition-assets@v1.0.0/models/letter-v3/model.json`,
  `https://raw.githubusercontent.com/jhs730127/pig-recognition-assets/v1.0.0/models/letter-v3/model.json`,
];

async function loadWithFallback() {
  for (const url of FALLBACK_URLS) {
    try {
      return await tf.loadLayersModel(url);
    } catch (e) {
      console.warn(`Failed ${url}, trying next...`);
    }
  }
  throw new Error("All model URLs failed");
}
```

`raw.githubusercontent.com` 沒 CDN 速度慢，但 jsdelivr 死掉時是 backup。

### 版本不存在 (404)

```ts
try {
  const r = await fetch(`${BASE}/models/letter-v3/model.json`);
  if (!r.ok) throw new Error(`HTTP ${r.status}: model not found`);
} catch (e) {
  // 顯示「離線模式」UI、降級走 Gemini API 等
}
```

### Model load fail（TF.js 內部錯誤）

```ts
try {
  await recognizer.load(...);
} catch (e) {
  console.error("model load failed:", e);
  // 通常原因：CSP 擋了 connect-src、CORS、TF.js 版本不相容
}
```

---

## TypeScript 型別

`classes.json` 的 interface 已在 [`sdk/ts/letter-recognizer.ts`](./sdk/ts/letter-recognizer.ts) export：

```ts
export interface LetterClassesMeta {
  version: string;
  num_classes: number;
  expected_polarity: "black_bg_white_ink";
  input_shape: [number, number, number];
  labels: string[];
  case_pairs: number[][];
  confusable_pairs: number[][];
}

export interface LetterPrediction {
  label: string;
  confidence: number;
  classIdx: number;
}
```

直接 import 用。

---

## TTS Audio 整合

### Audio key naming convention

```
audio/tts-zh-tw/{category}_{lang?}_{id}.mp3
```

| 範例 key | 內容 |
|---|---|
| `num_zh_0` | 零（中文） |
| `num_zh_100` | 一百（中文） |
| `num_en_5` | five（英文） |
| `op_add` | 加 |
| `op_equals` | 等於 |
| `feedback_correct_1` | 「答對了！」中文版 |
| `feedback_correct_en_1` | "Great!" 英文版 |
| `feedback_wrong_3` | 中文鼓勵重試 |
| `break_2` | 中場休息提示 |
| `problem_n_5` | 「第五題」 |
| `trace_zh_3` | 描紅引導「3」 |
| `streak_10` | 連對 10 題 |
| `q_equals_how` | 問句「等於多少」 |

manifest 全清單見 `audio/tts-zh-tw/manifest.json`。

### Sequence player（多個 audio 串播）

**推薦用 [`sdk/ts/prerendered-tts-player.ts`](./sdk/ts/prerendered-tts-player.ts)**（v1.2.0+）— 內建 Web Audio API + trim silence + 精準排程，組合句段間 0 gap，比 `new Audio().play()` 體驗好很多：

```ts
import { PrerenderedTtsPlayer } from "./prerendered-tts-player";

const tts = new PrerenderedTtsPlayer({ baseUrl: `${BASE}/audio/tts-zh-tw` });
tts.setupAudioUnlock();   // App mount 時呼叫一次（iOS Safari 解鎖 AudioContext）
await tts.loadManifest();

// 「三加五等於八」— 段與段精準對接、無 cold start gap
await tts.playSentence(["num_zh_3", "op_add", "num_zh_5", "op_equals", "num_zh_8"]);
```

不想用 SDK 的話最陽春版本（**會聽到字字停頓，不建議**）：

```ts
async function playSequence(keys: string[]) {
  for (const key of keys) {
    const audio = new Audio(`${BASE}/audio/tts-zh-tw/${key}.mp3`);
    await new Promise<void>((resolve, reject) => {
      audio.onended = () => resolve();
      audio.onerror = reject;
      audio.play();
    });
  }
}
```

差別：MP3 每段前後有 200~500ms 靜音 padding + `new Audio()` 有 cold start。SDK 用 Web Audio API decodeAudioData 一次 decode 起來 trim silence、再用 `AudioBufferSourceNode.start(when)` 精準排程，組合句聽起來像一句連續話。

### 「未來新專案的字母 TTS」

pig-recognition-assets v1.0.0 **只有數字 TTS**（沒英文字母）。如果你新專案要字母（"A"、"a"…）TTS：

選擇 1：自己用 edge-tts 生：
```bash
cd /path/to/pig-recognition-assets/scripts/gen-tts
# 開 phase1b_full_generation.ipynb，加 letter 內容 list 重跑
```

選擇 2：用 Web Speech API（瀏覽器內建，無檔案）：
```ts
const utter = new SpeechSynthesisUtterance("A");
utter.lang = "en-US";
speechSynthesis.speak(utter);
```

選擇 3：用第三方 TTS service（ResponsiveVoice、Microsoft TTS API、ElevenLabs）

---

## Production Checklist

部署到 prod 前確認：

- [ ] URL 鎖精準版本 `@v1.x.y`，不要用 `@main`
- [ ] CSP `connect-src` 含 `https://cdn.jsdelivr.net`
- [ ] Lazy load model（不全頁載入）
- [ ] IndexedDB cache 開啟
- [ ] 錯誤處理：jsdelivr 暫斷 / 404 / model load fail
- [ ] preload audio 給常用 phrase
- [ ] iOS Safari 測過（audio autoplay 要 user gesture trigger）
- [ ] CORS 沒問題（jsdelivr 預設 `Access-Control-Allow-Origin: *`）

---

## 常見問題

**Q: jsdelivr 會不會收費？**
A: 不會。Public repo 免費無限流量。<https://www.jsdelivr.com/network>

**Q: jsdelivr 多久同步 GitHub 變動？**
A: tag (v1.0.0) 是 immutable，永遠 cached。`@main` 約 12 小時更新一次。`?purge` API 可手動觸發更新。

**Q: 我自己的專案是 private，能用 public 的 jsdelivr URL 嗎？**
A: 可以。你的專案 visibility 跟 fetch 的 URL public 無關。

**Q: TF.js bundle 多大？**
A: `@tensorflow/tfjs` 約 400KB gzipped。加上 letter model 3.5MB 共約 4MB。

**Q: 沒網路怎麼辦？**
A: 第一次 load 後 IndexedDB cache 會留著（前述優化 #2），離線可用。第一次 load 強制要網路。
