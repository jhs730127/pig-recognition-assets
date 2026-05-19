# letter-stroke-order — 英文字母筆順資料

52 個字母（A-Z + a-z）的 SVG path stroke order 資料，給字母筆順教學 app 用。

由 [pig-english](https://github.com/jhs730127/pig-english) 自製，CC0 / public domain。

## 檔案

| 檔 | 內容 |
|---|---|
| `strokes.json` | 52 字母 × stroke order paths（含 viewBox、letterClass） |
| `grid-spec.json` | 四線格規格 — cap-line / midline / baseline / descender-line |

## 資料結構

```jsonc
// strokes.json
{
  "version": "1.0",
  "source": "pig-english",
  "viewBox": { "width": 320, "height": 320 },
  "letters": {
    "A": {
      "strokes": [
        { "path": "M 160 60 L 80 260", "start": { "x": 160, "y": 60 }, "length": 220 },
        { "path": "M 160 60 L 240 260", "start": { "x": 160, "y": 60 }, "length": 220 },
        { "path": "M 110 180 L 210 180", "start": { "x": 110, "y": 180 }, "length": 100 }
      ],
      "letterClass": "cap"
    },
    // ...
  }
}
```

`letterClass` 一定是 `cap` / `ascender` / `x-height` / `descender` 之一，決定字母佔據四線格的哪個區段。

## 四線格

```jsonc
// grid-spec.json
{
  "viewBox": { "width": 320, "height": 320 },
  "lines": {
    "capLine": 60,        // 大寫頂 / ascender 頂
    "midLine": 140,       // 小寫 x-height 頂
    "baseLine": 260,      // 所有字母底
    "descenderLine": 300  // g/j/p/q/y 底
  },
  "zones": {
    "cap":      { "top": 60, "bottom": 260 },
    "ascender": { "top": 60, "bottom": 260 },
    "xHeight":  { "top": 140, "bottom": 260 },
    "descender":{ "top": 140, "bottom": 300 }
  }
}
```

## 字母分類

- **cap** — A-Z 全部 26 個大寫
- **ascender** — `b d f h k l t`（7 個，撐到 cap-line）
- **x-height** — `a c e i m n o r s u v w x z`（14 個，x-height 範圍）
- **descender** — `g j p q y`（5 個，伸到 descender-line）

## 接入範例

```ts
const BASE = "https://cdn.jsdelivr.net/gh/jhs730127/pig-recognition-assets@v1.1.0";

const strokes = await fetch(`${BASE}/fonts/letter-stroke-order/strokes.json`)
  .then(r => r.json());
const grid = await fetch(`${BASE}/fonts/letter-stroke-order/grid-spec.json`)
  .then(r => r.json());

// 取字母 A 的筆畫
const letterA = strokes.letters["A"];
// letterA.strokes — Array of { path, start, length }
// 用 SVG <path d={s.path} /> 渲染，配合 stroke-dasharray={s.length} 做動畫
```

## 動畫提示

每個 stroke 的 `length` 是 path 長度估計值（手測），可直接用於 SVG `stroke-dasharray` 動畫：

```jsx
<path
  d={stroke.path}
  fill="none"
  stroke="currentColor"
  strokeWidth="12"
  strokeLinecap="round"
  style={{
    strokeDasharray: stroke.length,
    strokeDashoffset: stroke.length,
    animation: `draw ${stroke.length / 200}s linear forwards`,
  }}
/>
```

```css
@keyframes draw {
  to { stroke-dashoffset: 0; }
}
```

## 校對提醒

字母 path 是手動畫的、肉眼校過，但**不保證完全符合教育部標準寫法**。給 4-12 歲小孩學習 OK，給專業書寫教學請自行校對 `A B E M W a d e m w` 等多筆畫 / 曲線多的字母。
