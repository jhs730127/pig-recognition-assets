# vocab-en-preschool — 學齡前英文詞庫

152 個字（~4-6 歲），由 [pig-english](https://github.com/jhs730127/pig-english) 策展。

選字原則：**家裡 / 日常外面看得到的生活化單字**，小孩能指物命名（pointing & naming）。**不選罕見抽象字**（dragon、queen、umbrella、zebra 等）。

## 檔案

`words.json` — 完整資料。

## 資料結構

```jsonc
{
  "version": "1.0",
  "source": "pig-english",
  "count": 152,
  "words": [
    {
      "id": "cat",
      "en": "cat",
      "zh": "貓",
      "emoji": "🐱",
      "syllables": ["cat"],
      "tags": ["animal"]
    },
    {
      "id": "strawberry",
      "en": "strawberry",
      "zh": "草莓",
      "emoji": "🍓",
      "syllables": ["straw", "ber", "ry"],
      "tags": ["fruit", "food"]
    },
    // ...
  ]
}
```

## 欄位說明

| 欄 | 內容 |
|---|---|
| `id` | 唯一識別字，跟 `en` 通常同（少數例外如 `tv` → `TV`） |
| `en` | 英文書寫（小寫為主，`TV` 例外） |
| `zh` | 台灣繁體中文 |
| `emoji` | 對應 emoji（單字符，給 image-pick 題型用） |
| `syllables` | 音節拆分（教發音用） |
| `tags` | 主題標籤，用於 distractor 抽樣 |

## Tags

```
animal, fruit, food, drink, vehicle, nature, weather,
home, kitchen, bathroom, bedroom, clothes, school,
outdoor, toy, shape, color, number,
action, body, people, adjective, greeting
```

## 接入範例

```ts
const BASE = "https://cdn.jsdelivr.net/gh/jhs730127/pig-recognition-assets@v1.1.0";

const data = await fetch(`${BASE}/data/vocab-en-preschool/words.json`)
  .then(r => r.json());

// 取所有動物
const animals = data.words.filter(w => w.tags.includes("animal"));

// 取單字
const cat = data.words.find(w => w.id === "cat");
console.log(cat.emoji, cat.zh); // 🐱 貓
```

## 對應 TTS / 圖片

- **TTS**：`audio/tts-en/{voice}/words/{id}.mp3` — 同 repo 已有 3 voice × ~150 字
- **圖片**：emoji 直接可用（單字符）。若要更可愛的 3D 立體圖，可下 [Microsoft Fluent Emoji](https://github.com/microsoft/fluentui-emoji)（MIT 授權）

## 涵蓋分類

- 動物（pet + 公園 + 童書）— 16 字
- 水果 + 蔬菜 + 食物 + 點心 + 飲料 — 25 字
- 餐具 + 家具 + 浴室 — 18 字
- 衣物 + 玩具 + 學校用品 — 16 字
- 自然 + 天氣 + 交通 — 18 字
- 顏色 9 + 形狀 4 + 數字 10 — 23 字
- 家人 7 + 身體 8 + 動作 11 — 26 字
- 問候 8 + 形容詞 6 — 14 字
