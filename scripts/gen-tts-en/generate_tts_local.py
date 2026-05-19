#!/usr/bin/env python3
"""
Pig-English: 用 Microsoft Edge TTS (Azure neural voice) 生成所有預錄音檔

本機 CPU 即可跑，**不需要 GPU**。直接走 Azure 雲端服務，
~220 個 utterance 並發跑約 1-3 分鐘完成。

Usage:
    pip install edge-tts
    cd /Users/jerry.wu/Project/Pig-English
    python scripts/generate_tts_local.py

輸出位置：public/audio/{letters|numbers|praise|words}/*.mp3
已存在的檔案會 skip（可安全重跑補生成失敗的）。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

try:
    import edge_tts  # type: ignore
except ImportError:
    print("缺少套件，請先執行: pip install edge-tts", file=sys.stderr)
    sys.exit(1)

# 3 種風格 — slug 對應 public/audio/{slug}/ 子目錄
STYLES: list[dict] = [
    {
        "slug": "young-aria",
        "voice": "en-US-AriaNeural",
        "rate": "+0%",
        "pitch": "+5Hz",
    },
    {
        "slug": "warm-jenny",
        "voice": "en-US-JennyNeural",
        "rate": "-8%",
        "pitch": "+0Hz",
    },
    {
        "slug": "calm-guy",
        "voice": "en-US-GuyNeural",
        "rate": "-5%",
        "pitch": "-2Hz",
    },
]
CONCURRENCY = 5  # 同時發 5 個 request（Azure 對 anonymous 較嚴）

# === 與 src/lib/audio/manifest.ts 同步的清單 ===

LETTERS: list[str] = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
NUMBERS: list[str] = ["one", "two", "three", "four", "five", "six"]
PRAISE: dict[str, str] = {
    "great_job": "Great job!",
    "awesome": "Awesome!",
    "well_done": "Well done!",
    "perfect": "Perfect!",
    "yay": "Yay!",
    "amazing": "Amazing!",
    "good_try": "Good try!",
    "try_again": "Try again",
}
WORDS: list[str] = [
    "cat", "dog", "fish", "bird", "rabbit",
    "pig", "cow", "duck", "bee", "ant",
    "apple", "banana", "grape", "orange", "strawberry",
    "bread", "rice", "egg", "cake", "cookie", "noodle",
    "water", "milk", "juice", "tea",
    "cup", "spoon", "fork", "plate", "bowl",
    "chair", "bed", "sofa", "lamp", "door", "clock", "key", "phone",
    "soap", "towel", "toothbrush",
    "hat", "shirt", "pants", "shoes", "socks", "bag",
    "ball", "doll", "blocks", "kite", "drum",
    "book", "pen", "pencil", "desk", "scissors",
    "tree", "flower", "grass", "leaf", "rock",
    "sun", "moon", "star", "cloud", "rain", "snow", "wind",
    "car", "bus", "bike", "boat", "plane", "train",
    "red", "blue", "green", "yellow", "black", "white", "pink", "purple", "brown",
    "circle", "square", "triangle", "heart",
    "seven", "eight", "nine", "ten",
    "mom", "dad", "baby", "sister", "brother", "grandma", "grandpa",
    "eye", "ear", "nose", "mouth", "hand", "foot", "head", "hair",
    "eat", "drink", "sit", "stand", "run", "walk", "jump", "sleep", "wash", "sing", "play",
    "hello", "bye", "yes", "no", "please", "sorry", "thanks", "good",
    "big", "small", "hot", "cold", "happy", "sad",
    # === 新增詞庫（2026-05-18）===
    "carrot", "broccoli", "tomato", "corn",
    "pizza", "candy",
    "bath",
    "butterfly", "ladybug", "frog", "turtle",
    "horse", "sheep",
]
SPECIAL: dict[str, str] = {"tv": "T.V."}

# === 路徑 ===
# repo 結構：repo/scripts/gen-tts-en/ + repo/audio/tts-en/{voice}/{cat}/...
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = REPO_ROOT / "audio" / "tts-en"


def letter_prompt(letter: str) -> str:
    """字母 letter name (/eɪ/, /biː/...) 不分大小寫，統一用大寫 prompt 給 Azure。
    這樣 a.mp3 與 A.mp3 內容相同（letter name），但 file path 區分 case。
    """
    return letter.upper()


async def gen_one(
    text: str,
    out_path: Path,
    voice: str,
    rate: str,
    pitch: str,
    sem: asyncio.Semaphore,
) -> str:
    if out_path.exists():
        return "skipped"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    async with sem:
        try:
            tts = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            await tts.save(str(out_path))
            # Azure 偶爾回極小檔（< 1 KB 通常是空 mp3 header），刪掉讓下次重跑補生
            if out_path.stat().st_size < 500:
                out_path.unlink()
                return "FAILED: empty output"
            return "ok"
        except Exception as exc:  # noqa: BLE001
            # 失敗時清理半成品
            if out_path.exists():
                try:
                    out_path.unlink()
                except OSError:
                    pass
            return f"FAILED: {exc}"


async def main() -> int:
    base_items: list[tuple[str, str, str]] = []
    for letter in LETTERS:
        base_items.append((letter_prompt(letter), letter, "letters"))
    for num in NUMBERS:
        base_items.append((num, num, "numbers"))
    for key, text in PRAISE.items():
        base_items.append((text, key, "praise"))
    for word in WORDS:
        base_items.append((word, word, "words"))
    for key, text in SPECIAL.items():
        base_items.append((text, key, "words"))

    # 展開：每個 style × 每個 utterance
    items: list[tuple[str, str, str, str, str, str]] = []
    for style in STYLES:
        for text, key, subdir in base_items:
            items.append((text, key, subdir, style["slug"], style["voice"], style["rate"]))

    print(f"Styles: {[s['slug'] for s in STYLES]}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Total: {len(items)} utterances ({len(base_items)} × {len(STYLES)} styles)")
    print()

    sem = asyncio.Semaphore(CONCURRENCY)
    style_by_slug = {s["slug"]: s for s in STYLES}

    async def run(idx: int, item: tuple[str, str, str, str, str, str]) -> tuple[int, str, str, str, str, str]:
        text, key, subdir, slug, voice, rate = item
        pitch = style_by_slug[slug]["pitch"]
        out_path = OUTPUT_DIR / slug / subdir / f"{key}.mp3"
        status = await gen_one(text, out_path, voice, rate, pitch, sem)
        return idx, slug, subdir, key, status, text

    tasks = [asyncio.create_task(run(i, it)) for i, it in enumerate(items)]

    done = skipped = 0
    errors: list[tuple[str, str, str, str]] = []
    completed = 0
    for fut in asyncio.as_completed(tasks):
        idx, slug, subdir, key, status, text = await fut
        completed += 1
        if status == "skipped":
            skipped += 1
        elif status.startswith("FAILED"):
            errors.append((slug, subdir, key, status))
        else:
            done += 1
        if completed % 20 == 0 or completed == len(items):
            print(
                f"[{completed:3d}/{len(items)}] "
                f"done={done} skipped={skipped} errors={len(errors)}"
            )

    print()
    print(f"Finished — done {done}, skipped {skipped}, errors {len(errors)}")
    if errors:
        print("\nErrors (first 10):")
        for e in errors[:10]:
            print(" ", e)
        print("\n重新執行此 script 會自動補生失敗項目（已成功的會 skip）")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
