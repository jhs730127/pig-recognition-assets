#!/usr/bin/env python3
"""
TTS voice sampling — 產 3 個風格樣本給你試聽，挑完再回頭跑 generate_tts_local.py 全量。

Usage:
    cd /Users/jerry.wu/Project/Pig-English
    python scripts/tts_voice_samples.py

輸出：tmp/tts-samples/{young-aria,warm-jenny,calm-guy}/*.mp3
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

try:
    import edge_tts  # type: ignore
except ImportError:
    print("缺少套件：pip install edge-tts", file=sys.stderr)
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
OUT_DIR = REPO_ROOT / "scripts" / "gen-tts-en" / "_voice-samples"

STYLES: list[dict] = [
    {
        "slug": "young-aria",
        "label": "年輕活潑（姐姐/幼教老師）",
        "voice": "en-US-AriaNeural",
        "rate": "+0%",
        "pitch": "+5Hz",
    },
    {
        "slug": "warm-jenny",
        "label": "成熟溫暖（媽媽/阿姨）",
        "voice": "en-US-JennyNeural",
        "rate": "-8%",
        "pitch": "+0Hz",
    },
    {
        "slug": "calm-guy",
        "label": "沉穩（爸爸/哥哥）",
        "voice": "en-US-GuyNeural",
        "rate": "-5%",
        "pitch": "-2Hz",
    },
]

# 測試句：包含字母 / 單字 / 鼓勵語 / 完整短句，30 秒內聽完
SAMPLES: list[tuple[str, str]] = [
    ("letter_A", "A. apple."),
    ("letter_B", "B. ball."),
    ("word_cat", "cat"),
    ("word_strawberry", "strawberry"),
    ("word_butterfly", "butterfly"),
    ("praise_great", "Great job!"),
    ("praise_awesome", "Awesome!"),
    ("praise_try", "Good try!"),
    ("sentence", "Let's learn English together. Tap the picture you hear."),
]


async def gen_one(text: str, voice: str, rate: str, pitch: str, out: Path) -> str:
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        tts = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        await tts.save(str(out))
        if out.stat().st_size < 500:
            out.unlink()
            return "FAILED: empty"
        return "ok"
    except Exception as exc:  # noqa: BLE001
        if out.exists():
            out.unlink()
        return f"FAILED: {exc}"


async def main() -> int:
    if OUT_DIR.exists():
        # 清空舊樣本，免得新舊混淆
        for f in OUT_DIR.rglob("*.mp3"):
            f.unlink()

    tasks = []
    for style in STYLES:
        for key, text in SAMPLES:
            out = OUT_DIR / style["slug"] / f"{key}.mp3"
            tasks.append(
                gen_one(text, style["voice"], style["rate"], style["pitch"], out)
            )

    print(f"產生 {len(STYLES)} 風格 × {len(SAMPLES)} 句 = {len(tasks)} 個樣本")
    print(f"輸出：{OUT_DIR}\n")

    results = await asyncio.gather(*tasks)
    ok = sum(1 for r in results if r == "ok")
    fail = [r for r in results if r != "ok"]
    print(f"完成：ok={ok}  fail={len(fail)}")
    if fail:
        for r in fail[:5]:
            print(" ", r)
        return 1

    print("\n各風格說明：")
    for s in STYLES:
        print(f"  {s['slug']:<14}  {s['label']:<22}  voice={s['voice']} rate={s['rate']} pitch={s['pitch']}")
    print(f"\n試聽：open {OUT_DIR}")
    print("挑好後改 scripts/generate_tts_local.py 的 VOICE/RATE/PITCH，全量重產。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
