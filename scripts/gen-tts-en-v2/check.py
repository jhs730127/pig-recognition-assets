#!/usr/bin/env python3
"""自動檢查生成的語音，並產生本機試聽頁。

- 用 faster-whisper small 轉錄每個檔，跟原文比對（統一大小寫、去標點、數字轉英文）
- 長度異常：超過 2.5 秒或短於 0.15 秒
- 字母一律標記，給人工試聽

用法（在本資料夾內）：
  .venv/bin/python check.py --utterances utterances.json --dir tmp/out \
      --voices af_heart,af_bella [--only A,sofa] [--sample 20]
輸出：<dir>/report.json 與 <dir>/listen.html（音檔用相對路徑，直接用瀏覽器開）
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import subprocess
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

NUM_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
}
MAX_S = 2.5
MIN_S = 0.15

# 單獨念字母時 ASR 常聽成同音字，這些不算錯（字母仍一律列給人工聽）
LETTER_HOMOPHONES = {
    "b": {"be", "bee"}, "c": {"see", "sea"}, "g": {"gee"}, "i": {"eye", "aye"},
    "j": {"jay"}, "k": {"kay", "okay"}, "o": {"oh"}, "p": {"pee"}, "q": {"queue"},
    "r": {"are"}, "t": {"tea", "tee"}, "u": {"you"}, "y": {"why"}, "z": {"zee"},
}


# 單字的同音字／拼法差異，ASR 聽成這些不算錯
WORD_HOMOPHONES = {
    "bee": {"b"}, "tea": {"t"}, "eye": {"i", "aye"}, "wear": {"where"}, "sundae": {"sunday"},
    "to": {"two", "too"}, "two": {"to", "too"}, "four": {"for"}, "eight": {"ate"}, "one": {"won"},
    "no": {"know"}, "pajamas": {"pyjamas"}, "juice": {"jews"}, "horse": {"whores", "hoarse"},
    "flower": {"flour"}, "pear": {"pair", "pare"}, "sea": {"see", "c"}, "sun": {"son"},
    "eel": {"e"}, "doctor": {"dr"}, "mr": {"mister"}, "mrs": {"missus"},
}


def norm(s: str) -> str:
    s = s.lower().replace(".", "").replace("-", " ").replace("'", "")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(NUM_WORDS.get(w, w) for w in s.split())


def accepted(text: str, heard: str, category: str) -> bool:
    want, got = norm(text), norm(heard)
    if got == want or got.replace(" ", "") == want.replace(" ", ""):
        return True
    alts = LETTER_HOMOPHONES.get(want, set()) if category == "letter" else WORD_HOMOPHONES.get(want, set())
    return got in alts


def load_audio(path: Path, sr: int) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def transcribe(asr: WhisperModel, audio16: np.ndarray) -> str:
    # 前後補 0.5 秒靜音：片段太短、開頭太突然時 whisper 容易聽錯
    pad = np.zeros(8000, dtype=np.float32)
    segs, _ = asr.transcribe(np.concatenate([pad, audio16, pad]), language="en", beam_size=5)
    return " ".join(s.text.strip() for s in segs)


def rel_path(u: dict) -> str:
    return f"letters/{u['slug']}.mp3" if u["category"] == "letter" else f"{u['slug']}.mp3"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--utterances", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--voices", required=True, help="聲音資料夾名稱，逗號分隔")
    ap.add_argument("--only", default="")
    ap.add_argument("--sample", type=int, default=0, help="試聽頁每種聲音再隨機抽 N 句（0 = 全部列出）")
    ap.add_argument("--report-dir", default="", help="report.json 與 listen.html 的輸出位置（預設同 --dir）")
    args = ap.parse_args()

    root = Path(args.dir)
    report_dir = Path(args.report_dir) if args.report_dir else root
    report_dir.mkdir(parents=True, exist_ok=True)
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    utterances = json.loads(Path(args.utterances).read_text())
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    if only:
        utterances = [u for u in utterances if u["slug"] in only]

    asr = WhisperModel("small", device="cpu", compute_type="int8")
    report = []
    for v in voices:
        for u in utterances:
            f = root / v / rel_path(u)
            if not f.exists():
                report.append({"voice": v, "slug": u["slug"], "text": u["text"], "missing": True,
                               "flags": ["missing"]})
                continue
            dur = len(load_audio(f, 24000)) / 24000
            heard = transcribe(asr, load_audio(f, 16000))
            flags = []
            if not accepted(u["text"], heard, u["category"]):
                flags.append("asr")
            if dur > MAX_S:
                flags.append("long")
            if dur < MIN_S:
                flags.append("short")
            if u["category"] == "letter":
                flags.append("letter")
            report.append({"voice": v, "slug": u["slug"], "text": u["text"], "heard": heard,
                           "dur": round(dur, 2), "flags": flags,
                           "path": os.path.relpath(f.resolve(), report_dir.resolve())})
    (report_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))

    # 試聽頁：被標記的 asr/long/short 全列；字母全列；其他依 --sample 抽樣
    rng = random.Random(42)
    shown = []
    for v in voices:
        rows = [r for r in report if r["voice"] == v]
        must = [r for r in rows if r["flags"]]
        rest = [r for r in rows if not r["flags"]]
        if args.sample:
            rest = rng.sample(rest, min(args.sample, len(rest)))
        shown.append((v, must + rest))

    def cell(r: dict) -> str:
        if r.get("missing"):
            return '<div class="c bad">缺檔</div>'
        bad = any(f in r["flags"] for f in ("asr", "long", "short", "missing"))
        cls = "bad" if bad else ("letter" if "letter" in r["flags"] else "")
        return (
            f'<div class="c {cls}"><b>{html.escape(r["text"])}</b>'
            f'<audio controls preload="none" src="{html.escape(r["path"])}"></audio>'
            f'<small>聽成：{html.escape(r["heard"])} · {r["dur"]}s · {",".join(r["flags"]) or "ok"}</small></div>'
        )

    parts = [
        "<!doctype html><meta charset='utf-8'><title>語音試聽</title>",
        "<style>:root{color-scheme:light dark}body{font:15px system-ui;margin:16px;background:Canvas;color:CanvasText}"
        "h2{margin:24px 0 8px}.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:8px}"
        ".c{border:2px solid #8884;border-radius:12px;padding:8px;display:flex;flex-direction:column;gap:4px}"
        ".c audio{width:100%}.bad{border-color:#e63946}.letter{border-color:#f4a261}small{opacity:.75}</style>",
        "<p>紅框＝自動檢查有疑慮（聽成的字不同、太長或太短）；橘框＝字母（一律請人工聽）。</p>",
    ]
    for v, rows in shown:
        n_bad = sum(1 for r in rows if any(f in r["flags"] for f in ("asr", "long", "short", "missing")))
        parts.append(f"<h2>{html.escape(v)}（列出 {len(rows)} 句，紅框 {n_bad} 句）</h2><div class='g'>")
        parts += [cell(r) for r in rows]
        parts.append("</div>")
    (report_dir / "listen.html").write_text("\n".join(parts))

    for v in voices:
        rows = [r for r in report if r["voice"] == v]
        bad = [r for r in rows if any(f in r["flags"] for f in ("asr", "long", "short", "missing"))]
        print(f"{v}: {len(rows)} 句，紅框 {len(bad)}：" + "、".join(f"{r['text']}→{r.get('heard', '')!r}" for r in bad[:15]))


if __name__ == "__main__":
    main()
