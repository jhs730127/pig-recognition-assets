#!/usr/bin/env python3
"""重生自動檢查沒過的句子：依序試幾種變體，留第一個 ASR 聽對的。

Kokoro 依音素長度挑風格向量，單字太短時字首常含糊（fish 聽成 wish、key 聽成 I see）。
變體：改用較長輸入的風格向量、放慢、加驚嘆號。選中的變體寫進 <dir>/overrides.json，
generate.py 重產時照用。全部變體都沒過的句子保留原檔，列在輸出最後給人工聽。

用法（在本資料夾內）：
  .venv/bin/python fix_flagged.py --utterances ../../audio/tts-en-v2/utterances.json \
      --dir ../../audio/tts-en-v2 --report tmp/check/report.json --voices af_heart,af_bella,am_michael
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.signal as sps
from faster_whisper import WhisperModel

from check import accepted, load_audio, rel_path, transcribe
from generate import TARGET_SR, KokoroEngine, postprocess, write_mp3

VARIANTS = [
    {"style": 40},
    {"style": 20},
    {"style": 60},
    {"speed": 0.85},
    {"style": 40, "speed": 0.85},
    {"suffix": "!"},
    {"style": 40, "suffix": "!"},
    {"style": 100},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--utterances", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--voices", required=True, help="Kokoro 聲音名稱，逗號分隔")
    args = ap.parse_args()

    root = Path(args.dir)
    by_slug = {u["slug"]: u for u in json.loads(Path(args.utterances).read_text())}
    report = json.loads(Path(args.report).read_text())
    overrides_path = root / "overrides.json"
    overrides = json.loads(overrides_path.read_text()) if overrides_path.exists() else {}
    asr = WhisperModel("small", device="cpu", compute_type="int8")

    unresolved = []
    for voice in [v.strip() for v in args.voices.split(",") if v.strip()]:
        engine = KokoroEngine(voice, 1.0)
        todo = [r for r in report if r["voice"] == voice and "asr" in r["flags"]]
        fixed = already_ok = 0
        for r in todo:
            u = by_slug[r["slug"]]
            is_letter = u["category"] == "letter"
            dest = root / voice / rel_path(u)
            if accepted(u["text"], transcribe(asr, load_audio(dest, 16000)), u["category"]):
                already_ok += 1
                continue
            for var in VARIANTS:
                audio, sr = engine(u.get("spoken", u["text"]), is_letter, **var)
                audio = postprocess(audio, sr)
                a16 = sps.resample_poly(audio, 2, 3).astype(np.float32) if sr == TARGET_SR else audio
                heard = transcribe(asr, a16)
                if accepted(u["text"], heard, u["category"]):
                    write_mp3(audio, sr, dest)
                    overrides.setdefault(voice, {})[u["slug"]] = var
                    fixed += 1
                    break
            else:
                unresolved.append(f"{voice}:{u['text']}")
        overrides_path.write_text(json.dumps(overrides, ensure_ascii=False, indent=1, sort_keys=True))
        print(f"{voice}: 標記 {len(todo)}，補靜音後就聽對 {already_ok}，換變體修好 {fixed}，"
              f"沒修好 {len(todo) - already_ok - fixed}")
    print("沒修好：" + "、".join(unresolved))


if __name__ == "__main__":
    main()
