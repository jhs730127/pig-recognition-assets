#!/usr/bin/env python3
"""用開源模型生成 pig-english 的英文預錄語音（audio/tts-en-v2/）。

引擎：
  kokoro:<voice>  Kokoro-82M（Apache-2.0），經 mlx-audio 在 Apple Silicon 上跑
  piper:<voice>   Piper（GPL-3.0 程式；聲音模型各有授權，norman/john 為公有領域資料）

用法（在本資料夾內）：
  .venv/bin/python generate.py --utterances utterances.json --out tmp/out \
      --voices kokoro:af_heart,kokoro:af_bella,piper:en_US-norman-medium [--only A,sofa,tv]

輸出：<out>/<聲音資料夾>/<slug>.mp3。清單有字母時另輸出 <out>/<聲音資料夾>/letters/<A>.mp3
（v1.8.0 起清單不含字母，原因見 README）。
後製：頭尾去靜音（保留 50ms）、峰值正規化到 -3 dBFS、單聲道 24 kHz MP3 64 kbps。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
PIPER_VOICES_DIR = HERE / "tmp" / "piper-voices"
KOKORO_REPO = "prince-canuma/Kokoro-82M"

# misaki 把單獨的 A 念成冠詞 ɐ、I 只給次重音，改用行內音標指定字母名稱
LETTER_PHONEMES = {"A": "[A](/ˈA/)", "I": "[I](/ˈI/)"}

TARGET_SR = 24000
PEAK = 10 ** (-3 / 20)
TRIM_REL_DB = -40
PAD_S = 0.05


def folder_name(voice_spec: str) -> str:
    engine, voice = voice_spec.split(":", 1)
    if engine == "piper":
        return voice.removeprefix("en_US-").removesuffix("-medium").removesuffix("-high")
    return voice


class KokoroEngine:
    _model = None

    def __init__(self, voice: str, speed: float):
        self.voice = voice
        self.speed = speed
        if KokoroEngine._model is None:
            from mlx_audio.tts.utils import load_model

            KokoroEngine._model = load_model(KOKORO_REPO)

    def __call__(self, text: str, is_letter: bool, style: int | None = None,
                 speed: float | None = None, suffix: str = "") -> tuple[np.ndarray, int]:
        prompt = (LETTER_PHONEMES.get(text, text) if is_letter else text) + suffix
        if style is None and speed is None and not suffix:
            chunks = []
            sr = TARGET_SR
            for res in KokoroEngine._model.generate(
                text=prompt, voice=self.voice, speed=self.speed, lang_code="a", verbose=False
            ):
                chunks.append(np.array(res.audio, dtype=np.float32))
                sr = getattr(res, "sample_rate", sr)
            return np.concatenate(chunks), sr
        # 變體：Kokoro 依音素長度挑風格向量，單字太短時發音較含糊，改用指定位置的風格向量
        pipe = KokoroEngine._model._get_pipeline("a")
        pack = pipe.load_voice(self.voice)
        ps, _ = pipe.g2p(prompt)
        idx = style if style is not None else len(ps) - 1
        out = KokoroEngine._model(ps, pack[idx], speed or self.speed, return_output=True)
        return np.array(out.audio, dtype=np.float32).reshape(-1), TARGET_SR


class PiperEngine:
    def __init__(self, voice: str, length_scale: float):
        from piper import PiperVoice, SynthesisConfig

        self.model = PiperVoice.load(str(PIPER_VOICES_DIR / f"{voice}.onnx"))
        self.cfg = SynthesisConfig(length_scale=length_scale)

    def __call__(self, text: str, is_letter: bool) -> tuple[np.ndarray, int]:
        chunks = []
        sr = TARGET_SR
        for chunk in self.model.synthesize(text, syn_config=self.cfg):
            chunks.append(chunk.audio_float_array.astype(np.float32))
            sr = chunk.sample_rate
        return np.concatenate(chunks), sr


def make_engine(voice_spec: str, speed: float):
    engine, voice = voice_spec.split(":", 1)
    if engine == "kokoro":
        return KokoroEngine(voice, speed)
    if engine == "piper":
        return PiperEngine(voice, 1.0 / speed)
    raise ValueError(f"unknown engine: {engine}")


def postprocess(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    peak = float(np.max(np.abs(audio))) or 1.0
    thr = peak * 10 ** (TRIM_REL_DB / 20)
    idx = np.flatnonzero(np.abs(audio) > thr)
    if idx.size:
        pad = int(PAD_S * sr)
        audio = audio[max(0, idx[0] - pad) : min(len(audio), idx[-1] + pad)]
    return (audio / peak * PEAK).astype(np.float32)


def write_mp3(audio: np.ndarray, sr: int, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, audio, sr)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp.name,
             "-ac", "1", "-ar", str(TARGET_SR), "-b:a", "64k", str(dest)],
            check=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--utterances", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voices", required=True, help="逗號分隔，例如 kokoro:af_heart,piper:en_US-norman-medium")
    ap.add_argument("--only", default="", help="只生成這些 slug（逗號分隔）")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    utterances = json.loads(Path(args.utterances).read_text())
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    if only:
        utterances = [u for u in utterances if u["slug"] in only]

    out = Path(args.out)
    # fix_flagged.py 為個別句子挑出的變體（style / speed / suffix），重產時照用
    overrides_path = out / "overrides.json"
    overrides = json.loads(overrides_path.read_text()) if overrides_path.exists() else {}
    for spec in args.voices.split(","):
        engine = make_engine(spec, args.speed)
        folder = out / folder_name(spec)
        voice_overrides = overrides.get(folder_name(spec), {})
        for u in utterances:
            is_letter = u["category"] == "letter"
            dest = folder / "letters" / f"{u['slug']}.mp3" if is_letter else folder / f"{u['slug']}.mp3"
            if args.skip_existing and dest.exists():
                continue
            audio, sr = engine(u.get("spoken", u["text"]), is_letter, **voice_overrides.get(u["slug"], {}))
            write_mp3(postprocess(audio, sr), sr, dest)
        print(f"{spec}: {len(utterances)} → {folder}")


if __name__ == "__main__":
    main()
