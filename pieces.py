import json
import sys
from pathlib import Path

import wave

import numpy as np
from faster_whisper import decode_audio

# Uso: pieces.py OUTDIR audio1.flac [audio2.flac...]
# Corta cada audio en trozos de 8 a 20 s, cortando en el punto mas silencioso.
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
SR, MIN_S, MAX_S, WIN = 16000, 8, 20, 0.1
rows = []
for src in sys.argv[2:]:
    a = decode_audio(src, sampling_rate=SR)
    hop = int(WIN * SR)
    rms = np.sqrt(np.convolve(a ** 2, np.ones(hop) / hop, mode="same"))
    stem = Path(src).stem.split("-")[0]
    start, k = 0, 0
    while start < len(a):
        lo, hi = start + MIN_S * SR, start + MAX_S * SR
        cut = len(a) if hi >= len(a) else lo + int(np.argmin(rms[lo:hi:160])) * 160
        name = f"{stem}_{k:02d}.wav"
        with wave.open(str(out / name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((np.clip(a[start:cut], -1, 1) * 32767).astype("<i2").tobytes())
        rows.append({"audio": name, "duration": round((cut - start) / SR, 3), "start": round(start / SR, 2)})
        start, k = cut, k + 1
with open(out / "markdown.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(len(rows), "trozos")
