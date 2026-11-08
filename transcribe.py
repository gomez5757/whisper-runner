import json
import sys

from faster_whisper import WhisperModel, decode_audio

# Uso: transcribe.py audio cfg.json start end out.json
# No imprime nada del contenido: el log del repo es publico.
src, cfg_path, start, end, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
cfg = json.load(open(cfg_path, encoding="utf-8"))

sr = 16000
audio = decode_audio(src, sampling_rate=sr)
a = audio[int(start * sr): int(end * sr) if end > 0 else None]

model = WhisperModel(cfg.get("model", "large-v3"), device="cpu", compute_type="int8", cpu_threads=4)
segs, info = model.transcribe(
    a,
    language=cfg.get("language") or None,
    initial_prompt=cfg.get("prompt") or None,
    beam_size=5,
    best_of=5,
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    vad_filter=False,
    condition_on_previous_text=False,
    word_timestamps=True,
    hallucination_silence_threshold=2.0,
)

res = {"start": start, "end": end, "language": info.language, "segments": []}
for s in segs:
    res["segments"].append({
        "start": round(s.start + start, 2),
        "end": round(s.end + start, 2),
        "text": s.text.strip(),
        "avg_logprob": round(s.avg_logprob, 3),
        "no_speech_prob": round(s.no_speech_prob, 3),
        "compression_ratio": round(s.compression_ratio, 2),
        "words": [[round(w.start + start, 2), round(w.end + start, 2), w.word, round(w.probability, 3)] for w in (s.words or [])],
    })
json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False)
print("ok", len(res["segments"]), "segmentos")
