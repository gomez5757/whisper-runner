import json
import sys

import soundfile as sf

# Uso: engines.py ENGINE audio.flac out.json
# Motores distintos de Whisper. No imprime contenido: el log es publico.
engine, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
res = {"engine": engine, "segments": []}

if engine in ("canary", "parakeet"):
    import nemo.collections.asr as nemo_asr

    wav = src.replace(".flac", ".wav")
    a, sr = sf.read(src)
    sf.write(wav, a, sr)
    name = "nvidia/canary-1b-v2" if engine == "canary" else "nvidia/parakeet-tdt-0.6b-v3"
    m = nemo_asr.models.ASRModel.from_pretrained(model_name=name)
    kw = {"source_lang": "es", "target_lang": "es"} if engine == "canary" else {}
    o = m.transcribe([wav], timestamps=True, **kw)[0]
    for s in o.timestamp.get("segment", []):
        res["segments"].append({"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["segment"].strip()})
    res["words"] = [[round(w["start"], 2), round(w["end"], 2), w["word"]] for w in o.timestamp.get("word", [])]
    res["text"] = o.text
else:
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration
    from transformers.audio_utils import load_audio

    rid = "CohereLabs/cohere-transcribe-03-2026"
    proc = AutoProcessor.from_pretrained(rid)
    model = CohereAsrForConditionalGeneration.from_pretrained(rid)
    audio = load_audio(src, sampling_rate=16000)
    inp = proc(audio=audio, sampling_rate=16000, return_tensors="pt", language="es")
    idx = inp.get("audio_chunk_index")
    inp.to(model.device, dtype=model.dtype)
    t = proc.decode(model.generate(**inp, max_new_tokens=256), skip_special_tokens=True, audio_chunk_index=idx, language="es")
    res["text"] = (t[0] if isinstance(t, list) else t).strip()
    res["segments"].append({"start": 0, "end": len(audio) / 16000, "text": res["text"]})

json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False)
print("ok", len(res["segments"]), "segmentos")
