import json
import subprocess
import sys
from pathlib import Path

# Uso: prep.py KEYFILE WORKDIR LANG PROMPT VIDEO [VIDEO...]
# Saca el audio en dos versiones, lo cifra en data/ y escribe jobs.json.
key, work, lang, prompt, videos = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5:]
repo = Path(__file__).parent
data = repo / "data"
data.mkdir(exist_ok=True)
work.mkdir(parents=True, exist_ok=True)

CHUNK, OVERLAP = 300, 20
ENH = "highpass=f=80,lowpass=f=8000,afftdn=nf=-25,dynaudnorm=f=150:g=15:p=0.9:m=20,alimiter=limit=0.95"


def enc(src, dst):
    subprocess.run(["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-iter", "200000", "-salt",
                    "-pass", f"file:{key}", "-in", str(src), "-out", str(dst)], check=True)


cfg = work / "cfg.json"
cfg.write_text(json.dumps({"language": lang, "prompt": prompt, "model": "large-v3"}, ensure_ascii=False), encoding="utf-8")
enc(cfg, data / "cfg.bin")

jobs, manifest = [], {}
for i, v in enumerate(videos, 1):
    for variant, af in (("orig", None), ("enh", ENH)):
        name = f"v{i}-{variant}"
        out = work / f"{name}.flac"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", v, "-vn", "-ac", "1", "-ar", "16000"]
        if af:
            cmd += ["-af", af]
        subprocess.run(cmd + [str(out)], check=True)
        enc(out, data / f"{name}.bin")
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(work / f"v{i}-orig.flac")], capture_output=True, text=True, check=True).stdout)
    manifest[f"v{i}"] = {"video": v, "duration": dur}
    k = 0
    while k * CHUNK < dur:
        s, e = max(0, k * CHUNK - OVERLAP), min(dur, (k + 1) * CHUNK + OVERLAP)
        for variant in ("orig", "enh"):
            jobs.append({"id": f"v{i}-{variant}-{k:02d}", "file": f"v{i}-{variant}", "start": round(s, 2), "end": round(e, 2)})
        k += 1

(repo / "jobs.json").write_text(json.dumps(jobs, indent=1), encoding="utf-8")
(work / "manifest.json").write_text(json.dumps({"chunk": CHUNK, "videos": manifest}, ensure_ascii=False, indent=1), encoding="utf-8")
print(len(jobs), "trabajos;", {k: round(m["duration"] / 60, 1) for k, m in manifest.items()}, "min")
