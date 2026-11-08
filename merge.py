import json
import subprocess
import sys
from pathlib import Path

# Uso: merge.py KEYFILE WORKDIR ARTIFACTS_DIR
# Descifra los trozos, quita el solape y escribe un texto por video y version.
key, work, arts = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
man = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
chunk = man["chunk"]


def ts(t):
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}"


groups = {}
for b in sorted(arts.rglob("*.bin")):
    name = b.stem
    j = work / f"{name}.json"
    subprocess.run(["openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2", "-iter", "200000",
                    "-pass", f"file:{key}", "-in", str(b), "-out", str(j)], check=True)
    vid, variant, k = name.split("-")
    groups.setdefault(f"{vid}-{variant}", []).append((int(k), json.loads(j.read_text(encoding="utf-8"))))

for g, parts in sorted(groups.items()):
    segs = []
    for k, res in sorted(parts):
        lo, hi = k * chunk, (k + 1) * chunk
        segs += [s for s in res["segments"] if lo <= (s["start"] + s["end"]) / 2 < hi]
    lines = []
    for s in segs:
        words = s["words"]
        text = "".join(f" [{w[2].strip()}?]" if w[3] < 0.35 else w[2] for w in words).strip() if words else s["text"]
        flag = "  <<dudoso>>" if s["avg_logprob"] < -0.8 or s["no_speech_prob"] > 0.6 else ""
        lines.append(f"[{ts(s['start'])}] {text}{flag}")
    (work / f"{g}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    json.dump(segs, open(work / f"{g}.segments.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(g, len(segs), "frases,", sum("<<dudoso>>" in l for l in lines), "dudosas")
