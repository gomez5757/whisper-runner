import json
import subprocess
import sys

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Uso: score.py cands.json out.json MODEL
# Puntua frases candidatas contra el audio real: cuanto mas alta la nota, mejor encaja.
# No imprime contenido: el log es publico.
cfg, out, name = json.load(open(sys.argv[1], encoding="utf-8")), sys.argv[2], sys.argv[3]
proc = WhisperProcessor.from_pretrained(name)
model = WhisperForConditionalGeneration.from_pretrained(name).eval()
tok = proc.tokenizer
prefix = tok.convert_tokens_to_ids(["<|startoftranscript|>", "<|es|>", "<|transcribe|>", "<|notimestamps|>"])
res = []
for item in cfg:
    raw = subprocess.run(["ffmpeg", "-v", "quiet", "-ss", str(item["start"]), "-to", str(item["end"]), "-i", item["file"],
                          "-ac", "1", "-ar", "16000", "-f", "f32le", "-"], capture_output=True, check=True).stdout
    feats = proc(np.frombuffer(raw, np.float32), sampling_rate=16000, return_tensors="pt").input_features
    with torch.inference_mode():
        enc = model.model.encoder(feats).last_hidden_state
        for c in item["cands"]:
            ids = tok(" " + c.strip(), add_special_tokens=False).input_ids + [tok.eos_token_id]
            dec = torch.tensor([prefix + ids])
            logits = model(encoder_outputs=(enc,), decoder_input_ids=dec[:, :-1]).logits[0]
            lp = torch.log_softmax(logits.float(), -1)[len(prefix) - 1:].gather(1, torch.tensor(ids)[:, None]).squeeze(1)
            res.append({"id": item["id"], "file": item["file"], "cand": c, "sum": round(lp.sum().item(), 3),
                        "mean": round(lp.mean().item(), 3)})
json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False)
print("ok", len(res))
