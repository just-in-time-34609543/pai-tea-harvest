"""Embed task instructions with the frozen text tower of SmolVLM2 (the VLM inside SmolVLA). Run once; CPU is fine.

    python scripts/pcvla/embed_language.py  -> output/adeept/pcvla/lang.pt  {instruction: tensor(960)}
"""
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harvest_env as H  # noqa: E402

VLM = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
OUT = Path(__file__).resolve().parents[2] / "output" / "adeept" / "pcvla" / "lang.pt"


def main():
    proc = AutoProcessor.from_pretrained(VLM)
    model = AutoModelForImageTextToText.from_pretrained(VLM, torch_dtype=torch.float32).eval()
    text_model = model.model.text_model
    out = torch.load(OUT) if OUT.exists() else {}      # keep earlier instructions (older checkpoints use them)
    with torch.no_grad():
        for t in H.TASKS.values():
            ids = proc.tokenizer(t.language, return_tensors="pt")
            h = text_model(**ids).last_hidden_state[0]           # (L, 960)
            out[t.language] = h.mean(0)
            print(t.language, tuple(h.shape))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
