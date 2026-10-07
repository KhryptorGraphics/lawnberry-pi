"""W2 auto-labeling (workshop only, offline).

    python -m workshop.autolabel --frames DIR --out DATASET_DIR [--every-s 1.0]

1. Sample: perceptual-hash dedup keeps a frame only if it differs from the
   last kept frame by >= ``--min-hamming`` bits.
2. Propose: MM Grounding DINO large (Apache-2.0, open weights) with the
   prompts in ``workshop/classes.py``.
3. Write YOLO labels plus ``review.jsonl`` (every box ``reviewed: false``).
   Human review flips boxes to reviewed/corrected; ``workshop.review`` refuses
   to export a dataset with unreviewed boxes or missing spot checks.

The output folder is versioned by ``manifest.json`` (model id, revision,
thresholds, input hashes) so a run is reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .classes import CLASSES

MODEL_ID = "openmmlab-community/mm_grounding_dino_large_all"


def dedup(frames: list[Path], min_hamming: int) -> list[Path]:
    import imagehash
    from PIL import Image

    kept, last = [], None
    for f in frames:
        h = imagehash.phash(Image.open(f))
        if last is None or h - last >= min_hamming:
            kept.append(f)
            last = h
    return kept


def to_yolo(box, w: int, h: int) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = max(0.0, x0), min(float(w), x1)
    y0, y1 = max(0.0, y0), min(float(h), y1)
    return ((x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h)


def prompt_lookup() -> dict[str, int]:
    """Map every prompt phrase to its class index."""
    out = {}
    for i, c in enumerate(CLASSES):
        for phrase in c.prompt.split(" . "):
            out[phrase.strip()] = i
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-hamming", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch
    import transformers
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    frames = sorted(a.frames.glob("*.jpg")) + sorted(a.frames.glob("*.png"))
    kept = dedup(frames, a.min_hamming)
    (a.out / "images").mkdir(parents=True, exist_ok=True)
    (a.out / "labels").mkdir(parents=True, exist_ok=True)

    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(a.device).eval()
    lookup = prompt_lookup()
    phrases = [list(lookup)]
    review = (a.out / "review.jsonl").open("w")
    digest = hashlib.sha256()
    for f in kept:
        digest.update(f.read_bytes())
        img = Image.open(f).convert("RGB")
        inp = proc(images=img, text=phrases, return_tensors="pt").to(a.device)
        with torch.no_grad():
            out = model(**inp)
        r = proc.post_process_grounded_object_detection(
            out, threshold=a.threshold, target_sizes=[img.size[::-1]]
        )[0]
        rows = []
        for s, lab, box in zip(r["scores"], r["text_labels"], r["boxes"], strict=False):
            cls = lookup.get(lab.strip())
            if cls is None:
                continue
            cx, cy, bw, bh = to_yolo(box, *img.size)
            rows.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            review.write(
                json.dumps(
                    {
                        "image": f.name,
                        "cls": cls,
                        "score": round(float(s), 3),
                        "yolo": [cx, cy, bw, bh],
                        "reviewed": False,
                    }
                )
                + "\n"
            )
        img.save(a.out / "images" / f.name)
        (a.out / "labels" / f"{f.stem}.txt").write_text("\n".join(rows))
    review.close()
    (a.out / "manifest.json").write_text(
        json.dumps(
            {
                "stage": "autolabel",
                "model": MODEL_ID,
                "transformers": transformers.__version__,
                "threshold": a.threshold,
                "min_hamming": a.min_hamming,
                "frames_in": len(frames),
                "frames_kept": len(kept),
                "input_sha256": digest.hexdigest(),
                "classes": [c.name for c in CLASSES],
            },
            indent=2,
        )
    )
    print(f"kept {len(kept)}/{len(frames)} frames -> {a.out}")


if __name__ == "__main__":
    main()
