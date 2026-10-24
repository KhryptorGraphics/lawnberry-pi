"""W5 detector training: YOLO26s on real + synthetic YOLO-format datasets.

Runs in the workshop env ($LB/env/bin/python) on the V100 (default) or 3080 Ti.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      $LB/env/bin/python -m workshop.train_detector \
        --real data/datasets/v1 --sdg data/sdg/v1 --tag yard1

Dataset inputs are directories with ``images/`` and ``labels/`` (flat, or with
``images/{train,val}`` splits). The split rule for a flat ``--real`` dir is
deterministic: md5(filename) % 10 < 9 -> train. SDG frames NEVER appear in val
(domain gap would flatter metrics); the validation set is entirely real
imagery.

Writes ``$LB/data/models/yolo26s_<tag>/``: ``dataset/data.yaml``, weights,
``best.onnx`` (opset 17, simplified, end2end for the Hailo zoo yolo26 parser),
and ``run_manifest.json`` (split assignment counts + per-class recall vs gates).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from workshop.classes import CLASSES, NAMES, RECALL_GATE
from workshop.export_detector import export_hailo_onnx

LB = Path(os.environ.get("LB", str(Path.home() / "nvme2" / "lawnberrypiserver")))
IMG_EXTS = (".jpg", ".jpeg", ".png")


def _images(d: Path) -> list[Path]:
    roots = [d / "images"]
    if (d / "images" / "train").is_dir():
        roots = [d / "images" / s for s in ("train", "val")]
    return sorted(p for r in roots for p in r.glob("*") if p.suffix.lower() in IMG_EXTS)


def _split_of(name: str) -> bool:
    """True -> train (90%), by deterministic md5 of the filename."""
    return (int(hashlib.md5(name.encode()).hexdigest(), 16) % 10) < 9


def _label_for(src: Path, split: str, stem: str) -> Path | None:
    for cand in (src / "labels" / split / f"{stem}.txt", src / "labels" / f"{stem}.txt"):
        if cand.exists():
            return cand
    return None


def stage_dataset(out: Path, sources: list[tuple[Path, bool]], *, link: bool) -> dict:
    """Build a YOLO dataset dir from (dir, is_real) sources.

    Flat real dirs get the md5 split; dirs already carrying images/{train,val}
    keep their assignment; SDG sources are train-only. Files are copied (or
    hardlinked where possible) so the run dir is self-contained.

    Staged files are prefixed ``s<source index>_``: every SDG run numbers its frames
    ``000000.jpg`` upward, so unprefixed names collide between sources (v1 + v2) and the later
    source's frames used to be skipped silently while still being counted. The staged file counts
    are checked against the counted ones before returning.
    """
    stats: dict = {"train": 0, "val": 0, "per_source": {}}
    for idx, (src, is_real) in enumerate(sources):
        prefix = f"s{idx}_"
        has_splits = (src / "images" / "train").is_dir()
        n_tr = n_va = 0
        for img in _images(src):
            if not is_real:
                split = "train"  # SDG never validates
            elif has_splits:
                lab = _label_for(src, "val", img.stem)
                split = "val" if (lab and (src / "images" / "val" / img.name).exists()) else "train"
            else:
                split = "train" if _split_of(img.name) else "val"
            lab = _label_for(src, split, img.stem) or _label_for(src, "", img.stem)
            if lab is None:
                continue  # unlabeled image: not part of the training set
            for sub, sfile in (("images", img), ("labels", lab)):
                dst = out / sub / split / (prefix + sfile.name)
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    if link:
                        try:
                            dst.hardlink_to(sfile)
                            continue
                        except OSError:
                            pass
                    shutil.copy2(sfile, dst)
            if split == "train":
                n_tr += 1
            else:
                n_va += 1
        stats["train"] += n_tr
        stats["val"] += n_va
        stats["per_source"][str(src)] = {
            "train": n_tr,
            "val": n_va,
            "real": is_real,
            "prefix": prefix,
        }
    for split in ("train", "val"):
        staged_dir = out / "images" / split
        staged = sum(1 for _ in staged_dir.iterdir()) if staged_dir.is_dir() else 0
        if staged != stats[split]:
            raise SystemExit(
                f"staged {staged} {split} images but counted {stats[split]}: "
                "a file name occurs twice within one source"
            )
    return stats


def write_yaml(out: Path) -> None:
    lines = [
        f"path: {out.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(NAMES)}",
        "names:",
    ]
    lines += [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    (out / "data.yaml").write_text("\n".join(lines) + "\n")


def per_class_metrics(box, names: dict[int, str] | list[str]) -> dict[str, dict]:
    """Per-class AP50 / recall / precision from an Ultralytics ``DetMetrics.box``.

    ``box.ap50``, ``box.r`` and ``box.p`` are compact: one entry per class that occurs in the
    validation set, in the order of ``box.ap_class_index``. Indexing them by class id mislabels
    every class after an absent one, so classes are looked up through ``ap_class_index``; a class
    absent from validation is omitted (the caller reports it as "no samples"). ``recall`` and
    ``precision`` are Ultralytics' values at the F1-maximising confidence, not at the deployed
    threshold: ``scripts/eval_detector.py`` is the fixed-threshold measurement.
    """
    out: dict[str, dict] = {}
    for pos, cls in enumerate(box.ap_class_index):
        cls = int(cls)
        if isinstance(names, dict):
            name = names.get(cls)
        else:
            name = names[cls] if cls < len(names) else None
        out[name or (NAMES[cls] if cls < len(NAMES) else str(cls))] = {
            "ap50": float(box.ap50[pos]),
            "recall": float(box.r[pos]),
            "precision": float(box.p[pos]),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--real",
        action="append",
        type=Path,
        default=[],
        help="real YOLO-format dataset dir (repeatable)",
    )
    ap.add_argument(
        "--sdg",
        action="append",
        type=Path,
        default=[],
        help="SDG dataset dir (train-only; repeatable)",
    )
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0", help="torch device within the env")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--model", default=None, help="yolo26s.pt path (default: ultralytics auto-download)"
    )
    ap.add_argument("--out-root", type=Path, default=LB / "data" / "models")
    ap.add_argument("--link", action="store_true", help="hardlink dataset files instead of copying")
    ap.add_argument(
        "--smoke-sdg-val",
        action="store_true",
        help="recipe smoke only: md5-split SDG into train/val when no real "
        "data exists yet; metrics are NOT a recall gate",
    )
    a = ap.parse_args()

    sdg_split = a.smoke_sdg_val and not a.real
    sources = [(d.resolve(), True) for d in a.real] + [(d.resolve(), sdg_split) for d in a.sdg]
    if not sources:
        raise SystemExit("no datasets given (--real/--sdg)")
    for d, _ in sources:
        if not (d / "images").is_dir():
            raise SystemExit(f"{d} has no images/ dir")

    run_dir = a.out_root / f"yolo26s_{a.tag}"
    ds_dir = run_dir / "dataset"
    if ds_dir.exists():
        raise SystemExit(f"{ds_dir} exists; pick a new --tag or remove it")
    ds_dir.mkdir(parents=True)
    stats = stage_dataset(ds_dir, sources, link=a.link)
    write_yaml(ds_dir)
    print(json.dumps(stats, indent=2))
    if stats["val"] == 0:
        raise SystemExit("no real validation images; val must be entirely real")

    from ultralytics import YOLO

    weights = a.model or "yolo26s.pt"
    model = YOLO(weights)
    model.train(
        data=str(ds_dir / "data.yaml"),
        imgsz=a.imgsz,
        batch=a.batch,
        epochs=a.epochs,
        device=a.device,
        project=str(a.out_root),
        name=f"yolo26s_{a.tag}",
        exist_ok=True,
        seed=a.seed,
        patience=0 if a.epochs <= 3 else 50,
    )
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        cand = sorted(run_dir.glob("**/best.pt"))
        if not cand:
            raise SystemExit(f"no best.pt under {run_dir}")
        best = cand[0]
    model = YOLO(str(best))

    metrics = model.val(split="val", device=a.device, verbose=False)
    per_class: dict[str, dict] = {}
    per_class_error = None
    try:
        per_class = per_class_metrics(metrics.box, metrics.names or {})
    except Exception as exc:  # metrics layout differs across ultralytics versions
        per_class_error = f"{type(exc).__name__}: {exc}"
        print(f"WARNING: per-class extraction failed: {per_class_error}")

    print(f"{'class':16} {'tier':4} {'recall':7} {'gate':6} verdict")
    failures = []
    for c in CLASSES:
        m = per_class.get(c.name)
        if m is None or m.get("recall") is None:
            print(f"{c.name:16} {c.tier:<4} {'-':7} {RECALL_GATE[c.tier]:<6} no samples")
            continue
        ok = m["recall"] >= RECALL_GATE[c.tier]
        if not ok:
            failures.append(c.name)
        print(
            f"{c.name:16} {c.tier:<4} {m['recall']:<7.3f} {RECALL_GATE[c.tier]:<6} "
            f"{'PASS' if ok else 'BELOW GATE'}"
        )

    onnx_path = export_hailo_onnx(best, device=a.device)

    manifest = {
        "schema": "train-v1",
        "tag": a.tag,
        "base_weights": str(weights),
        "datasets": stats,
        "split_rule": "flat real dirs: md5(filename)%10<9 -> train; dirs with "
        "images/{train,val} keep their split; SDG train-only; "
        "val is entirely real",
        "hyper": {
            "epochs": a.epochs,
            "imgsz": a.imgsz,
            "batch": a.batch,
            "device": a.device,
            "seed": a.seed,
        },
        "smoke_sdg_val": sdg_split,
        "per_class_val": per_class,
        "per_class_val_error": per_class_error,
        "gate_failures": failures,
        "onnx": str(onnx_path),
        "onnx_opset": 17,
        "onnx_end2end": True,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"run manifest + ONNX: {run_dir}")


if __name__ == "__main__":
    main()
