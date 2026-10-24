"""Merge sharded ``isaac_sdg.py`` runs into one dataset directory (W4).

On the V100 one Isaac Sim process leaves the card idle between host-side steps (~20 s/frame,
GPU well under 100 %), so a 2000-frame set is rendered as several concurrent shards with distinct
``--seed`` and ``--out``. This joins them into the single directory the rest of the pipeline
(``dataset_stats``, ``train_detector``) expects:

    python -m workshop.merge_sdg --out data/sdg/v2-full data/sdg/v2-full-s0 data/sdg/v2-full-s1 ...

Frames are renamed ``sh<k>_<NNNNNN>`` (``k`` = position on the command line) because every shard
numbers its frames from ``000000``; files are hardlinked when the filesystem allows, else copied.
Shards are only merged when their manifests agree on everything that changes what an image
means (renderer, resolution, label mode, asset config, class order), when each image has a
label file, and when the manifest's per-class instance counts equal what is on disk, which also
catches a shard that was killed mid-run. The merged ``manifest.json`` sums the counters and keeps
each shard's manifest under ``shards``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from workshop.classes import NAMES

IMG_EXTS = (".jpg", ".jpeg", ".png")
# Manifest fields that must be identical in every shard.
SAME_ACROSS_SHARDS = (
    "stage",
    "isaac_sim",
    "gpu",
    "width",
    "height",
    "accum",
    "renderer",
    "spp",
    "labels",
    "assets_sha256",
    "classes",
)
SUMMED = ("frames", "rejected_frames", "rejected_exposure")


def _place(src: Path, dst: Path, link: bool) -> None:
    if link:
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _shard_manifest(shard: Path) -> dict:
    path = shard / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"{shard}: no manifest.json (the shard did not finish)")
    return json.loads(path.read_text(encoding="utf-8"))


def _label_counts(labels: list[Path]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for path in labels:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    return counts


def _check_shard(shard: Path, manifest: dict) -> tuple[list[Path], list[Path]]:
    """Return (images, labels) of one shard after verifying it against its manifest."""
    if manifest.get("classes") != list(NAMES):
        raise SystemExit(f"{shard}: manifest class order differs from workshop.classes.NAMES")
    images = sorted(p for p in (shard / "images").glob("*") if p.suffix.lower() in IMG_EXTS)
    labels = []
    for img in images:
        label = shard / "labels" / f"{img.stem}.txt"
        if not label.is_file():
            raise SystemExit(f"{shard}: image {img.name} has no label file")
        labels.append(label)
    on_disk = _label_counts(labels)
    declared = {n: c for n, c in manifest["instance_counts"].items() if c}
    found = {NAMES[i]: c for i, c in on_disk.items()}
    if declared != found:
        raise SystemExit(
            f"{shard}: manifest instance_counts {declared} != label files {found} "
            "(shard killed mid-run or edited)"
        )
    return images, labels


def merge(out: Path, shards: list[Path], *, link: bool = True) -> dict:
    if not shards:
        raise SystemExit("no shards given")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    manifests = [_shard_manifest(s) for s in shards]
    for key in SAME_ACROSS_SHARDS:
        values = {json.dumps(m.get(key), sort_keys=True) for m in manifests}
        if len(values) != 1:
            raise SystemExit(f"shards disagree on {key!r}: {sorted(values)}")
    checked = [_check_shard(s, m) for s, m in zip(shards, manifests, strict=True)]

    for sub in ("images", "labels", "rejected"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for k, (shard, (images, labels)) in enumerate(zip(shards, checked, strict=True)):
        for src in [*images, *labels]:
            sub = "images" if src.suffix.lower() in IMG_EXTS else "labels"
            _place(src, out / sub / f"sh{k}_{src.name}", link)
        rejected = shard / "rejected"
        for src in sorted(rejected.glob("*")) if rejected.is_dir() else []:
            _place(src, out / "rejected" / f"sh{k}_{src.name}", link)

    merged = {key: manifests[0].get(key) for key in SAME_ACROSS_SHARDS}
    merged["stage"] = f"{manifests[0]['stage']}+merge_sdg"
    merged["assets_root"] = manifests[0].get("assets_root")
    merged["seeds"] = [m["seed"] for m in manifests]
    for key in SUMMED:
        merged[key] = sum(m[key] for m in manifests)
    merged["images"] = sum(len(images) for images, _ in checked)
    merged["instance_counts"] = {
        n: sum(m["instance_counts"].get(n, 0) for m in manifests) for n in NAMES
    }
    merged["asset_sources"] = {
        k: sum(m.get("asset_sources", {}).get(k, 0) for m in manifests) for k in ("local", "s3")
    }
    merged["asset_sources_note"] = "sum over shards; a file used by n shards counts n times"
    merged["seconds"] = max(m["seconds"] for m in manifests)
    merged["seconds_note"] = "wall clock of the slowest shard (shards ran concurrently)"
    merged["shards"] = [
        {"dir": str(s), "images": len(images), **m}
        for s, (images, _), m in zip(shards, checked, manifests, strict=True)
    ]
    (out / "manifest.json").write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return merged


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("shards", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--copy", action="store_true", help="copy files instead of hardlinking")
    a = ap.parse_args(argv)
    merged = merge(a.out, a.shards, link=not a.copy)
    present = sorted(n for n, c in merged["instance_counts"].items() if c)
    missing = [n for n in NAMES if n not in present]
    print(
        f"merged {len(a.shards)} shards -> {a.out}: {merged['images']} images, "
        f"{sum(merged['instance_counts'].values())} instances, "
        f"{len(present)}/{len(NAMES)} classes present"
    )
    if missing:
        print(f"WARNING: zero instances of: {', '.join(missing)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
