"""Build a reproducible calibration image directory for ``scripts/hailo_compile.sh`` (W5).

    python -m workshop.make_calib --out data/hailo/calib/v1v2 --count 1024 --seed 1 \
        --from data/sdg/v1/images --from data/sdg/v2-full/images

The model zoo (v2.19.1 ``ImageFeed``) lists the calibration folder sorted by path and takes the
first ``calibset_size`` (1024) files. Pointing ``--calib`` at a single SDG run therefore
calibrates on its first 1024 frames only, and v1 has no instance of ten classes, so the
activation ranges of houses, walls or drop-offs would never be seen. This draws an equal share
from every ``--from`` directory (seeded, without replacement) into ``--out`` -- exactly
``--count`` files, so the sorted-prefix rule cannot skip any of them -- and writes
``calib_manifest.json`` (sources, seed, file list, digest) next to the images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png")


def _images(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("*") if p.suffix.lower() in IMG_EXTS)


def _place(src: Path, dst: Path, link: bool) -> None:
    if link:
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def build(out: Path, sources: list[Path], count: int, seed: int, *, link: bool = True) -> dict:
    if not sources:
        raise SystemExit("no --from directories")
    if count <= 0:
        raise SystemExit("--count must be positive")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    # Equal shares; the remainder goes to the first sources.
    shares = [count // len(sources) + (i < count % len(sources)) for i in range(len(sources))]
    rng = random.Random(seed)
    chosen: list[tuple[int, Path]] = []
    for k, (src, share) in enumerate(zip(sources, shares, strict=True)):
        pool = _images(src)
        if len(pool) < share:
            raise SystemExit(f"{src}: {len(pool)} images, need {share}")
        chosen += [(k, p) for p in rng.sample(pool, share)]
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for k, src in chosen:
        name = f"c{k}_{src.name}"
        _place(src, out / name, link)
        files.append({"name": name, "source": str(src), "sha256": _sha256(src)})
    files.sort(key=lambda f: f["name"])
    manifest = {
        "schema": "calib-set/1",
        "count": len(files),
        "seed": seed,
        "sources": [str(s) for s in sources],
        "per_source": shares,
        "list_sha256": hashlib.sha256("\n".join(f["name"] for f in files).encode()).hexdigest(),
        "files": files,
    }
    (out / "calib_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--from", dest="sources", type=Path, action="append", default=[])
    ap.add_argument("--count", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--copy", action="store_true", help="copy files instead of hardlinking")
    a = ap.parse_args(argv)
    m = build(a.out, a.sources, a.count, a.seed, link=not a.copy)
    print(f"{m['count']} calibration images ({m['per_source']} per source) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
