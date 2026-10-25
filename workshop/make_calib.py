"""Build a reproducible calibration image directory for ``scripts/hailo_compile.sh`` (W5).

    python -m workshop.make_calib --out data/hailo/calib/v1v2 --count 1024 --seed 1 \
        --from data/sdg/v1/images --from data/sdg/v2-full/images
    python -m workshop.make_calib --verify data/hailo/calib/v1v2     # what hailo_compile.sh runs

The model zoo (v2.19.1 ``ImageFeed``) lists the calibration folder sorted by path and takes the
first ``calibset_size`` (1024) files. Pointing ``--calib`` at a single SDG run therefore
calibrates on its first 1024 frames only, and v1 has no instance of ten classes, so the
activation ranges of houses, walls or drop-offs would never be seen. This draws an equal share
from every ``--from`` directory (seeded, without replacement) into ``--out`` and writes
``calib_manifest.json`` (sources, seed, file list, digest) next to the images.

The files are named ``<position>_c<source>_<original name>`` and the positions go round-robin
over the sources, so the sorted order alternates between them and *every* sorted prefix holds an
equal share (to within one file) from each source. That is what makes ``--count`` above 1024
harmless: the zoo reads only the first 1024 sorted files, and those are still balanced. (Named
by source first, as before, a folder of 2048 files calibrated on the first source alone.)
``--verify`` re-checks a folder against its manifest and against that rule: the same image files
by name AND by content (every file is re-hashed against the sha256 the manifest recorded, so a
file replaced under its old name is found), and the sorted prefix balanced. It also accepts the
older ``c<source>_...`` names, which are balanced only while the folder holds 1024 files or fewer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png")
# How many sorted files the zoo's ``calibset_size`` makes it read (cfg/alls/generic/yolo26s.alls).
CALIBSET_SIZE = 1024
SCHEMA = "calib-set/2"
SCHEMAS = ("calib-set/1", SCHEMA)  # /1 named its files c<source>_<name>, without a position
_SOURCE_OF = re.compile(r"(?:^|_)c(\d+)_")


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


def interleave(shares: list[int]) -> list[int]:
    """Source index of every output position: round-robin, a source leaves when its share is used.

    ``[4, 3, 3]`` -> ``0 1 2 0 1 2 0 1 2 0``: any prefix of the result has, per source, within one
    of an equal share.
    """
    left = list(shares)
    order: list[int] = []
    while any(left):
        for k, n in enumerate(left):
            if n:
                order.append(k)
                left[k] -= 1
    return order


def calib_name(position: int, source_index: int, original: str, count: int) -> str:
    """``<position>_c<source>_<original>``; positions are zero-padded so names sort as positions."""
    width = max(5, len(str(count - 1)))
    return f"{position:0{width}d}_c{source_index}_{original}"


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
    picks: list[list[Path]] = []
    for src, share in zip(sources, shares, strict=True):
        pool = _images(src)
        if len(pool) < share:
            raise SystemExit(f"{src}: {len(pool)} images, need {share}")
        picks.append(rng.sample(pool, share))
    out.mkdir(parents=True, exist_ok=True)
    cursor = [0] * len(sources)
    files = []
    for position, k in enumerate(interleave(shares)):
        src = picks[k][cursor[k]]
        cursor[k] += 1
        name = calib_name(position, k, src.name, count)
        _place(src, out / name, link)
        files.append({"name": name, "source": str(src), "source_index": k, "sha256": _sha256(src)})
    files.sort(key=lambda f: f["name"])
    prefix = files[:CALIBSET_SIZE]
    manifest = {
        "schema": SCHEMA,
        "count": len(files),
        "seed": seed,
        "sources": [str(s) for s in sources],
        "per_source": shares,
        "calibset_size": CALIBSET_SIZE,
        "zoo_prefix": {
            "files": len(prefix),
            "per_source": [
                sum(f["source_index"] == k for f in prefix) for k in range(len(sources))
            ],
        },
        "list_sha256": hashlib.sha256("\n".join(f["name"] for f in files).encode()).hexdigest(),
        "files": files,
    }
    (out / "calib_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def verify(folder: Path, calibset_size: int = CALIBSET_SIZE) -> dict:
    """Check ``folder`` against its manifest and the zoo's sorted-prefix rule.

    Raises ``SystemExit`` with the reason unless the images on disk are exactly the manifest's
    files, each with the bytes (sha256) the manifest recorded, and the first ``calibset_size`` of
    them, in sorted order (all the zoo reads), hold every source with an equal share to within one
    file. Returns what was found.
    """
    path = folder / "calib_manifest.json"
    if not path.is_file():
        raise SystemExit(f"{folder}: no calib_manifest.json (build the folder with make_calib)")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        listed = sorted(f["name"] for f in manifest["files"])
        recorded = {f["name"]: f.get("sha256") for f in manifest["files"]}
        n_sources = len(manifest["sources"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"{path}: unreadable manifest ({type(exc).__name__}: {exc})") from exc
    if manifest.get("schema") not in SCHEMAS:
        raise SystemExit(f"{path}: schema {manifest.get('schema')!r} is not one of {SCHEMAS}")
    on_disk = sorted(p.name for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)
    if on_disk != listed:
        gone, extra = sorted(set(listed) - set(on_disk)), sorted(set(on_disk) - set(listed))
        raise SystemExit(
            f"{folder}: the images are not the manifest's files ({len(gone)} missing, "
            f"{len(extra)} not in the manifest; first: {(gone + extra)[:3]})"
        )
    unrecorded = [n for n in on_disk if not isinstance(recorded.get(n), str)]
    if unrecorded:
        raise SystemExit(
            f"{folder}: the manifest records no sha256 for {len(unrecorded)} files (first: "
            f"{unrecorded[:3]}), so their content cannot be verified: rebuild the folder"
        )
    changed = [n for n in on_disk if _sha256(folder / n) != recorded[n]]
    if changed:
        raise SystemExit(
            f"{folder}: {len(changed)} files are not the bytes the manifest recorded (first: "
            f"{changed[:3]}): the folder was edited after it was built, rebuild it"
        )
    prefix = on_disk[:calibset_size]
    shares = Counter()
    for name in prefix:
        match = _SOURCE_OF.search(name)
        if match is None or int(match.group(1)) >= n_sources:
            raise SystemExit(f"{folder}: {name!r} does not say which of the {n_sources} sources")
        shares[int(match.group(1))] += 1
    per_source = [shares[k] for k in range(n_sources)]
    fair = len(prefix) / n_sources
    if any(abs(n - fair) > 1 for n in per_source):
        raise SystemExit(
            f"{folder}: the zoo reads only the first {len(prefix)} of {len(on_disk)} sorted files "
            f"and they come {per_source} from the {n_sources} sources (equal would be "
            f"{fair:.1f} each): rebuild the folder with make_calib"
        )
    return {"files": len(on_disk), "zoo_prefix": len(prefix), "per_source": per_source}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--from", dest="sources", type=Path, action="append", default=[])
    ap.add_argument("--count", type=int, default=CALIBSET_SIZE)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--copy", action="store_true", help="copy files instead of hardlinking")
    ap.add_argument("--verify", type=Path, metavar="DIR", help="check DIR instead of building")
    ap.add_argument("--calibset-size", type=int, default=CALIBSET_SIZE, help="with --verify")
    a = ap.parse_args(argv)
    if a.verify:
        found = verify(a.verify, a.calibset_size)
        print(
            f"calibration folder ok: {found['files']} images, the zoo reads the first "
            f"{found['zoo_prefix']} sorted, per source {found['per_source']}"
        )
        return 0
    if not a.out:
        ap.error("--out is required to build a folder (or use --verify DIR)")
    m = build(a.out, a.sources, a.count, a.seed, link=not a.copy)
    print(f"{m['count']} calibration images ({m['per_source']} per source) -> {a.out}")
    if m["count"] != CALIBSET_SIZE:
        print(
            f"note: the zoo reads the first {CALIBSET_SIZE} sorted files "
            f"({m['zoo_prefix']['per_source']} per source); "
            + (
                "the rest are never used"
                if m["count"] > CALIBSET_SIZE
                else f"hailo_compile.sh needs at least {CALIBSET_SIZE}"
            ),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
