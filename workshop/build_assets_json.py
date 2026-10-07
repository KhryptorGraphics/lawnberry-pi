"""Merge curated Objaverse USD assets into the SDG asset config (W4).

    python -m workshop.build_assets_json \
        --manifest ~/nvme2/lawnberrypiserver/data/assets/objaverse/manifest.json

Reads ``workshop/sdg_assets.json`` and the Objaverse ``manifest.json`` written by
``workshop.objaverse_select``, appends every converted entry's ``usd/<uid>.usd`` under
its class (sorted by uid, so the output is deterministic), and writes
``workshop/sdg_assets.generated.json`` for ``isaac_sdg.py --assets``.

Paths are written as the Isaac container sees them: the launchers mount the host data
root (``--data-root``, default ``~/nvme2/lawnberrypiserver/data``) at ``/data``, so a
host path under it is rewritten to ``/data/...``. A manifest read from ``/data/...``
inside the container is already in that form. The output is machine-path dependent
and is not committed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from workshop.classes import NAMES

CONTAINER_DATA = Path("/data")
HERE = Path(__file__).resolve().parent


def container_path(path: Path, data_root: Path) -> str:
    """``path`` as seen inside the Isaac container (data root mounted at ``/data``)."""
    try:
        return str(CONTAINER_DATA / path.relative_to(data_root))
    except ValueError:
        if path.is_relative_to(CONTAINER_DATA):
            return str(path)
        raise SystemExit(
            f"{path} is outside the data root {data_root}; the container cannot see it"
        ) from None


def build(
    base: dict[str, list[str]], manifest: list[dict], manifest_dir: Path, data_root: Path
) -> dict[str, list[str]]:
    unknown = sorted({e["class"] for e in manifest} - set(NAMES))
    if unknown:
        raise SystemExit(f"manifest has classes not in workshop.classes: {unknown}")
    out = {name: list(base.get(name, [])) for name in NAMES}
    for entry in sorted(manifest, key=lambda e: e["uid"]):
        usd = entry.get("usd")
        if not usd or not (manifest_dir / usd).is_file():
            continue  # not converted yet
        path = container_path(manifest_dir / usd, data_root)
        if path not in out[entry["class"]]:
            out[entry["class"]].append(path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=HERE / "sdg_assets.json")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument(
        "--data-root", type=Path, default=Path("~/nvme2/lawnberrypiserver/data").expanduser()
    )
    ap.add_argument("--out", type=Path, default=HERE / "sdg_assets.generated.json")
    a = ap.parse_args()
    manifest_path = a.manifest.expanduser().resolve()
    cfg = build(
        json.loads(a.base.read_text()),
        json.loads(manifest_path.read_text()),
        manifest_path.parent,
        a.data_root.expanduser().resolve(),
    )
    a.out.write_text(json.dumps(cfg, indent=2) + "\n")
    print(json.dumps({k: len(v) for k, v in cfg.items() if v}))


if __name__ == "__main__":
    main()
