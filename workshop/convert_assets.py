"""Convert curated Objaverse GLBs to USD inside Isaac Sim (W4).

    scripts/isaacsim-headless.sh workshop/convert_assets.py data/assets/objaverse

Reads ``manifest.json`` written by ``workshop.objaverse_select --curation``,
converts each ``glb/<uid>.glb`` to ``usd/<uid>.usd`` with Omniverse's asset
converter, and adds ``usd`` paths to the manifest. Already-converted files
are kept.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp(
    {
        "headless": True,
        "extra_args": ["--/renderer/gpuEnumeration/rtxRequired=false"],
    }
)

import omni.kit.app  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("omni.kit.asset_converter")
import omni.kit.asset_converter as converter  # noqa: E402

root = Path(sys.argv[1]).resolve()
manifest = json.loads((root / "manifest.json").read_text())


async def convert(src: Path, dst: Path) -> bool:
    ctx = converter.AssetConverterContext()
    ctx.ignore_materials = False
    ctx.ignore_animations = True
    ctx.ignore_camera = True
    ctx.ignore_light = True
    ctx.single_mesh = False
    ctx.use_meter_as_world_unit = True
    task = converter.get_instance().create_converter_task(str(src), str(dst), None, ctx)
    ok = await task.wait_until_finished()
    if not ok:
        print(f"CONVERT FAIL {src.name}: {task.get_error_message()}", flush=True)
    return bool(ok)


async def main() -> None:
    (root / "usd").mkdir(exist_ok=True)
    for e in manifest:
        dst = root / "usd" / f"{e['uid']}.usd"
        if dst.exists() or await convert(root / e["glb"], dst):
            e["usd"] = str(dst.relative_to(root))
        status = "ok" if "usd" in e else "FAILED"
        print(f"CONVERT {e['class']:15s} {e['name'][:40]:40s} {status}", flush=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))


task = asyncio.ensure_future(main())
while not task.done():
    omni.kit.app.get_app().update()
task.result()
app.close()
