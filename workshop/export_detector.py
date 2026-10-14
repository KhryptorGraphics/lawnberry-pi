"""W5 detector export for Hailo: end2end ONNX, HEF sidecar spec, CPU reference boxes.

Run from the repo root with the workshop env ($LB/env/bin/python):

    python -m workshop.export_detector onnx --tag smoke [--device cpu]
    python -m workshop.export_detector check --onnx best.onnx
    python -m workshop.export_detector spec --names-from best.onnx --hef x.hef --tag T \
        [--calib DIR] --out x.json
    python -m workshop.export_detector reference --weights yolo26s.pt --image bus.jpg \
        [--conf 0.5] --out ref.json

The Hailo model zoo v2.19.1 ``yolo26`` config parses the ONNX up to the six
``one2one`` head convolutions, which exist only in an ``end2end=True`` export.
The sidecar spec (schema ``lawnberry-detector/1``) is what
``backend/src/drivers/ai/yolo26.py`` loads next to the HEF on the Pi.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

LB = Path(os.environ.get("LB", str(Path.home() / "nvme2" / "lawnberrypiserver")))
SPEC_SCHEMA = "lawnberry-detector/1"
DFC_VERSION = "3.34.0"
MODEL_ZOO_VERSION = "2.19.1"

# Zoo v2.19.1 cfg/base/yolo26.yaml parser end nodes: box heads (stride 8/16/32),
# then class heads (stride 8/16/32).
YOLO26_END_NODES: tuple[str, ...] = tuple(
    f"/model.23/one2one_{head}.{i}/one2one_{head}.{i}.2/Conv"
    for head in ("cv2", "cv3")
    for i in range(3)
)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def missing_end_nodes(onnx_path: Path) -> list[str]:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    present = {n.name for n in model.graph.node}
    return [n for n in YOLO26_END_NODES if n not in present]


def check_end_nodes(onnx_path: Path) -> None:
    missing = missing_end_nodes(onnx_path)
    if missing:
        raise SystemExit(f"missing yolo26 end nodes: {', '.join(missing)}")


def export_hailo_onnx(best_pt: Path, device: str) -> Path:
    """Export ``best_pt`` as an end2end ONNX the zoo yolo26 parser accepts."""
    import onnx
    from ultralytics import YOLO

    onnx_path = Path(
        YOLO(str(best_pt)).export(
            format="onnx", opset=17, simplify=True, end2end=True, device=device
        )
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)), full_check=False)
    check_end_nodes(onnx_path)
    return onnx_path


def class_names_from(path: Path) -> list[str]:
    """Class names in index order from ONNX metadata ``names`` or ``YOLO(pt).names``."""
    if path.suffix == ".onnx":
        import onnx

        props = onnx.load(str(path), load_external_data=False).metadata_props
        meta = {p.key: p.value for p in props}
        if "names" not in meta:
            raise SystemExit(f"{path}: no 'names' in ONNX metadata")
        names = ast.literal_eval(meta["names"])
    else:
        from ultralytics import YOLO

        names = YOLO(str(path)).names
    if isinstance(names, dict):
        keys = sorted(int(k) for k in names)
        if keys != list(range(len(keys))):
            raise SystemExit(f"{path}: class ids are not contiguous: {keys}")
        return [str(names[k]) for k in keys]
    return [str(n) for n in names]


def build_spec(names_from: Path, hef: Path, tag: str, calib: Path | None) -> dict:
    return {
        "schema": SPEC_SCHEMA,
        "arch": "yolo26",
        "input_size": 640,
        "class_names": class_names_from(names_from),
        "conf_threshold": 0.25,
        "max_det": 100,
        "tag": tag,
        "hef_sha256": sha256_of(hef),
        "names_source": str(names_from),
        "names_source_sha256": sha256_of(names_from),
        "dfc_version": DFC_VERSION,
        "model_zoo_version": MODEL_ZOO_VERSION,
        "calibration_dir": str(calib) if calib else None,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def reference_detections(weights: Path, image: Path, conf: float) -> list[dict]:
    from ultralytics import YOLO

    res = YOLO(str(weights)).predict(str(image), conf=conf, device="cpu", verbose=False)[0]
    names = res.names
    dets = []
    for xyxy, c, s in zip(
        res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist(), strict=True
    ):
        dets.append(
            {
                "class": str(names[int(c)]),
                "class_id": int(c),
                "confidence": float(s),
                "bbox": [float(v) for v in xyxy],
            }
        )
    dets.sort(key=lambda d: d["confidence"], reverse=True)
    return dets


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def cmd_onnx(a) -> int:
    run_dir = LB / "data" / "models" / f"yolo26s_{a.tag}"
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"no weights: {best}")
    onnx_path = export_hailo_onnx(best, a.device)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({"onnx": str(onnx_path), "onnx_opset": 17, "onnx_end2end": True})
    _write_json(manifest_path, manifest)
    print(f"end2end ONNX with yolo26 end nodes: {onnx_path}")
    return 0


def cmd_check(a) -> int:
    missing = missing_end_nodes(Path(a.onnx))
    if missing:
        print("missing yolo26 end nodes:", *missing, sep="\n  ", file=sys.stderr)
        return 1
    print(f"ok: all {len(YOLO26_END_NODES)} yolo26 end nodes present in {a.onnx}")
    return 0


def cmd_spec(a) -> int:
    spec = build_spec(Path(a.names_from), Path(a.hef), a.tag, Path(a.calib) if a.calib else None)
    _write_json(Path(a.out), spec)
    print(f"spec ({len(spec['class_names'])} classes, tag {a.tag}): {a.out}")
    return 0


def cmd_reference(a) -> int:
    dets = reference_detections(Path(a.weights), Path(a.image), a.conf)
    _write_json(Path(a.out), dets)
    print(f"{len(dets)} reference detections: {a.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="YOLO26 export for Hailo")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("onnx", help="re-export a run's best.pt as end2end ONNX")
    p.add_argument("--tag", required=True)
    p.add_argument("--device", default="cpu")
    p.set_defaults(fn=cmd_onnx)

    p = sub.add_parser("check", help="verify the six yolo26 end nodes")
    p.add_argument("--onnx", required=True)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("spec", help="write the HEF sidecar spec")
    p.add_argument("--names-from", required=True)
    p.add_argument("--hef", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--calib")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_spec)

    p = sub.add_parser("reference", help="CPU reference detections for one image")
    p.add_argument("--weights", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_reference)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
