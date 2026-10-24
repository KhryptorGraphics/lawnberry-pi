#!/usr/bin/env python3
"""Evaluate a YOLO26 detector: HEF on the Hailo, ONNX on CPU, or DFC emulation of the HAR.

Run with the repo on PYTHONPATH:

    # one image, optionally checked against reference boxes (exit 1 on mismatch)
    eval_detector.py --backend hef --model x.hef --predict bus.jpg \
        [--expect ref.json --min-iou 0.8]

    # per-class recall/precision over a YOLO-format split
    eval_detector.py --backend hef|onnx|har --model P [--spec J] --images DIR --labels DIR \
        [--conf 0.25] [--iou 0.5] [--gate] \
        [--baseline REPORT.json --max-recall-drop 0.10 --min-gt 20] --out REPORT.json

``hef`` needs /dev/hailo0 free (``pi_detector_ctl.py release`` first); its spec
defaults to the HEF sidecar. ``onnx`` and ``har`` need ``--spec`` for class names. All use
the same letterbox, so their boxes are directly comparable. ``har`` runs the Dataflow Compiler's
fixed-point emulation of the quantized model (``--har-context quantized|bit-exact``) in the DFC
env (``hailo_sdk_client``; pick the GPU with ``CUDA_VISIBLE_DEVICES``): what the HEF computes,
but NOT a device measurement (the report says ``"kind": "dfc-emulation"``). Exit codes: 0 ok,
1 gate/baseline/expect failure, 2 setup error.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from backend.src.drivers.ai.yolo26 import (
    DetectorSpec,
    Letterbox,
    decode_yolo26,
    letterbox,
    load_detector_spec,
    load_spec_file,
)
from workshop.detector_eval import iou, load_yolo_labels, match, merge, summarize

IMG_EXTS = (".jpg", ".jpeg", ".png")


class SetupError(Exception):
    """Backend/model problem: exit 2."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_rgb(path: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SetupError(f"cannot read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class HefBackend:
    def __init__(self, model: Path, spec: DetectorSpec):
        self.model = model
        self.spec = spec
        self.driver = None

    async def __aenter__(self):
        from backend.src.drivers.ai.hailo_driver import HailoDriver

        self.driver = HailoDriver({"hef_path": str(self.model), "input_format": "uint8"})
        await self.driver.initialize()
        if not self.driver.hardware_available:
            raise SetupError(f"Hailo unavailable: {self.driver.hardware_error}")
        await self.driver.start()
        return self

    async def __aexit__(self, *exc):
        if self.driver is not None:
            await self.driver.stop()

    async def predict(self, rgb: np.ndarray) -> tuple[list[dict], float]:
        tensor, lb = letterbox(rgb, self.spec.input_size)
        t0 = time.perf_counter()
        result = await self.driver.infer(tensor)
        ms = (time.perf_counter() - t0) * 1000
        return decode_yolo26(result.outputs, self.spec, lb), ms


class HarBackend:
    """DFC emulation of the quantized (or compiled) HAR; not the device."""

    CONTEXTS = {"quantized": "SDK_QUANTIZED", "bit-exact": "SDK_BIT_EXACT"}

    def __init__(self, model: Path, spec: DetectorSpec, context: str = "quantized"):
        self.model = model
        self.spec = spec
        self.context = context
        self.runner = None
        self._cm = None
        self._ctx = None

    async def __aenter__(self):
        try:
            from hailo_sdk_client import ClientRunner, InferenceContext
        except ImportError as exc:
            raise SetupError(f"--backend har needs the DFC env (hailo_sdk_client): {exc}") from exc
        self.runner = ClientRunner(har=str(self.model))
        self._cm = self.runner.infer_context(getattr(InferenceContext, self.CONTEXTS[self.context]))
        self._ctx = self._cm.__enter__()
        return self

    async def __aexit__(self, *exc):
        if self._cm is not None:
            self._cm.__exit__(*exc)

    async def predict(self, rgb: np.ndarray) -> tuple[list[dict], float]:
        tensor, lb = letterbox(rgb, self.spec.input_size)
        t0 = time.perf_counter()
        raw = self.runner.infer(self._ctx, tensor.astype(np.float32))
        ms = (time.perf_counter() - t0) * 1000
        items = raw.items() if isinstance(raw, dict) else enumerate(raw)
        outputs = {f"out{k}": np.asarray(v) for k, v in items}
        return decode_yolo26(outputs, self.spec, lb), ms


class OnnxBackend:
    def __init__(self, model: Path, spec: DetectorSpec):
        import onnxruntime as ort

        self.spec = spec
        self.session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def predict(self, rgb: np.ndarray) -> tuple[list[dict], float]:
        tensor, lb = letterbox(rgb, self.spec.input_size)
        x = np.ascontiguousarray(tensor.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0
        t0 = time.perf_counter()
        out = self.session.run(None, {self.input_name: x})[0]
        ms = (time.perf_counter() - t0) * 1000
        if out.ndim != 3 or out.shape[0] != 1 or out.shape[2] != 6:
            raise SetupError(f"ONNX output shape {out.shape} is not end2end (1, N, 6)")
        return self._decode(out[0], lb), ms

    def _decode(self, rows: np.ndarray, lb: Letterbox) -> list[dict]:
        dets = []
        for x1, y1, x2, y2, score, cls in rows.tolist():
            if score < self.spec.conf_threshold:
                continue
            box = [
                min(max((x1 - lb.pad_x) / lb.scale, 0.0), lb.src_w),
                min(max((y1 - lb.pad_y) / lb.scale, 0.0), lb.src_h),
                min(max((x2 - lb.pad_x) / lb.scale, 0.0), lb.src_w),
                min(max((y2 - lb.pad_y) / lb.scale, 0.0), lb.src_h),
            ]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            c = int(cls)
            dets.append(
                {
                    "class": self.spec.class_names[c],
                    "class_id": c,
                    "confidence": float(score),
                    "bbox": box,
                }
            )
        dets.sort(key=lambda d: d["confidence"], reverse=True)
        return dets


def _load_spec(a) -> DetectorSpec:
    if a.backend in ("onnx", "har") and not a.spec:
        raise SetupError(f"--backend {a.backend} needs --spec (class names come from it)")
    try:
        spec = load_spec_file(a.spec) if a.spec else load_detector_spec(a.model)
    except (FileNotFoundError, ValueError) as exc:
        raise SetupError(str(exc)) from exc
    if a.conf is not None:
        spec = dataclasses.replace(spec, conf_threshold=a.conf)
    return spec


def _backend(a, spec: DetectorSpec):
    model = Path(a.model)
    if not model.exists():
        raise SetupError(f"model not found: {model}")
    if a.backend == "hef":
        return HefBackend(model, spec)
    if a.backend == "har":
        return HarBackend(model, spec, a.har_context)
    return OnnxBackend(model, spec)


async def run_predict(a, spec: DetectorSpec) -> int:
    rgb = _read_rgb(Path(a.predict))
    async with _backend(a, spec) as be:
        dets, ms = await be.predict(rgb)
    print(json.dumps(dets, indent=2))
    print(f"[eval] {len(dets)} detections in {ms:.1f} ms", file=sys.stderr)
    if not a.expect:
        return 0
    ref = json.loads(Path(a.expect).read_text(encoding="utf-8"))
    failures = []
    for r in ref:
        best = max(
            (iou(r["bbox"], d["bbox"]) for d in dets if d["class"] == r["class"]), default=0.0
        )
        status = "ok" if best >= a.min_iou else "MISSING"
        print(
            f"[eval] ref {r['class']:<12} conf {r['confidence']:.2f} best IoU {best:.3f} {status}",
            file=sys.stderr,
        )
        if best < a.min_iou:
            failures.append(r)
    matched = len(ref) - len(failures)
    print(f"[eval] expect: {matched}/{len(ref)} reference boxes matched", file=sys.stderr)
    return 1 if failures else 0


def _gate_failures(per_class: dict) -> list[dict]:
    from workshop.classes import CLASSES, RECALL_GATE

    tiers = {c.name: c.tier for c in CLASSES}
    failures = []
    for name, m in per_class.items():
        if name not in tiers or m["gt"] == 0:
            continue
        need = RECALL_GATE[tiers[name]]
        if m["recall"] < need:
            failures.append({"class": name, "recall": m["recall"], "gate": need})
    return failures


def _baseline_failures(per_class: dict, baseline: dict, max_drop: float, min_gt: int) -> list:
    failures = []
    for name, m in per_class.items():
        b = baseline.get("per_class", {}).get(name)
        if m["gt"] < min_gt or b is None or b.get("recall") is None:
            continue
        drop = b["recall"] - m["recall"]
        if drop > max_drop:
            failures.append(
                {"class": name, "recall": m["recall"], "baseline": b["recall"], "drop": drop}
            )
    return failures


async def run_eval(a, spec: DetectorSpec) -> int:
    images = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SetupError(f"no images in {a.images}")
    labels = Path(a.labels)
    total: dict = {}
    times = []
    async with _backend(a, spec) as be:
        for img in images:
            rgb = _read_rgb(img)
            h, w = rgb.shape[:2]
            dets, ms = await be.predict(rgb)
            times.append(ms)
            merge(total, match(dets, load_yolo_labels(labels / f"{img.stem}.txt", w, h), a.iou))
    per_class = summarize(total, spec.class_names)
    gate = _gate_failures(per_class) if a.gate else []
    base = []
    if a.baseline:
        baseline = json.loads(Path(a.baseline).read_text(encoding="utf-8"))
        base = _baseline_failures(per_class, baseline, a.max_recall_drop, a.min_gt)
    report = {
        "schema": "detector-eval/1",
        "backend": a.backend,
        "kind": {"hef": "device", "har": "dfc-emulation", "onnx": "float-cpu"}[a.backend],
        "model": str(a.model),
        "model_sha256": _sha256(Path(a.model)),
        "images": len(images),
        "conf": spec.conf_threshold,
        "iou": a.iou,
        "per_class": per_class,
        "gate": {"applied": bool(a.gate), "failures": gate},
        "baseline": {"failures": base},
        "mean_infer_ms": round(sum(times) / len(times), 2),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def fmt(v):
        return "-" if v is None else f"{v:.3f}"

    print(f"{'class':18} {'gt':>5} {'tp':>5} {'fp':>5} {'fn':>5} {'recall':>7} {'prec':>7}")
    for name, m in per_class.items():
        print(
            f"{name:18} {m['gt']:>5} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5} "
            f"{fmt(m['recall']):>7} {fmt(m['precision']):>7}"
        )
    print(f"[eval] {len(images)} images, mean infer {report['mean_infer_ms']} ms -> {out}")
    for f in gate:
        print(f"[eval] GATE FAIL {f['class']}: recall {f['recall']:.3f} < {f['gate']}")
    for f in base:
        print(
            f"[eval] BASELINE FAIL {f['class']}: recall {f['recall']:.3f} "
            f"vs {f['baseline']:.3f} (drop {f['drop']:.3f})"
        )
    return 1 if gate or base else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=("hef", "onnx", "har"), required=True)
    ap.add_argument("--har-context", choices=("quantized", "bit-exact"), default="quantized")
    ap.add_argument("--model", required=True)
    ap.add_argument("--spec")
    ap.add_argument("--conf", type=float, help="override the spec confidence threshold")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predict", metavar="IMG")
    mode.add_argument("--images", metavar="DIR")
    ap.add_argument("--expect", metavar="REF.json")
    ap.add_argument("--min-iou", type=float, default=0.8)
    ap.add_argument("--labels", metavar="DIR")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--baseline", metavar="REPORT.json")
    ap.add_argument("--max-recall-drop", type=float, default=0.10)
    ap.add_argument("--min-gt", type=int, default=20)
    ap.add_argument("--out", metavar="REPORT.json")
    a = ap.parse_args(argv)
    if a.images and not (a.labels and a.out):
        ap.error("--images needs --labels and --out")
    try:
        spec = _load_spec(a)
        return asyncio.run(run_predict(a, spec) if a.predict else run_eval(a, spec))
    except SetupError as exc:
        print(f"[eval] error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
