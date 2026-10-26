"""``workshop.train_detector``: the manifest says ``passed`` only when a gate was really measured.

``gate_failures: []`` used to be written when the per-class extraction failed and when no tier-1
class occurred in the validation set, i.e. exactly when nothing had been measured (audit review
TOOL-03): an empty list of failures read as a pass. The ``gate`` block is now a tri-state.
"""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from workshop import train_detector as td
from workshop.classes import CLASSES, NAMES, RECALL_GATE
from workshop.train_detector import evaluate_gate

TIER = {c.name: c.tier for c in CLASSES}


def measured(**recall: float) -> dict:
    """A ``per_class`` table as ``per_class_metrics`` builds it, for the classes given."""
    return {n: {"ap50": 0.5, "recall": r, "precision": 0.5} for n, r in recall.items()}


def everything_passing(**over: float) -> dict:
    table = {c.name: RECALL_GATE[c.tier] + 0.001 for c in CLASSES}
    table.update(over)
    return measured(**table)


def test_a_clean_real_validation_run_passes():
    gate = evaluate_gate(everything_passing(), None, smoke_sdg_val=False)

    assert gate == {
        "status": "passed",
        "reasons": [],
        "failures": [],
        "classes_without_samples": [],
    }


def test_a_class_below_its_tier_gate_fails_and_is_named_with_its_numbers():
    gate = evaluate_gate(everything_passing(person=0.90, rock=0.50), None, smoke_sdg_val=False)

    assert gate["status"] == "failed"
    assert gate["failures"] == ["person", "rock"]
    assert gate["reasons"] == [
        "person (tier 1) recall 0.900 < 0.98",
        "rock (tier 2) recall 0.500 < 0.85",
    ]


@pytest.mark.parametrize("name", [c.name for c in CLASSES])
def test_each_class_is_held_to_its_own_tier_gate(name):
    need = RECALL_GATE[TIER[name]]

    at_gate = evaluate_gate(everything_passing(**{name: need}), None, smoke_sdg_val=False)
    below = evaluate_gate(everything_passing(**{name: need - 0.001}), None, smoke_sdg_val=False)

    assert at_gate["status"] == "passed"  # the gate is inclusive
    assert below["status"] == "failed" and below["failures"] == [name]


def test_a_failed_extraction_is_not_evaluated_not_passed():
    # Review repro: broken metrics -> per_class_val {} and gate_failures [] (read as "passed").
    gate = evaluate_gate(
        {}, "AttributeError: 'Metric' object has no attribute 'r'", smoke_sdg_val=False
    )

    assert gate["status"] == "not_evaluated"
    assert gate["failures"] == []
    assert gate["classes_without_samples"] == NAMES
    assert any(r.startswith("per-class extraction failed: AttributeError") for r in gate["reasons"])


def test_tier1_classes_nobody_measured_are_not_evaluated_even_when_the_rest_is_perfect():
    table = everything_passing()
    for name in ("person", "pet"):
        del table[name]

    gate = evaluate_gate(table, None, smoke_sdg_val=False)

    assert gate["status"] == "not_evaluated"
    assert gate["classes_without_samples"] == ["person", "pet"]
    assert gate["reasons"] == ["tier-1 classes without validation samples: person, pet"]


def test_review_repro_only_rock_in_validation_is_a_failure_and_says_tier1_is_missing():
    # Old manifest: gate_failures ['rock'], with person/pet/wildlife printed "no samples" and
    # never counted. The measured failure stays a failure; the missing tier-1 classes are reported.
    gate = evaluate_gate(measured(rock=0.0), None, smoke_sdg_val=False)

    assert gate["status"] == "failed" and gate["failures"] == ["rock"]
    assert "tier-1 classes without validation samples: person, pet, wildlife" in gate["reasons"]


def test_only_rock_in_validation_and_found_is_not_evaluated_not_passed():
    gate = evaluate_gate(measured(rock=0.99), None, smoke_sdg_val=False)

    assert gate["status"] == "not_evaluated" and gate["failures"] == []


def test_a_synthetic_validation_set_cannot_pass_a_gate():
    gate = evaluate_gate(everything_passing(), None, smoke_sdg_val=True)

    assert gate["status"] == "not_evaluated"
    assert gate["reasons"] == [
        "the validation set is synthetic (--smoke-sdg-val): a recipe check, not a recall gate"
    ]


def test_a_synthetic_validation_set_does_not_hide_a_measured_failure():
    gate = evaluate_gate(everything_passing(wildlife=0.1), None, smoke_sdg_val=True)

    assert gate["status"] == "failed" and gate["failures"] == ["wildlife"]
    assert len(gate["reasons"]) == 2


def test_lower_tier_classes_without_samples_are_listed_and_do_not_block_a_pass():
    table = everything_passing()
    del table["dropoff"], table["water"]

    gate = evaluate_gate(table, None, smoke_sdg_val=False)

    assert gate["status"] == "passed"
    assert gate["classes_without_samples"] == ["water", "dropoff"]


@pytest.mark.parametrize("recall", [None, math.nan, math.inf, -math.inf, "0.9"])
def test_a_recall_that_is_not_a_number_counts_as_not_measured(recall):
    table = everything_passing()
    table["person"]["recall"] = recall

    gate = evaluate_gate(table, None, smoke_sdg_val=False)

    assert gate["status"] == "not_evaluated" and gate["classes_without_samples"] == ["person"]


def test_a_class_entry_without_a_recall_key_counts_as_not_measured():
    table = everything_passing()
    table["pet"] = {"ap50": 0.9}

    assert evaluate_gate(table, None, smoke_sdg_val=False)["classes_without_samples"] == ["pet"]


# ---- the whole script, with a fake ultralytics --------------------------------------------------


class FakeBox:
    """``metrics.box`` of a validation run; ``recalls`` maps class id -> recall (compact arrays)."""

    def __init__(self, recalls: dict[int, float] | None):
        self.recalls = recalls

    @property
    def ap_class_index(self):
        if self.recalls is None:
            raise AttributeError("'Metric' object has no attribute 'ap_class_index'")
        return np.array(sorted(self.recalls))

    @property
    def ap50(self):
        return [0.5 for _ in self.recalls]

    @property
    def r(self):
        return [self.recalls[k] for k in sorted(self.recalls)]

    @property
    def p(self):
        return [0.5 for _ in self.recalls]


def run_main(tmp_path, monkeypatch, recalls, *extra, real=True):
    """``train_detector.main`` on a tiny dataset with a fake YOLO; returns the run manifest."""
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    (data / "labels").mkdir()
    for i in range(40):
        (data / "images" / f"f{i:03d}.jpg").write_bytes(b"x")
        (data / "labels" / f"f{i:03d}.txt").write_text("7 0.5 0.5 0.1 0.1\n")

    class FakeYOLO:
        def __init__(self, weights):
            self.weights = weights

        def train(self, **kw):
            weights = Path(kw["project"]) / kw["name"] / "weights"
            weights.mkdir(parents=True, exist_ok=True)
            (weights / "best.pt").write_bytes(b"w")

        def val(self, **kw):
            return types.SimpleNamespace(box=FakeBox(recalls), names=dict(enumerate(NAMES)))

    fake = types.ModuleType("ultralytics")
    fake.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    monkeypatch.setattr(
        td, "export_hailo_onnx", lambda best, device=None: Path(best).with_suffix(".onnx")
    )
    out_root = tmp_path / "models"
    flag = ["--real" if real else "--sdg", str(data)]
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", *flag, "--tag", "t", "--epochs", "1", "--out-root", str(out_root), *extra],
    )
    td.main()
    return json.loads((out_root / "yolo26s_t" / "run_manifest.json").read_text())


def test_the_manifest_of_a_failed_extraction_is_not_evaluated(tmp_path, monkeypatch, capsys):
    manifest = run_main(tmp_path, monkeypatch, None)

    assert manifest["schema"] == "train-v2"
    assert manifest["per_class_val"] == {} and "AttributeError" in manifest["per_class_val_error"]
    assert manifest["gate"]["status"] == "not_evaluated"
    assert "gate_failures" not in manifest  # the field that read "[]" as "passed" is gone
    assert "gate: not_evaluated" in capsys.readouterr().out


def test_the_manifest_of_a_run_that_only_validated_rock_is_not_a_pass(tmp_path, monkeypatch):
    manifest = run_main(tmp_path, monkeypatch, {NAMES.index("rock"): 0.99})

    assert manifest["gate"]["status"] == "not_evaluated"
    assert manifest["gate"]["classes_without_samples"][:3] == ["person", "pet", "wildlife"]


def test_the_manifest_records_a_failed_gate_and_the_printed_table_agrees(
    tmp_path, monkeypatch, capsys
):
    manifest = run_main(
        tmp_path, monkeypatch, {NAMES.index("person"): 0.5, NAMES.index("rock"): 0.9}
    )

    assert manifest["gate"]["status"] == "failed" and manifest["gate"]["failures"] == ["person"]
    out = capsys.readouterr().out
    assert "person" in out and "BELOW GATE" in out and "no samples" in out and "PASS" in out


def test_a_complete_real_validation_run_is_a_pass(tmp_path, monkeypatch):
    recalls = {i: RECALL_GATE[c.tier] + 0.001 for i, c in enumerate(CLASSES)}

    manifest = run_main(tmp_path, monkeypatch, recalls)

    assert manifest["gate"]["status"] == "passed" and manifest["smoke_sdg_val"] is False


def test_the_smoke_split_run_is_never_a_pass(tmp_path, monkeypatch):
    recalls = {i: RECALL_GATE[c.tier] + 0.001 for i, c in enumerate(CLASSES)}

    manifest = run_main(tmp_path, monkeypatch, recalls, "--smoke-sdg-val", real=False)

    assert manifest["smoke_sdg_val"] is True
    assert manifest["gate"]["status"] == "not_evaluated"
    assert any("--smoke-sdg-val" in r for r in manifest["gate"]["reasons"])
