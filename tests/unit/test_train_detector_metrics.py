"""``workshop.train_detector.per_class_metrics``: Ultralytics' per-class arrays are compact.

Ultralytics' ``DetMetrics.box`` keeps ``ap50`` / ``r`` / ``p`` only for the classes that occur in
the validation set, in ``ap_class_index`` order. The training script used to index them by class
id and to read a ``recall`` attribute that does not exist, so every class reported ``recall=None``
("no samples") and, once a class was missing from validation, every later class got the wrong name.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from workshop.classes import NAMES
from workshop.train_detector import per_class_metrics


def _box(index: list[int], ap50: list[float], r: list[float], p: list[float]) -> SimpleNamespace:
    return SimpleNamespace(
        ap_class_index=np.array(index),
        ap50=np.array(ap50),
        r=np.array(r),
        p=np.array(p),
    )


def test_absent_classes_do_not_shift_names():
    # val contains person (0), wildlife (2) and toy (5); pet (1), hose (3) and sprinkler (4) absent.
    box = _box([0, 2, 5], [0.9, 0.5, 0.2], [0.99, 0.6, 0.3], [0.95, 0.7, 0.4])

    got = per_class_metrics(box, dict(enumerate(NAMES)))

    assert set(got) == {"person", "wildlife", "toy"}
    assert got["person"] == {"ap50": pytest.approx(0.9), "recall": 0.99, "precision": 0.95}
    assert got["wildlife"]["recall"] == pytest.approx(0.6)
    assert got["toy"]["ap50"] == pytest.approx(0.2)


def test_recall_is_reported_not_none():
    box = _box([3], [0.4], [0.75], [0.5])

    assert per_class_metrics(box, dict(enumerate(NAMES)))["hose"]["recall"] == pytest.approx(0.75)


def test_names_may_be_a_list_and_unknown_ids_fall_back():
    box = _box([1, 40], [0.1, 0.2], [0.3, 0.4], [0.5, 0.6])

    got = per_class_metrics(box, list(NAMES))

    assert list(got) == ["pet", "40"]


def test_empty_validation_set_gives_no_classes():
    assert per_class_metrics(_box([], [], [], []), dict(enumerate(NAMES))) == {}
