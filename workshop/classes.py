"""Detector class set (W2/W5), in priority order.

``id`` order is the YOLO class index and must never be reordered once a
dataset version exists; append only. ``prompt`` is the open-vocabulary text
given to MM Grounding DINO for auto-labeling.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DetClass:
    name: str
    tier: int  # 1 hard-stop, 2 damage, 3 structure, 4 terrain, 5 unknown
    prompt: str


CLASSES: tuple[DetClass, ...] = (
    DetClass("person", 1, "person"),
    DetClass("pet", 1, "dog . cat"),
    DetClass("wildlife", 1, "deer . rabbit . squirrel . bird . turtle . snake"),
    DetClass("hose", 2, "garden hose"),
    DetClass("sprinkler_head", 2, "sprinkler head"),
    DetClass("toy", 2, "toy . ball"),
    DetClass("tool", 2, "rake . shovel . hand tool"),
    DetClass("rock", 2, "rock"),
    DetClass("root", 2, "exposed tree root"),
    DetClass("furniture", 2, "chair . table . bench"),
    DetClass("planter", 2, "planter pot"),
    DetClass("house", 3, "house"),
    DetClass("fence", 3, "fence"),
    DetClass("shed", 3, "shed"),
    DetClass("wall", 3, "wall"),
    DetClass("bed", 3, "flower bed"),
    DetClass("edging", 3, "lawn edging"),
    DetClass("tree", 3, "tree trunk"),
    DetClass("path", 4, "paved path"),
    DetClass("driveway", 4, "driveway"),
    DetClass("mulch", 4, "mulch"),
    DetClass("water", 4, "water . pond"),
    DetClass("dropoff", 4, "step . drop-off"),
    DetClass("unknown_obstacle", 5, "obstacle"),
)

NAMES = [c.name for c in CLASSES]
HARD_STOP = [c.name for c in CLASSES if c.tier == 1]
# Compiled-model recall gates on the real validation set (W5).
RECALL_GATE = {1: 0.98, 2: 0.85, 3: 0.80, 4: 0.70, 5: 0.80}
