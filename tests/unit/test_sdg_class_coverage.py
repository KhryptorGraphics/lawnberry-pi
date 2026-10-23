"""Every class must be reachable in SDG: by an asset or by a procedural spawner.

The v1 synthetic set silently contained zero instances of ten classes because their
entries in the asset map were empty lists and nothing else supplied geometry. This
test is the guard: adding a class to `workshop.classes` without an asset or a
generator now fails here instead of showing up as a class that simply never renders.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from workshop.classes import NAMES

WORKSHOP = Path(__file__).resolve().parents[2] / "workshop"


def _procedural_keys() -> set[str]:
    tree = ast.parse((WORKSHOP / "isaac_sdg.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "PROCEDURAL" for t in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    raise AssertionError("PROCEDURAL dict not found in workshop/isaac_sdg.py")


def _asset_keys() -> set[str]:
    keys: set[str] = set()
    for name in ("sdg_assets.json", "sdg_assets.generated.json"):
        path = WORKSHOP / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        keys |= {k for k, v in data.items() if v}
    return keys


def test_every_class_has_an_asset_or_a_generator():
    covered = _asset_keys() | _procedural_keys()
    missing = sorted(set(NAMES) - covered)

    assert not missing, f"classes with no SDG source (will never render): {missing}"


def test_generators_only_reference_real_classes():
    unknown = sorted(_procedural_keys() - set(NAMES))

    assert not unknown, f"PROCEDURAL keys that are not classes: {unknown}"


def test_generated_asset_config_covers_the_objaverse_classes():
    data = json.loads((WORKSHOP / "sdg_assets.generated.json").read_text(encoding="utf-8"))

    # The generated config is what produced data/sdg/v1; if it is lost, the three
    # classes that only exist there (pet, wildlife, root) silently disappear again.
    for name in ("pet", "wildlife", "root"):
        assert data.get(name), f"generated asset config lost its {name} assets"
