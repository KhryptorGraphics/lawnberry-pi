#!/usr/bin/env python3
"""Run pytest in a seeded random order to expose tests that depend on the one before them.

    python scripts/run_tests_shuffled.py --seed 11                  # whole suite, files shuffled
    python scripts/run_tests_shuffled.py --seed 22 --mode item      # every test shuffled
    python scripts/run_tests_shuffled.py --seed 0 --mode reverse -- tests/unit -x

``--mode module`` (default) shuffles the order of the test FILES and, inside each file, the order
of its tests, parametrized cases included. ``item`` shuffles every collected test globally.
``reverse`` runs the default order backwards. The same seed always gives the same order, so a
failure found here is replayed by repeating the command; the first line printed carries a digest
of the whole order to prove it. Everything after ``--`` goes to pytest unchanged (default
``tests``). Exit status is pytest's.

The default order is what CI runs, so a green default run says nothing about tests that only
pass after a particular predecessor (a module-level singleton keeping failed-login counts, a TTL
armed by an earlier E-stop test, ...). ``tests/conftest.py`` resets the known shared state before
every test; this script is how that is checked.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys

import pytest

MODES = ("module", "item", "reverse")


def shuffled(items: list, mode: str, seed: int) -> list:
    """The order ``mode`` and ``seed`` give ``items`` (collected tests), without touching them.

    Pure, so the ordering is testable without running pytest.
    """
    rnd = random.Random(seed)
    out = list(items)
    if mode == "item":
        rnd.shuffle(out)
    elif mode == "reverse":
        out.reverse()
    elif mode == "module":
        groups: dict[str, list] = {}
        for item in out:
            groups.setdefault(item.nodeid.split("::")[0], []).append(item)
        files = list(groups)
        rnd.shuffle(files)
        out = []
        for name in files:
            tests = groups[name]
            rnd.shuffle(tests)
            out.extend(tests)
    else:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
    return out


class ShufflePlugin:
    """Reorders the collected tests last, after pytest's own fixture-based grouping."""

    def __init__(self, mode: str, seed: int):
        self.mode = mode
        self.seed = seed

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, session, config, items):
        items[:] = shuffled(items, self.mode, self.seed)
        digest = hashlib.sha256("\n".join(i.nodeid for i in items).encode()).hexdigest()[:16]
        first = items[0].nodeid if items else None
        print(
            f"[shuffle] mode={self.mode} seed={self.seed} tests={len(items)} "
            f"order_sha256={digest} first={first}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, required=True, help="same seed, same order")
    ap.add_argument("--mode", choices=MODES, default="module")
    ap.add_argument(
        "pytest_args", nargs=argparse.REMAINDER, help="after `--`: arguments for pytest"
    )
    a = ap.parse_args(argv)
    rest = a.pytest_args[1:] if a.pytest_args[:1] == ["--"] else a.pytest_args
    return int(pytest.main(rest or ["tests"], plugins=[ShufflePlugin(a.mode, a.seed)]))


if __name__ == "__main__":
    sys.exit(main())
