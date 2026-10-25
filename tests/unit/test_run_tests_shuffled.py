"""``scripts/run_tests_shuffled.py``: a seeded random order, the same every time.

The runner is how order-dependent tests are found (audit review of PR #20); it is only useful if a
seed names one order, the modes do what they say, and the pytest wiring really runs the tests in
that order.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_tests_shuffled import MODES, main, shuffled

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "run_tests_shuffled.py"


def items(spec: dict[str, int]) -> list[SimpleNamespace]:
    """Fake collected tests: ``{"a.py": 3}`` -> ``a.py::t0 .. a.py::t2``."""
    return [SimpleNamespace(nodeid=f"{f}::t{i}") for f, n in spec.items() for i in range(n)]


def ids(seq) -> list[str]:
    return [i.nodeid for i in seq]


SPEC = {"a.py": 4, "b.py": 4, "c.py": 4, "d.py": 4}


@pytest.mark.parametrize("mode", MODES)
def test_the_same_seed_gives_the_same_order_and_every_test_exactly_once(mode):
    first = ids(shuffled(items(SPEC), mode, seed=5))
    again = ids(shuffled(items(SPEC), mode, seed=5))

    assert first == again
    assert sorted(first) == sorted(ids(items(SPEC)))  # a permutation: nothing lost or doubled


def test_different_seeds_give_different_orders():
    orders = {tuple(ids(shuffled(items(SPEC), "item", seed=s))) for s in range(8)}

    assert len(orders) > 1


def test_module_mode_keeps_each_file_together_and_shuffles_inside_it():
    got = ids(shuffled(items(SPEC), "module", seed=3))

    files = [i.split("::")[0] for i in got]
    blocks = [f for n, f in enumerate(files) if n == 0 or files[n - 1] != f]
    assert sorted(blocks) == sorted(SPEC)  # one contiguous block per file
    orders = {
        tuple(f.split("::")[0] for f in ids(shuffled(items(SPEC), "module", s))[::4])
        for s in range(8)
    }
    assert len(orders) > 1  # the files themselves are reordered, not just the tests in them
    inside = [i for i in got if i.startswith(blocks[0])]
    assert sorted(inside) == sorted(i for i in ids(items(SPEC)) if i.startswith(blocks[0]))


def test_module_mode_changes_the_order_inside_a_file_for_some_seed():
    natural = ids(items({"a.py": 8}))

    assert any(ids(shuffled(items({"a.py": 8}), "module", s)) != natural for s in range(6))


def test_item_mode_interleaves_files():
    got = ids(shuffled(items(SPEC), "item", seed=1))
    files = [i.split("::")[0] for i in got]

    assert any(files[n] != files[n + 1] for n in range(len(files) - 1))
    assert len({f for f in files[:4]}) > 1  # a block of four would be one file


def test_reverse_mode_is_the_default_order_backwards_whatever_the_seed():
    natural = ids(items(SPEC))

    assert ids(shuffled(items(SPEC), "reverse", seed=1)) == natural[::-1]
    assert ids(shuffled(items(SPEC), "reverse", seed=99)) == natural[::-1]


def test_the_input_is_not_modified_and_an_unknown_mode_is_refused():
    original = items(SPEC)
    before = ids(original)

    shuffled(original, "item", seed=2)

    assert ids(original) == before
    with pytest.raises(ValueError, match="unknown mode 'sideways'"):
        shuffled(original, "sideways", seed=2)


def test_an_empty_collection_stays_empty():
    assert shuffled([], "module", seed=1) == []


PROBE = textwrap.dedent(
    """
    import pytest

    @pytest.mark.parametrize("n", range(6))
    def test_param(n):
        print(f"RAN param {n}")

    def test_a():
        print("RAN a")

    def test_b():
        print("RAN b")
    """
)


def run_probe(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    (tmp_path / "test_probe.py").write_text(PROBE)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args]
        + [
            "--",
            str(tmp_path / "test_probe.py"),
            "-s",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )


def order_of(done: subprocess.CompletedProcess) -> list[str]:
    return re.findall(r"RAN (\S+(?: \d)?)", done.stdout)


def test_pytest_really_runs_the_tests_in_the_order_the_seed_names(tmp_path):
    one = run_probe(tmp_path, "--seed", "7", "--mode", "item")
    two = run_probe(tmp_path, "--seed", "7", "--mode", "item")
    other = [order_of(run_probe(tmp_path, "--seed", str(s), "--mode", "item")) for s in (1, 2, 3)]

    assert one.returncode == 0 and "8 passed" in one.stdout
    assert order_of(one) == order_of(two) and len(order_of(one)) == 8
    assert re.search(r"\[shuffle\] mode=item seed=7 tests=8 order_sha256=[0-9a-f]{16}", one.stdout)
    assert len({tuple(o) for o in other + [order_of(one)]}) > 1  # the seed matters to pytest too


def test_the_printed_digest_identifies_the_order(tmp_path):
    digest = lambda d: re.search(r"order_sha256=([0-9a-f]+)", d.stdout).group(1)  # noqa: E731

    a = digest(run_probe(tmp_path, "--seed", "7", "--mode", "item"))
    b = digest(run_probe(tmp_path, "--seed", "7", "--mode", "item"))
    c = digest(run_probe(tmp_path, "--seed", "8", "--mode", "item"))

    assert a == b and a != c


def test_reverse_mode_through_pytest_runs_the_last_test_first(tmp_path):
    done = run_probe(tmp_path, "--seed", "0", "--mode", "reverse")

    assert order_of(done) == ["b", "a", *[f"param {n}" for n in (5, 4, 3, 2, 1, 0)]]


def test_a_failing_test_gives_a_failing_exit_status(tmp_path):
    (tmp_path / "test_bad.py").write_text("def test_bad():\n    assert False\n")

    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--seed", "1", "--", str(tmp_path / "test_bad.py"), "-q"]
        + ["-o", "addopts=", "-p", "no:cacheprovider"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert done.returncode == 1 and "1 failed" in done.stdout


def test_the_seed_is_required(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a run that wrongly goes ahead finds no `tests` here, fast
    with pytest.raises(SystemExit) as exc:
        main(["--mode", "item"])

    assert exc.value.code == 2


def test_the_default_mode_is_module_and_the_default_target_is_tests(tmp_path):
    (tmp_path / "tests" / "sub").mkdir(parents=True)
    (tmp_path / "tests" / "test_one.py").write_text("def test_one():\n    pass\n")
    (tmp_path / "tests" / "sub" / "test_two.py").write_text("def test_two():\n    pass\n")
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "test_three.py").write_text("def test_three():\n    pass\n")

    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--seed", "3"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTEST_ADDOPTS": "-o addopts= -p no:cacheprovider"},
    )

    assert done.returncode == 0, done.stdout + done.stderr
    assert "[shuffle] mode=module seed=3 tests=2" in done.stdout, (
        "module mode, both tests of tests/"
    )
    assert "2 passed" in done.stdout


def test_the_shuffle_is_the_last_word_on_the_order_after_every_other_plugin(tmp_path):
    """A plugin registered before the shuffler that re-sorts the items must not undo the shuffle."""
    (tmp_path / "test_many.py").write_text(
        "import pytest\n\n@pytest.mark.parametrize('n', range(12))\n"
        "def test_n(n):\n    print(f'RAN {n:02d}')\n"
    )
    driver = tmp_path / "driver.py"
    driver.write_text(
        textwrap.dedent(
            f"""
            import sys
            import pytest
            sys.path.insert(0, {str(REPO)!r})
            from scripts.run_tests_shuffled import ShufflePlugin

            class SortsByName:
                def pytest_collection_modifyitems(self, items):
                    items.sort(key=lambda i: i.nodeid)

            sys.exit(pytest.main(
                ["test_many.py", "-s", "-q", "-o", "addopts=", "-p", "no:cacheprovider"],
                plugins=[SortsByName(), ShufflePlugin("item", 7)],
            ))
            """
        )
    )

    done = subprocess.run(
        [sys.executable, str(driver)], cwd=tmp_path, capture_output=True, text=True, timeout=120
    )

    order = re.findall(r"RAN (\d\d)", done.stdout)
    assert done.returncode == 0 and len(order) == 12, done.stdout + done.stderr
    assert order != sorted(order), "the sort ran after the shuffle and put the tests back in order"
    fake = [SimpleNamespace(nodeid=f"test_many.py::test_n[{n}]") for n in range(12)]
    expected = shuffled(sorted(fake, key=lambda i: i.nodeid), "item", 7)
    assert order == [re.search(r"\[(\d+)\]", i.nodeid).group(1).zfill(2) for i in expected]
