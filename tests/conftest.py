from __future__ import annotations

from pathlib import Path

import pytest


BASELINE_OPTION = "--upstream-baseline"
MANIFEST = Path(__file__).with_name("upstream_baseline.txt")


def _manifest() -> set[str]:
    return {
        line.strip()
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        BASELINE_OPTION,
        action="store_true",
        help="run only the upstream CI baseline manifest",
    )


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    if not config.getoption(BASELINE_OPTION):
        return False
    if collection_path.name == "conftest.py" or collection_path.is_dir():
        return False
    relative = collection_path.relative_to(Path(__file__).parent.parent).as_posix()
    return relative not in {node.split("::", 1)[0] for node in _manifest()}


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not config.getoption(BASELINE_OPTION):
        return
    manifest = _manifest()
    selected = [item for item in items if item.nodeid in manifest]
    deselected = [item for item in items if item.nodeid not in manifest]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected
    if len(items) != 248:
        raise pytest.UsageError(
            f"upstream baseline manifest must collect 248 tests, got {len(items)}"
        )
@pytest.fixture(autouse=True)
def isolate_external_runtime_dependencies(monkeypatch):
    """Keep unit tests independent from an externally running go2rtc service."""
    monkeypatch.setattr(
        "frigate.infrastructure.config.config.auto_detect_hwaccel",
        lambda: "",
    )
