"""brian2 process setup — the reference-simulator framework (peer of jax / torch / nest).

brian2 keeps global simulation state (the "magic" device / default clock), so like NEST it
is a process-isolation boundary and gets its own framework directory; ``run_all.py`` spawns
it separately. This conftest reports the version once and skips cleanly if brian2 is absent.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _brian2_session_info() -> None:
    brian2 = pytest.importorskip("brian2")
    brian2.prefs.logging.console_log_level = "WARNING"
    print(f"\nbrian2 version : {brian2.__version__}")
