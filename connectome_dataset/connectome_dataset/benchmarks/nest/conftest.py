"""NEST process setup — the reference-simulator framework (peer of jax / torch).

NEST is a hard process-isolation boundary like jax/torch (its kernel is a global
singleton), so it gets its own framework directory and ``run_all.py`` spawns it in a
separate process. This conftest quiets the kernel banner and reports the version once.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYNEST_QUIET", "1")

import pytest


@pytest.fixture(scope="session", autouse=True)
def _nest_session_info() -> None:
    nest = pytest.importorskip("nest")
    print(f"\nNEST version : {nest.__version__}")
