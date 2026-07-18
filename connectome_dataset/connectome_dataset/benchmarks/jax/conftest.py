"""JAX process setup — sets XLA env vars before any JAX import."""
from __future__ import annotations

import os

# Must be set before jax is imported anywhere in this process.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

from connectome_dataset.benchmarks.backend_utils import (
    configure_conda_cuda_toolchain,
    patch_brainevent_conda_nvcc,
)

configure_conda_cuda_toolchain()

try:
    import brainevent as _brainevent
except Exception:
    _brainevent = None

patch_brainevent_conda_nvcc(_brainevent)

import pytest  # noqa: E402  — must follow the XLA/CUDA env setup above


@pytest.fixture(scope="session", autouse=True)
def _jax_session_info() -> None:
    import jax

    print(f"\nJAX backend : {jax.default_backend()}")
    print(f"JAX devices : {jax.devices()}")
    if _brainevent is not None:
        print(f"brainevent  : {_brainevent.__version__}")
    else:
        print("brainevent  : unavailable")
