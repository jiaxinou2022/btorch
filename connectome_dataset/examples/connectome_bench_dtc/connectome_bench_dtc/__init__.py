"""Out-of-tree DTC-SpMM provider — the same weighted tensor-core kernel as the in-tree
leaf, reached through the entry-point registry. The kernel wrapper lives in
``connectome_dataset.benchmarks.torch.dtc.spmm``; this package only advertises it. Build
the extension with ``scripts/build_dtc.sh`` and set ``DTC_ROOT`` if needed.
"""
from __future__ import annotations

from connectome_dataset.benchmarks.providers import SpmmProvider
from connectome_dataset.benchmarks.torch.dtc import spmm as dtc


def get_providers() -> list[SpmmProvider]:
    return [
        SpmmProvider(
            framework="torch",  # DTC-SpMM is a torch CUDA extension
            provider="dtc_spmm",
            make_fn=lambda matrix, **kw: dtc.make_fn(matrix, **kw),
            variants=["csr"],
            self_timed=True,
            supports=dtc.supports,
        )
    ]
