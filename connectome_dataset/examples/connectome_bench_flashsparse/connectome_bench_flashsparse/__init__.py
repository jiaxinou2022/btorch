"""Out-of-tree FlashSparse provider — a real tensor-core weighted SpMM (Cao et al.,
FlashSparse), reached through the entry-point registry.

The kernel wrapper (blocked-format conversion + self-timing) is the single in-tree
implementation in ``connectome_dataset.benchmarks.torch.flashsparse.spmm``; this package
only advertises it through the ``connectome_bench.spmm`` entry point, so the same code
validates both integration paths. Build the extensions with ``scripts/build_flashsparse.sh``
and point ``FLASHSPARSE_ROOT`` at ``external/FlashSparse/FlashSparse``.
"""
from __future__ import annotations

from connectome_dataset.benchmarks.providers import SpmmProvider
from connectome_dataset.benchmarks.torch.flashsparse import spmm as fs


def get_providers() -> list[SpmmProvider]:
    return [
        SpmmProvider(
            framework="torch",  # invoked via torch (FS_SpMM is a torch CUDA extension)
            provider="flashsparse",
            make_fn=lambda matrix, **kw: fs.make_fn(matrix, **kw),
            variants=["csr"],
            self_timed=True,  # FlashSparse returns its own kernel-only CUDA-event time
            supports=fs.supports,
        )
    ]
