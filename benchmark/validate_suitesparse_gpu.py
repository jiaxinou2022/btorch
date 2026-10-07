"""Validate SuiteSparse CSR construction and its GPU memory footprint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.benchmark_rsnn_cudagraph_compare import (  # noqa: E402
    make_torch_csr_weight,
)
from benchmark.suitesparse_datasets import (  # noqa: E402
    SUITESPARSE_DATASETS,
    load_suitesparse_csr,
)


def mib(value: int) -> float:
    """Convert bytes to mebibytes."""

    return value / 2**20


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=SUITESPARSE_DATASETS)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--weight-scale", type=float, default=0.15)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Construct both CSR orientations and execute one GPU SpMV."""

    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    matrix = load_suitesparse_csr(
        args.dataset,
        args.root,
        weight_scale=args.weight_scale,
        device=device,
    )
    torch.cuda.synchronize()
    after_loaded = torch.cuda.memory_allocated()
    torch_weight = make_torch_csr_weight(matrix)
    torch.cuda.synchronize()
    after_torch = torch.cuda.memory_allocated()
    vector = torch.zeros((matrix.shape[0], 1), device=device)
    result = torch.sparse.mm(torch_weight, vector)
    torch.cuda.synchronize()
    report = {
        "dataset": args.dataset,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_memory_mib": mib(
            torch.cuda.get_device_properties(device).total_memory
        ),
        "shape": list(matrix.shape),
        "nnz": matrix.indices.numel(),
        "weight_scale": args.weight_scale,
        "source_csr_storage_mib": mib(
            sum(
                tensor.numel() * tensor.element_size()
                for tensor in (
                    matrix.indptr,
                    matrix.indices,
                    matrix.data,
                    matrix._row,
                )
            )
        ),
        "transposed_torch_csr_storage_mib": mib(
            sum(
                tensor.numel() * tensor.element_size()
                for tensor in (
                    torch_weight.crow_indices(),
                    torch_weight.col_indices(),
                    torch_weight.values(),
                )
            )
        ),
        "t256_input_plus_reference_mib": mib(2 * 256 * matrix.shape[0] * 4),
        "loaded_graph_representations_mib": mib(after_loaded),
        "cached_torch_csr_incremental_mib": mib(after_torch - after_loaded),
        "spmv_result_shape": list(result.shape),
        "spmv_result_finite": bool(torch.isfinite(result).all()),
        "peak_allocated_mib": mib(torch.cuda.max_memory_allocated()),
        "peak_reserved_mib": mib(torch.cuda.max_memory_reserved()),
        "elapsed_s": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
