#pragma once

#include <memory>

#include "connectome_bench/interfaces.h"
#include "connectome_bench/registry.h"

namespace connectome_bench {

// cuSPARSE SpMM baseline (C = A*X, dense X of batch_size cols; batch=1 -> SpMV).
// Precision is taken from RunConfig::dtype ("float32" | "float16" | "bfloat16");
// reduced precision uses fp32 accumulation (CUDA_R_32F compute type), the standard
// tensor-core mixed-precision mode. This is the real cuSPARSE half-precision path
// that CuPy's sparse layer cannot express.
std::unique_ptr<SpmvImplementation> MakeCusparseSpmv();

// Register the cuSPARSE baselines into a registry (called from the runner when the
// CUDA build is enabled).
void RegisterCusparse(Registry& registry);

}  // namespace connectome_bench
