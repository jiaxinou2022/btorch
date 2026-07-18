#pragma once

#include <memory>

#include "connectome_bench/interfaces.h"
#include "connectome_bench/registry.h"

namespace connectome_bench {

// Sputnik SpMM baseline (Gale et al., "Sparse GPU Kernels for Deep Learning").
// A CUDA-core, load-balanced 1-D tiling SpMM — the canonical non-tensor-core SOTA
// baseline. Built from the vendored external/sputnik repo (fetched separately).
std::unique_ptr<SpmvImplementation> MakeSputnikSpmv();

void RegisterSputnik(Registry& registry);

}  // namespace connectome_bench
