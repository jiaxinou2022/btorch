#pragma once

#include <memory>

#include "connectome_bench/interfaces.h"
#include "connectome_bench/registry.h"

namespace connectome_bench {

// Reference (single-thread Eigen CSR) implementations — the numerical oracle
// every accelerated kernel is validated against. See
// docs/benchmark_design/design.md §1 (Reference executor).
std::unique_ptr<SpmvImplementation> MakeReferenceSpmv();
std::unique_ptr<SpgemmImplementation> MakeReferenceSpgemm();

// Register all in-tree reference baselines into a registry.
void RegisterBuiltins(Registry& registry);

}  // namespace connectome_bench
