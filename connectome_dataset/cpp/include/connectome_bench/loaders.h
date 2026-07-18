#pragma once

#include <filesystem>

#include "connectome_bench/cases.h"

namespace connectome_bench {

// Load a Matrix Market (.mtx) file into a row-major float CSR matrix.
Eigen::SparseMatrix<float, Eigen::RowMajor> LoadMatrixMarket(
    const std::filesystem::path& path);

SpmvCase LoadSpmvCase(const std::filesystem::path& path);
SpgemmCase LoadSpgemmCase(const std::filesystem::path& path);
RSNNCase LoadRSNNCase(const std::filesystem::path& path);

}  // namespace connectome_bench
