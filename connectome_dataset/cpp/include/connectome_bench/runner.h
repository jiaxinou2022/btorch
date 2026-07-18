#pragma once

#include <filesystem>
#include <vector>

#include "connectome_bench/records.h"

namespace connectome_bench {

std::filesystem::path EnsureRunDirectory(const std::filesystem::path& root,
                                         const std::string& run_id);
void EmitConsoleSummary(const std::vector<BenchmarkRecord>& records);

}  // namespace connectome_bench
