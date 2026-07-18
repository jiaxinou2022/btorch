#pragma once

#include <filesystem>
#include <vector>

#include "connectome_bench/records.h"

namespace connectome_bench {

void WriteRecordsJsonl(const std::vector<BenchmarkRecord>& records,
                       const std::filesystem::path& path);
void WriteRecordsCsv(const std::vector<BenchmarkRecord>& records,
                     const std::filesystem::path& path);
void WriteManifest(const RunManifest& manifest, const std::filesystem::path& path);

}  // namespace connectome_bench
