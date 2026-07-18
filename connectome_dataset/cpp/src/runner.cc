#include "connectome_bench/runner.h"

#include <filesystem>
#include <iostream>

namespace connectome_bench {

std::filesystem::path EnsureRunDirectory(const std::filesystem::path& root,
                                         const std::string& run_id) {
  auto run_dir = root / run_id;
  std::filesystem::create_directories(run_dir / "reports");
  std::filesystem::create_directories(run_dir / "artifacts");
  return run_dir;
}

void EmitConsoleSummary(const std::vector<BenchmarkRecord>& records) {
  std::cout << "Records: " << records.size() << std::endl;
}

}  // namespace connectome_bench
