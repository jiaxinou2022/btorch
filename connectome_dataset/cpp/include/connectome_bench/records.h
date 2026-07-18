#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace connectome_bench {

struct BenchmarkRecord {
  std::string target;
  std::string alg;
  std::string variant;
  std::string case_id;
  std::string graph_id;
  int replicate = 1;
  std::string device;
  std::string dtype;
  std::int64_t n = 0;
  std::int64_t nnz = 0;
  double density = 0.0;
  std::optional<int> batch_size;
  std::optional<int> timesteps;
  std::optional<std::string> pass_name;
  std::optional<double> median_ms;
  std::optional<double> p25_ms;
  std::optional<double> p75_ms;
  std::optional<double> min_ms;
  std::optional<double> max_ms;
  std::optional<double> compile_ms;
  std::optional<double> gbps;
  std::optional<double> gflops_s;
  std::optional<double> cpu_delta_mb;
  std::optional<double> gpu_delta_mb;
  std::string status = "ok";
  std::string error;
  std::string extra_json = "{}";
};

struct RunManifest {
  std::string run_id;
  std::string target;
  std::string entrypoint;
  std::string created_at;
  std::string command;
  std::string host;
  std::string platform;
  std::string records_path;
  std::string reports_dir;
  std::string artifacts_dir;
};

}  // namespace connectome_bench
