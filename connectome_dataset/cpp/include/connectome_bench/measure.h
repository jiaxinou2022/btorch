#pragma once

#include <cstdint>
#include <functional>
#include <optional>

namespace connectome_bench {

struct MeasureConfig {
  int warmup = 10;
  int rep = 50;
  std::optional<double> io_bytes;
  std::optional<double> total_flops;
};

struct MeasureResult {
  double median_ms = 0.0;
  double p25_ms = 0.0;
  double p75_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  std::optional<double> compile_ms;
  std::optional<double> gbps;
  std::optional<double> gflops_s;
  std::optional<double> cpu_delta_mb;
  std::optional<double> gpu_delta_mb;
};

MeasureResult MeasureFunction(const std::function<void()>& fn, const MeasureConfig& config);

}  // namespace connectome_bench
