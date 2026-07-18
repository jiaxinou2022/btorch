#include "connectome_bench/measure.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace connectome_bench {

namespace {

double Percentile(std::vector<double> values, double fraction) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double index = fraction * static_cast<double>(values.size() - 1);
  const auto lower = static_cast<std::size_t>(index);
  const auto upper = static_cast<std::size_t>(std::ceil(index));
  if (lower == upper) {
    return values[lower];
  }
  const double weight = index - static_cast<double>(lower);
  return values[lower] * (1.0 - weight) + values[upper] * weight;
}

}  // namespace

MeasureResult MeasureFunction(const std::function<void()>& fn, const MeasureConfig& config) {
  using Clock = std::chrono::steady_clock;
  auto compile_start = Clock::now();
  fn();
  auto compile_end = Clock::now();

  for (int i = 0; i < config.warmup; ++i) {
    fn();
  }

  std::vector<double> samples;
  samples.reserve(config.rep);
  for (int i = 0; i < config.rep; ++i) {
    auto start = Clock::now();
    fn();
    auto end = Clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }

  MeasureResult result;
  result.compile_ms = std::chrono::duration<double, std::milli>(compile_end - compile_start).count();
  result.min_ms = *std::min_element(samples.begin(), samples.end());
  result.max_ms = *std::max_element(samples.begin(), samples.end());
  result.p25_ms = Percentile(samples, 0.25);
  result.median_ms = Percentile(samples, 0.50);
  result.p75_ms = Percentile(samples, 0.75);
  if (config.io_bytes.has_value() && result.median_ms > 0.0) {
    result.gbps = *config.io_bytes / (result.median_ms / 1000.0) / 1e9;
  }
  if (config.total_flops.has_value() && result.median_ms > 0.0) {
    result.gflops_s = *config.total_flops / (result.median_ms / 1000.0) / 1e9;
  }
  return result;
}

}  // namespace connectome_bench
