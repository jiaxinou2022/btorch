#pragma once

#include <memory>
#include <string>

#include "connectome_bench/cases.h"

namespace connectome_bench {

struct RunConfig {
  std::string device = "cpu";
  std::string dtype = "float32";
  int warmup = 10;
  int rep = 50;
};

class SpmvLoadedData {
 public:
  virtual ~SpmvLoadedData() = default;
};

class SpmvPreparedData {
 public:
  virtual ~SpmvPreparedData() = default;
};

class SpgemmLoadedData {
 public:
  virtual ~SpgemmLoadedData() = default;
};

class SpgemmPreparedData {
 public:
  virtual ~SpgemmPreparedData() = default;
};

class RSNNLoadedData {
 public:
  virtual ~RSNNLoadedData() = default;
};

class RSNNPreparedData {
 public:
  virtual ~RSNNPreparedData() = default;
};

class SpmvImplementation {
 public:
  virtual ~SpmvImplementation() = default;
  virtual std::string Name() const = 0;
  virtual std::unique_ptr<SpmvLoadedData> LoadGraph(
      const SpmvCase& spmv_case, const RunConfig& config) = 0;
  virtual std::unique_ptr<SpmvPreparedData> Preprocess(
      const SpmvLoadedData& loaded, int batch_size, const RunConfig& config) = 0;
  virtual void Compute(SpmvPreparedData& prepared) = 0;
};

class SpgemmImplementation {
 public:
  virtual ~SpgemmImplementation() = default;
  virtual std::string Name() const = 0;
  virtual std::unique_ptr<SpgemmLoadedData> LoadGraph(
      const SpgemmCase& spgemm_case, const RunConfig& config) = 0;
  virtual std::unique_ptr<SpgemmPreparedData> Preprocess(
      const SpgemmLoadedData& loaded, const RunConfig& config) = 0;
  virtual void Compute(SpgemmPreparedData& prepared) = 0;
};

class RSNNImplementation {
 public:
  virtual ~RSNNImplementation() = default;
  virtual std::string Name() const = 0;
  virtual std::unique_ptr<RSNNLoadedData> LoadCase(
      const RSNNCase& rsnn_case, const RunConfig& config) = 0;
  virtual std::unique_ptr<RSNNPreparedData> Preprocess(
      const RSNNLoadedData& loaded, const RunConfig& config) = 0;
  virtual void ComputeFwd(const RSNNPreparedData& prepared) = 0;
  virtual void ComputeFwdBwd(const RSNNPreparedData& prepared) = 0;
};

}  // namespace connectome_bench
