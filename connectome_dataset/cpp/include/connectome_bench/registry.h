#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "connectome_bench/interfaces.h"

namespace connectome_bench {

class Registry {
 public:
  void RegisterSpmv(std::unique_ptr<SpmvImplementation> implementation);
  void RegisterSpgemm(std::unique_ptr<SpgemmImplementation> implementation);
  void RegisterRSNN(std::unique_ptr<RSNNImplementation> implementation);

  SpmvImplementation* FindSpmv(const std::string& name) const;
  SpgemmImplementation* FindSpgemm(const std::string& name) const;
  RSNNImplementation* FindRSNN(const std::string& name) const;

 private:
  std::unordered_map<std::string, std::unique_ptr<SpmvImplementation>> spmv_;
  std::unordered_map<std::string, std::unique_ptr<SpgemmImplementation>> spgemm_;
  std::unordered_map<std::string, std::unique_ptr<RSNNImplementation>> rsnn_;
};

}  // namespace connectome_bench
