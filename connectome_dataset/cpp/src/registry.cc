#include "connectome_bench/registry.h"

namespace connectome_bench {

void Registry::RegisterSpmv(std::unique_ptr<SpmvImplementation> implementation) {
  spmv_[implementation->Name()] = std::move(implementation);
}

void Registry::RegisterSpgemm(std::unique_ptr<SpgemmImplementation> implementation) {
  spgemm_[implementation->Name()] = std::move(implementation);
}

void Registry::RegisterRSNN(std::unique_ptr<RSNNImplementation> implementation) {
  rsnn_[implementation->Name()] = std::move(implementation);
}

SpmvImplementation* Registry::FindSpmv(const std::string& name) const {
  auto it = spmv_.find(name);
  return it == spmv_.end() ? nullptr : it->second.get();
}

SpgemmImplementation* Registry::FindSpgemm(const std::string& name) const {
  auto it = spgemm_.find(name);
  return it == spgemm_.end() ? nullptr : it->second.get();
}

RSNNImplementation* Registry::FindRSNN(const std::string& name) const {
  auto it = rsnn_.find(name);
  return it == rsnn_.end() ? nullptr : it->second.get();
}

}  // namespace connectome_bench
