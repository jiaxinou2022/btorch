#include <Eigen/Sparse>

#include "reference/reference.h"

namespace connectome_bench {

namespace {

using SpMat = Eigen::SparseMatrix<float, Eigen::RowMajor>;

class RefSpgemmLoaded : public SpgemmLoadedData {
 public:
  explicit RefSpgemmLoaded(const SpMat* matrix) : matrix(matrix) {}
  const SpMat* matrix;
};

class RefSpgemmPrepared : public SpgemmPreparedData {
 public:
  const SpMat* matrix = nullptr;
  SpMat result;
};

// C = A * A.T  (two-hop / common-neighbour / Markov-clustering expansion).
class ReferenceSpgemm : public SpgemmImplementation {
 public:
  std::string Name() const override { return "reference.eigen_csr"; }

  std::unique_ptr<SpgemmLoadedData> LoadGraph(const SpgemmCase& spgemm_case,
                                              const RunConfig&) override {
    return std::make_unique<RefSpgemmLoaded>(&spgemm_case.matrix);
  }

  std::unique_ptr<SpgemmPreparedData> Preprocess(const SpgemmLoadedData& loaded,
                                                 const RunConfig&) override {
    const auto& l = static_cast<const RefSpgemmLoaded&>(loaded);
    auto prepared = std::make_unique<RefSpgemmPrepared>();
    prepared->matrix = l.matrix;
    return prepared;
  }

  void Compute(SpgemmPreparedData& prepared) override {
    auto& p = static_cast<RefSpgemmPrepared&>(prepared);
    p.result = (*p.matrix) * p.matrix->transpose();
  }
};

}  // namespace

std::unique_ptr<SpgemmImplementation> MakeReferenceSpgemm() {
  return std::make_unique<ReferenceSpgemm>();
}

void RegisterBuiltins(Registry& registry) {
  registry.RegisterSpmv(MakeReferenceSpmv());
  registry.RegisterSpgemm(MakeReferenceSpgemm());
}

}  // namespace connectome_bench
