#include <algorithm>

#include <Eigen/Sparse>

#include "reference/reference.h"

namespace connectome_bench {

namespace {

using SpMat = Eigen::SparseMatrix<float, Eigen::RowMajor>;

class RefSpmvLoaded : public SpmvLoadedData {
 public:
  explicit RefSpmvLoaded(const SpMat* matrix) : matrix(matrix) {}
  const SpMat* matrix;
};

class RefSpmvPrepared : public SpmvPreparedData {
 public:
  const SpMat* matrix = nullptr;
  Eigen::MatrixXf x;
  Eigen::MatrixXf y;
};

// y = A * X, with a dense right-hand side of `batch_size` columns (batch=1 -> SpMV).
class ReferenceSpmv : public SpmvImplementation {
 public:
  std::string Name() const override { return "reference.eigen_csr"; }

  std::unique_ptr<SpmvLoadedData> LoadGraph(const SpmvCase& spmv_case,
                                            const RunConfig&) override {
    return std::make_unique<RefSpmvLoaded>(&spmv_case.matrix);
  }

  std::unique_ptr<SpmvPreparedData> Preprocess(const SpmvLoadedData& loaded, int batch_size,
                                               const RunConfig&) override {
    const auto& l = static_cast<const RefSpmvLoaded&>(loaded);
    const int b = std::max(batch_size, 1);
    auto prepared = std::make_unique<RefSpmvPrepared>();
    prepared->matrix = l.matrix;
    prepared->x = Eigen::MatrixXf::Ones(l.matrix->cols(), b);
    prepared->y = Eigen::MatrixXf::Zero(l.matrix->rows(), b);
    return prepared;
  }

  void Compute(SpmvPreparedData& prepared) override {
    auto& p = static_cast<RefSpmvPrepared&>(prepared);
    p.y.noalias() = (*p.matrix) * p.x;
  }
};

}  // namespace

std::unique_ptr<SpmvImplementation> MakeReferenceSpmv() {
  return std::make_unique<ReferenceSpmv>();
}

}  // namespace connectome_bench
