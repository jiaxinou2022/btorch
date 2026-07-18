// Sputnik SpMM baseline. We call sputnik::CudaSpmm (a host entry point) and manage the
// device buffers ourselves, so this is plain C++ linked against the CUDA-compiled
// sputnik object + cudart — no nvcc needed for this translation unit.
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "sputnik/sputnik_baseline.h"

namespace sputnik {
// Declared here (not via sputnik's headers) to keep this TU decoupled from the vendored
// tree; resolved at link time from the CUDA-compiled cuda_spmm object.
cudaError_t CudaSpmm(int m, int k, int n, int nonzeros, const int* row_indices, const float* values,
                     const int* row_offsets, const int* column_indices, const float* dense_matrix,
                     float* output_matrix, cudaStream_t stream);
}  // namespace sputnik

namespace connectome_bench {

namespace {

using SpMat = Eigen::SparseMatrix<float, Eigen::RowMajor>;

void Check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("cuda: ") + what + ": " + cudaGetErrorString(e));
}

void* DeviceCopy(const void* host, std::size_t bytes) {
  void* d = nullptr;
  Check(cudaMalloc(&d, bytes), "malloc");
  Check(cudaMemcpy(d, host, bytes, cudaMemcpyHostToDevice), "memcpy H2D");
  return d;
}

class SputnikLoaded : public SpmvLoadedData {
 public:
  explicit SputnikLoaded(const SpMat* m) : matrix(m) {}
  const SpMat* matrix;
};

class SputnikPrepared : public SpmvPreparedData {
 public:
  int m = 0, k = 0, n = 0, nnz = 0;
  int* d_row_offsets = nullptr;
  int* d_col_indices = nullptr;
  int* d_row_indices = nullptr;  // load-balancing swizzle
  float* d_values = nullptr;
  float* d_x = nullptr;
  float* d_y = nullptr;

  ~SputnikPrepared() override {
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_row_indices);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
  }
};

void ValidateAgainstOracle(const SpMat& A, const SputnikPrepared& p);

class SputnikSpmv : public SpmvImplementation {
 public:
  std::string Name() const override { return "sputnik.csr"; }

  std::unique_ptr<SpmvLoadedData> LoadGraph(const SpmvCase& c, const RunConfig&) override {
    return std::make_unique<SputnikLoaded>(&c.matrix);
  }

  std::unique_ptr<SpmvPreparedData> Preprocess(const SpmvLoadedData& loaded, int batch_size,
                                               const RunConfig&) override {
    const auto& l = static_cast<const SputnikLoaded&>(loaded);
    SpMat A = *l.matrix;
    A.makeCompressed();
    auto p = std::make_unique<SputnikPrepared>();
    p->m = static_cast<int>(A.rows());
    p->k = static_cast<int>(A.cols());
    p->n = batch_size < 1 ? 1 : batch_size;
    p->nnz = static_cast<int>(A.nonZeros());

    // Sputnik's load balancing: process rows longest-first (this is its whole point on
    // heavy-tailed connectome degree). We compute the swizzle on host.
    std::vector<int> swizzle(p->m);
    std::iota(swizzle.begin(), swizzle.end(), 0);
    const int* off = A.outerIndexPtr();
    std::sort(swizzle.begin(), swizzle.end(),
              [&](int a, int b) { return (off[a + 1] - off[a]) > (off[b + 1] - off[b]); });

    p->d_row_offsets = static_cast<int*>(DeviceCopy(off, (p->m + 1) * sizeof(int)));
    p->d_col_indices = static_cast<int*>(DeviceCopy(A.innerIndexPtr(), p->nnz * sizeof(int)));
    p->d_values = static_cast<float*>(DeviceCopy(A.valuePtr(), p->nnz * sizeof(float)));
    p->d_row_indices = static_cast<int*>(DeviceCopy(swizzle.data(), p->m * sizeof(int)));

    const std::vector<float> x(static_cast<std::size_t>(p->k) * p->n, 1.0f);  // row-major k x n
    p->d_x = static_cast<float*>(DeviceCopy(x.data(), x.size() * sizeof(float)));
    Check(cudaMalloc(&p->d_y, static_cast<std::size_t>(p->m) * p->n * sizeof(float)), "malloc Y");

    Compute(*p);
    ValidateAgainstOracle(A, *p);
    return p;
  }

  void Compute(SpmvPreparedData& prepared) override {
    auto& p = static_cast<SputnikPrepared&>(prepared);
    Check(sputnik::CudaSpmm(p.m, p.k, p.n, p.nnz, p.d_row_indices, p.d_values, p.d_row_offsets,
                            p.d_col_indices, p.d_x, p.d_y, /*stream=*/0),
          "sputnik spmm");
    Check(cudaDeviceSynchronize(), "sync");
  }
};

void ValidateAgainstOracle(const SpMat& A, const SputnikPrepared& p) {
  Eigen::VectorXf ref = A * Eigen::VectorXf::Ones(A.cols());  // Y column 0 = row sums
  std::vector<float> got(static_cast<std::size_t>(p.m) * p.n);
  Check(cudaMemcpy(got.data(), p.d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy Y");
  const float scale = ref.cwiseAbs().maxCoeff() + 1e-6f;
  int bad = 0;
  for (int i = 0; i < p.m; ++i)
    if (std::abs(got[static_cast<std::size_t>(i) * p.n] - ref[i]) > 1e-3f * scale) ++bad;
  if (bad) std::fprintf(stderr, "[warn] sputnik.csr: %d/%d rows exceed tolerance vs oracle\n", bad, p.m);
}

}  // namespace

std::unique_ptr<SpmvImplementation> MakeSputnikSpmv() { return std::make_unique<SputnikSpmv>(); }

void RegisterSputnik(Registry& registry) { registry.RegisterSpmv(MakeSputnikSpmv()); }

}  // namespace connectome_bench
