// Eigen CSR SpMV / SpMM benchmark — Google Benchmark.
//
// Build:
//   cmake -B build -DCONNECTOME_BENCH_BUILD_BENCHMARKS=ON && cmake --build build -t bench_spmv
//
// Run all:
//   ./build/bench_spmv
//
// Filter (Google Benchmark hierarchy via '/'):
//   ./build/bench_spmv --benchmark_filter='eigen/csr/spmv'
//   ./build/bench_spmv --benchmark_filter='eigen/csr/spmm.*bs=32'
//
// Compare two runs:
//   ./build/bench_spmv --benchmark_out=run1.json --benchmark_out_format=json
//   tools/compare.py benchmarks run1.json run2.json

#include <benchmark/benchmark.h>

#include <Eigen/Sparse>
#include <random>
#include <vector>

namespace {

// Build an n×n random CSR matrix at the requested density.
Eigen::SparseMatrix<float, Eigen::RowMajor>
MakeRandomCSR(int n, float density = 0.04f, unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> val(-1.0f, 1.0f);
  std::bernoulli_distribution keep(density);

  std::vector<Eigen::Triplet<float>> trips;
  trips.reserve(static_cast<size_t>(n) * n * density * 1.2f);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      if (keep(rng)) trips.emplace_back(i, j, val(rng));

  Eigen::SparseMatrix<float, Eigen::RowMajor> mat(n, n);
  mat.setFromTriplets(trips.begin(), trips.end());
  mat.makeCompressed();
  return mat;
}

// ── SpMV: y = A * x ─────────────────────────────────────────────────────────

void EigenCSRSpMV(benchmark::State& state) {
  const int n = static_cast<int>(state.range(0));
  auto A = MakeRandomCSR(n);
  Eigen::VectorXf x(n), y(n);
  x.setRandom();

  for (auto _ : state) {
    y.noalias() = A * x;
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }

  const double nnz = A.nonZeros();
  state.counters["n"]      = n;
  state.counters["nnz"]    = nnz;
  // kIsIterationInvariantRate: multiplies by iterations, divides by elapsed time
  state.counters["GFLOP/s"] = benchmark::Counter(
      2.0 * nnz / 1e9,
      benchmark::Counter::kIsIterationInvariantRate);
  state.counters["GB/s"] = benchmark::Counter(
      // read: nnz values + nnz col-indices + (n+1) row-ptrs + n x-values + n y-values
      (nnz * (sizeof(float) + sizeof(int)) + (n + 1) * sizeof(int) +
       2.0 * n * sizeof(float)) / 1e9,
      benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(EigenCSRSpMV)
    ->Name("eigen/csr/spmv")
    ->Arg(256)
    ->Arg(1024)
    ->Arg(4096)
    ->Unit(benchmark::kMicrosecond);

// ── SpMM: Y = A * X (batched SpMV with column-major dense RHS) ───────────────

void EigenCSRSpMM(benchmark::State& state) {
  const int n  = static_cast<int>(state.range(0));
  const int bs = static_cast<int>(state.range(1));
  auto A = MakeRandomCSR(n);
  // Column-major X so each column is contiguous — natural for Eigen SpMM.
  Eigen::MatrixXf X(n, bs), Y(n, bs);
  X.setRandom();

  for (auto _ : state) {
    Y.noalias() = A * X;
    benchmark::DoNotOptimize(Y.data());
    benchmark::ClobberMemory();
  }

  const double nnz = A.nonZeros();
  state.counters["n"]       = n;
  state.counters["nnz"]     = nnz;
  state.counters["bs"]      = bs;
  state.counters["GFLOP/s"] = benchmark::Counter(
      2.0 * nnz * bs / 1e9,
      benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(EigenCSRSpMM)
    ->Name("eigen/csr/spmm")
    ->ArgsProduct({{256, 1024}, {1, 8, 32}})
    ->Unit(benchmark::kMicrosecond);

}  // namespace

BENCHMARK_MAIN();
