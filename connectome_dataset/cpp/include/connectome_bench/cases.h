#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Sparse>

namespace connectome_bench {

using MetadataMap = std::map<std::string, std::string>;

struct SpmvCase {
  std::string case_id;
  std::string graph_id;
  int replicate = 1;
  Eigen::SparseMatrix<float, Eigen::RowMajor> matrix;
  std::int64_t n = 0;
  std::int64_t nnz = 0;
  double density = 0.0;
  std::vector<int> default_batch_sizes;
  MetadataMap metadata;
};

struct SpgemmCase {
  // C = A @ A.T (two-hop / common-neighbour / Markov-clustering expansion).
  std::string case_id;
  std::string graph_id;
  int replicate = 1;
  Eigen::SparseMatrix<float, Eigen::RowMajor> matrix;
  std::int64_t n = 0;
  std::int64_t nnz = 0;
  double density = 0.0;
  std::int64_t intermediate_products = 0;  // for FLOP counting; see metrics
  MetadataMap metadata;
};

struct RSNNCase {
  std::string case_id;
  std::string graph_id;
  int replicate = 1;
  Eigen::SparseMatrix<float, Eigen::RowMajor> matrix;
  std::int64_t n = 0;
  std::int64_t nnz = 0;
  double density = 0.0;
  int timesteps = 50;
  int batch_size = 4;
  int num_input = 64;
  int num_output = 10;
  MetadataMap metadata;
};

}  // namespace connectome_bench
