#include "connectome_bench/loaders.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace connectome_bench {

namespace {

struct MmHeader {
  bool pattern = false;
  bool symmetric = false;
};

MmHeader ParseBanner(const std::string& line) {
  // %%MatrixMarket matrix coordinate <field> <symmetry>
  std::istringstream iss(line);
  std::string tag, object, format, field, symmetry;
  iss >> tag >> object >> format >> field >> symmetry;
  if (format != "coordinate") {
    throw std::runtime_error("Only coordinate (sparse) Matrix Market files are supported: " + format);
  }
  MmHeader h;
  h.pattern = (field == "pattern");
  if (symmetry == "symmetric") {
    h.symmetric = true;
  } else if (symmetry == "skew-symmetric" || symmetry == "hermitian") {
    // These mirror with a sign/conjugate we don't apply; reject rather than corrupt.
    throw std::runtime_error("Unsupported Matrix Market symmetry: " + symmetry);
  }
  return h;
}

std::string CaseName(const std::filesystem::path& path) {
  return path.stem().string();
}

double Density(const Eigen::SparseMatrix<float, Eigen::RowMajor>& m) {
  const double cells = static_cast<double>(m.rows()) * static_cast<double>(m.cols());
  return cells > 0 ? static_cast<double>(m.nonZeros()) / cells : 0.0;
}

}  // namespace

Eigen::SparseMatrix<float, Eigen::RowMajor> LoadMatrixMarket(
    const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Cannot open Matrix Market file: " + path.string());
  }

  std::string line;
  if (!std::getline(in, line)) {
    throw std::runtime_error("Empty Matrix Market file: " + path.string());
  }
  const MmHeader header = ParseBanner(line);

  // Skip comment lines, then read the dimension line.
  while (std::getline(in, line)) {
    if (!line.empty() && line[0] != '%') break;
  }
  std::istringstream dims(line);
  Eigen::Index rows = 0, cols = 0;
  std::int64_t declared_nnz = 0;
  dims >> rows >> cols >> declared_nnz;

  std::vector<Eigen::Triplet<float>> triplets;
  triplets.reserve(static_cast<size_t>(header.symmetric ? declared_nnz * 2 : declared_nnz));

  Eigen::Index r = 0, c = 0;
  double v = 1.0;
  std::int64_t read = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '%') continue;
    std::istringstream entry(line);
    if (!(entry >> r >> c)) continue;  // skip a malformed coordinate line
    v = 1.0;
    if (!header.pattern && !(entry >> v)) continue;
    triplets.emplace_back(static_cast<int>(r - 1), static_cast<int>(c - 1), static_cast<float>(v));
    if (header.symmetric && r != c) {
      triplets.emplace_back(static_cast<int>(c - 1), static_cast<int>(r - 1), static_cast<float>(v));
    }
    ++read;
  }
  if (read != declared_nnz) {
    std::cerr << "[warn] " << path.filename().string() << ": read " << read
              << " entries but header declared " << declared_nnz << "\n";
  }

  Eigen::SparseMatrix<float, Eigen::RowMajor> mat(rows, cols);
  mat.setFromTriplets(triplets.begin(), triplets.end());
  mat.makeCompressed();
  return mat;
}

SpmvCase LoadSpmvCase(const std::filesystem::path& path) {
  SpmvCase c;
  c.matrix = LoadMatrixMarket(path);
  c.case_id = CaseName(path) + "_spmv";
  c.graph_id = CaseName(path);
  c.n = c.matrix.rows();
  c.nnz = c.matrix.nonZeros();
  c.density = Density(c.matrix);
  c.default_batch_sizes = {1, 8, 32, 128};
  return c;
}

SpgemmCase LoadSpgemmCase(const std::filesystem::path& path) {
  SpgemmCase c;
  c.matrix = LoadMatrixMarket(path);
  c.case_id = CaseName(path) + "_spgemm";
  c.graph_id = CaseName(path);
  c.n = c.matrix.rows();
  c.nnz = c.matrix.nonZeros();
  c.density = Density(c.matrix);
  // Intermediate products of C = A @ A.T: for each nonzero (i,k) of A, row k of
  // A.T contributes col_nnz(k) products. Column counts come from the CSC view.
  const Eigen::SparseMatrix<float, Eigen::ColMajor> csc = c.matrix;
  std::int64_t intermediate = 0;
  const int* inner = c.matrix.innerIndexPtr();
  for (int k = 0; k < c.matrix.nonZeros(); ++k) {
    const int col = inner[k];
    intermediate += csc.outerIndexPtr()[col + 1] - csc.outerIndexPtr()[col];
  }
  c.intermediate_products = intermediate;
  return c;
}

RSNNCase LoadRSNNCase(const std::filesystem::path& path) {
  RSNNCase c;
  c.matrix = LoadMatrixMarket(path);
  c.case_id = CaseName(path) + "_rsnn";
  c.graph_id = CaseName(path);
  c.n = c.matrix.rows();
  c.nnz = c.matrix.nonZeros();
  c.density = Density(c.matrix);
  return c;
}

}  // namespace connectome_bench
