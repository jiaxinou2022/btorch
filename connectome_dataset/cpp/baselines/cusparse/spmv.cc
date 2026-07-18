// cuSPARSE SpMM baseline. cuSPARSE is a host-side library, so this compiles with the
// ordinary C++ compiler (no nvcc) as long as CUDA::cusparse / CUDA::cudart are linked.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include "cusparse/cusparse_baseline.h"

namespace connectome_bench {

namespace {

using SpMat = Eigen::SparseMatrix<float, Eigen::RowMajor>;

void Check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("cuda: ") + what + ": " + cudaGetErrorString(e));
}
void Check(cusparseStatus_t s, const char* what) {
  if (s != CUSPARSE_STATUS_SUCCESS)
    throw std::runtime_error(std::string("cusparse: ") + what + ": " + cusparseGetErrorString(s));
}

// Map the RunConfig dtype string to the cuSPARSE data type and host element size.
// Reduced precision accumulates in fp32 (compute type CUDA_R_32F), the tensor-core mode.
cudaDataType DataTypeOf(const std::string& dtype) {
  if (dtype == "float32" || dtype == "fp32") return CUDA_R_32F;
  if (dtype == "float16" || dtype == "fp16") return CUDA_R_16F;
  if (dtype == "bfloat16" || dtype == "bf16") return CUDA_R_16BF;
  throw std::runtime_error("cusparse: unsupported dtype " + dtype);
}

// Host-side cast of the fp32 CSR values into the target precision, as raw bytes.
std::vector<std::uint8_t> CastValues(const float* vals, std::int64_t nnz, cudaDataType dt) {
  if (dt == CUDA_R_32F) {
    const auto* b = reinterpret_cast<const std::uint8_t*>(vals);
    return std::vector<std::uint8_t>(b, b + nnz * sizeof(float));
  }
  if (dt == CUDA_R_16F) {
    std::vector<__half> h(nnz);
    for (std::int64_t i = 0; i < nnz; ++i) h[i] = __float2half(vals[i]);
    const auto* b = reinterpret_cast<const std::uint8_t*>(h.data());
    return std::vector<std::uint8_t>(b, b + nnz * sizeof(__half));
  }
  std::vector<__nv_bfloat16> h(nnz);
  for (std::int64_t i = 0; i < nnz; ++i) h[i] = __float2bfloat16(vals[i]);
  const auto* b = reinterpret_cast<const std::uint8_t*>(h.data());
  return std::vector<std::uint8_t>(b, b + nnz * sizeof(__nv_bfloat16));
}

std::size_t ElemSize(cudaDataType dt) {
  return dt == CUDA_R_32F ? sizeof(float) : 2;  // fp16 / bf16 are 2 bytes
}

class CusparseLoaded : public SpmvLoadedData {
 public:
  explicit CusparseLoaded(const SpMat* m) : matrix(m) {}
  const SpMat* matrix;
};

class CusparsePrepared : public SpmvPreparedData {
 public:
  cusparseHandle_t handle = nullptr;
  cusparseSpMatDescr_t matA = nullptr;
  cusparseDnMatDescr_t matB = nullptr, matC = nullptr;
  void* dRowPtr = nullptr;
  void* dColInd = nullptr;
  void* dVals = nullptr;
  void* dX = nullptr;
  void* dY = nullptr;
  void* buffer = nullptr;
  cudaDataType dtype = CUDA_R_32F;
  float alpha = 1.0f, beta = 0.0f;

  ~CusparsePrepared() override {
    if (matA) cusparseDestroySpMat(matA);
    if (matB) cusparseDestroyDnMat(matB);
    if (matC) cusparseDestroyDnMat(matC);
    if (handle) cusparseDestroy(handle);
    cudaFree(dRowPtr);
    cudaFree(dColInd);
    cudaFree(dVals);
    cudaFree(dX);
    cudaFree(dY);
    cudaFree(buffer);
  }
};

void* DeviceCopy(const void* host, std::size_t bytes) {
  void* d = nullptr;
  Check(cudaMalloc(&d, bytes), "malloc");
  Check(cudaMemcpy(d, host, bytes, cudaMemcpyHostToDevice), "memcpy H2D");
  return d;
}

// Validate one result against the Eigen row-sum oracle (X is all ones -> Y[i] = sum of row i).
// Reduced precision is checked with a looser tolerance; a failure is reported to stderr.
void ValidateAgainstOracle(const SpMat& A, const CusparsePrepared& p, int batch, const std::string& dtype);

class CusparseSpmv : public SpmvImplementation {
 public:
  std::string Name() const override { return "cusparse.csr"; }

  std::unique_ptr<SpmvLoadedData> LoadGraph(const SpmvCase& c, const RunConfig&) override {
    return std::make_unique<CusparseLoaded>(&c.matrix);
  }

  std::unique_ptr<SpmvPreparedData> Preprocess(const SpmvLoadedData& loaded, int batch_size,
                                               const RunConfig& config) override {
    const auto& l = static_cast<const CusparseLoaded&>(loaded);
    SpMat A = *l.matrix;
    A.makeCompressed();
    const int rows = static_cast<int>(A.rows());
    const int cols = static_cast<int>(A.cols());
    const std::int64_t nnz = A.nonZeros();
    const int batch = batch_size < 1 ? 1 : batch_size;

    auto p = std::make_unique<CusparsePrepared>();
    p->dtype = DataTypeOf(config.dtype);
    const std::size_t es = ElemSize(p->dtype);

    Check(cusparseCreate(&p->handle), "create");

    p->dRowPtr = DeviceCopy(A.outerIndexPtr(), (rows + 1) * sizeof(int));
    p->dColInd = DeviceCopy(A.innerIndexPtr(), nnz * sizeof(int));
    const std::vector<std::uint8_t> vbytes = CastValues(A.valuePtr(), nnz, p->dtype);
    p->dVals = DeviceCopy(vbytes.data(), vbytes.size());

    // Dense X of ones (matches the reference oracle), zero-initialised Y — column-major.
    const std::vector<std::uint8_t> xbytes = CastValues(std::vector<float>(cols * batch, 1.0f).data(),
                                                         static_cast<std::int64_t>(cols) * batch, p->dtype);
    p->dX = DeviceCopy(xbytes.data(), xbytes.size());
    Check(cudaMalloc(&p->dY, static_cast<std::size_t>(rows) * batch * es), "malloc Y");
    Check(cudaMemset(p->dY, 0, static_cast<std::size_t>(rows) * batch * es), "memset Y");

    Check(cusparseCreateCsr(&p->matA, rows, cols, nnz, p->dRowPtr, p->dColInd, p->dVals,
                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, p->dtype),
          "createCsr");
    Check(cusparseCreateDnMat(&p->matB, cols, batch, cols, p->dX, p->dtype, CUSPARSE_ORDER_COL), "createDnB");
    Check(cusparseCreateDnMat(&p->matC, rows, batch, rows, p->dY, p->dtype, CUSPARSE_ORDER_COL), "createDnC");

    std::size_t buffer_size = 0;
    Check(cusparseSpMM_bufferSize(p->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, &p->alpha, p->matA, p->matB, &p->beta,
                                  p->matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size),
          "bufferSize");
    if (buffer_size) Check(cudaMalloc(&p->buffer, buffer_size), "malloc buffer");

    // One validated warm compute, checked against the Eigen oracle.
    Compute(*p);
    ValidateAgainstOracle(A, *p, batch, config.dtype);
    return p;
  }

  void Compute(SpmvPreparedData& prepared) override {
    auto& p = static_cast<CusparsePrepared&>(prepared);
    Check(cusparseSpMM(p.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                       &p.alpha, p.matA, p.matB, &p.beta, p.matC, CUDA_R_32F,
                       CUSPARSE_SPMM_ALG_DEFAULT, p.buffer),
          "spmm");
    Check(cudaDeviceSynchronize(), "sync");  // the multiply is async; sync so timing is real
  }
};

void ValidateAgainstOracle(const SpMat& A, const CusparsePrepared& p, int batch, const std::string& dtype) {
  const int rows = static_cast<int>(A.rows());
  Eigen::VectorXf ref = A * Eigen::VectorXf::Ones(A.cols());  // Y column 0 = row sums

  std::vector<float> got(static_cast<std::size_t>(rows));
  if (p.dtype == CUDA_R_32F) {
    Check(cudaMemcpy(got.data(), p.dY, rows * sizeof(float), cudaMemcpyDeviceToHost), "copy Y");
  } else if (p.dtype == CUDA_R_16F) {
    std::vector<__half> h(rows);
    Check(cudaMemcpy(h.data(), p.dY, rows * sizeof(__half), cudaMemcpyDeviceToHost), "copy Y");
    for (int i = 0; i < rows; ++i) got[i] = __half2float(h[i]);
  } else {
    std::vector<__nv_bfloat16> h(rows);
    Check(cudaMemcpy(h.data(), p.dY, rows * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "copy Y");
    for (int i = 0; i < rows; ++i) got[i] = __bfloat162float(h[i]);
  }

  const float rtol = (dtype == "float32" || dtype == "fp32") ? 1e-4f : (dtype.find("bf") != std::string::npos ? 1e-1f : 6e-2f);
  const float scale = ref.cwiseAbs().maxCoeff() + 1e-6f;
  int bad = 0;
  for (int i = 0; i < rows; ++i)
    if (std::abs(got[i] - ref[i]) > rtol * scale) ++bad;
  if (bad)
    std::fprintf(stderr, "[warn] cusparse.csr(%s): %d/%d rows exceed tolerance vs oracle\n",
                 dtype.c_str(), bad, rows);
}

}  // namespace

std::unique_ptr<SpmvImplementation> MakeCusparseSpmv() { return std::make_unique<CusparseSpmv>(); }

void RegisterCusparse(Registry& registry) { registry.RegisterSpmv(MakeCusparseSpmv()); }

}  // namespace connectome_bench
