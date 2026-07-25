// C-ABI wrapper around MH-SpGEMM (Yang et al., "MH-SpGEMM: Efficient Sparse General
// Matrix-Matrix Multiplication on Modern GPUs via Masking and Hashing Cooperative
// Optimization") for the out-of-tree Python provider (loaded via ctypes).
//
// Computes C = A·Aᵀ (the repo's SpGEMM target) in fp64. MH_spgemm(A, B, C) is a general
// C = A·B where only B is tiled/masked, so passing B = transpose(A) yields A·Aᵀ with no
// change to the kernels (verified against the source). A and B live on the device across
// timed iterations; the timed region is MH-SpGEMM's own phase-summed runtime (mem-alloc +
// mask + binning + symbolic + numeric), matching how the upstream harness reports it.
//
// The MH_spgemm orchestration below is copied from external/MH-SpGEMM/src/main.cu (which
// also defines main() and cannot be linked) with the per-iteration printf removed. Built
// into libconnectome_mh_spgemm.so; compile with CSR.cu Tool.cu Timing.cpp utils.cpp.
#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include <cuda_fp16.h>  // VALUE_TYPE may be __half (fp16 build); harmless otherwise

#include "common.h"
#include "CSR.h"
#include "MH_spgemm.cuh"
#include "Timing.h"
#include "Tool.h"
#include "utils.h"

// ---- MH_spgemm: verbatim from external/MH-SpGEMM/src/main.cu (printf removed) ----
static void MH_spgemm(const CSR &A, CSR &B, CSR &C, Timing &Timing, Tool &tools) {
  double t0;
  t0 = fast_clock_time();
  C.M = A.M;
  C.N = B.N;
  tools.allocate(B, C);
  CHECK_ERROR(cudaDeviceSynchronize());
  Timing.mem_alloc = (fast_clock_time() - t0) * 1000;

  t0 = fast_clock_time();
  Form_mask_matrix_B(A, B, C, tools);
  CHECK_ERROR(cudaDeviceSynchronize());
  Timing.Form_mask_matrix_B = (fast_clock_time() - t0) * 1000;

  t0 = fast_clock_time();
  binning<2>(tools, tools.d_bins_C, C.d_ptr, C.M);
  CHECK_ERROR(cudaDeviceSynchronize());
  Timing.symbolic_binning = (fast_clock_time() - t0) * 1000;

  t0 = fast_clock_time();
  Calculate_C_nnz(A, B, C, tools);
  CHECK_ERROR(cudaDeviceSynchronize());
  Timing.Calculate_C_nnz = (fast_clock_time() - t0) * 1000;

  t0 = fast_clock_time();
  binning<4>(tools, tools.d_bins_C, C.d_ptr, C.M);
  CHECK_ERROR(cudaDeviceSynchronize());
  Timing.numeric_binning = (fast_clock_time() - t0) * 1000;

#if ADAPTIVE_GROUPING
  t0 = fast_clock_time();
  int GS = (A.M + 127) / 128;
  k_calculate_flop_tmp<<<GS, 512>>>(A.d_ptr, A.d_col, B.d_ptr, A.M, C.d_tileptr);
  int rows = C.M - tools.h_bin_offset[3];
  k_init_group_size<2><<<(rows + 511) / 512, 512>>>(A.d_ptr, C.d_ptr, C.d_tileptr, rows,
                                                    tools.d_bins_C + tools.h_bin_offset[3],
                                                    tools.group_size);
  Timing.Numeric += (fast_clock_time() - t0) * 1000;
#endif

  t0 = fast_clock_time();
  cub::DeviceScan::ExclusiveSum(tools.d_cub_storage, tools.cub_temp_storage, C.d_ptr, C.d_ptr,
                                C.M + 1, 0);
  CHECK_ERROR(cudaMemcpy(tools.count, C.d_ptr + C.M, sizeof(int), cudaMemcpyDeviceToHost));
  C.nnz = *tools.count;
  CHECK_ERROR(cudaMalloc(&C.d_col, C.nnz * sizeof(int)));
  CHECK_ERROR(cudaMalloc(&C.d_val, C.nnz * sizeof(VALUE_TYPE)));
  Timing.Malloc_C_col_val = (fast_clock_time() - t0) * 1000;

  t0 = fast_clock_time();
  h_numeric(A, B, C, tools);
  CHECK_ERROR(cudaDeviceSynchronize());
  Timing.Numeric = (fast_clock_time() - t0) * 1000;
}

namespace {
struct Handle {
  CSR A, B;              // A and B = Aᵀ, resident on device across timed iterations
  int m = 0;             // rows of C = A.M
  int c_nnz = 0;         // nnz of the last computed C
  int c_ncols = 0;       // columns of C (= N; equals M for the A·Aᵀ path)
  int* c_ptr = nullptr;  // last computed C, on host, owned by the handle
  int* c_col = nullptr;
  VALUE_TYPE* c_val = nullptr;  // value type is the build-time VALUE_TYPE (float / __half)
};

// Flushed to stderr so a fault is visible even when the process is about to die.
void Diag(const char* where) {
  cudaError_t e = cudaGetLastError();
  std::fprintf(stderr, "[mh-spgemm] fault in %s: cuda='%s'\n", where, cudaGetErrorString(e));
  std::fflush(stderr);
}

void FreeHostC(Handle* h) {
  delete[] h->c_ptr;
  delete[] h->c_col;
  delete[] h->c_val;
  h->c_ptr = nullptr;
  h->c_col = nullptr;
  h->c_val = nullptr;
}
}  // namespace

extern "C" {

// Build A (M×K, column indices ascending) and B = Aᵀ, upload both. C = A·B = A·Aᵀ.
void* cbn_mh_prepare(int M, int K, int A_nnz, const int* A_ptr, const int* A_col,
                     const VALUE_TYPE* A_val) {
  auto* h = new Handle();
  try {
    h->m = M;
    h->A.alloc(M, K, A_nnz);
    std::copy(A_ptr, A_ptr + M + 1, h->A.ptr);
    std::copy(A_col, A_col + A_nnz, h->A.col);
    std::copy(A_val, A_val + A_nnz, h->A.val);
    matrix_transposition(h->A, h->B);  // B = Aᵀ (K×M), columns ascending
    warm_gpu();
    h->A.H2D();
    h->B.H2D();
    return h;
  } catch (...) {  // never let a CUDA/alloc failure cross the C ABI into terminate()
    Diag("prepare");
    delete h;
    return nullptr;
  }
}

// General C = A·B with A (M×K) and B (K×N) both given (no transpose). This is the RSNN
// spike-delivery path: A = W (connectivity), B = S (a batch of sparse spike vectors,
// K×N), C = delivered current. Both operands must have ascending column indices per row.
void* cbn_mh_prepare_ab(int M, int K, int N, int A_nnz, const int* A_ptr, const int* A_col,
                        const VALUE_TYPE* A_val, int B_nnz, const int* B_ptr, const int* B_col,
                        const VALUE_TYPE* B_val) {
  auto* h = new Handle();
  try {
    h->m = M;
    h->A.alloc(M, K, A_nnz);
    std::copy(A_ptr, A_ptr + M + 1, h->A.ptr);
    std::copy(A_col, A_col + A_nnz, h->A.col);
    std::copy(A_val, A_val + A_nnz, h->A.val);
    h->B.alloc(K, N, B_nnz);
    std::copy(B_ptr, B_ptr + K + 1, h->B.ptr);
    std::copy(B_col, B_col + B_nnz, h->B.col);
    std::copy(B_val, B_val + B_nnz, h->B.val);
    warm_gpu();
    h->A.H2D();
    h->B.H2D();
    return h;
  } catch (...) {  // never let a CUDA/alloc failure cross the C ABI into terminate()
    Diag("prepare_ab");
    delete h;
    return nullptr;
  }
}

// Run the full MH-SpGEMM pipeline `iters` times; return the mean phase-summed runtime (ms)
// and keep the last C on the host for copy-out. Mirrors the upstream per-iter cleanup.
int cbn_mh_compute_timed(void* handle, int iters, double* out_ms, int* out_nnz) {
  auto* h = static_cast<Handle*>(handle);
  if (iters < 1) iters = 1;
  double total = 0.0;
  // The whole body is guarded: the vendored kernel can fault on degenerate operands, and a
  // bad C.nnz can make the host copy-out throw std::bad_alloc. Returning an error code (never
  // aborting) lets the caller record the case as failed instead of killing the process.
  for (int it = 0; it < iters; ++it) {
    CSR C;
    Timing timing;
    Tool tools;
    try {
      MH_spgemm(h->A, h->B, C, timing, tools);
      total += timing.getTotal();
      if (it == iters - 1) {
        if (C.nnz < 0) throw std::exception();
        C.D2H();  // allocates + fills host C.ptr/col/val
        FreeHostC(h);
        h->c_nnz = C.nnz;
        h->c_ncols = C.N;
        h->c_ptr = new int[C.M + 1];
        h->c_col = new int[C.nnz];
        h->c_val = new VALUE_TYPE[C.nnz];
        std::copy(C.ptr, C.ptr + C.M + 1, h->c_ptr);
        std::copy(C.col, C.col + C.nnz, h->c_col);
        std::copy(C.val, C.val + C.nnz, h->c_val);
      }
      C.d_release_csr();
      cudaFree(C.d_tileptr);
      tools.release();
      h->B.d_release_tile();  // Form_mask_matrix_B reallocates B's tiles every call
    } catch (...) {
      Diag("compute_timed");  // report the CUDA error, then recover instead of aborting
      return -1;
    }
  }
  *out_ms = total / iters;
  *out_nnz = h->c_nnz;
  return static_cast<int>(cudaGetLastError());
}

int cbn_mh_result_nnz(void* handle) { return static_cast<Handle*>(handle)->c_nnz; }
int cbn_mh_result_cols(void* handle) { return static_cast<Handle*>(handle)->c_ncols; }

// Copy the last computed C (host CSR, fp64) out for oracle validation.
void cbn_mh_copy_out(void* handle, int* c_ptr, int* c_col, VALUE_TYPE* c_val) {
  auto* h = static_cast<Handle*>(handle);
  std::copy(h->c_ptr, h->c_ptr + h->m + 1, c_ptr);
  std::copy(h->c_col, h->c_col + h->c_nnz, c_col);
  std::copy(h->c_val, h->c_val + h->c_nnz, c_val);
}

void cbn_mh_free(void* handle) {
  auto* h = static_cast<Handle*>(handle);
  FreeHostC(h);
  delete h;  // A, B destructors free their host + device CSR arrays
}
}  // extern "C"
