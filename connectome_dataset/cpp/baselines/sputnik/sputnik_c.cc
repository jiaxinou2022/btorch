// C-ABI wrapper around Sputnik SpMM for the out-of-tree Python provider (loaded via
// ctypes). Device buffers persist across compute() calls so the benchmark times only the
// kernel, not host<->device transfers. Built into libconnectome_sputnik.so.
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

namespace sputnik {
cudaError_t CudaSpmm(int m, int k, int n, int nonzeros, const int* row_indices, const float* values,
                     const int* row_offsets, const int* column_indices, const float* dense_matrix,
                     float* output_matrix, cudaStream_t stream);
}

namespace {
struct Handle {
  int m = 0, k = 0, n = 0, nnz = 0;
  int *row_offsets = nullptr, *col_indices = nullptr, *row_indices = nullptr;
  float *values = nullptr, *x = nullptr, *y = nullptr;
};

template <class T>
T* Upload(const T* host, std::size_t count) {
  T* d = nullptr;
  if (cudaMalloc(&d, count * sizeof(T)) != cudaSuccess) return nullptr;
  cudaMemcpy(d, host, count * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
}  // namespace

extern "C" {

// row_indices is the load-balancing swizzle, computed host-side by the caller.
void* cbn_sputnik_prepare(int m, int k, int n, int nnz, const int* row_offsets,
                          const int* col_indices, const float* values, const int* row_indices) {
  auto* h = new Handle();
  h->m = m; h->k = k; h->n = n; h->nnz = nnz;
  h->row_offsets = Upload(row_offsets, m + 1);
  h->col_indices = Upload(col_indices, nnz);
  h->values = Upload(values, nnz);
  h->row_indices = Upload(row_indices, m);
  std::vector<float> ones(static_cast<std::size_t>(k) * n, 1.0f);
  h->x = Upload(ones.data(), ones.size());
  cudaMalloc(&h->y, static_cast<std::size_t>(m) * n * sizeof(float));
  return h;
}

// Run the kernel and synchronize, so the caller's timing reflects real compute.
int cbn_sputnik_compute(void* handle) {
  auto* h = static_cast<Handle*>(handle);
  cudaError_t e = sputnik::CudaSpmm(h->m, h->k, h->n, h->nnz, h->row_indices, h->values,
                                    h->row_offsets, h->col_indices, h->x, h->y, /*stream=*/0);
  if (e != cudaSuccess) return static_cast<int>(e);
  return static_cast<int>(cudaDeviceSynchronize());
}

// Launch with caller-owned device operands on the caller's stream. This is the
// graph-capturable entry point used by the end-to-end RSNN benchmark: it does
// not allocate, synchronize, or copy through host memory.
int cbn_sputnik_compute_device(void* handle, const float* x_device,
                               float* y_device, void* stream) {
  auto* h = static_cast<Handle*>(handle);
  cudaError_t e = sputnik::CudaSpmm(
      h->m, h->k, h->n, h->nnz, h->row_indices, h->values, h->row_offsets,
      h->col_indices, x_device, y_device,
      reinterpret_cast<cudaStream_t>(stream));
  return static_cast<int>(e);
}

// Copy the m x n result to host (for oracle validation, off the timed path).
void cbn_sputnik_copy_out(void* handle, float* y_host) {
  auto* h = static_cast<Handle*>(handle);
  cudaMemcpy(y_host, h->y, static_cast<std::size_t>(h->m) * h->n * sizeof(float), cudaMemcpyDeviceToHost);
}

void cbn_sputnik_free(void* handle) {
  auto* h = static_cast<Handle*>(handle);
  cudaFree(h->row_offsets); cudaFree(h->col_indices); cudaFree(h->row_indices);
  cudaFree(h->values); cudaFree(h->x); cudaFree(h->y);
  delete h;
}
}  // extern "C"
