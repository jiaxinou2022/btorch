// VDHA — Vector-Driven Hash Aggregation for SpMSpV (y = A·x, x sparse) on GPUs.
//
// Reimplemented from Li, Pan, Qu, Zhang, "VDHA: Vector-Driven Hash Aggregation for
// Sparse Matrix-Sparse Vector Multiplication on GPUs" (PPoPP '26). This is the
// weighted, vector-driven (column-selection) formulation: for each nonzero (c, s) of x
// we scale CSC column c of A by the scalar s and scatter-accumulate its entries into y.
//
// The three ablation components of the paper are here:
//   * hash aggregation — each CTA owns a private shared-memory hash table (row-index
//     keyed, linear probing with a fixed odd stride, atomicCAS to claim a slot) that
//     locally combines partial products before a coalesced flush into global y;
//   * long-column split — a column longer than SPLIT_SIZE is cut into ≤SPLIT_SIZE
//     segments so no CTA is overloaded by a skewed hub column;
//   * segment reorder — segments are sorted by the row index of their first nonzero so
//     CTAs that touch nearby rows run together and their flushes coalesce.
//
// The vector-processing pass (classify / split / sort) depends only on x and is done
// once host-side in prepare(); the timed compute() is the device aggregation. Two paper
// refinements are intentionally omitted as pure performance tweaks that do not affect
// correctness: block-mapped grouping of many short columns per CTA, and cp.async
// double-buffered fetch. Built into libconnectome_vdha.so and reached via ctypes.
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Stored matrix-value type: fp32 by default, __half for the fp16 build (-DVDHA_VAL=__half).
// Accumulation and the sparse-vector scalar stay fp32 regardless, so fp16 only halves the
// matrix value-memory traffic (the index arrays and output are unchanged).
#ifndef VDHA_VAL
#define VDHA_VAL float
#endif

namespace {

constexpr int kTableSize = 2048;    // shared hash table entries per CTA (16 KB: 4B key + 4B val)
constexpr int kSplitSize = 256;     // max nonzeros per column segment (= threads per CTA)
constexpr int kThreads = 256;
constexpr int kProbeStride = 7;     // odd => coprime with 2048 => probing visits every slot
constexpr int kFallbackIter = 256;  // probe cap before falling back to a direct global atomic

__global__ void VdhaAggregate(const int* __restrict__ seg_off, const int* __restrict__ seg_len,
                              const float* __restrict__ seg_scalar, const int* __restrict__ csc_row,
                              const VDHA_VAL* __restrict__ csc_val, float* __restrict__ y) {
  __shared__ int hkey[kTableSize];
  __shared__ float hval[kTableSize];

  const int seg = blockIdx.x;
  const int off = seg_off[seg];
  const int len = seg_len[seg];
  const float s = seg_scalar[seg];

  for (int i = threadIdx.x; i < kTableSize; i += blockDim.x) {
    hkey[i] = -1;
    hval[i] = 0.0f;
  }
  __syncthreads();

  for (int j = threadIdx.x; j < len; j += blockDim.x) {
    const int row = csc_row[off + j];
    const float v = static_cast<float>(csc_val[off + j]) * s;  // fp16 loads widen to fp32
    int h = row % kTableSize;
    bool done = false;
    for (int cnt = 0; cnt < kFallbackIter; ++cnt) {
      const int old = atomicCAS(&hkey[h], -1, row);
      if (old == -1 || old == row) {
        atomicAdd(&hval[h], v);
        done = true;
        break;
      }
      h += kProbeStride;
      if (h >= kTableSize) h -= kTableSize;
    }
    if (!done) atomicAdd(&y[row], v);  // probe chain too long: skip the table
  }
  __syncthreads();

  for (int i = threadIdx.x; i < kTableSize; i += blockDim.x) {
    if (hkey[i] != -1) atomicAdd(&y[hkey[i]], hval[i]);  // coalesced-ish bulk flush
  }
}

__global__ void VdhaAggregateDense(
    const int* __restrict__ seg_off, const int* __restrict__ seg_len,
    const int* __restrict__ seg_col, const int* __restrict__ csc_row,
    const VDHA_VAL* __restrict__ csc_val, const float* __restrict__ x,
    float* __restrict__ y) {
  __shared__ int hkey[kTableSize];
  __shared__ float hval[kTableSize];

  const int seg = blockIdx.x;
  const int off = seg_off[seg];
  const int len = seg_len[seg];
  const float s = x[seg_col[seg]];

  for (int i = threadIdx.x; i < kTableSize; i += blockDim.x) {
    hkey[i] = -1;
    hval[i] = 0.0f;
  }
  __syncthreads();

  for (int j = threadIdx.x; j < len; j += blockDim.x) {
    const int row = csc_row[off + j];
    const float v = static_cast<float>(csc_val[off + j]) * s;
    int h = row % kTableSize;
    bool done = false;
    for (int cnt = 0; cnt < kFallbackIter; ++cnt) {
      const int old = atomicCAS(&hkey[h], -1, row);
      if (old == -1 || old == row) {
        atomicAdd(&hval[h], v);
        done = true;
        break;
      }
      h += kProbeStride;
      if (h >= kTableSize) h -= kTableSize;
    }
    if (!done) atomicAdd(&y[row], v);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < kTableSize; i += blockDim.x) {
    if (hkey[i] != -1) atomicAdd(&y[hkey[i]], hval[i]);
  }
}

struct Handle {
  int n = 0;              // rows of A = length of y
  int num_segments = 0;
  int* d_seg_off = nullptr;
  int* d_seg_len = nullptr;
  float* d_seg_scalar = nullptr;
  int* d_seg_col = nullptr;
  int* d_csc_row = nullptr;
  VDHA_VAL* d_csc_val = nullptr;
  float* d_y = nullptr;
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

void cbn_vdha_free(void* handle);

// A is CSC: col_ptr[num_cols+1], csc_row[nnz], csc_val[nnz]; x is sparse: x_idx/x_val[x_nnz].
// Builds the reordered segment work-list host-side (once, since x is fixed across timed
// iterations) and uploads everything. Returns an opaque handle or nullptr on failure.
void* cbn_vdha_prepare(int n, int num_cols, int nnz, const int* col_ptr, const int* csc_row,
                       const VDHA_VAL* csc_val, int x_nnz, const int* x_idx, const float* x_val) {
  auto* h = new Handle();
  h->n = n;

  std::vector<int> seg_off, seg_len;
  std::vector<float> seg_scalar;
  std::vector<int> seg_firstrow;
  seg_off.reserve(x_nnz);
  for (int k = 0; k < x_nnz; ++k) {
    const int c = x_idx[k];
    if (c < 0 || c >= num_cols) continue;
    const int cstart = col_ptr[c];
    const int clen = col_ptr[c + 1] - cstart;
    for (int s = 0; s < clen; s += kSplitSize) {
      const int off = cstart + s;
      seg_off.push_back(off);
      seg_len.push_back(std::min(kSplitSize, clen - s));
      seg_scalar.push_back(x_val[k]);
      seg_firstrow.push_back(csc_row[off]);
    }
  }

  const int m = static_cast<int>(seg_off.size());
  h->num_segments = m;

  // Reorder: sort segments by the row index of their first nonzero (locality for flush).
  std::vector<int> order(m);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return seg_firstrow[a] < seg_firstrow[b]; });

  std::vector<int> off_r(m), len_r(m);
  std::vector<float> scal_r(m);
  for (int i = 0; i < m; ++i) {
    off_r[i] = seg_off[order[i]];
    len_r[i] = seg_len[order[i]];
    scal_r[i] = seg_scalar[order[i]];
  }

  h->d_csc_row = Upload(csc_row, nnz ? nnz : 1);
  h->d_csc_val = Upload(csc_val, nnz ? nnz : 1);
  h->d_seg_off = Upload(off_r.data(), m ? m : 1);
  h->d_seg_len = Upload(len_r.data(), m ? m : 1);
  h->d_seg_scalar = Upload(scal_r.data(), m ? m : 1);
  cudaMalloc(&h->d_y, static_cast<std::size_t>(n) * sizeof(float));
  if (!h->d_csc_row || !h->d_csc_val || !h->d_y || (m && !h->d_seg_off)) {
    return nullptr;
  }
  return h;
}

// Prepare a fixed all-column work envelope. The values of x remain dynamic and
// are read from caller-owned device memory by cbn_vdha_compute_dense_device.
// A fixed envelope is required because a CUDA graph cannot change its grid
// topology when the spike set changes between recurrent timesteps.
void* cbn_vdha_prepare_dense(int n, int num_cols, int nnz,
                             const int* col_ptr, const int* csc_row,
                             const VDHA_VAL* csc_val) {
  auto* h = new Handle();
  h->n = n;
  std::vector<int> seg_off, seg_len, seg_col, seg_firstrow;
  for (int c = 0; c < num_cols; ++c) {
    const int cstart = col_ptr[c];
    const int clen = col_ptr[c + 1] - cstart;
    for (int s = 0; s < clen; s += kSplitSize) {
      const int off = cstart + s;
      seg_off.push_back(off);
      seg_len.push_back(std::min(kSplitSize, clen - s));
      seg_col.push_back(c);
      seg_firstrow.push_back(csc_row[off]);
    }
  }
  const int count = static_cast<int>(seg_off.size());
  h->num_segments = count;
  std::vector<int> order(count);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return seg_firstrow[a] < seg_firstrow[b];
  });
  std::vector<int> off_r(count), len_r(count), col_r(count);
  for (int i = 0; i < count; ++i) {
    off_r[i] = seg_off[order[i]];
    len_r[i] = seg_len[order[i]];
    col_r[i] = seg_col[order[i]];
  }
  h->d_csc_row = Upload(csc_row, nnz ? nnz : 1);
  h->d_csc_val = Upload(csc_val, nnz ? nnz : 1);
  h->d_seg_off = Upload(off_r.data(), count ? count : 1);
  h->d_seg_len = Upload(len_r.data(), count ? count : 1);
  h->d_seg_col = Upload(col_r.data(), count ? count : 1);
  cudaMalloc(&h->d_y, static_cast<std::size_t>(n) * sizeof(float));
  if (!h->d_csc_row || !h->d_csc_val || !h->d_y ||
      (count && (!h->d_seg_off || !h->d_seg_len || !h->d_seg_col))) {
    cbn_vdha_free(h);
    return nullptr;
  }
  return h;
}

int cbn_vdha_compute_dense_device(void* handle, const float* x_device,
                                  float* y_device, void* stream) {
  auto* h = static_cast<Handle*>(handle);
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  cudaError_t error = cudaMemsetAsync(
      y_device, 0, static_cast<std::size_t>(h->n) * sizeof(float),
      cuda_stream);
  if (error != cudaSuccess || h->num_segments == 0) {
    return static_cast<int>(error);
  }
  VdhaAggregateDense<<<h->num_segments, kThreads, 0, cuda_stream>>>(
      h->d_seg_off, h->d_seg_len, h->d_seg_col, h->d_csc_row,
      h->d_csc_val, x_device, y_device);
  return static_cast<int>(cudaGetLastError());
}

// Run the kernel `iters` times under CUDA events; return mean kernel ms via `out_ms`.
// y is zeroed inside the timed region (accumulation requires a clean output each call).
int cbn_vdha_compute_timed(void* handle, int iters, double* out_ms) {
  auto* h = static_cast<Handle*>(handle);
  if (iters < 1) iters = 1;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int it = 0; it < iters; ++it) {
    cudaMemsetAsync(h->d_y, 0, static_cast<std::size_t>(h->n) * sizeof(float));
    if (h->num_segments > 0) {
      VdhaAggregate<<<h->num_segments, kThreads>>>(h->d_seg_off, h->d_seg_len, h->d_seg_scalar,
                                                   h->d_csc_row, h->d_csc_val, h->d_y);
    }
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  *out_ms = static_cast<double>(ms) / iters;
  return static_cast<int>(cudaGetLastError());
}

// Copy the length-n result vector to host (for oracle validation, off the timed path).
void cbn_vdha_copy_out(void* handle, float* y_host) {
  auto* h = static_cast<Handle*>(handle);
  cudaMemcpy(y_host, h->d_y, static_cast<std::size_t>(h->n) * sizeof(float),
             cudaMemcpyDeviceToHost);
}

void cbn_vdha_free(void* handle) {
  auto* h = static_cast<Handle*>(handle);
  cudaFree(h->d_seg_off);
  cudaFree(h->d_seg_len);
  cudaFree(h->d_seg_scalar);
  cudaFree(h->d_seg_col);
  cudaFree(h->d_csc_row);
  cudaFree(h->d_csc_val);
  cudaFree(h->d_y);
  delete h;
}
}  // extern "C"
