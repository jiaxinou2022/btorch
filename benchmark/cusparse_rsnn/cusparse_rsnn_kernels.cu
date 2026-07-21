#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;

__global__ void lif_step_kernel(
    const float* __restrict__ input,
    float* __restrict__ v,
    const float* __restrict__ psc,
    float* __restrict__ spikes,
    int count,
    float dt,
    float tau_mem,
    float v_threshold,
    float v_reset,
    float c_m) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const float current = psc[index] + input[index];
    const float v_pre =
        v[index] + dt * (-(v[index] - v_reset) / tau_mem + current / c_m);
    const float spike = v_pre >= v_threshold ? 1.0f : 0.0f;
    v[index] = v_pre - (v_threshold - v_reset) * spike;
    spikes[index] = spike;
}

__global__ void psc_step_kernel(
    float* __restrict__ psc,
    const float* __restrict__ recurrent,
    int count,
    float decay) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        psc[index] = psc[index] * decay + recurrent[index];
    }
}

}  // namespace

void launch_lif_step(
    const float* input,
    float* v,
    const float* psc,
    float* spikes,
    int count,
    float dt,
    float tau_mem,
    float v_threshold,
    float v_reset,
    float c_m,
    cudaStream_t stream) {
    const int blocks = (count + kThreads - 1) / kThreads;
    lif_step_kernel<<<blocks, kThreads, 0, stream>>>(
        input,
        v,
        psc,
        spikes,
        count,
        dt,
        tau_mem,
        v_threshold,
        v_reset,
        c_m);
}

void launch_psc_step(
    float* psc,
    const float* recurrent,
    int count,
    float decay,
    cudaStream_t stream) {
    const int blocks = (count + kThreads - 1) / kThreads;
    psc_step_kernel<<<blocks, kThreads, 0, stream>>>(
        psc,
        recurrent,
        count,
        decay);
}
