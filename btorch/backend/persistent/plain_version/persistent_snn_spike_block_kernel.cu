#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <math_constants.h>

namespace cg = cooperative_groups;

namespace {

constexpr int kFanoutThreshold = 256;
constexpr int kLaneRowThreshold = 16;
constexpr int kSegmentSize = 1024;
constexpr unsigned kFullWarpMask = 0xffffffffu;

__device__ __forceinline__ void process_edge_range(
    const int* graph_indices,
    const float* graph_weight,
    float* psc,
    int start,
    int end,
    int lane) {
    for (int edge_base = start; edge_base < end; edge_base += 32) {
        const int edge = edge_base + lane;
        const bool valid = edge < end;
        if (valid) {
            const int post = graph_indices[edge];
            atomicAdd(psc + post, graph_weight[edge]);
        }
    }
}

template <bool ReturnDense>
__global__ void persistent_snn_spike_block_kernel(
    const int* __restrict__ event_offsets,
    const int* __restrict__ event_indices,
    const float* __restrict__ event_values,
    bool has_event_values,
    const int* __restrict__ graph_indptr,
    const int* __restrict__ graph_indices,
    const float* __restrict__ graph_weight,
    float* __restrict__ v,
    float* __restrict__ psc,
    float* __restrict__ dense_spikes,
    float* __restrict__ input_current,
    int* __restrict__ task_queue_batch,
    int* __restrict__ task_queue_start,
    int* __restrict__ task_queue_end_or_mask,
    int* __restrict__ task_counts,
    int* __restrict__ work_counters,
    int* __restrict__ event_counts,
    int* __restrict__ event_indices_full,
    bool return_events,
    int t_steps,
    int batch_size,
    int n_neuron,
    int queue_capacity,
    float dt,
    float tau_mem,
    float tau_syn,
    float v_threshold,
    float v_reset,
    float c_m) {
    cg::grid_group grid = cg::this_grid();
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int lane = global_tid & 31;
    const float decay = expf(-dt / tau_syn);
    const float reset_delta = v_threshold - v_reset;

    for (int t = 0; t < t_steps; ++t) {
        if (global_tid == 0) {
            task_counts[0] = 0;
            task_counts[1] = 0;
            work_counters[0] = 0;
            work_counters[1] = 0;
        }
        for (int b = 0; b < batch_size; ++b) {
            const int bucket = t * batch_size + b;
            const int start = event_offsets[bucket];
            const int end = event_offsets[bucket + 1];
            for (int event = start + global_tid; event < end; event += stride) {
                const int pre = event_indices[event];
                if (pre >= 0 && pre < n_neuron) {
                    const float value =
                        has_event_values ? event_values[event] : 1.0f;
                    atomicAdd(input_current + b * n_neuron + pre, value);
                }
            }
        }
        grid.sync();

        for (int b = 0; b < batch_size; ++b) {
            const int batch_base = b * n_neuron;
            for (int n = global_tid; n < n_neuron; n += stride) {
                const int cell = batch_base + n;
                const float current = psc[cell] + input_current[cell];
                input_current[cell] = 0.0f;
                const float v_pre = v[cell] +
                    dt * (-(v[cell] - v_reset) / tau_mem + current / c_m);
                const bool fired = v_pre >= v_threshold;
                const float spike = fired ? 1.0f : 0.0f;
                v[cell] = v_pre - reset_delta * spike;
                if constexpr (ReturnDense) {
                    dense_spikes[(t * batch_size + b) * n_neuron + n] = spike;
                }
                psc[cell] *= decay;

                const int edge_start = graph_indptr[n];
                const int edge_end = graph_indptr[n + 1];
                const int degree = edge_end - edge_start;
                const bool block_spike = fired && degree < kFanoutThreshold;
                const unsigned warp_active = __activemask();
                const unsigned spike_mask =
                    __ballot_sync(warp_active, block_spike);

                if (lane == 0 && spike_mask != 0) {
                    const int task = atomicAdd(task_counts + 1, 1);
                    const int slot = queue_capacity - 1 - task;
                    task_queue_batch[slot] = b;
                    task_queue_start[slot] = n;
                    task_queue_end_or_mask[slot] =
                        static_cast<int>(spike_mask);
                }

                if (fired && degree >= kFanoutThreshold) {
                    const int segment_count =
                        (degree + kSegmentSize - 1) / kSegmentSize;
                    const int first_task =
                        atomicAdd(task_counts, segment_count);
                    for (int segment = 0; segment < segment_count; ++segment) {
                        const int task = first_task + segment;
                        const int segment_start =
                            edge_start + segment * kSegmentSize;
                        task_queue_batch[task] = b;
                        task_queue_start[task] = segment_start;
                        task_queue_end_or_mask[task] =
                            min(segment_start + kSegmentSize, edge_end);
                    }
                }

                if (fired && return_events) {
                    const int bucket = t * batch_size + b;
                    const int rank = atomicAdd(event_counts + bucket, 1);
                    if (rank < n_neuron) {
                        event_indices_full[bucket * n_neuron + rank] = n;
                    }
                }
            }
        }
        grid.sync();

        const int segment_count = task_counts[0];
        while (true) {
            int task = 0;
            if (lane == 0) {
                task = atomicAdd(work_counters, 1);
            }
            task = __shfl_sync(kFullWarpMask, task, 0);
            if (task >= segment_count) {
                break;
            }
            const int b = task_queue_batch[task];
            process_edge_range(
                graph_indices,
                graph_weight,
                psc + b * n_neuron,
                task_queue_start[task],
                task_queue_end_or_mask[task],
                lane);
        }

        const int block_count = task_counts[1];
        while (true) {
            int task = 0;
            if (lane == 0) {
                task = atomicAdd(work_counters + 1, 1);
            }
            task = __shfl_sync(kFullWarpMask, task, 0);
            if (task >= block_count) {
                break;
            }
            const int slot = queue_capacity - 1 - task;
            const int b = task_queue_batch[slot];
            const int block_start = task_queue_start[slot];
            const unsigned spike_mask =
                static_cast<unsigned>(task_queue_end_or_mask[slot]);
            const bool lane_fired =
                (spike_mask & (1u << lane)) != 0;
            const int neuron_id = block_start + lane;
            const int relative_id = neuron_id & 31;
            const int lane_pre = block_start + relative_id;
            const int lane_edge_start =
                lane_fired ? graph_indptr[lane_pre] : 0;
            const int lane_edge_end =
                lane_fired ? graph_indptr[lane_pre + 1] : 0;
            const int lane_degree = lane_edge_end - lane_edge_start;
            // Very short rows are cheaper when their owning lane walks them
            // directly. Longer rows retain full-warp traversal below.
            const bool lane_row =
                lane_fired && lane_degree <= kLaneRowThreshold;
            unsigned warp_row_mask = __ballot_sync(
                kFullWarpMask,
                lane_fired && !lane_row);

            if (lane_row) {
                for (int edge = lane_edge_start; edge < lane_edge_end; ++edge) {
                    const int post = graph_indices[edge];
                    atomicAdd(
                        psc + b * n_neuron + post,
                        graph_weight[edge]);
                }
            }
            while (warp_row_mask != 0) {
                const int spike_lane = __ffs(warp_row_mask) - 1;
                const int pre = block_start + spike_lane;
                process_edge_range(
                    graph_indices,
                    graph_weight,
                    psc + b * n_neuron,
                    graph_indptr[pre],
                    graph_indptr[pre + 1],
                    lane);
                warp_row_mask &= warp_row_mask - 1;
            }
        }
        grid.sync();
    }
}

}  // namespace

void launch_persistent_snn_spike_block_kernel(
    const int* event_offsets,
    const int* event_indices,
    const float* event_values,
    bool has_event_values,
    const int* graph_indptr,
    const int* graph_indices,
    const float* graph_weight,
    float* v,
    float* psc,
    float* dense_spikes,
    float* input_current,
    int* task_queue_batch,
    int* task_queue_start,
    int* task_queue_end_or_mask,
    int* task_counts,
    int* work_counters,
    int* event_counts,
    int* event_indices_full,
    bool return_dense,
    bool return_events,
    int t_steps,
    int batch_size,
    int n_neuron,
    int queue_capacity,
    float dt,
    float tau_mem,
    float tau_syn,
    float v_threshold,
    float v_reset,
    float c_m,
    int grid_dim,
    int block_dim,
    cudaStream_t stream) {
    void* args[] = {
        &event_offsets,
        &event_indices,
        &event_values,
        &has_event_values,
        &graph_indptr,
        &graph_indices,
        &graph_weight,
        &v,
        &psc,
        &dense_spikes,
        &input_current,
        &task_queue_batch,
        &task_queue_start,
        &task_queue_end_or_mask,
        &task_counts,
        &work_counters,
        &event_counts,
        &event_indices_full,
        &return_events,
        &t_steps,
        &batch_size,
        &n_neuron,
        &queue_capacity,
        &dt,
        &tau_mem,
        &tau_syn,
        &v_threshold,
        &v_reset,
        &c_m,
    };
    const void* kernel = return_dense
        ? reinterpret_cast<void*>(persistent_snn_spike_block_kernel<true>)
        : reinterpret_cast<void*>(persistent_snn_spike_block_kernel<false>);
    cudaLaunchCooperativeKernel(
        kernel,
        grid_dim,
        block_dim,
        args,
        0,
        stream);
}

int persistent_snn_spike_block_max_active_blocks_per_sm(
    int block_dim, bool return_dense, bool return_events) {
    int active_blocks = 0;
    const void* kernel = return_dense
        ? reinterpret_cast<void*>(persistent_snn_spike_block_kernel<true>)
        : reinterpret_cast<void*>(persistent_snn_spike_block_kernel<false>);
    const cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks,
        kernel,
        block_dim,
        0);
    if (error != cudaSuccess) {
        return 0;
    }
    return active_blocks;
}
