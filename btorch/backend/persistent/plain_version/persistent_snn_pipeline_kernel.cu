#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cuda_runtime.h>

#include <cstdint>

namespace cg = cooperative_groups;

namespace {

constexpr int kEdgesPerTask = 1024;
constexpr uint32_t kFragmentMask = 0xffffu;
constexpr uint32_t kEpochCount = 0xffffu;
constexpr int kInitialBackoffCycles = 32;
constexpr int kMaximumBackoffCycles = 1024;

enum PipelineDebugCounter : int {
    kTicketWaitTail = 0,
    kTicketWaitReady = 1,
    kInvalidFinalTickets = 2,
    kProcessedTasks = 3,
};

template <typename T>
__device__ __forceinline__ T device_load_acquire(T* ptr) {
    cuda::atomic_ref<T, cuda::thread_scope_device> ref(*ptr);
    return ref.load(cuda::memory_order_acquire);
}

__device__ __forceinline__ void state_store_release(
    uint32_t* ptr, uint32_t state) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> ref(*ptr);
    ref.store(state, cuda::memory_order_release);
}

template <bool ReturnDense>
__global__ void persistent_snn_kernel(
    const int* __restrict__ event_offsets,
    const int* __restrict__ event_indices,
    const float* __restrict__ event_values,
    bool has_event_values,
    const int* __restrict__ graph_indptr,
    const int* __restrict__ graph_indices,
    const float* __restrict__ graph_weight,
    float* __restrict__ v,
    float* __restrict__ psc,
    float* __restrict__ recurrent_delta_0,
    float* __restrict__ recurrent_delta_1,
    float* __restrict__ dense_spikes,
    float* __restrict__ input_current,
    int* __restrict__ spike_queue_neuron,
    uint32_t* __restrict__ spike_queue_state,
    int* __restrict__ queue_tail,
    int* __restrict__ next_ticket,
    int* __restrict__ update_done_blocks,
    int* __restrict__ propagation_done_tasks,
    unsigned long long* __restrict__ debug_counters,
    int* __restrict__ event_counts,
    int* __restrict__ event_indices_full,
    bool return_events,
    int t_steps,
    int n_neuron,
    int update_block_count,
    int consumer_warps_per_block,
    int ticket_chunk,
    uint32_t forward_epoch_base,
    float dt,
    float tau_mem,
    float tau_syn,
    float v_threshold,
    float v_reset,
    float c_m) {
    cg::grid_group grid = cg::this_grid();
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_stride = blockDim.x * gridDim.x;
    const bool is_update_block = blockIdx.x < update_block_count;
    const int update_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int update_stride = update_block_count * blockDim.x;
    const int lane = threadIdx.x & 31;
    const float decay = expf(-dt / tau_syn);
    const float reset_delta = v_threshold - v_reset;

    for (int t = 0; t < t_steps; ++t) {
        float* delta_read =
            (t & 1) ? recurrent_delta_1 : recurrent_delta_0;
        float* delta_write =
            (t & 1) ? recurrent_delta_0 : recurrent_delta_1;
        const uint32_t expected_epoch =
            (forward_epoch_base + static_cast<uint32_t>(t)) % kEpochCount + 1;

        if (global_tid == 0) {
            *queue_tail = 0;
            *next_ticket = 0;
            *update_done_blocks = 0;
            *propagation_done_tasks = 0;
        }
        grid.sync();

        const int event_start = event_offsets[t];
        const int event_end = event_offsets[t + 1];
        for (int event = event_start + global_tid; event < event_end;
             event += global_stride) {
            const int pre = event_indices[event];
            if (pre >= 0 && pre < n_neuron) {
                const float value =
                    has_event_values ? event_values[event] : 1.0f;
                atomicAdd(input_current + pre, value);
            }
        }
        grid.sync();

        if (is_update_block) {
            for (int n = update_tid; n < n_neuron; n += update_stride) {
                const float recurrent = psc[n] + delta_read[n];
                delta_read[n] = 0.0f;
                const float current = recurrent + input_current[n];
                input_current[n] = 0.0f;
                const float v_pre =
                    v[n] +
                    dt * (-(v[n] - v_reset) / tau_mem + current / c_m);
                const bool fired = v_pre >= v_threshold;
                const float spike = fired ? 1.0f : 0.0f;
                v[n] = v_pre - reset_delta * spike;
                psc[n] = recurrent * decay;

                if constexpr (ReturnDense) {
                    dense_spikes[t * n_neuron + n] = spike;
                }

                if (fired) {
                    const int row_start = graph_indptr[n];
                    const int row_end = graph_indptr[n + 1];
                    const int fragment_count =
                        (row_end - row_start + kEdgesPerTask - 1) /
                        kEdgesPerTask;
                    if (fragment_count > 0) {
                        const int first_task =
                            atomicAdd(queue_tail, fragment_count);
                        for (int fragment = 0; fragment < fragment_count;
                             ++fragment) {
                            const int task = first_task + fragment;
                            spike_queue_neuron[task] = n;
                            const uint32_t state =
                                (expected_epoch << 16) |
                                static_cast<uint32_t>(fragment + 1);
                            state_store_release(
                                spike_queue_state + task, state);
                        }
                    }

                    if (return_events) {
                        const int rank = atomicAdd(event_counts + t, 1);
                        if (rank < n_neuron) {
                            event_indices_full[t * n_neuron + rank] = n;
                        }
                    }
                }
            }

            __syncthreads();
            if (threadIdx.x == 0) {
                atomicAdd(update_done_blocks, 1);
            }
            __syncthreads();
        }

        const int warp_in_block = threadIdx.x >> 5;
        const bool active_consumer =
            warp_in_block < consumer_warps_per_block;
        if (active_consumer) {
            int backoff_cycles = kInitialBackoffCycles;
            bool terminate_consumer = false;
            while (!terminate_consumer) {
                int first_task = -1;
                if (lane == 0) {
                    first_task = atomicAdd(next_ticket, ticket_chunk);
                }
                first_task =
                    __shfl_sync(0xffffffffu, first_task, 0);

                for (int ticket_offset = 0;
                     ticket_offset < ticket_chunk;
                     ++ticket_offset) {
                    const int task = first_task + ticket_offset;
                    uint32_t state = 0;
                    bool ready = false;
                    bool invalid_final = false;

                    while (true) {
                        if (lane == 0) {
                            const int tail = device_load_acquire(queue_tail);
                            if (task < tail) {
                                state = device_load_acquire(
                                    spike_queue_state + task);
                                ready =
                                    (state >> 16) == expected_epoch;
                                if (!ready && debug_counters != nullptr) {
                                    atomicAdd(
                                        debug_counters + kTicketWaitReady,
                                        1ull);
                                }
                            } else {
                                const int update_done = device_load_acquire(
                                    update_done_blocks);
                                invalid_final =
                                    update_done == update_block_count;
                                if (invalid_final) {
                                    if (debug_counters != nullptr) {
                                        atomicAdd(
                                            debug_counters +
                                                kInvalidFinalTickets,
                                            1ull);
                                    }
                                } else if (debug_counters != nullptr) {
                                    atomicAdd(
                                        debug_counters + kTicketWaitTail,
                                        1ull);
                                }
                            }

                            if (!ready && !invalid_final) {
                                __nanosleep(backoff_cycles);
                                backoff_cycles = min(
                                    backoff_cycles << 1,
                                    kMaximumBackoffCycles);
                            }
                        }
                        state = __shfl_sync(0xffffffffu, state, 0);
                        ready = __shfl_sync(0xffffffffu, ready, 0);
                        invalid_final = __shfl_sync(
                            0xffffffffu, invalid_final, 0);
                        if (ready || invalid_final) {
                            break;
                        }
                    }

                    if (invalid_final) {
                        terminate_consumer = true;
                        break;
                    }

                    backoff_cycles = kInitialBackoffCycles;
                    const int fragment =
                        static_cast<int>(state & kFragmentMask) - 1;
                    const int neuron = spike_queue_neuron[task];
                    const int row_start = graph_indptr[neuron];
                    const int row_end = graph_indptr[neuron + 1];
                    const int edge_start =
                        row_start + fragment * kEdgesPerTask;
                    const int edge_end =
                        min(edge_start + kEdgesPerTask, row_end);
                    for (int edge = edge_start + lane; edge < edge_end;
                         edge += 32) {
                        const int post = graph_indices[edge];
                        atomicAdd(delta_write + post, graph_weight[edge]);
                    }
                    __syncwarp();
                    if (lane == 0) {
                        atomicAdd(propagation_done_tasks, 1);
                        if (debug_counters != nullptr) {
                            atomicAdd(
                                debug_counters + kProcessedTasks, 1ull);
                        }
                    }
                }
            }
        }

        // Non-consumer warps wait here until this block's ticket holders have
        // processed every valid task assigned to them.
        __syncthreads();

        // The next timestep cannot consume delta_write until every propagation
        // warp has completed all of its recurrent-current atomic additions.
        grid.sync();
    }
}

__global__ void compact_event_indices_kernel(
    const int* __restrict__ event_counts,
    const int* __restrict__ event_offsets,
    const int* __restrict__ event_indices_full,
    int* __restrict__ event_indices,
    int n_buckets,
    int n_neuron) {
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int bucket = global_tid; bucket < n_buckets; bucket += stride) {
        const int count = event_counts[bucket];
        const int src_base = bucket * n_neuron;
        const int dst_base = event_offsets[bucket];
        for (int i = 0; i < count; ++i) {
            event_indices[dst_base + i] = event_indices_full[src_base + i];
        }
    }
}

}  // namespace

void launch_persistent_snn_kernel(
    const int* event_offsets,
    const int* event_indices,
    const float* event_values,
    bool has_event_values,
    const int* graph_indptr,
    const int* graph_indices,
    const float* graph_weight,
    float* v,
    float* psc,
    float* recurrent_delta_0,
    float* recurrent_delta_1,
    float* dense_spikes,
    float* input_current,
    int* spike_queue_neuron,
    uint32_t* spike_queue_state,
    int* queue_tail,
    int* next_ticket,
    int* update_done_blocks,
    int* propagation_done_tasks,
    unsigned long long* debug_counters,
    int* event_counts,
    int* event_indices_full,
    bool return_dense,
    bool return_events,
    int t_steps,
    int n_neuron,
    int update_block_count,
    int consumer_warps_per_block,
    int ticket_chunk,
    uint32_t forward_epoch_base,
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
        &recurrent_delta_0,
        &recurrent_delta_1,
        &dense_spikes,
        &input_current,
        &spike_queue_neuron,
        &spike_queue_state,
        &queue_tail,
        &next_ticket,
        &update_done_blocks,
        &propagation_done_tasks,
        &debug_counters,
        &event_counts,
        &event_indices_full,
        &return_events,
        &t_steps,
        &n_neuron,
        &update_block_count,
        &consumer_warps_per_block,
        &ticket_chunk,
        &forward_epoch_base,
        &dt,
        &tau_mem,
        &tau_syn,
        &v_threshold,
        &v_reset,
        &c_m,
    };
    const void* kernel = return_dense
        ? reinterpret_cast<void*>(persistent_snn_kernel<true>)
        : reinterpret_cast<void*>(persistent_snn_kernel<false>);
    cudaLaunchCooperativeKernel(
        kernel, grid_dim, block_dim, args, 0, stream);
}

int persistent_snn_max_active_blocks_per_sm(
    int block_dim, bool return_dense, bool return_events) {
    int active_blocks = 0;
    const void* kernel = return_dense
        ? reinterpret_cast<void*>(persistent_snn_kernel<true>)
        : reinterpret_cast<void*>(persistent_snn_kernel<false>);
    const cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, kernel, block_dim, 0);
    if (error != cudaSuccess) {
        return 0;
    }
    return active_blocks;
}

void launch_compact_event_indices_kernel(
    const int* event_counts,
    const int* event_offsets,
    const int* event_indices_full,
    int* event_indices,
    int n_buckets,
    int n_neuron,
    int grid_dim,
    int block_dim,
    cudaStream_t stream) {
    compact_event_indices_kernel<<<grid_dim, block_dim, 0, stream>>>(
        event_counts,
        event_offsets,
        event_indices_full,
        event_indices,
        n_buckets,
        n_neuron);
}
