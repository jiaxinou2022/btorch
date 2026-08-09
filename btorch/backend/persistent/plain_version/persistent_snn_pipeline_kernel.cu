#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cuda_runtime.h>

#include <cstdint>

namespace cg = cooperative_groups;

namespace {

#ifndef BTORCH_PIPELINE_LOW_SUBWARP_SIZE
#define BTORCH_PIPELINE_LOW_SUBWARP_SIZE 8
#endif
#ifndef BTORCH_PIPELINE_HIGH_LOW_RATIO
#define BTORCH_PIPELINE_HIGH_LOW_RATIO 2
#endif

constexpr int kEdgesPerTask = 1024;
constexpr int kLowSubwarpSize = BTORCH_PIPELINE_LOW_SUBWARP_SIZE;
constexpr int kLowTasksPerWarp = 32 / kLowSubwarpSize;
constexpr int kHighTasksPerLowGroup = BTORCH_PIPELINE_HIGH_LOW_RATIO;
constexpr uint32_t kFragmentMask = 0xffffu;
constexpr uint32_t kEpochCount = 0xffffu;
constexpr int kInitialBackoffCycles = 32;
constexpr int kMaximumBackoffCycles = 1024;

enum PipelineDebugCounter : int {
    kTicketWaitTail = 0,
    kTicketWaitReady = 1,
    kInvalidFinalTickets = 2,
    kProcessedTasks = 3,
    kHighTaskCount = 4,
    kLowTaskCount = 5,
    kHighProcessed = 6,
    kLowProcessed = 7,
    kHighClaims = 8,
    kLowGroupClaims = 9,
    kLowPartialGroupClaims = 10,
    kQueueOverflow = 11,
    kHighEdges = 12,
    kLowEdges = 13,
};

#ifdef ENABLE_PIPELINE_TIMING
constexpr int kPipelineTimingColumns = 9;

enum PipelineTimestamp : int {
    kTimestampBegin = 0,
    kTimestampFirstPublish = 1,
    kTimestampFirstConsume = 2,
    kTimestampUpdateDone = 3,
    kTimestampPipelineDone = 4,
    kPublicationQ25 = 5,
    kPublicationQ50 = 6,
    kPublicationQ75 = 7,
    kPublicationQ100 = 8,
};

enum PipelineTimingCounter : int {
    kFirstPublishClaimed = 0,
    kFirstConsumeClaimed = 1,
    kPipelineDoneBlocks = 2,
};

__device__ __forceinline__ unsigned long long global_timestamp_ns() {
    unsigned long long timestamp;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp));
    return timestamp;
}
#endif

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
    unsigned long long* __restrict__ debug_counters,
    unsigned long long* __restrict__ timing_stats,
    int* __restrict__ timing_counters,
    int* __restrict__ event_counts,
    int* __restrict__ event_indices_full,
    bool return_events,
    int t_steps,
    int n_neuron,
    int queue_capacity,
    int update_block_count,
    int dedicated_consumer_warps,
    int helper_consumer_warps,
    int ticket_chunk,
    int static_waves,
    int high_fanout_threshold,
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
    const int warp_in_block = threadIdx.x >> 5;
    const int propagation_block_count = gridDim.x - update_block_count;
    const int total_dedicated_warps =
        propagation_block_count * dedicated_consumer_warps;
    const int static_task_count = total_dedicated_warps * static_waves;
    const bool is_dedicated_consumer =
        !is_update_block && warp_in_block < dedicated_consumer_warps;
    const int dedicated_warp_rank =
        (blockIdx.x - update_block_count) * dedicated_consumer_warps +
        warp_in_block;
    const float decay = expf(-dt / tau_syn);
    const float reset_delta = v_threshold - v_reset;
    const bool pipeline_binned = high_fanout_threshold > 0;

    for (int t = 0; t < t_steps; ++t) {
        float* delta_read =
            (t & 1) ? recurrent_delta_1 : recurrent_delta_0;
        float* delta_write =
            (t & 1) ? recurrent_delta_0 : recurrent_delta_1;
        const uint32_t expected_epoch =
            (forward_epoch_base + static_cast<uint32_t>(t)) % kEpochCount + 1;

        if (global_tid == 0) {
            *queue_tail = 0;
            queue_tail[1] = 0;
            *next_ticket = pipeline_binned ? 0 : static_task_count;
            next_ticket[1] = 0;
            *update_done_blocks = 0;
#ifdef ENABLE_PIPELINE_TIMING
            timing_counters[kFirstPublishClaimed] = 0;
            timing_counters[kFirstConsumeClaimed] = 0;
            timing_counters[kPipelineDoneBlocks] = 0;
#endif
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
#ifdef ENABLE_PIPELINE_TIMING
        if (global_tid == 0) {
            timing_stats[t * kPipelineTimingColumns + kTimestampBegin] =
                global_timestamp_ns();
        }
#endif
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
                    const int fanout = row_end - row_start;
                    const bool high_task =
                        !pipeline_binned || fanout >= high_fanout_threshold;
                    if (fanout > 0 && high_task) {
                        const int fragment_count =
                            (fanout + kEdgesPerTask - 1) / kEdgesPerTask;
                        const int first_task =
                            atomicAdd(queue_tail, fragment_count);
                        if (debug_counters != nullptr) {
                            atomicAdd(
                                debug_counters + kHighTaskCount,
                                static_cast<unsigned long long>(fragment_count));
                            atomicAdd(
                                debug_counters + kHighEdges,
                                static_cast<unsigned long long>(fanout));
                            if (first_task + fragment_count +
                                    device_load_acquire(queue_tail + 1) >
                                queue_capacity) {
                                atomicExch(
                                    debug_counters + kQueueOverflow, 1ull);
                            }
                        }
                        for (int fragment = 0; fragment < fragment_count;
                             ++fragment) {
                            const int task = first_task + fragment;
                            spike_queue_neuron[task] = n;
                            const uint32_t state =
                                (expected_epoch << 16) |
                                static_cast<uint32_t>(fragment + 1);
#ifdef ENABLE_PIPELINE_TIMING
                            if (device_load_acquire(
                                    timing_counters +
                                    kFirstPublishClaimed) == 0 &&
                                atomicCAS(
                                    timing_counters + kFirstPublishClaimed,
                                    0,
                                    1) == 0) {
                                timing_stats[
                                    t * kPipelineTimingColumns +
                                    kTimestampFirstPublish] =
                                    global_timestamp_ns();
                            }
#endif
                            state_store_release(
                                spike_queue_state + task, state);
                        }
                    } else if (fanout > 0) {
                        const int low_task = atomicAdd(queue_tail + 1, 1);
                        const int slot = queue_capacity - 1 - low_task;
                        if (debug_counters != nullptr) {
                            atomicAdd(debug_counters + kLowTaskCount, 1ull);
                            atomicAdd(
                                debug_counters + kLowEdges,
                                static_cast<unsigned long long>(fanout));
                            if (device_load_acquire(queue_tail) + low_task + 1 >
                                queue_capacity) {
                                atomicExch(
                                    debug_counters + kQueueOverflow, 1ull);
                            }
                        }
                        spike_queue_neuron[slot] = n;
#ifdef ENABLE_PIPELINE_TIMING
                        if (device_load_acquire(
                                timing_counters + kFirstPublishClaimed) == 0 &&
                            atomicCAS(
                                timing_counters + kFirstPublishClaimed,
                                0,
                                1) == 0) {
                            timing_stats[
                                t * kPipelineTimingColumns +
                                kTimestampFirstPublish] = global_timestamp_ns();
                        }
#endif
                        state_store_release(
                            spike_queue_state + slot,
                            (expected_epoch << 16) | 1u);
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
                const int old = atomicAdd(update_done_blocks, 1);
#ifdef ENABLE_PIPELINE_TIMING
                const int completed = old + 1;
                const int q25_blocks = max(1, (update_block_count + 3) / 4);
                const int q50_blocks = max(1, (update_block_count + 1) / 2);
                const int q75_blocks =
                    max(1, (update_block_count * 3 + 3) / 4);
                if (completed == q25_blocks) {
                    timing_stats[
                        t * kPipelineTimingColumns + kPublicationQ25] =
                        device_load_acquire(queue_tail) +
                        (pipeline_binned
                             ? device_load_acquire(queue_tail + 1)
                             : 0);
                }
                if (completed == q50_blocks) {
                    timing_stats[
                        t * kPipelineTimingColumns + kPublicationQ50] =
                        device_load_acquire(queue_tail) +
                        (pipeline_binned
                             ? device_load_acquire(queue_tail + 1)
                             : 0);
                }
                if (completed == q75_blocks) {
                    timing_stats[
                        t * kPipelineTimingColumns + kPublicationQ75] =
                        device_load_acquire(queue_tail) +
                        (pipeline_binned
                             ? device_load_acquire(queue_tail + 1)
                             : 0);
                }
                if (completed == update_block_count) {
                    timing_stats[
                        t * kPipelineTimingColumns + kPublicationQ100] =
                        device_load_acquire(queue_tail) +
                        (pipeline_binned
                             ? device_load_acquire(queue_tail + 1)
                             : 0);
                }
                if (old == update_block_count - 1) {
                    timing_stats[
                        t * kPipelineTimingColumns +
                        kTimestampUpdateDone] = global_timestamp_ns();
                }
#endif
            }
            __syncthreads();
        }

        const int consumer_warp_limit =
            is_update_block ? helper_consumer_warps
                            : dedicated_consumer_warps;
        const bool active_consumer =
            warp_in_block < consumer_warp_limit;
        if (active_consumer) {
          if (!pipeline_binned) {
            int backoff_cycles = kInitialBackoffCycles;
            int static_wave = 0;
            bool terminate_consumer = false;
            while (!terminate_consumer) {
                int first_task = -1;
                const bool static_claim =
                    is_dedicated_consumer && static_wave < static_waves;
                if (static_claim) {
                    first_task =
                        dedicated_warp_rank +
                        static_wave * total_dedicated_warps;
                    ++static_wave;
                } else if (lane == 0) {
                    first_task = atomicAdd(next_ticket, ticket_chunk);
                }
                first_task =
                    __shfl_sync(0xffffffffu, first_task, 0);
                const int tasks_in_claim =
                    static_claim ? 1 : ticket_chunk;

                for (int ticket_offset = 0;
                     ticket_offset < tasks_in_claim;
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
#ifdef ENABLE_PIPELINE_TIMING
                    if (lane == 0 &&
                        device_load_acquire(
                            timing_counters + kFirstConsumeClaimed) == 0 &&
                        atomicCAS(
                            timing_counters + kFirstConsumeClaimed,
                            0,
                            1) == 0) {
                        timing_stats[
                            t * kPipelineTimingColumns +
                            kTimestampFirstConsume] = global_timestamp_ns();
                    }
#endif
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
                    if (lane == 0 && debug_counters != nullptr) {
                        atomicAdd(
                            debug_counters + kProcessedTasks, 1ull);
                        atomicAdd(
                            debug_counters + kHighProcessed, 1ull);
                        atomicAdd(debug_counters + kHighClaims, 1ull);
                    }
                }
            }
          } else {
            int backoff_cycles = kInitialBackoffCycles;
            int high_tasks_before_low = kHighTasksPerLowGroup;
            bool high_finished = false;
            bool low_finished = false;
            while (!high_finished || !low_finished) {
                const bool process_high =
                    !high_finished &&
                    (high_tasks_before_low > 0 || low_finished);
                if (process_high) {
                    int task = -1;
                    if (lane == 0) {
                        task = atomicAdd(next_ticket, 1);
                    }
                    task = __shfl_sync(0xffffffffu, task, 0);
                    uint32_t state = 0;
                    bool ready = false;
                    bool invalid_final = false;
                    while (true) {
                        if (lane == 0) {
                            const int tail =
                                device_load_acquire(queue_tail);
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
                                invalid_final =
                                    device_load_acquire(update_done_blocks) ==
                                    update_block_count;
                                if (!invalid_final &&
                                    debug_counters != nullptr) {
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
                        high_finished = true;
                        if (lane == 0 && debug_counters != nullptr) {
                            atomicAdd(
                                debug_counters + kInvalidFinalTickets,
                                1ull);
                        }
                        high_tasks_before_low = 0;
                        continue;
                    }

                    backoff_cycles = kInitialBackoffCycles;
#ifdef ENABLE_PIPELINE_TIMING
                    if (lane == 0 &&
                        device_load_acquire(
                            timing_counters + kFirstConsumeClaimed) == 0 &&
                        atomicCAS(
                            timing_counters + kFirstConsumeClaimed,
                            0,
                            1) == 0) {
                        timing_stats[
                            t * kPipelineTimingColumns +
                            kTimestampFirstConsume] = global_timestamp_ns();
                    }
#endif
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
                    if (lane == 0 && debug_counters != nullptr) {
                        atomicAdd(
                            debug_counters + kProcessedTasks, 1ull);
                        atomicAdd(
                            debug_counters + kHighProcessed, 1ull);
                        atomicAdd(debug_counters + kHighClaims, 1ull);
                    }
                    --high_tasks_before_low;
                } else {
                    int base_task = -1;
                    if (lane == 0) {
                        base_task = atomicAdd(
                            next_ticket + 1, kLowTasksPerWarp);
                    }
                    base_task =
                        __shfl_sync(0xffffffffu, base_task, 0);
                    int group_count = 0;
                    bool ready = false;
                    bool invalid_final = false;
                    while (true) {
                        if (lane == 0) {
                            const int tail =
                                device_load_acquire(queue_tail + 1);
                            const bool updates_done =
                                device_load_acquire(update_done_blocks) ==
                                update_block_count;
                            if (base_task < tail) {
                                group_count = updates_done
                                    ? min(kLowTasksPerWarp, tail - base_task)
                                    : (base_task + kLowTasksPerWarp <= tail
                                           ? kLowTasksPerWarp
                                           : 0);
                                ready = group_count > 0;
                                for (int offset = 0;
                                     offset < group_count;
                                     ++offset) {
                                    const int slot =
                                        queue_capacity - 1 -
                                        (base_task + offset);
                                    const uint32_t low_state =
                                        device_load_acquire(
                                            spike_queue_state + slot);
                                    ready = ready &&
                                        (low_state >> 16) == expected_epoch;
                                }
                                if (!ready && group_count > 0 &&
                                    debug_counters != nullptr) {
                                    atomicAdd(
                                        debug_counters + kTicketWaitReady,
                                        1ull);
                                }
                            } else {
                                invalid_final = updates_done;
                            }
                            if (!ready && !invalid_final) {
                                if (debug_counters != nullptr) {
                                    atomicAdd(
                                        debug_counters + kTicketWaitTail,
                                        1ull);
                                }
                                __nanosleep(backoff_cycles);
                                backoff_cycles = min(
                                    backoff_cycles << 1,
                                    kMaximumBackoffCycles);
                            }
                        }
                        group_count =
                            __shfl_sync(0xffffffffu, group_count, 0);
                        ready = __shfl_sync(0xffffffffu, ready, 0);
                        invalid_final = __shfl_sync(
                            0xffffffffu, invalid_final, 0);
                        if (ready || invalid_final) {
                            break;
                        }
                    }
                    if (invalid_final) {
                        low_finished = true;
                        if (lane == 0 && debug_counters != nullptr) {
                            atomicAdd(
                                debug_counters + kInvalidFinalTickets,
                                1ull);
                        }
                        high_tasks_before_low = kHighTasksPerLowGroup;
                        continue;
                    }

                    backoff_cycles = kInitialBackoffCycles;
#ifdef ENABLE_PIPELINE_TIMING
                    if (lane == 0 &&
                        device_load_acquire(
                            timing_counters + kFirstConsumeClaimed) == 0 &&
                        atomicCAS(
                            timing_counters + kFirstConsumeClaimed,
                            0,
                            1) == 0) {
                        timing_stats[
                            t * kPipelineTimingColumns +
                            kTimestampFirstConsume] = global_timestamp_ns();
                    }
#endif
                    const int subgroup = lane / kLowSubwarpSize;
                    const int sublane = lane % kLowSubwarpSize;
                    if (subgroup < group_count) {
                        const int slot =
                            queue_capacity - 1 - (base_task + subgroup);
                        const int neuron = spike_queue_neuron[slot];
                        const int row_start = graph_indptr[neuron];
                        const int row_end = graph_indptr[neuron + 1];
                        for (int edge = row_start + sublane;
                             edge < row_end;
                             edge += kLowSubwarpSize) {
                            const int post = graph_indices[edge];
                            atomicAdd(
                                delta_write + post, graph_weight[edge]);
                        }
                    }
                    if (lane == 0 && debug_counters != nullptr) {
                        atomicAdd(
                            debug_counters + kProcessedTasks,
                            static_cast<unsigned long long>(group_count));
                        atomicAdd(
                            debug_counters + kLowProcessed,
                            static_cast<unsigned long long>(group_count));
                        atomicAdd(
                            debug_counters + kLowGroupClaims, 1ull);
                        if (group_count < kLowTasksPerWarp) {
                            atomicAdd(
                                debug_counters + kLowPartialGroupClaims,
                                1ull);
                        }
                    }
                    high_tasks_before_low = kHighTasksPerLowGroup;
                }
            }
          }
        }

        // Non-consumer warps wait here until this block's ticket holders have
        // processed every valid task assigned to them.
        __syncthreads();

#ifdef ENABLE_PIPELINE_TIMING
        if (threadIdx.x == 0) {
            const int old = atomicAdd(
                timing_counters + kPipelineDoneBlocks, 1);
            if (old == gridDim.x - 1) {
                timing_stats[
                    t * kPipelineTimingColumns +
                    kTimestampPipelineDone] = global_timestamp_ns();
            }
        }
#endif

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
    unsigned long long* debug_counters,
    unsigned long long* timing_stats,
    int* timing_counters,
    int* event_counts,
    int* event_indices_full,
    bool return_dense,
    bool return_events,
    int t_steps,
    int n_neuron,
    int queue_capacity,
    int update_block_count,
    int dedicated_consumer_warps,
    int helper_consumer_warps,
    int ticket_chunk,
    int static_waves,
    int high_fanout_threshold,
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
        &debug_counters,
        &timing_stats,
        &timing_counters,
        &event_counts,
        &event_indices_full,
        &return_events,
        &t_steps,
        &n_neuron,
        &queue_capacity,
        &update_block_count,
        &dedicated_consumer_warps,
        &helper_consumer_warps,
        &ticket_chunk,
        &static_waves,
        &high_fanout_threshold,
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
