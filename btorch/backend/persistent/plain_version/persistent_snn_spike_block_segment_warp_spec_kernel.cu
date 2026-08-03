#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cuda_runtime.h>

#include <math_constants.h>

namespace cg = cooperative_groups;

namespace {

#ifndef BTORCH_BLOCK_EDGE_BUDGET
#define BTORCH_BLOCK_EDGE_BUDGET 0
#endif

#ifndef BTORCH_LONG_SEGMENT_SIZE
#define BTORCH_LONG_SEGMENT_SIZE 1024
#endif

#ifndef BTORCH_TILE_REDUCE_MODE
#define BTORCH_TILE_REDUCE_MODE 0
#endif

#ifndef BTORCH_BLOCK_HASH_AGGREGATION
#define BTORCH_BLOCK_HASH_AGGREGATION 512
#endif

#ifndef BTORCH_BLOCK_HASH_CAPACITY
#define BTORCH_BLOCK_HASH_CAPACITY 512
#endif

#ifndef BTORCH_BLOCK_HASH_MAX_PROBE
#define BTORCH_BLOCK_HASH_MAX_PROBE 4
#endif

#ifndef BTORCH_BLOCK_HASH_MIN_EDGES
#define BTORCH_BLOCK_HASH_MIN_EDGES 256
#endif

#ifndef BTORCH_BLOCK_HASH_USED_SLOTS
#define BTORCH_BLOCK_HASH_USED_SLOTS 1
#endif

#ifndef BTORCH_LONG_WARP_SPEC_MODE
#define BTORCH_LONG_WARP_SPEC_MODE 0
#endif

#ifndef BTORCH_LONG_WARP_SPEC_CHUNK
#define BTORCH_LONG_WARP_SPEC_CHUNK 128
#endif

#ifndef BTORCH_LONG_WARP_SPEC_STAGES
#define BTORCH_LONG_WARP_SPEC_STAGES 3
#endif

#ifndef BTORCH_LONG_WARP_SPEC_THRESHOLD
#define BTORCH_LONG_WARP_SPEC_THRESHOLD 256
#endif

constexpr int kFanoutThreshold = 256;
constexpr int kLaneRowThreshold = 4;
constexpr int kBlockEdgeBudget = BTORCH_BLOCK_EDGE_BUDGET;
constexpr int kSegmentSize = BTORCH_LONG_SEGMENT_SIZE;
constexpr int kTileReduceMode = BTORCH_TILE_REDUCE_MODE;
constexpr int kHashAggregation = BTORCH_BLOCK_HASH_AGGREGATION;
constexpr int kHashSize = BTORCH_BLOCK_HASH_CAPACITY;
constexpr int kHashMaxProbe = BTORCH_BLOCK_HASH_MAX_PROBE;
constexpr int kHashMinEdges = BTORCH_BLOCK_HASH_MIN_EDGES;
constexpr bool kHashUsedSlots = BTORCH_BLOCK_HASH_USED_SLOTS != 0;
constexpr int kLongWarpSpecMode = BTORCH_LONG_WARP_SPEC_MODE;
constexpr int kLongChunkEdges = BTORCH_LONG_WARP_SPEC_CHUNK;
constexpr int kLongStages = BTORCH_LONG_WARP_SPEC_STAGES;
constexpr int kLongPipelineThreshold = BTORCH_LONG_WARP_SPEC_THRESHOLD;
constexpr int kLegacyHashMaxEdges = 96;
constexpr int kWarpsPerBlock = 8;
constexpr int kLongConsumerWarps = 7;
constexpr int kEmptyHashKey = -1;
constexpr unsigned kFullWarpMask = 0xffffffffu;
constexpr int kBlockIndexBits = 24;
constexpr unsigned kBlockIndexMask = (1u << kBlockIndexBits) - 1u;
#ifdef ENABLE_BLOCK_HASH
constexpr int kBlockTaskSpan =
    kHashAggregation > kBlockEdgeBudget
    ? kHashAggregation
    : kBlockEdgeBudget;
#else
constexpr int kBlockTaskSpan = kBlockEdgeBudget;
#endif

static_assert(
    kBlockEdgeBudget == 0
        || kBlockEdgeBudget == 128
        || kBlockEdgeBudget == 256
        || kBlockEdgeBudget == 512
        || kBlockEdgeBudget == 1024);
static_assert(
    kSegmentSize == 128
        || kSegmentSize == 256
        || kSegmentSize == 512
        || kSegmentSize == 1024
        || kSegmentSize == 2048);
static_assert(kTileReduceMode >= 0 && kTileReduceMode <= 2);
static_assert(
    kHashAggregation == 128
        || kHashAggregation == 256
        || kHashAggregation == 512);
static_assert(kHashSize == 128 || kHashSize == 256 || kHashSize == 512);
static_assert(
    kHashMaxProbe == 4 || kHashMaxProbe == 8 || kHashMaxProbe == 16);
static_assert(
    kHashMinEdges == 0
        || kHashMinEdges == 64
        || kHashMinEdges == 128
        || kHashMinEdges == 192
        || kHashMinEdges == 256);
static_assert(
    BTORCH_BLOCK_HASH_USED_SLOTS == 0
        || BTORCH_BLOCK_HASH_USED_SLOTS == 1);
static_assert(kLongWarpSpecMode >= 0 && kLongWarpSpecMode <= 5);
static_assert(
    kLongChunkEdges == 64
        || kLongChunkEdges == 128
        || kLongChunkEdges == 256);
static_assert(kLongStages >= 2 && kLongStages <= 4);
static_assert(
    kLongPipelineThreshold == 128
        || kLongPipelineThreshold == 256
        || kLongPipelineThreshold == 384
        || kLongPipelineThreshold == 512);

#ifdef ENABLE_BLOCK_STATS
constexpr int kBlockStatsColumns = 64;
constexpr int kV4InputEdgesColumn = 13;
constexpr int kV4GlobalAtomicsColumn = 14;
constexpr int kV4ReduceTasksColumn = 15;
constexpr int kV4ReduceEdgesColumn = 16;
constexpr int kHashTasksColumn = 17;
constexpr int kHashWindowsColumn = 18;
constexpr int kHashInputEdgesColumn = 19;
constexpr int kHashFlushAtomicsColumn = 20;
constexpr int kHashFallbackAtomicsColumn = 21;
constexpr int kHashProbeAttemptsColumn = 22;
constexpr int kLongTasksColumn = 23;
constexpr int kLongEdgesColumn = 24;
constexpr int kLongFullSegmentsColumn = 25;
constexpr int kLongPartialSegmentsColumn = 26;
constexpr int kLong128To255Column = 27;
constexpr int kLong256To383Column = 28;
constexpr int kLong384To511Column = 29;
constexpr int kLong512Column = 30;
constexpr int kLongEligibleTasksColumn = 31;
constexpr int kLongEligibleEdgesColumn = 32;
constexpr int kLongChunksProducedColumn = 33;
constexpr int kLongChunksConsumedColumn = 34;
constexpr int kLongProducerIdleColumn = 35;
constexpr int kLongProducerNoSlotColumn = 36;
constexpr int kLongProducerNoConsumerColumn = 37;
constexpr int kLongConsumerTaskWaitColumn = 38;
constexpr int kLongConsumerChunkWaitColumn = 39;
constexpr int kLongMaxActiveConsumersColumn = 40;
constexpr int kLongActiveConsumerCycleKColumn = 41;
constexpr int kLongPipelineCycleKColumn = 42;
constexpr int kLongPathCycleKColumn = 43;
constexpr int kBlockPathCycleKColumn = 44;
constexpr int kUpdatePathCycleKColumn = 45;
constexpr int kLongSlotUseBaseColumn = 46;

// One diagnostic thread reconstructs one potential 32-neuron task after the
// persistent launch. This deliberately lives outside the timed/production
// path; the normal build contains neither this kernel nor its output buffer.
__global__ void collect_block_stats_kernel(
    const float* __restrict__ dense_spikes,
    const int* __restrict__ graph_indptr,
    int* __restrict__ stats,
    int t_steps,
    int batch_size,
    int n_neuron,
    int blocks_per_batch) {
    const int record = blockIdx.x * blockDim.x + threadIdx.x;
    const int record_count = t_steps * batch_size * blocks_per_batch;
    if (record >= record_count) {
        return;
    }
    const int block = record % blocks_per_batch;
    const int bucket = record / blocks_per_batch;
    const int block_start = block * 32;
    const int block_end = min(block_start + 32, n_neuron);
    const float* bucket_spikes = dense_spikes + bucket * n_neuron;

    unsigned active_mask = 0;
    int active_rows = 0;
    int active_edges = 0;
    int active_runs = 0;
    int max_row_degree = 0;
    int first_active = -1;
    int last_active = -1;
    int long_segment_tasks = 0;
    bool previous_active = false;
    bool all_direct_lane = true;
    for (int neuron = block_start; neuron < block_end; ++neuron) {
        if (bucket_spikes[neuron] == 0.0f) {
            previous_active = false;
            continue;
        }
        const int degree = graph_indptr[neuron + 1] - graph_indptr[neuron];
        if (degree >= kFanoutThreshold) {
            long_segment_tasks +=
                (degree + kSegmentSize - 1) / kSegmentSize;
            previous_active = false;
            continue;
        }
        const int lane = neuron - block_start;
        active_mask |= 1u << lane;
        ++active_rows;
        active_edges += degree;
        max_row_degree = max(max_row_degree, degree);
        all_direct_lane = all_direct_lane && degree <= kLaneRowThreshold;
        if (!previous_active) {
            ++active_runs;
        }
        previous_active = true;
        first_active = first_active < 0 ? neuron : first_active;
        last_active = neuron;
    }

    int* row = stats + record * kBlockStatsColumns;
    row[0] = active_rows;
    row[1] = active_edges;
    row[2] = active_runs;
    row[3] = first_active < 0
        ? 0
        : graph_indptr[last_active + 1] - graph_indptr[first_active];
    row[4] = max_row_degree;
    row[5] = active_edges;  // current spike-block path: one atomic per edge
    row[6] = active_rows > 0 && all_direct_lane ? 1 : 0;
    row[7] = active_rows > active_runs ? 1 : 0;
    row[8] = long_segment_tasks;
    row[9] = bucket / batch_size;
    row[10] = bucket % batch_size;
    row[11] = block_start;
    row[12] = static_cast<int>(active_mask);
}
#endif

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

__device__ __forceinline__ int block_load_acquire(const int* value) {
    cuda::atomic_ref<int, cuda::thread_scope_block> ref(
        *const_cast<int*>(value));
    return ref.load(cuda::memory_order_acquire);
}

__device__ __forceinline__ void block_store_release(int* value, int next) {
    cuda::atomic_ref<int, cuda::thread_scope_block> ref(*value);
    ref.store(next, cuda::memory_order_release);
}

#ifdef ENABLE_BLOCK_STATS
__device__ __forceinline__ void record_long_task_stats(
    int* block_stats,
    int edge_count,
    bool pipeline_eligible) {
    if (block_stats == nullptr) {
        return;
    }
    atomicAdd(block_stats + kLongTasksColumn, 1);
    atomicAdd(block_stats + kLongEdgesColumn, edge_count);
    atomicAdd(
        block_stats
            + (edge_count == kSegmentSize
                   ? kLongFullSegmentsColumn
                   : kLongPartialSegmentsColumn),
        1);
    if (edge_count < 256) {
        atomicAdd(block_stats + kLong128To255Column, 1);
    } else if (edge_count < 384) {
        atomicAdd(block_stats + kLong256To383Column, 1);
    } else if (edge_count < 512) {
        atomicAdd(block_stats + kLong384To511Column, 1);
    } else {
        atomicAdd(block_stats + kLong512Column, 1);
    }
    if (pipeline_eligible) {
        atomicAdd(block_stats + kLongEligibleTasksColumn, 1);
        atomicAdd(block_stats + kLongEligibleEdgesColumn, edge_count);
    }
}
#endif

__device__ __forceinline__ int map_packed_edge(
    const int* packed_prefix,
    const int* packed_starts,
    int logical_edge) {
    int low = 0;
    int high = 31;
#pragma unroll
    for (int step = 0; step < 5; ++step) {
        const int middle = (low + high) >> 1;
        if (packed_prefix[middle] > logical_edge) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    const int previous_end = low == 0 ? 0 : packed_prefix[low - 1];
    return packed_starts[low] + logical_edge - previous_end;
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
    int* __restrict__ block_stats,
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
    const int warp_in_block = threadIdx.x >> 5;
    __shared__ int packed_prefix[kWarpsPerBlock][32];
    __shared__ int packed_starts[kWarpsPerBlock][32];
#ifdef ENABLE_BLOCK_HASH
    __shared__ int hash_keys[kWarpsPerBlock][kHashSize];
    __shared__ float hash_values[kWarpsPerBlock][kHashSize];
    __shared__ unsigned short
        hash_used_slots[kWarpsPerBlock][kHashSize];
#else
    __shared__ int hash_keys[1][1];
    __shared__ float hash_values[1][1];
    __shared__ unsigned short hash_used_slots[1][1];
#endif
#if BTORCH_LONG_WARP_SPEC_MODE >= 2
    __shared__ int long_consumer_batch[kLongConsumerWarps];
    __shared__ int long_consumer_edge_begin[kLongConsumerWarps];
    __shared__ int long_consumer_edge_count[kLongConsumerWarps];
    __shared__ int long_consumer_task_epoch[kLongConsumerWarps];
    __shared__ int long_consumer_done_epoch[kLongConsumerWarps];
    __shared__ int long_producer_done;
#endif
#if BTORCH_LONG_WARP_SPEC_MODE >= 3
    __shared__ int long_consumer_next_chunk[kLongConsumerWarps];
    __shared__ int long_consumer_chunk_count[kLongConsumerWarps];
    __shared__ int long_consumer_chunks_consumed[kLongConsumerWarps];
    __shared__ int long_slot_post[kLongStages][kLongChunkEdges];
    __shared__ float long_slot_weight[kLongStages][kLongChunkEdges];
    __shared__ int long_slot_consumer[kLongStages];
    __shared__ int long_slot_count[kLongStages];
    __shared__ int long_slot_chunk_epoch[kLongStages];
    __shared__ int long_slot_ready_epoch[kLongStages];
    __shared__ int long_slot_consumed_epoch[kLongStages];
#endif
    const float decay = expf(-dt / tau_syn);
    const float reset_delta = v_threshold - v_reset;

#ifdef ENABLE_BLOCK_HASH
    if constexpr (kHashUsedSlots) {
        for (int hash_slot = lane;
             hash_slot < kHashSize;
             hash_slot += 32) {
            hash_keys[warp_in_block][hash_slot] = kEmptyHashKey;
        }
        __syncwarp();
    }
#endif

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

#ifdef ENABLE_BLOCK_STATS
        const unsigned long long update_path_begin =
            threadIdx.x == 0 ? clock64() : 0;
#endif
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
                const bool block_spike =
                    fired && degree > 0 && degree < kFanoutThreshold;
                const unsigned warp_active = __activemask();
                const unsigned spike_mask =
                    __ballot_sync(warp_active, block_spike);
                if constexpr (kBlockEdgeBudget > 0) {
                    int inclusive_edges = block_spike ? degree : 0;
#pragma unroll
                    for (int offset = 1; offset < 32; offset <<= 1) {
                        const int value = __shfl_up_sync(
                            warp_active, inclusive_edges, offset);
                        if (lane >= offset) {
                            inclusive_edges += value;
                        }
                    }
                    const int final_lane = 31 - __clz(warp_active);
                    const int active_edges = __shfl_sync(
                        warp_active, inclusive_edges, final_lane);
                    if (lane == 0 && active_edges > 0) {
                        const int logical_task_count =
                            (active_edges + kBlockTaskSpan - 1)
                            / kBlockTaskSpan;
                        const int first_task = atomicAdd(
                            task_counts + 1, logical_task_count);
                        const int block_index = n >> 5;
                        for (int segment = 0;
                             segment < logical_task_count;
                             ++segment) {
                            const int task = first_task + segment;
                            const int slot = queue_capacity - 1 - task;
                            const unsigned descriptor =
                                static_cast<unsigned>(block_index)
                                | (static_cast<unsigned>(segment)
                                   << kBlockIndexBits);
                            task_queue_batch[slot] = b;
                            task_queue_start[slot] =
                                static_cast<int>(descriptor);
                            task_queue_end_or_mask[slot] =
                                static_cast<int>(spike_mask);
                        }
                    }
                } else if (lane == 0 && spike_mask != 0) {
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
#ifdef ENABLE_BLOCK_STATS
        if (threadIdx.x == 0 && block_stats != nullptr) {
            atomicAdd(
                block_stats + kUpdatePathCycleKColumn,
                static_cast<int>(
                    (clock64() - update_path_begin) >> 10));
        }
#endif
        grid.sync();

        const int segment_count = task_counts[0];
#ifdef ENABLE_BLOCK_STATS
        const unsigned long long long_path_begin =
            threadIdx.x == 0 ? clock64() : 0;
#endif
#if BTORCH_LONG_WARP_SPEC_MODE == 0
        while (true) {
            int task = 0;
            if (lane == 0) {
                task = atomicAdd(work_counters, 1);
            }
            task = __shfl_sync(kFullWarpMask, task, 0);
            if (task >= segment_count) {
                break;
            }
            const int edge_begin = task_queue_start[task];
            const int edge_end = task_queue_end_or_mask[task];
#ifdef ENABLE_BLOCK_STATS
            if (lane == 0) {
                record_long_task_stats(
                    block_stats, edge_end - edge_begin, false);
            }
#endif
            const int b = task_queue_batch[task];
            process_edge_range(
                graph_indices,
                graph_weight,
                psc + b * n_neuron,
                edge_begin,
                edge_end,
                lane);
        }
#elif BTORCH_LONG_WARP_SPEC_MODE == 1
        if (warp_in_block != 0) {
            while (true) {
                int task = 0;
                if (lane == 0) {
                    task = atomicAdd(work_counters, 1);
                }
                task = __shfl_sync(kFullWarpMask, task, 0);
                if (task >= segment_count) {
                    break;
                }
                const int edge_begin = task_queue_start[task];
                const int edge_end = task_queue_end_or_mask[task];
#ifdef ENABLE_BLOCK_STATS
                if (lane == 0) {
                    record_long_task_stats(
                        block_stats, edge_end - edge_begin, false);
                }
#endif
                const int b = task_queue_batch[task];
                process_edge_range(
                    graph_indices,
                    graph_weight,
                    psc + b * n_neuron,
                    edge_begin,
                    edge_end,
                    lane);
            }
        }
#else
        if (threadIdx.x < kLongConsumerWarps) {
            long_consumer_batch[threadIdx.x] = 0;
            long_consumer_edge_begin[threadIdx.x] = 0;
            long_consumer_edge_count[threadIdx.x] = 0;
            long_consumer_task_epoch[threadIdx.x] = 0;
            long_consumer_done_epoch[threadIdx.x] = 0;
#if BTORCH_LONG_WARP_SPEC_MODE >= 3
            long_consumer_next_chunk[threadIdx.x] = 0;
            long_consumer_chunk_count[threadIdx.x] = 0;
            long_consumer_chunks_consumed[threadIdx.x] = 0;
#endif
        }
        if (threadIdx.x == 0) {
            long_producer_done = 0;
        }
#if BTORCH_LONG_WARP_SPEC_MODE >= 3
        if (threadIdx.x < kLongStages) {
            long_slot_consumer[threadIdx.x] = -1;
            long_slot_count[threadIdx.x] = 0;
            long_slot_chunk_epoch[threadIdx.x] = 0;
            long_slot_ready_epoch[threadIdx.x] = 0;
            long_slot_consumed_epoch[threadIdx.x] = 0;
        }
#endif
        __syncthreads();

#if BTORCH_LONG_WARP_SPEC_MODE == 2
        if (warp_in_block == 0 && lane == 0) {
            bool no_more_tasks = false;
            while (true) {
                bool progress = false;
                bool any_active = false;
                for (int consumer = 0;
                     consumer < kLongConsumerWarps;
                     ++consumer) {
                    const int epoch =
                        block_load_acquire(
                            long_consumer_task_epoch + consumer);
                    const int done =
                        block_load_acquire(
                            long_consumer_done_epoch + consumer);
                    if (epoch != done) {
                        any_active = true;
                        continue;
                    }
                    if (no_more_tasks) {
                        continue;
                    }
                    const int task = atomicAdd(work_counters, 1);
                    if (task >= segment_count) {
                        no_more_tasks = true;
                        continue;
                    }
                    const int edge_begin = task_queue_start[task];
                    const int edge_end = task_queue_end_or_mask[task];
                    long_consumer_batch[consumer] =
                        task_queue_batch[task];
                    long_consumer_edge_begin[consumer] = edge_begin;
                    long_consumer_edge_count[consumer] =
                        edge_end - edge_begin;
#ifdef ENABLE_BLOCK_STATS
                    record_long_task_stats(
                        block_stats, edge_end - edge_begin, false);
#endif
                    block_store_release(
                        long_consumer_task_epoch + consumer,
                        epoch + 1);
                    any_active = true;
                    progress = true;
                }
                if (no_more_tasks && !any_active) {
                    break;
                }
                if (!progress) {
#ifdef ENABLE_BLOCK_STATS
                    if (block_stats != nullptr) {
                        atomicAdd(
                            block_stats + kLongProducerIdleColumn, 1);
                    }
#endif
                    __nanosleep(64);
                }
            }
            block_store_release(&long_producer_done, 1);
        } else if (warp_in_block > 0) {
            const int consumer = warp_in_block - 1;
            int observed_epoch = 0;
            while (true) {
                int epoch = block_load_acquire(
                    long_consumer_task_epoch + consumer);
                if (epoch == observed_epoch) {
                    if (block_load_acquire(&long_producer_done) != 0) {
                        break;
                    }
#ifdef ENABLE_BLOCK_STATS
                    if (lane == 0 && block_stats != nullptr) {
                        atomicAdd(
                            block_stats + kLongConsumerTaskWaitColumn,
                            1);
                    }
#endif
                    __nanosleep(64);
                    continue;
                }
                const int b = long_consumer_batch[consumer];
                const int edge_begin =
                    long_consumer_edge_begin[consumer];
                const int edge_count =
                    long_consumer_edge_count[consumer];
                process_edge_range(
                    graph_indices,
                    graph_weight,
                    psc + b * n_neuron,
                    edge_begin,
                    edge_begin + edge_count,
                    lane);
                observed_epoch = epoch;
                if (lane == 0) {
                    block_store_release(
                        long_consumer_done_epoch + consumer,
                        observed_epoch);
                }
            }
        }
#else
        if (warp_in_block == 0) {
            bool no_more_tasks = false;
            int consumer_cursor = 0;
            int slot_cursor = 0;
#ifdef ENABLE_BLOCK_STATS
            unsigned long long pipeline_cycle_sum = 0;
            unsigned long long active_consumer_cycle_sum = 0;
            int max_active_consumers = 0;
#endif
            while (true) {
                int selected_consumer = -1;
                int selected_slot = -1;
                int selected_source = 0;
                int selected_count = 0;
                int selected_chunk_epoch = 0;
                bool progress = false;
                bool done = false;
                int active_consumers = 0;
#ifdef ENABLE_BLOCK_STATS
                const unsigned long long iteration_begin =
                    lane == 0 ? clock64() : 0;
#endif
                if (lane == 0) {
                    for (int offset = 0;
                         offset < kLongConsumerWarps;
                         ++offset) {
                        const int consumer =
                            (consumer_cursor + offset)
                            % kLongConsumerWarps;
                        const int epoch = block_load_acquire(
                            long_consumer_task_epoch + consumer);
                        const int completed = block_load_acquire(
                            long_consumer_done_epoch + consumer);
                        if (epoch != completed) {
                            ++active_consumers;
                            continue;
                        }
                        if (no_more_tasks) {
                            continue;
                        }
                        const int task = atomicAdd(work_counters, 1);
                        if (task >= segment_count) {
                            no_more_tasks = true;
                            continue;
                        }
                        const int edge_begin = task_queue_start[task];
                        const int edge_count =
                            task_queue_end_or_mask[task] - edge_begin;
                        long_consumer_batch[consumer] =
                            task_queue_batch[task];
                        long_consumer_edge_begin[consumer] = edge_begin;
                        long_consumer_edge_count[consumer] = edge_count;
                        long_consumer_next_chunk[consumer] = 0;
                        long_consumer_chunk_count[consumer] =
                            (edge_count + kLongChunkEdges - 1)
                            / kLongChunkEdges;
                        long_consumer_chunks_consumed[consumer] = 0;
#ifdef ENABLE_BLOCK_STATS
                        record_long_task_stats(
                            block_stats,
                            edge_count,
                            edge_count >= kLongPipelineThreshold);
#endif
                        block_store_release(
                            long_consumer_task_epoch + consumer,
                            epoch + 1);
                        ++active_consumers;
                        progress = true;
                        consumer_cursor =
                            (consumer + 1) % kLongConsumerWarps;
                    }

                    for (int offset = 0;
                         offset < kLongConsumerWarps;
                         ++offset) {
                        const int consumer =
                            (consumer_cursor + offset)
                            % kLongConsumerWarps;
                        const int edge_count =
                            long_consumer_edge_count[consumer];
                        const int produced =
                            long_consumer_next_chunk[consumer];
                        const int consumed = block_load_acquire(
                            long_consumer_chunks_consumed + consumer);
                        const int chunk_count =
                            long_consumer_chunk_count[consumer];
                        const int epoch = block_load_acquire(
                            long_consumer_task_epoch + consumer);
                        const int completed = block_load_acquire(
                            long_consumer_done_epoch + consumer);
                        if (epoch == completed
                            || edge_count < kLongPipelineThreshold
                            || produced >= chunk_count
                            || produced != consumed) {
                            continue;
                        }
                        selected_consumer = consumer;
                        consumer_cursor =
                            (consumer + 1) % kLongConsumerWarps;
                        break;
                    }

                    if (selected_consumer >= 0) {
                        for (int offset = 0;
                             offset < kLongStages;
                             ++offset) {
                            const int slot =
                                (slot_cursor + offset) % kLongStages;
                            const int ready = block_load_acquire(
                                long_slot_ready_epoch + slot);
                            const int consumed = block_load_acquire(
                                long_slot_consumed_epoch + slot);
                            if (ready == consumed) {
                                selected_slot = slot;
                                slot_cursor = (slot + 1) % kLongStages;
                                break;
                            }
                        }
                    }

                    if (selected_consumer >= 0
                        && selected_slot >= 0) {
                        const int chunk =
                            long_consumer_next_chunk[selected_consumer];
                        const int edge_count =
                            long_consumer_edge_count[selected_consumer];
                        selected_source =
                            long_consumer_edge_begin[selected_consumer]
                            + chunk * kLongChunkEdges;
                        selected_count = min(
                            kLongChunkEdges,
                            edge_count - chunk * kLongChunkEdges);
                        const int epoch = block_load_acquire(
                            long_consumer_task_epoch
                                + selected_consumer);
                        selected_chunk_epoch =
                            (epoch << 6)
                            | (selected_consumer << 3)
                            | (chunk + 1);
                    } else {
#ifdef ENABLE_BLOCK_STATS
                        if (selected_consumer < 0
                            && active_consumers > 0
                            && block_stats != nullptr) {
                            atomicAdd(
                                block_stats
                                    + kLongProducerNoConsumerColumn,
                                1);
                        } else if (selected_slot < 0
                                   && selected_consumer >= 0
                                   && block_stats != nullptr) {
                            atomicAdd(
                                block_stats + kLongProducerNoSlotColumn,
                                1);
                        }
#endif
                    }
                    done = no_more_tasks && active_consumers == 0;
                }

                selected_consumer = __shfl_sync(
                    kFullWarpMask, selected_consumer, 0);
                selected_slot = __shfl_sync(
                    kFullWarpMask, selected_slot, 0);
                selected_source = __shfl_sync(
                    kFullWarpMask, selected_source, 0);
                selected_count = __shfl_sync(
                    kFullWarpMask, selected_count, 0);
                selected_chunk_epoch = __shfl_sync(
                    kFullWarpMask, selected_chunk_epoch, 0);
                done = __shfl_sync(kFullWarpMask, done, 0);

                if (selected_slot >= 0) {
                    for (int index = lane;
                         index < selected_count;
                         index += 32) {
                        long_slot_post[selected_slot][index] =
                            graph_indices[selected_source + index];
                        long_slot_weight[selected_slot][index] =
                            graph_weight[selected_source + index];
                    }
                    __syncwarp();
                    if (lane == 0) {
                        long_slot_consumer[selected_slot] =
                            selected_consumer;
                        long_slot_count[selected_slot] = selected_count;
                        long_slot_chunk_epoch[selected_slot] =
                            selected_chunk_epoch;
                        ++long_consumer_next_chunk[selected_consumer];
                        block_store_release(
                            long_slot_ready_epoch + selected_slot,
                            selected_chunk_epoch);
#ifdef ENABLE_BLOCK_STATS
                        if (block_stats != nullptr) {
                            atomicAdd(
                                block_stats
                                    + kLongChunksProducedColumn,
                                1);
                            atomicAdd(
                                block_stats
                                    + kLongSlotUseBaseColumn
                                    + selected_slot,
                                1);
                        }
#endif
                    }
                    progress = true;
                }
#ifdef ENABLE_BLOCK_STATS
                if (lane == 0) {
                    const unsigned long long iteration_cycles =
                        clock64() - iteration_begin;
                    pipeline_cycle_sum += iteration_cycles;
                    active_consumer_cycle_sum +=
                        iteration_cycles * active_consumers;
                    max_active_consumers = max(
                        max_active_consumers, active_consumers);
                }
#endif
                if (done) {
                    break;
                }
                if (!progress) {
#ifdef ENABLE_BLOCK_STATS
                    if (lane == 0 && block_stats != nullptr) {
                        atomicAdd(
                            block_stats + kLongProducerIdleColumn, 1);
                    }
#endif
                    __nanosleep(64);
                }
            }
            if (lane == 0) {
#ifdef ENABLE_BLOCK_STATS
                if (block_stats != nullptr) {
                    atomicMax(
                        block_stats + kLongMaxActiveConsumersColumn,
                        max_active_consumers);
                    atomicAdd(
                        block_stats
                            + kLongActiveConsumerCycleKColumn,
                        static_cast<int>(
                            active_consumer_cycle_sum >> 10));
                    atomicAdd(
                        block_stats + kLongPipelineCycleKColumn,
                        static_cast<int>(pipeline_cycle_sum >> 10));
                }
#endif
                block_store_release(&long_producer_done, 1);
            }
        } else {
            const int consumer = warp_in_block - 1;
            int observed_epoch = 0;
            while (true) {
                int epoch = block_load_acquire(
                    long_consumer_task_epoch + consumer);
                if (epoch == observed_epoch) {
                    if (block_load_acquire(&long_producer_done) != 0) {
                        break;
                    }
#ifdef ENABLE_BLOCK_STATS
                    if (lane == 0 && block_stats != nullptr) {
                        atomicAdd(
                            block_stats + kLongConsumerTaskWaitColumn,
                            1);
                    }
#endif
                    __nanosleep(64);
                    continue;
                }
                const int b = long_consumer_batch[consumer];
                const int edge_begin =
                    long_consumer_edge_begin[consumer];
                const int edge_count =
                    long_consumer_edge_count[consumer];
                if (edge_count < kLongPipelineThreshold) {
                    process_edge_range(
                        graph_indices,
                        graph_weight,
                        psc + b * n_neuron,
                        edge_begin,
                        edge_begin + edge_count,
                        lane);
                } else {
                    const int chunk_count =
                        (edge_count + kLongChunkEdges - 1)
                        / kLongChunkEdges;
                    for (int chunk = 0; chunk < chunk_count; ++chunk) {
                        const int expected_epoch =
                            (epoch << 6)
                            | (consumer << 3)
                            | (chunk + 1);
                        int slot = -1;
                        while (slot < 0) {
#pragma unroll
                            for (int candidate = 0;
                                 candidate < kLongStages;
                                 ++candidate) {
                                const int ready = block_load_acquire(
                                    long_slot_ready_epoch + candidate);
                                if (ready == expected_epoch
                                    && long_slot_consumer[candidate]
                                        == consumer
                                    && long_slot_chunk_epoch[candidate]
                                        == expected_epoch) {
                                    slot = candidate;
                                    break;
                                }
                            }
                            if (slot < 0) {
#ifdef ENABLE_BLOCK_STATS
                                if (lane == 0
                                    && block_stats != nullptr) {
                                    atomicAdd(
                                        block_stats
                                            + kLongConsumerChunkWaitColumn,
                                        1);
                                }
#endif
                                __nanosleep(64);
                            }
                        }
                        const int count = long_slot_count[slot];
                        int post_registers[kLongChunkEdges / 32];
                        float weight_registers[kLongChunkEdges / 32];
#pragma unroll
                        for (int item = 0;
                             item < kLongChunkEdges / 32;
                             ++item) {
                            const int index = lane + item * 32;
                            if (index < count) {
                                post_registers[item] =
                                    long_slot_post[slot][index];
                                weight_registers[item] =
                                    long_slot_weight[slot][index];
                            }
                        }
                        __syncwarp();
                        if (lane == 0) {
                            block_store_release(
                                long_slot_consumed_epoch + slot,
                                expected_epoch);
                            block_store_release(
                                long_consumer_chunks_consumed + consumer,
                                chunk + 1);
#ifdef ENABLE_BLOCK_STATS
                            if (block_stats != nullptr) {
                                atomicAdd(
                                    block_stats
                                        + kLongChunksConsumedColumn,
                                    1);
                            }
#endif
                        }
#pragma unroll
                        for (int item = 0;
                             item < kLongChunkEdges / 32;
                             ++item) {
                            const int index = lane + item * 32;
                            if (index < count) {
                                atomicAdd(
                                    psc + b * n_neuron
                                        + post_registers[item],
                                    weight_registers[item]);
                            }
                        }
                    }
                }
                observed_epoch = epoch;
                if (lane == 0) {
                    block_store_release(
                        long_consumer_done_epoch + consumer,
                        observed_epoch);
                }
            }
        }
#endif
        __syncthreads();
#endif
#ifdef ENABLE_BLOCK_STATS
        if (threadIdx.x == 0 && block_stats != nullptr) {
            atomicAdd(
                block_stats + kLongPathCycleKColumn,
                static_cast<int>((clock64() - long_path_begin) >> 10));
        }
#endif

#ifdef ENABLE_BLOCK_STATS
        const unsigned long long block_path_begin =
            threadIdx.x == 0 ? clock64() : 0;
#endif
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
            const unsigned descriptor =
                static_cast<unsigned>(task_queue_start[slot]);
            const int block_start = kBlockEdgeBudget > 0
                ? static_cast<int>(descriptor & kBlockIndexMask) << 5
                : static_cast<int>(descriptor);
            const int logical_segment = kBlockEdgeBudget > 0
                ? static_cast<int>(descriptor >> kBlockIndexBits)
                : 0;
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
            if constexpr (kBlockEdgeBudget > 0) {
                int inclusive_end = lane_degree;
#pragma unroll
                for (int offset = 1; offset < 32; offset <<= 1) {
                    const int value = __shfl_up_sync(
                        kFullWarpMask, inclusive_end, offset);
                    if (lane >= offset) {
                        inclusive_end += value;
                    }
                }
                packed_prefix[warp_in_block][lane] = inclusive_end;
                packed_starts[warp_in_block][lane] = lane_edge_start;
                __syncwarp();

                const int total_edges = packed_prefix[warp_in_block][31];
                const int logical_begin =
                    logical_segment * kBlockTaskSpan;
                const int logical_end = min(
                    total_edges, logical_begin + kBlockTaskSpan);
                const int task_edges = logical_end - logical_begin;
#ifdef ENABLE_BLOCK_HASH
#ifdef ENABLE_BLOCK_STATS
                if (lane == 0 && block_stats != nullptr) {
                    atomicAdd(
                        block_stats + kV4InputEdgesColumn,
                        task_edges);
                }
#endif
                bool task_used_hash = false;
                for (int window_begin = logical_begin;
                     window_begin < logical_end;
                     window_begin += kHashAggregation) {
                    const int window_end = min(
                        logical_end, window_begin + kHashAggregation);
                    const int window_edges = window_end - window_begin;
                    const bool use_hash = window_edges >= kHashMinEdges;
                    if (!use_hash) {
                        for (int edge_base = window_begin;
                             edge_base < window_end;
                             edge_base += 32) {
                            const int logical_edge = edge_base + lane;
                            if (logical_edge < window_end) {
                                const int edge = map_packed_edge(
                                    packed_prefix[warp_in_block],
                                    packed_starts[warp_in_block],
                                    logical_edge);
                                atomicAdd(
                                    psc + b * n_neuron
                                        + graph_indices[edge],
                                    graph_weight[edge]);
                            }
                        }
#ifdef ENABLE_BLOCK_STATS
                        if (lane == 0 && block_stats != nullptr) {
                            atomicAdd(
                                block_stats + kV4GlobalAtomicsColumn,
                                window_edges);
                        }
#endif
                        continue;
                    }

                    task_used_hash = true;
                    if constexpr (!kHashUsedSlots) {
                        for (int hash_slot = lane;
                             hash_slot < kHashSize;
                             hash_slot += 32) {
                            hash_keys[warp_in_block][hash_slot] =
                                kEmptyHashKey;
                            hash_values[warp_in_block][hash_slot] = 0.0f;
                        }
                        __syncwarp();
                    }
                    int used_count = 0;

#ifdef ENABLE_BLOCK_STATS
                    if (lane == 0 && block_stats != nullptr) {
                        atomicAdd(
                            block_stats + kHashWindowsColumn, 1);
                        atomicAdd(
                            block_stats + kHashInputEdgesColumn,
                            window_edges);
                    }
#endif
                    for (int edge_base = window_begin;
                         edge_base < window_end;
                         edge_base += 32) {
                        const int logical_edge = edge_base + lane;
                        const bool valid = logical_edge < window_end;
                        int post = 0;
                        float weight = 0.0f;
                        if (valid) {
                            const int edge = map_packed_edge(
                                packed_prefix[warp_in_block],
                                packed_starts[warp_in_block],
                                logical_edge);
                            post = graph_indices[edge];
                            weight = graph_weight[edge];
                        }

                        int destination_slot = -1;
                        int probe_attempts = 0;
                        bool claimed_new = false;
                        if (valid) {
                            int hash_slot = static_cast<int>(
                                (static_cast<unsigned>(post)
                                 * 2654435761u)
                                & (kHashSize - 1));
#pragma unroll
                            for (int probe = 0;
                                 probe < kHashMaxProbe;
                                 ++probe) {
                                ++probe_attempts;
                                const int old = atomicCAS(
                                    &hash_keys
                                        [warp_in_block][hash_slot],
                                    kEmptyHashKey,
                                    post);
                                if (old == kEmptyHashKey
                                    || old == post) {
                                    destination_slot = hash_slot;
                                    claimed_new =
                                        old == kEmptyHashKey;
                                    if constexpr (kHashUsedSlots) {
                                        if (claimed_new) {
                                            hash_values
                                                [warp_in_block][hash_slot] =
                                                0.0f;
                                        }
                                    }
                                    break;
                                }
                                hash_slot =
                                    (hash_slot + 1) & (kHashSize - 1);
                            }
                        }

                        if constexpr (kHashUsedSlots) {
                            const unsigned new_key_mask =
                                __ballot_sync(
                                    kFullWarpMask, claimed_new);
                            if (claimed_new) {
                                const unsigned lower_lanes =
                                    lane == 0
                                    ? 0u
                                    : (1u << lane) - 1u;
                                const int used_rank = __popc(
                                    new_key_mask & lower_lanes);
                                hash_used_slots
                                    [warp_in_block]
                                    [used_count + used_rank] =
                                    static_cast<unsigned short>(
                                        destination_slot);
                            }
                            used_count += __popc(new_key_mask);
                        }

                        // Key publication, value initialization, and used-slot
                        // recording complete before any lane accumulates.
                        __syncwarp();
                        if (destination_slot >= 0) {
                            atomicAdd(
                                &hash_values
                                    [warp_in_block][destination_slot],
                                weight);
                        } else if (valid) {
                            atomicAdd(
                                psc + b * n_neuron + post,
                                weight);
                        }
                        __syncwarp();

#ifdef ENABLE_BLOCK_STATS
                        const int warp_probe_attempts = __reduce_add_sync(
                            kFullWarpMask, probe_attempts);
                        const unsigned fallback_mask = __ballot_sync(
                            kFullWarpMask,
                            valid && destination_slot < 0);
                        if (lane == 0 && block_stats != nullptr) {
                            atomicAdd(
                                block_stats + kHashProbeAttemptsColumn,
                                warp_probe_attempts);
                            const int fallback_count =
                                __popc(fallback_mask);
                            atomicAdd(
                                block_stats
                                    + kHashFallbackAtomicsColumn,
                                fallback_count);
                            atomicAdd(
                                block_stats + kV4GlobalAtomicsColumn,
                                fallback_count);
                        }
#endif
                    }

                    int flush_count = 0;
                    if constexpr (kHashUsedSlots) {
                        flush_count = used_count;
                        for (int used_index = lane;
                             used_index < used_count;
                             used_index += 32) {
                            const int hash_slot = hash_used_slots
                                [warp_in_block][used_index];
                            const int post =
                                hash_keys[warp_in_block][hash_slot];
                            atomicAdd(
                                psc + b * n_neuron + post,
                                hash_values[warp_in_block][hash_slot]);
                            hash_keys[warp_in_block][hash_slot] =
                                kEmptyHashKey;
                        }
                    } else {
                        const bool occupied =
                            hash_keys[warp_in_block][lane]
                            != kEmptyHashKey;
                        flush_count = __popc(__ballot_sync(
                            kFullWarpMask, occupied));
                        for (int hash_slot = lane;
                             hash_slot < kHashSize;
                             hash_slot += 32) {
                            if (hash_slot >= 32) {
                                const bool extra_occupied =
                                    hash_keys
                                        [warp_in_block][hash_slot]
                                    != kEmptyHashKey;
                                flush_count += __popc(__ballot_sync(
                                    kFullWarpMask, extra_occupied));
                            }
                            const int post =
                                hash_keys[warp_in_block][hash_slot];
                            if (post != kEmptyHashKey) {
                                atomicAdd(
                                    psc + b * n_neuron + post,
                                    hash_values
                                        [warp_in_block][hash_slot]);
                            }
                        }
                    }
#ifdef ENABLE_BLOCK_STATS
                    if (lane == 0 && block_stats != nullptr) {
                        atomicAdd(
                            block_stats + kHashFlushAtomicsColumn,
                            flush_count);
                        atomicAdd(
                            block_stats + kV4GlobalAtomicsColumn,
                            flush_count);
                    }
#endif
                    __syncwarp();
                }
#ifdef ENABLE_BLOCK_STATS
                if (lane == 0
                    && task_used_hash
                    && block_stats != nullptr) {
                    atomicAdd(block_stats + kHashTasksColumn, 1);
                }
#endif
                __syncwarp();
                continue;
#else
                const bool reduce_task =
                    kTileReduceMode == 1
                    || (kTileReduceMode == 2 && task_edges >= 64);
#ifdef ENABLE_BLOCK_STATS
                if (lane == 0 && block_stats != nullptr) {
                    atomicAdd(
                        block_stats + kV4InputEdgesColumn,
                        task_edges);
                    if (reduce_task) {
                        atomicAdd(
                            block_stats + kV4ReduceTasksColumn,
                            1);
                        atomicAdd(
                            block_stats + kV4ReduceEdgesColumn,
                            task_edges);
                    }
                }
#endif
                for (int edge_base = logical_begin;
                     edge_base < logical_end;
                     edge_base += 32) {
                    const int logical_edge = edge_base + lane;
                    const bool valid = logical_edge < logical_end;
                    const int lookup_edge = valid
                        ? logical_edge
                        : logical_end - 1;
                    int low = 0;
                    int high = 31;
#pragma unroll
                    for (int step = 0; step < 5; ++step) {
                        const int middle = (low + high) >> 1;
                        if (packed_prefix[warp_in_block][middle]
                            > lookup_edge) {
                            high = middle;
                        } else {
                            low = middle + 1;
                        }
                    }

                    int post = 0;
                    float weight = 0.0f;
                    if (valid) {
                        const int previous_end = low == 0
                            ? 0
                            : packed_prefix[warp_in_block][low - 1];
                        const int edge =
                            packed_starts[warp_in_block][low]
                            + logical_edge - previous_end;
                        post = graph_indices[edge];
                        weight = graph_weight[edge];
                    }
                    const unsigned valid_mask =
                        __ballot_sync(kFullWarpMask, valid);
                    if (!reduce_task) {
#ifdef ENABLE_BLOCK_STATS
                        if (lane == 0 && block_stats != nullptr) {
                            atomicAdd(
                                block_stats + kV4GlobalAtomicsColumn,
                                __popc(valid_mask));
                        }
#endif
                        if (valid) {
                            atomicAdd(
                                psc + b * n_neuron + post,
                                weight);
                        }
                        continue;
                    }

                    unsigned peers = 0;
                    if (valid) {
                        peers = __match_any_sync(valid_mask, post);
                    }
                    const bool is_leader =
                        valid && lane == __ffs(peers) - 1;
                    const unsigned leader_mask =
                        __ballot_sync(kFullWarpMask, is_leader);
#ifdef ENABLE_BLOCK_STATS
                    if (lane == 0 && block_stats != nullptr) {
                        atomicAdd(
                            block_stats + kV4GlobalAtomicsColumn,
                            __popc(leader_mask));
                    }
#endif
                    if (valid) {
                        unsigned remaining = peers;
                        float reduced_weight = 0.0f;
                        while (remaining != 0) {
                            const int source = __ffs(remaining) - 1;
                            reduced_weight += __shfl_sync(
                                peers, weight, source);
                            remaining &= remaining - 1;
                        }
                        if (is_leader) {
                            atomicAdd(
                                psc + b * n_neuron + post,
                                reduced_weight);
                        }
                    }
                }
                __syncwarp();
                continue;
#endif
            }

            // Tiny rows stay lane-owned. The remaining medium rows are either
            // consumed as strict adjacent CSR runs or packed into one logical
            // edge stream so scattered masks no longer serialize per row.
            const bool tiny_row =
                lane_fired && lane_degree <= kLaneRowThreshold;
            const unsigned tiny_mask = __ballot_sync(kFullWarpMask, tiny_row);
            const unsigned medium_mask = spike_mask & ~tiny_mask;

            if (tiny_row) {
                for (int edge = lane_edge_start; edge < lane_edge_end; ++edge) {
                    const int post = graph_indices[edge];
                    atomicAdd(
                        psc + b * n_neuron + post,
                        graph_weight[edge]);
                }
            }

            if (medium_mask != 0) {
                const int medium_rows = __popc(medium_mask);
                const unsigned run_starts =
                    medium_mask & ~(medium_mask << 1);
                const int run_count = __popc(run_starts);
                const int medium_degree =
                    (medium_mask & (1u << lane)) != 0 ? lane_degree : 0;
                int inclusive_end = medium_degree;
#pragma unroll
                for (int offset = 1; offset < 32; offset <<= 1) {
                    const int value = __shfl_up_sync(
                        kFullWarpMask, inclusive_end, offset);
                    if (lane >= offset) {
                        inclusive_end += value;
                    }
                }
                const int medium_edges = __shfl_sync(
                    kFullWarpMask, inclusive_end, 31);
#ifdef ENABLE_BLOCK_HASH
                const bool use_hash =
                    medium_rows >= 2
                    && medium_edges >= kHashMinEdges
                    && medium_edges <= kLegacyHashMaxEdges;
#else
                constexpr bool use_hash = false;
#endif
                if (use_hash) {
                    for (int slot = lane; slot < kHashSize; slot += 32) {
                        hash_keys[warp_in_block][slot] = kEmptyHashKey;
                        hash_values[warp_in_block][slot] = 0.0f;
                    }
                    __syncwarp();
                }
                if (run_count * 2 <= medium_rows) {
                    // P1: adjacent active CSR rows are one physical edge run.
                    unsigned remaining = medium_mask;
                    while (remaining != 0) {
                        const int first_lane = __ffs(remaining) - 1;
                        const unsigned shifted = remaining >> first_lane;
                        const int first_zero = __ffs(~shifted);
                        const int run_length = first_zero == 0
                            ? 32
                            : first_zero - 1;
                        const int end_lane = first_lane + run_length;
                        const int run_begin =
                            graph_indptr[block_start + first_lane];
                        const int run_end =
                            graph_indptr[block_start + end_lane];
                        for (int edge_base = run_begin;
                             edge_base < run_end;
                             edge_base += 32) {
                            const int edge = edge_base + lane;
                            if (edge < run_end) {
                                const int post = graph_indices[edge];
                                const float weight = graph_weight[edge];
                                if (use_hash) {
                                    int slot = static_cast<int>(
                                        (static_cast<unsigned>(post)
                                         * 2654435761u)
                                        & (kHashSize - 1));
                                    bool inserted = false;
#pragma unroll
                                    for (int probe = 0;
                                         probe < kHashMaxProbe;
                                         ++probe) {
                                        const int old = atomicCAS(
                                            &hash_keys[warp_in_block][slot],
                                            kEmptyHashKey,
                                            post);
                                        if (old == kEmptyHashKey
                                            || old == post) {
                                            atomicAdd(
                                                &hash_values
                                                    [warp_in_block][slot],
                                                weight);
                                            inserted = true;
                                            break;
                                        }
                                        slot = (slot + 1) & (kHashSize - 1);
                                    }
                                    if (!inserted) {
                                        atomicAdd(
                                            psc + b * n_neuron + post,
                                            weight);
                                    }
                                } else {
                                    atomicAdd(
                                        psc + b * n_neuron + post,
                                        weight);
                                }
                            }
                        }
                        const unsigned run_mask = run_length == 32
                            ? kFullWarpMask
                            : ((1u << run_length) - 1u) << first_lane;
                        remaining &= ~run_mask;
                    }
                } else {
                    // P3: prefix-pack scattered rows into one logical stream.
                    packed_prefix[warp_in_block][lane] = inclusive_end;
                    packed_starts[warp_in_block][lane] = lane_edge_start;
                    __syncwarp();
                    const int total_edges =
                        packed_prefix[warp_in_block][31];
                    for (int edge_base = 0;
                         edge_base < total_edges;
                         edge_base += 32) {
                        const int logical_edge = edge_base + lane;
                        const bool valid = logical_edge < total_edges;
                        const int lookup_edge = valid
                            ? logical_edge
                            : total_edges - 1;
                        int low = 0;
                        int high = 31;
#pragma unroll
                        for (int step = 0; step < 5; ++step) {
                            const int middle = (low + high) >> 1;
                            if (packed_prefix[warp_in_block][middle]
                                > lookup_edge) {
                                high = middle;
                            } else {
                                low = middle + 1;
                            }
                        }
                        if (valid) {
                            const int previous_end = low == 0
                                ? 0
                                : packed_prefix[warp_in_block][low - 1];
                            const int edge =
                                packed_starts[warp_in_block][low]
                                + logical_edge - previous_end;
                            const int post = graph_indices[edge];
                            const float weight = graph_weight[edge];
                            if (use_hash) {
                                int slot = static_cast<int>(
                                    (static_cast<unsigned>(post)
                                     * 2654435761u)
                                    & (kHashSize - 1));
                                bool inserted = false;
#pragma unroll
                                for (int probe = 0;
                                     probe < kHashMaxProbe;
                                     ++probe) {
                                    const int old = atomicCAS(
                                        &hash_keys[warp_in_block][slot],
                                        kEmptyHashKey,
                                        post);
                                    if (old == kEmptyHashKey || old == post) {
                                        atomicAdd(
                                            &hash_values
                                                [warp_in_block][slot],
                                            weight);
                                        inserted = true;
                                        break;
                                    }
                                    slot = (slot + 1) & (kHashSize - 1);
                                }
                                if (!inserted) {
                                    atomicAdd(
                                        psc + b * n_neuron + post,
                                        weight);
                                }
                            } else {
                                atomicAdd(
                                    psc + b * n_neuron + post,
                                    weight);
                            }
                        }
                    }
                    __syncwarp();
                }
                if (use_hash) {
                    __syncwarp();
                    for (int slot = lane; slot < kHashSize; slot += 32) {
                        const int post = hash_keys[warp_in_block][slot];
                        if (post != kEmptyHashKey) {
                            atomicAdd(
                                psc + b * n_neuron + post,
                                hash_values[warp_in_block][slot]);
                        }
                    }
                    __syncwarp();
                }
            }
        }
#ifdef ENABLE_BLOCK_STATS
        if (threadIdx.x == 0 && block_stats != nullptr) {
            atomicAdd(
                block_stats + kBlockPathCycleKColumn,
                static_cast<int>(
                    (clock64() - block_path_begin) >> 10));
        }
#endif
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
    int* block_stats,
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
        &block_stats,
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
#ifdef ENABLE_BLOCK_STATS
    if (block_stats != nullptr) {
        const int blocks_per_batch = (n_neuron + 31) / 32;
        const int record_count = t_steps * batch_size * blocks_per_batch;
        constexpr int kStatsThreads = 256;
        collect_block_stats_kernel<<<
            (record_count + kStatsThreads - 1) / kStatsThreads,
            kStatsThreads,
            0,
            stream>>>(
            dense_spikes,
            graph_indptr,
            block_stats,
            t_steps,
            batch_size,
            n_neuron,
            blocks_per_batch);
    }
#else
    (void)block_stats;
#endif
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
