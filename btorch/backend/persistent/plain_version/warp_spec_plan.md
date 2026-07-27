# 第一版 TMA Warp Specialization 实施计划

## 1. 目标

在现有 Block V4 + warp-private shared hash 的基础上，将普通 block task 的处理流程由：

```text
fetch task
→ map logical edge
→ load post/weight
→ shared hash accumulate
→ global atomic flush
→ fetch next task
```

改为：

```text
Producer warp:
    fetch task k+1
    prepare edge tile k+1
    asynchronous copy tile k+1 to shared memory

Consumer warps:
    shared hash accumulate tile k
    global atomic flush tile k
```

形成两阶段流水线：

[
\text{edge load}*{k+1}
\parallel
\left(
\text{hash accumulate}*{k}
+
\text{global flush}_{k}
\right)
]

第一版的核心问题是验证：

1. edge load 是否可以被 hash 和 flush 覆盖；
2. 专用 producer warp 是否能提高 eligible warp 和 issue active；
3. shared-memory 增量是否会破坏当前 2 blocks/SM 的 residency；
4. 当前 512-edge aggregation 是否适合作为异步搬运 tile。

第一版**不实现独立 flush warp**。现有实验已经表明，512-edge task 合并和 warp-private hash 共同构成稳定收益；其中约 4.6% 来自 512-edge 调度粒度，hash 相对匹配的 512-edge direct 路径还能再快约 10.4%。因此不能为了简化流水线退回 256-edge task。

---

## 2. 保留与修改的部分

### 2.1 保留不变

以下逻辑第一版保持不变：

* neuron update 和 spike mask 构建；
* long-row task queue；
* ordinary block task 的 512-edge logical chunk；
* `aggregation=512`；
* `capacity=512`；
* `max_probe=4`；
* `min_edges=256`；
* warp-private shared hash；
* probe 失败后的 global atomic fallback；
* consumer warp 自己执行 hash flush；
* timestep 末尾的 `grid.sync()`；
* 当前数值正确性标准。

推荐继续使用当前实验最佳配置：

```text
aggregation = 512
capacity    = 512
max probes = 4
min edges   = 256
used slots = on
```

512-edge 实际 atomic reduction 为约 6.31%，并在不同活动率和 batch 下保持稳定正收益。

### 2.2 第一版新增

新增：

* warp 0 作为 producer/scheduler；
* warp 1–7 作为 consumer；
* 两个 shared-memory edge stage；
* producer/consumer stage 状态；
* producer 发起异步 edge copy；
* consumer 等待 stage ready；
* timestep 结束时 drain pipeline；
* pipeline stall 统计；
* 非 TMA 的 producer-copy 对照版本。

### 2.3 第一版暂不实现

不实现：

* 独立 global flush warp；
* hash-to-flush compact queue；
* CTA-wide shared hash；
* full-block aggregation；
* 三重或更多 buffering；
* 动态调整 producer/consumer warp 数；
* 对 long-row queue 使用 TMA；
* 跨 timestep pipeline；
* 稀疏 gather 型 TMA。

---

## 3. 线程块划分

线程块仍使用：

```text
256 threads
8 warps
```

固定划分：

```text
warp 0:
    producer
    task scheduler
    edge tile preparation
    asynchronous copy issue

warp 1–7:
    consumers
    shared hash accumulate
    fallback global atomic
    hash global flush
```

对应：

```cpp
constexpr int kProducerWarp = 0;
constexpr int kConsumerWarps = 7;

const int warp_id = threadIdx.x >> 5;
const int lane = threadIdx.x & 31;

const bool is_producer = warp_id == kProducerWarp;
const bool is_consumer = warp_id != kProducerWarp;

const int consumer_id = warp_id - 1;
```

producer warp不分配 hash table，由此释放一份：

```text
hash_keys[512]
hash_values[512]
hash_used_slots[512]
```

约 5 KB shared memory，用来抵消 edge staging buffer 的成本。

---

## 4. Edge tile 定义

第一版固定：

```text
tile size = 512 logical edges
stage count = 2
```

每个 tile 至少需要包含：

```cpp
post[512]
weight[512]
edge_count
batch
task_id
```

概念结构：

```cpp
struct EdgeTileMetadata {
    int batch;
    int edge_count;
    int task_id;
};
```

实际 shared memory建议使用 SoA，而不是数组结构体：

```cpp
__shared__ int stage_post[2][512];
__shared__ float stage_weight[2][512];

__shared__ int stage_batch[2];
__shared__ int stage_edge_count[2];
__shared__ int stage_task_id[2];
__shared__ int stage_state[2];
```

状态定义：

```cpp
enum StageState {
    STAGE_EMPTY = 0,
    STAGE_LOADING = 1,
    STAGE_READY = 2,
    STAGE_BUSY = 3
};
```

第一版只有两个 stage：

```text
stage 0
stage 1
```

producer轮流写：

```text
0 → 1 → 0 → 1
```

consumer动态领取 ready stage。

---

## 5. 数据布局前提

### 5.1 需要先区分两种实现

当前代码中的 512-edge chunk 是“逻辑连续”的：

```cpp
logical_edge
→ map_packed_edge()
→ physical CSR edge
```

但实际 CSR 地址可能来自最多 32 条不同 row，因此未必物理连续。

TMA不适合直接完成任意 gather。因此第一版分成两个层次。

### Version P：producer普通 gather copy

producer warp执行：

```cpp
edge = map_packed_edge(...)
stage_post[...] = graph_indices[edge]
stage_weight[...] = graph_weight[edge]
```

然后 consumer从 shared memory读取。

这个版本不使用真正 TMA，但它是必要对照，用于判断：

* warp specialization 本身是否有效；
* shared staging 是否有效；
* producer是否能跟上七个 consumer；
* pipeline同步成本是否过高。

### Version A：连续 packed layout + asynchronous copy

预处理为每个 block构造连续 packed edge stream：

```text
packed_block_edge_offsets
packed_block_posts
packed_block_weights
```

运行时一个 512-edge task直接对应：

```cpp
packed_offset + segment * 512
```

不再逐 edge调用 `map_packed_edge()`。

此时 producer可以通过 TMA 或 bulk asynchronous copy 将连续区域搬入 shared memory。

第一版应先完成 Version P，再实现 Version A。否则很难区分收益来自：

* warp specialization；
* 消除 `map_packed_edge()`；
* 连续布局；
* TMA；
* 双缓冲。

---

## 6. Shared memory 设计

### 6.1 Hash table

仅为七个 consumer分配：

```cpp
__shared__ int hash_keys[7][512];
__shared__ float hash_values[7][512];
__shared__ unsigned short hash_used_slots[7][512];
```

每个 consumer使用：

```cpp
hash_keys[consumer_id]
hash_values[consumer_id]
hash_used_slots[consumer_id]
```

不能继续使用：

```cpp
hash_keys[warp_in_block]
```

否则 producer warp会占用一份无用 hash。

### 6.2 Edge stage

双缓冲成本：

```text
post:
2 × 512 × 4 B = 4 KB

weight:
2 × 512 × 4 B = 4 KB

total:
8 KB
```

减少一份 hash约节省：

```text
key:       2 KB
value:     2 KB
used slot: 1 KB
total:     5 KB
```

预计总 shared memory比当前增加约 3 KB，加少量 metadata。

目标：

```text
shared memory/block < 49 KB
```

最好控制在：

```text
45–48 KB/block
```

以继续维持：

```text
2 blocks/SM
16 active warps/SM
约 33% occupancy
```

如果编译后超过 residency 临界值，第一优先级不是减小 aggregation，而是将 `hash_used_slots` 压缩成 bitmap。

---

## 7. Producer任务

producer warp负责：

1. 从 block task queue领取下一个普通 block task；
2. 读取 task descriptor；
3. 重建 active row 的 prefix/start；
4. 确定当前 512-edge logical范围；
5. 等待一个 empty stage；
6. 将 edge tile填入 stage；
7. 发布 stage ready；
8. 重复直到 task耗尽；
9. 发布 producer done。

producer不处理：

* long-row queue；
* shared hash；
* global hash flush。

long-row queue仍可在 block-task pipeline 前由全部 warp按原逻辑处理。

---

## 8. Consumer任务

每个 consumer warp负责：

1. 领取一个 ready stage；
2. 标记 stage busy；
3. 对 stage 中的 edge执行现有 hash路径；
4. probe失败时直接 global atomic fallback；
5. flush当前 warp的 used slots；
6. 清理 hash keys；
7. 将 stage标记为 empty；
8. 领取下一个 ready stage；
9. producer结束且所有 stage为空后退出。

consumer仍以一个 warp处理一个完整 512-edge tile。

这样保留现有语义：

```text
one warp
→ one 512-edge chunk
→ one warp-private hash
→ own flush
```

---

## 9. 总体伪代码

### 9.1 Kernel主结构

```cpp
for t in timesteps:

    reset_task_counters()
    process_external_input()
    grid.sync()

    update_neurons()
    emit_spikes()
    build_long_row_tasks()
    build_block_tasks()
    grid.sync()

    process_long_row_tasks_original_path()

    __syncthreads()

    initialize_pipeline_state()

    if producer warp:
        run_block_task_producer()

    else:
        run_block_task_consumer()

    __syncthreads()

    assert_pipeline_drained()

    grid.sync()
```

---

## 10. Pipeline初始化伪代码

```cpp
if threadIdx.x == 0:
    stage_state[0] = STAGE_EMPTY
    stage_state[1] = STAGE_EMPTY

    producer_done = 0
    ready_counter = 0

__syncthreads()
```

hash keys初始化：

```cpp
if is_consumer:
    for slot = lane; slot < HASH_SIZE; slot += 32:
        hash_keys[consumer_id][slot] = EMPTY_KEY

__syncwarp()
```

如果使用现有 used-slot机制，key仍然可以只在 kernel开始时初始化一次，然后每次 flush清理实际使用 slot。

---

## 11. Producer伪代码：普通 gather staging版本

```cpp
function run_block_task_producer():

    int next_stage = 0

    while true:

        task = fetch_next_block_task()

        if task >= block_count:
            break

        stage = wait_for_empty_stage(next_stage)

        if lane == 0:
            stage_state[stage] = STAGE_LOADING

        __syncwarp()

        descriptor = load_task_descriptor(task)

        batch = descriptor.batch
        block_index = descriptor.block_index
        segment = descriptor.segment
        spike_mask = descriptor.spike_mask

        build_packed_prefix_and_starts(
            block_index,
            spike_mask,
            producer_prefix,
            producer_starts
        )

        total_edges = producer_prefix[last_active_lane]

        logical_begin = segment * 512
        logical_end = min(total_edges, logical_begin + 512)
        edge_count = logical_end - logical_begin

        for local_edge = lane;
            local_edge < edge_count;
            local_edge += 32:

            logical_edge = logical_begin + local_edge

            physical_edge = map_packed_edge(
                producer_prefix,
                producer_starts,
                logical_edge
            )

            stage_post[stage][local_edge] =
                graph_indices[physical_edge]

            stage_weight[stage][local_edge] =
                graph_weight[physical_edge]

        __syncwarp()

        if lane == 0:
            stage_batch[stage] = batch
            stage_edge_count[stage] = edge_count
            stage_task_id[stage] = task

            memory_fence_block()

            stage_state[stage] = STAGE_READY

        next_stage ^= 1

    if lane == 0:
        memory_fence_block()
        producer_done = 1
```

这一版虽然没有真正隐藏 producer warp内部的 global load latency，但能验证生产者—消费者架构和 shared staging成本。

---

## 12. Producer伪代码：连续布局异步版本

预处理后：

```text
task_queue_start[slot]
```

可直接或间接给出 packed edge offset。

伪代码：

```cpp
function run_block_task_producer_async():

    int next_stage = 0

    while true:

        task = fetch_next_block_task()

        if task >= block_count:
            break

        stage = wait_for_empty_stage(next_stage)

        descriptor = load_task_descriptor(task)

        packed_begin =
            packed_block_offsets[descriptor.block]
            + descriptor.segment * 512

        edge_count = min(
            512,
            packed_block_end - packed_begin
        )

        if lane == 0:
            stage_state[stage] = STAGE_LOADING
            stage_batch[stage] = descriptor.batch
            stage_edge_count[stage] = edge_count
            stage_task_id[stage] = task

        __syncwarp()

        issue_async_copy(
            source_posts =
                packed_graph_indices + packed_begin,
            destination_posts =
                stage_post[stage],
            bytes =
                edge_count * sizeof(int)
        )

        issue_async_copy(
            source_weights =
                packed_graph_weights + packed_begin,
            destination_weights =
                stage_weight[stage],
            bytes =
                edge_count * sizeof(float)
        )

        async_copy_commit(stage)
        async_copy_wait(stage)

        if lane == 0:
            memory_fence_block()
            stage_state[stage] = STAGE_READY

        next_stage ^= 1

    if lane == 0:
        producer_done = 1
```

对于真正 TMA：

```text
issue_async_copy
async_copy_commit
async_copy_wait
```

应替换为对应的 tensor map、transaction barrier和 bulk tensor copy。

第一版只需一维连续 tile，不需要设计复杂多维 tensor map。

---

## 13. Consumer伪代码

```cpp
function run_block_task_consumer():

    while true:

        stage = try_acquire_ready_stage()

        if stage < 0:
            if producer_done && all_stages_empty():
                break

            backoff_or_retry()
            continue

        batch = stage_batch[stage]
        edge_count = stage_edge_count[stage]

        process_hash_tile(
            stage,
            batch,
            edge_count,
            consumer_id
        )

        if lane == 0:
            memory_fence_block()
            stage_state[stage] = STAGE_EMPTY
```

stage领取需要保证一个 stage只能被一个 consumer warp领取。

可以使用：

```cpp
if lane == 0:
    old = atomicCAS(
        &stage_state[stage],
        STAGE_READY,
        STAGE_BUSY
    )

acquired = shfl(old == STAGE_READY)
```

伪代码：

```cpp
function try_acquire_ready_stage():

    for candidate in {0, 1}:

        acquired = false

        if lane == 0:
            old = atomicCAS(
                &stage_state[candidate],
                STAGE_READY,
                STAGE_BUSY
            )

            acquired = old == STAGE_READY

        acquired = shfl(acquired, 0)

        if acquired:
            return candidate

    return -1
```

由于只有两个 stage、七个 consumer，这里可能有多个 consumer竞争同一 stage，但 `atomicCAS` 能保证唯一所有者。

---

## 14. Hash tile处理伪代码

基本沿用当前路径：

```cpp
function process_hash_tile(
    stage,
    batch,
    edge_count,
    consumer_id
):
    used_count = 0

    if edge_count < HASH_MIN_EDGES:
        process_direct_atomic(stage)
        release_stage()
        return

    for edge_base = 0;
        edge_base < edge_count;
        edge_base += 32:

        local_edge = edge_base + lane
        valid = local_edge < edge_count

        if valid:
            post = stage_post[stage][local_edge]
            weight = stage_weight[stage][local_edge]

        destination_slot = -1
        claimed_new = false

        if valid:
            slot = hash(post)

            for probe in 0 .. MAX_PROBE - 1:
                old = atomicCAS(
                    &hash_keys[consumer_id][slot],
                    EMPTY_KEY,
                    post
                )

                if old == EMPTY_KEY or old == post:
                    destination_slot = slot
                    claimed_new = old == EMPTY_KEY

                    if claimed_new:
                        hash_values[consumer_id][slot] = 0

                    break

                slot = next_slot(slot)

        record_new_used_slot(
            claimed_new,
            destination_slot,
            used_count
        )

        __syncwarp()

        if destination_slot >= 0:
            atomicAdd(
                &hash_values[consumer_id][destination_slot],
                weight
            )

        else if valid:
            atomicAdd(
                psc + batch * n_neuron + post,
                weight
            )

        __syncwarp()

    flush_hash(
        batch,
        consumer_id,
        used_count
    )
```

Flush：

```cpp
function flush_hash(
    batch,
    consumer_id,
    used_count
):
    for i = lane; i < used_count; i += 32:

        slot =
            hash_used_slots[consumer_id][i]

        post =
            hash_keys[consumer_id][slot]

        value =
            hash_values[consumer_id][slot]

        atomicAdd(
            psc + batch * n_neuron + post,
            value
        )

        hash_keys[consumer_id][slot] =
            EMPTY_KEY

    __syncwarp()
```

---

## 15. Pipeline drain

在进入 timestep末尾的 `grid.sync()` 前，必须满足：

```text
producer_done == true
stage 0 == EMPTY
stage 1 == EMPTY
所有 consumer 已完成 flush
```

伪代码：

```cpp
if is_consumer:
    while true:
        stage = try_acquire_ready_stage()

        if stage >= 0:
            process_hash_tile(stage)
            release_stage(stage)
            continue

        if producer_done && all_stages_empty():
            break

__syncthreads()

grid.sync()
```

不能让 producer在最后一个 copy完成前发布 done。

正确顺序：

```text
finish final copy
publish final stage READY
then producer_done = 1
```

而不是：

```text
issue final copy
producer_done = 1
```

---

## 16. 预处理连续 layout

如果 Version P证明流水线本身有价值，再加入 TMA-friendly packed layout。

### 16.1 数据结构

对于每个 32-neuron block：

```cpp
packed_block_offsets[block_count + 1]
packed_graph_indices[total_edges]
packed_graph_weights[total_edges]
packed_edge_owner[total_edges]   // 可选
```

最简单布局：

```text
block 0:
    row 0 edges
    row 1 edges
    ...
    row 31 edges

block 1:
    row 0 edges
    ...
```

但运行时只有 spike mask中的 row有效。

因此还有两种方案。

### 方案 A：搬完整 block span，再用 owner mask过滤

每条 packed edge附带：

```text
owner_lane ∈ [0, 31]
```

consumer判断：

```cpp
valid = spike_mask & (1u << owner_lane)
```

优点：

* 完全连续；
* TMA简单；
* 不需要运行时 gather。

缺点：

* 会加载未发放 neuron的 edge；
* 低活动率下浪费严重；
* tile含义不再是 512 active edges。

第一版不建议直接采用，除非离线统计显示 active edge覆盖率足够高。

### 方案 B：运行时仍构造 active packed buffer

producer根据 spike mask收集 active row，并将逻辑 512-edge task复制进 shared。

这仍然需要 gather，但 producer可以集中完成，consumer不再承担 mapping和load。

它未必能使用真正 TMA，但可作为第一阶段有效实现。

### 方案 C：预生成常见 active组合

不建议第一版实现。32-bit spike mask组合空间太大。

---

## 17. 推荐实际开发顺序

### Step 1：重构 hash数组索引

将：

```cpp
hash_keys[kWarpsPerBlock][kHashSize]
```

改为：

```cpp
hash_keys[kConsumerWarps][kHashSize]
```

并增加：

```cpp
consumer_id = warp_in_block - 1
```

确认：

* producer不访问 hash；
* 原有正确性测试通过；
* shared memory下降约一份 hash；
* 暂时仍让 warp 1–7通过原 task queue直接工作。

这一阶段先不做 pipeline。

### Step 2：建立双 stage状态机

只加入：

```text
EMPTY
READY
BUSY
producer_done
```

producer生成假 tile或普通 shared copy，consumer领取 tile。

先验证：

* 没有死锁；
* 没有重复领取；
* 没有遗漏 task；
* stage最终都回到 EMPTY；
* spike/PSC正确。

### Step 3：实现 producer普通 gather

producer执行现有：

```cpp
map_packed_edge
graph_indices load
graph_weight load
```

consumer只读 shared stage并做 hash。

得到：

```text
WS-gather-single-buffer
```

先只用一个 stage，隔离 specialization本身。

### Step 4：加入双缓冲

得到：

```text
WS-gather-double-buffer
```

比较：

```text
current hash
vs
single-buffer producer
vs
double-buffer producer
```

如果双缓冲没有优于单缓冲，说明：

* producer没有形成可重叠工作；
* producer成为瓶颈；
* stage同步成本过高；
* consumer hash/flush不够长；
* shared staging本身没有价值。

此时不应立刻继续实现 TMA。

### Step 5：压缩 shared memory

检查编译和 NCU：

```text
shared memory/block
active blocks/SM
achieved occupancy
registers/thread
```

如果只能 1 block/SM：

1. 先将 used-slot list改 bitmap；
2. 再压缩 stage metadata；
3. 再考虑只双缓冲 post；
4. 最后才考虑减小 edge tile。

不要首先把 aggregation退回 256。

### Step 6：加入连续 packed edge layout

消除 consumer路径中的：

```cpp
map_packed_edge()
graph_indices[edge]
graph_weight[edge]
```

将 task descriptor改为：

```text
packed edge offset
edge count
batch
```

### Step 7：替换为 TMA/bulk asynchronous copy

得到：

```text
WS-TMA-double-buffer
```

完成完整消融。

---

## 18. 必须完成的消融实验

| 版本 | Producer     | Buffer | Edge layout       | 目的               |
| -- | ------------ | -----: | ----------------- | ---------------- |
| H0 | 无            |      0 | 当前 CSR            | 当前 512 hash基线    |
| H1 | 专用 warp      |      1 | 当前 gather         | specialization开销 |
| H2 | 专用 warp      |      2 | 当前 gather         | 双缓冲收益            |
| H3 | 专用 warp      |      1 | packed continuous | layout收益         |
| H4 | 专用 warp      |      2 | packed continuous | layout + overlap |
| H5 | TMA producer |      2 | packed continuous | TMA增量收益          |

不能只比较 H0 和 H5，否则无法确认性能来源。

---

## 19. Benchmark配置

首先保持与 block/hash最佳实验完全一致：

```text
GPU: RTX 5090
dataset: FlyBrain / FlyWire 783
neurons: 138,639
edges: 15,091,983
reorder: global_similarity
block edge budget: 256
hash aggregation: 512
hash capacity: 512
max probe: 4
min edges: 256
event rate: 0.02
batch: 1
timesteps: 128
warmup: 10
repeat: 50
```

然后补充：

```text
event rate:
0.005
0.020
0.100

batch:
1
4

reorder:
identity
global_similarity
```

当前 hash在低活动率、较高活动率和 batch=4 下均保持正收益，因此新流水线不能只在单一活动率有效。

---

## 20. 新增运行时统计

建议增加：

```cpp
producer_tasks
producer_edges
consumer_tasks
consumer_edges

producer_wait_empty_iterations
consumer_wait_ready_iterations

stage_0_acquires
stage_1_acquires

partial_tiles
full_tiles

pipeline_drain_iterations
pipeline_fallback_direct_tasks
```

派生指标：

```text
average edges/tile
partial tile ratio
producer wait ratio
consumer wait ratio
stage balance
```

解释：

### producer wait高

```text
consumer hash/flush慢
stage经常占满
edge load可能已被充分覆盖
```

### consumer wait高

```text
producer gather/load慢
producer warp成为瓶颈
双 stage不足或布局不连续
```

### producer和consumer wait都低，但性能无提升

```text
原实现已有足够 latency hiding
新增 shared copy和同步抵消收益
```

---

## 21. NCU重点指标

### Occupancy和资源

```text
launch__shared_mem_per_block
launch__registers_per_thread
launch__occupancy_limit_shared_mem
sm__warps_active
```

硬性目标：

```text
active blocks/SM = 2
```

如果下降到 1，需要将其作为独立对照，而不是直接接受。

### 调度效率

重点比较：

```text
smsp__warps_eligible.avg.per_cycle_active
smsp__issue_active.avg.pct_of_peak_sustained_active
smsp__warps_active.avg.per_cycle_active
```

目标：

```text
eligible warps明显增加
issue active明显增加
```

即使 achieved occupancy不变，只要 eligible warp增加，也说明 specialization有效。

### Stall变化

重点看：

```text
Long Scoreboard
Short Scoreboard
Barrier
MIO Throttle
LG Throttle
Wait
```

理想结果：

```text
Long Scoreboard ↓
Eligible Warps  ↑
Issue Active    ↑
Kernel Time     ↓
```

失败模式：

```text
Long Scoreboard ↓
Barrier         ↑
MIO Throttle    ↑
Kernel Time     ≈ 或 ↑
```

说明只是把 global等待转移成 shared同步开销。

---

## 22. 正确性测试

每个阶段都必须通过：

1. 单 timestep PyTorch对照；
2. 多 timestep对照；
3. spike mismatch为 0；
4. PSC误差保持在现有容差；
5. 空 spike mask；
6. 只有一个 active neuron；
7. tile正好 512 edges；
8. tile为 513 edges；
9. 最后一个 partial tile；
10. producer结束时一个 stage仍处于 READY；
11. 两个 consumer竞争同一 stage；
12. batch大于 1；
13. hash fallback；
14. long-row和block-row同时存在；
15. timestep drain后再进入下一步。

特别需要增加：

```text
producer提前结束
consumer仍在flush
```

以及：

```text
最后一个tile copy完成后才发布producer_done
```

两个回归测试。

---

## 23. 第一版成功标准

满足以下条件，才继续拆独立 flush warp。

### 必须满足

```text
正确性通过
无死锁
2 blocks/SM保持
512-edge aggregation保持
```

### 性能标准

以当前最佳 512 hash为基线：

```text
kernel speedup >= 5%
```

或至少同时满足：

```text
kernel speedup >= 3%
eligible warps显著增加
long scoreboard显著下降
```

### 停止条件

若出现以下任一情况，应停止继续复杂化：

```text
只能保持1 block/SM且性能下降
producer wait接近0但consumer长期等待producer
double buffer不优于single buffer
packed layout后仍无正收益
TMA相对普通async copy无显著增益
```

这说明当前瓶颈不适合继续通过 load pipeline优化。

---

## 24. 后续进入独立 flush的条件

只有同时观察到：

```text
edge load相关long scoreboard下降
producer经常等待empty stage
consumer主要停在global atomic/flush
hash阶段明显短于flush阶段
```

才进入第二版：

```text
1 producer
5–6 hash consumer
1–2 flush warp
```

第二版再加入：

```text
hash → compact flush list
flush queue
独立 global flush
```

否则第一版的：

```text
1 producer + 7 hash/flush consumer
```

应作为最终 warp-specialized结构。

---

## 25. 第一版最终结构摘要

```text
Block: 256 threads

Warp 0:
    fetch ordinary block task
    construct 512-edge tile
    copy post/weight to shared stage
    publish READY

Warp 1–7:
    acquire READY stage
    hash 512 edges
    fallback global atomic
    flush used hash slots
    release EMPTY stage

Shared:
    7 warp-private 512-entry hash tables
    2 × 512-edge staging buffers
    stage states and metadata

Pipeline:
    load tile k+1
        overlaps
    hash + flush tile k

Constraints:
    keep shared memory below 49 KB/block
    preserve 2 blocks/SM
    preserve 512-edge aggregation
```

第一版的主要贡献不是立刻建立完整三阶段流水线，而是先验证：

> 将 edge mapping/load 从七个计算 warp中剥离到一个 producer warp，并通过双缓冲让下一块 edge准备与当前块的 hash和global flush重叠，是否能提高真正可发射的 warp数量。

这一步成立后，再考虑将 global flush进一步解耦。
