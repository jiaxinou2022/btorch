# Persistent RSNN：UPDATE–Propagation Block 流水化执行计划

## 1. 改造目标

当前单个时间步执行顺序为：

```text
初始化计数器
→ 注入外部事件
→ grid.sync()
→ 全 grid UPDATE + 生成 spike task
→ grid.sync()
→ 全 grid 传播 recurrent current
→ grid.sync()
```

计划改为：

```text
初始化计数器
→ 注入外部事件
→ grid.sync()
→ UPDATE blocks 更新神经元并持续发布 fragment task
  与
  PROPAGATION blocks 持续消费 task、计算 recurrent current
→ UPDATE 全部结束且所有 task 全部处理完成
→ grid.sync()
→ 下一时间步
```

即消除：

```text
UPDATE 完成 → grid.sync() → propagation 开始
```

保留时间步末同步，以保证当前时间步生成的 recurrent current 在下一时间步 UPDATE 前全部完成。

由于 RSNN 推理始终满足：

```cpp
batch_size == 1
```

后续 kernel 内不再保留 batch 循环，也不再在任务队列中记录 batch。

---

# 2. 必须首先修正 PSC 数据依赖

当前代码中：

```cpp
const float current = psc[n] + input_current[n];
psc[n] *= decay;
```

随后 propagation 对同一个 `psc[post]` 执行：

```cpp
atomicAdd(psc + post, weight);
```

原来的中间 `grid.sync()` 保证：

```text
所有 psc 衰减完成
→ propagation 才开始 atomicAdd
```

删除同步后，UPDATE 对 `psc[n]` 的普通写和 propagation 的 `atomicAdd` 会发生竞争。

因此必须把当前 PSC 状态和本时间步新产生的 recurrent 增量分离。

## 2.1 新的状态定义

保留：

```cpp
float* psc;
```

其语义改为：

```text
经过历史累积和衰减后的 PSC 状态
```

新增两个缓冲区：

```cpp
float* recurrent_delta_0;
float* recurrent_delta_1;
```

记为：

```cpp
float* delta_read  = (t & 1) ? recurrent_delta_1
                             : recurrent_delta_0;

float* delta_write = (t & 1) ? recurrent_delta_0
                             : recurrent_delta_1;
```

其中：

```text
delta_read[n]
    上一个时间步的 spike propagation 产生、
    当前 UPDATE 要消费的增量

delta_write[n]
    当前时间步 spike propagation 正在生成、
    下一时间步 UPDATE 才会消费的增量
```

## 2.2 UPDATE 新语义

当前：

```cpp
current = psc[n] + input_current[n];
psc[n] *= decay;
```

改为：

```cpp
const float recurrent = psc[n] + delta_read[n];
delta_read[n] = 0.0f;

const float current = recurrent + input_current[n];
input_current[n] = 0.0f;

update_voltage(...);

psc[n] = recurrent * decay;
```

Propagation 不再写 `psc`，而是：

```cpp
atomicAdd(delta_write + post, graph_weight[edge]);
```

于是 UPDATE 和 propagation 访问不同的输出数组：

```text
UPDATE：
    读写 psc
    读并清空 delta_read

PROPAGATION：
    atomicAdd delta_write
```

二者可以安全并发。

---

# 3. Kernel 参数与辅助存储修改

## 3.1 删除任务元数据数组

删除：

```cpp
int* spike_queue_batch;
int* spike_queue_edge_start;
int* spike_queue_edge_end;
```

由于 B 固定为 1，任务只需要保存来源 neuron：

```cpp
int* spike_queue_neuron;
```

新增任务状态：

```cpp
uint32_t* spike_queue_state;
```

其状态定义为：

```text
0：
    当前队列槽尚未发布完成

非零：
    fragment_id + 1
```

第一版每个时间步都会从任务槽 0 开始重新使用。由于时间步末存在 `grid.sync()`，并且只会读取当前 `spike_count` 范围内的槽，可以在时间步开始时清空上一轮使用的 state。

暂不引入 epoch 编码，先保证实现简单、便于验证。

## 3.2 新增同步计数器

当前：

```cpp
int* spike_count;
int* work_counter;
```

建议明确改名：

```cpp
int* queue_tail;             // 已 reserve 的任务总数
int* queue_head;             // 下一个待领取 task
int* update_done_blocks;     // 完成 UPDATE 的 block 数
int* propagation_done_tasks; // 已完整处理的 task 数
```

其中：

```text
queue_tail
    producer 已经预留的任务数量

queue_head
    consumer 已经领取的任务数量

update_done_blocks
    判断之后是否还可能出现新任务

propagation_done_tasks
    判断已领取任务是否真正完成
```

暂时忽略 queue overflow，默认 `queue_capacity` 足够。

## 3.3 新增角色划分参数

Kernel 新增：

```cpp
int update_block_count;
```

固定：

```cpp
const bool is_update_block =
    blockIdx.x < update_block_count;

const bool is_propagation_block =
    blockIdx.x >= update_block_count;

const int propagation_block_count =
    gridDim.x - update_block_count;
```

要求：

```text
1 <= update_block_count < gridDim.x
```

---

# 4. Block 级角色划分

## 4.1 UPDATE block 的线程编号

当前 UPDATE 使用全 grid 的：

```cpp
global_tid
stride = blockDim.x * gridDim.x
```

改造后只有 UPDATE blocks 参与神经元更新，因此必须使用角色内部编号：

```cpp
const int update_tid =
    blockIdx.x * blockDim.x + threadIdx.x;

const int update_stride =
    update_block_count * blockDim.x;
```

UPDATE 遍历：

```cpp
for (int n = update_tid;
     n < n_neuron;
     n += update_stride) {
    ...
}
```

## 4.2 Propagation block 的 warp

Propagation 使用 block 内 warp：

```cpp
const int lane = threadIdx.x & 31;
const int warp_in_block = threadIdx.x >> 5;
```

每个 warp 独立领取一个 task。

只要求：

```cpp
blockDim.x % 32 == 0
```

---

# 5. 时间步初始化

时间步开始时，需要初始化本轮计数器，并清理上一时间步使用过的 task state。

由于上一轮任务数量会在初始化时被覆盖，需要额外保留：

```cpp
int previous_task_count = *queue_tail;
```

更直接的做法是在时间步末清理，或者增加：

```cpp
int* previous_queue_tail;
```

第一版推荐时间步开始时：

```cpp
if (global_tid == 0) {
    *previous_queue_tail = *queue_tail;

    *queue_tail = 0;
    *queue_head = 0;
    *update_done_blocks = 0;
    *propagation_done_tasks = 0;
}
grid.sync();
```

然后清空上一轮 state：

```cpp
const int previous_count = *previous_queue_tail;

for (int task = global_tid;
     task < previous_count;
     task += blockDim.x * gridDim.x) {
    spike_queue_state[task] = 0;
}
grid.sync();
```

但这会增加一次全 grid 同步。

更简洁的第一版可以把 state 清理放在上一时间步末尾：

```cpp
grid.sync();

const int finished_task_count = *queue_tail;

for (int task = global_tid;
     task < finished_task_count;
     task += stride) {
    spike_queue_state[task] = 0;
}

grid.sync();
```

这样每一时间步结构变成：

```text
流水化 UPDATE + propagation
→ grid.sync()
→ 清空已用 state
→ grid.sync()
```

这会多出一个同步，因此性能实验阶段应进一步改成 epoch state。

更推荐直接使用 epoch 编码，避免清空。

## 5.1 推荐采用 epoch + fragment 编码

定义：

```cpp
constexpr int kFragmentBits = 16;
constexpr uint32_t kFragmentMask = 0xffffu;
```

状态：

```text
state[31:16] = timestep + 1
state[15:0]  = fragment_id + 1
```

Producer：

```cpp
const uint32_t task_state =
    (static_cast<uint32_t>(t + 1) << 16) |
    static_cast<uint32_t>(fragment + 1);
```

Consumer 只接受：

```cpp
(state >> 16) == t + 1
```

这样：

* 不需要清空 `spike_queue_state`；
* 旧时间步的非零值不会被误认为当前任务已发布；
* 时间步初始化只需清零四个计数器。

第一版即采用该方案。

---

# 6. 外部事件注入

目前事件注入使用整个 cooperative grid：

```cpp
for (int event = start + global_tid;
     event < end;
     event += stride) {
    atomicAdd(input_current + pre, value);
}
```

这一阶段仍允许所有 blocks 参与，因为此时还未开始角色分流。

B 固定为 1 后：

```cpp
const int start = event_offsets[t];
const int end = event_offsets[t + 1];

for (int event = start + global_tid;
     event < end;
     event += global_stride) {
    const int pre = event_indices[event];

    if (pre >= 0 && pre < n_neuron) {
        const float value =
            has_event_values ? event_values[event] : 1.0f;

        atomicAdd(input_current + pre, value);
    }
}
```

之后保留：

```cpp
grid.sync();
```

保证所有外部输入都已经写入，UPDATE 才能读取。

---

# 7. UPDATE producer 逻辑

## 7.1 神经元更新

每个 UPDATE thread 独占其分配到的 neuron：

```cpp
if (is_update_block) {
    for (int n = update_tid;
         n < n_neuron;
         n += update_stride) {

        const float recurrent =
            psc[n] + delta_read[n];

        delta_read[n] = 0.0f;

        const float current =
            recurrent + input_current[n];

        input_current[n] = 0.0f;

        const float v_pre =
            v[n] +
            dt * (
                -(v[n] - v_reset) / tau_mem +
                current / c_m);

        const bool fired =
            v_pre >= v_threshold;

        const float spike =
            fired ? 1.0f : 0.0f;

        v[n] =
            v_pre - reset_delta * spike;

        psc[n] =
            recurrent * decay;

        if constexpr (ReturnDense) {
            dense_spikes[t * n_neuron + n] =
                spike;
        }

        if (fired) {
            publish_fragments(n, t);
            record_event_if_needed(n, t);
        }
    }
}
```

## 7.2 Fragment 数量

对于发放的 neuron：

```cpp
const int row_start = graph_indptr[n];
const int row_end = graph_indptr[n + 1];
const int fanout = row_end - row_start;

const int fragment_count =
    (fanout + kEdgesPerTask - 1) /
    kEdgesPerTask;
```

若：

```cpp
fragment_count == 0
```

无需发布任务。

## 7.3 预留任务槽

```cpp
const int first_task =
    atomicAdd(queue_tail, fragment_count);
```

暂不处理：

```text
first_task + fragment_count > queue_capacity
```

假设容量充分。

## 7.4 写入 neuron ID

同一个 fired neuron 的所有 fragment 都对应同一个 neuron：

```cpp
for (int fragment = 0;
     fragment < fragment_count;
     ++fragment) {
    spike_queue_neuron[first_task + fragment] = n;
}
```

## 7.5 发布任务

必须保证 consumer 看到 state 后，能够读到正确的 neuron ID。

因此：

```cpp
__threadfence();
```

随后发布每个 fragment：

```cpp
const uint32_t epoch =
    static_cast<uint32_t>(t + 1);

for (int fragment = 0;
     fragment < fragment_count;
     ++fragment) {
    const int task =
        first_task + fragment;

    const uint32_t state =
        (epoch << 16) |
        static_cast<uint32_t>(fragment + 1);

    atomicExch(
        spike_queue_state + task,
        state);
}
```

这里一个 fired neuron 只执行一次 `__threadfence()`。

任务从 state 写入完成的时刻开始对 consumer 可见。

## 7.6 UPDATE block 完成标志

不能由每个 thread 单独更新完成计数器。

UPDATE block 内所有线程完成自己的 neuron 循环后：

```cpp
__syncthreads();

if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd(update_done_blocks, 1);
}
```

`__threadfence()` 保证该 block 发布的 task 和其他全局写在完成标志前可见。

随后第一版中 UPDATE block 不再执行其他工作，等待时间步末 `grid.sync()`。

后续可扩展为 UPDATE block 转 consumer helper，但不纳入本轮实现。

---

# 8. Propagation consumer 逻辑

Propagation block 中每个 warp不断领取任务。

## 8.1 领取任务

不能仅使用：

```cpp
task = atomicAdd(queue_head, 1);
```

因为 consumer 可能在当前没有任务时把 `queue_head` 推到 `queue_tail` 之外。

使用 CAS 领取：

```cpp
int task = -1;

if (lane == 0) {
    while (true) {
        const int head =
            atomicAdd(queue_head, 0);

        const int tail =
            atomicAdd(queue_tail, 0);

        if (head >= tail) {
            break;
        }

        const int old =
            atomicCAS(
                queue_head,
                head,
                head + 1);

        if (old == head) {
            task = head;
            break;
        }
    }
}

task = __shfl_sync(
    0xffffffffu,
    task,
    0);
```

此时：

```text
task >= 0
    成功领取任务

task == -1
    当前暂时没有可领取任务
```

## 8.2 等待任务发布

`queue_tail` 在 producer 写 descriptor 之前已经增长，因此 consumer 可能领取一个尚未完成发布的槽。

领取后等待 state epoch 匹配当前时间步：

```cpp
uint32_t state = 0;

if (lane == 0) {
    const uint32_t expected_epoch =
        static_cast<uint32_t>(t + 1);

    do {
        state = atomicAdd(
            spike_queue_state + task,
            0u);
    } while ((state >> 16) != expected_epoch);
}

state = __shfl_sync(
    0xffffffffu,
    state,
    0);
```

暂不处理无限自旋问题。

## 8.3 恢复 fragment 信息

```cpp
const int fragment =
    static_cast<int>(
        state & 0xffffu) - 1;

const int neuron =
    spike_queue_neuron[task];

const int row_start =
    graph_indptr[neuron];

const int row_end =
    graph_indptr[neuron + 1];

const int edge_start =
    row_start +
    fragment * kEdgesPerTask;

const int edge_end =
    min(
        edge_start + kEdgesPerTask,
        row_end);
```

因此不再需要：

```cpp
spike_queue_edge_start
spike_queue_edge_end
spike_queue_batch
```

## 8.4 执行传播

```cpp
for (int edge = edge_start + lane;
     edge < edge_end;
     edge += 32) {

    const int post =
        graph_indices[edge];

    const float weight =
        graph_weight[edge];

    atomicAdd(
        delta_write + post,
        weight);
}
```

## 8.5 标记任务真正完成

warp 内 propagation 完成后：

```cpp
__syncwarp();

if (lane == 0) {
    __threadfence();
    atomicAdd(
        propagation_done_tasks,
        1);
}
```

`queue_head` 只表示任务已领取，不能用于判断传播是否结束。

必须使用：

```cpp
propagation_done_tasks
```

表示已经执行完所有 edge atomicAdd 的任务数量。

---

# 9. Consumer 退出条件

Propagation block 在当前暂时领不到任务时，检查：

```cpp
const bool all_updates_done =
    atomicAdd(update_done_blocks, 0) ==
    update_block_count;

const int tail =
    atomicAdd(queue_tail, 0);

const int done =
    atomicAdd(propagation_done_tasks, 0);
```

只有满足：

```cpp
all_updates_done &&
done == tail
```

才能退出 consumer loop。

不能使用：

```cpp
queue_head == queue_tail
```

因为它只表示所有任务已被领取，最后几个任务可能仍在计算。

伪代码：

```cpp
while (true) {
    int task = try_claim_task();

    if (task >= 0) {
        wait_until_ready(task, t);
        process_task(task);
        mark_task_done();
        continue;
    }

    const bool producers_finished =
        atomicAdd(update_done_blocks, 0) ==
        update_block_count;

    if (producers_finished) {
        const int tail =
            atomicAdd(queue_tail, 0);

        const int done =
            atomicAdd(
                propagation_done_tasks,
                0);

        if (done >= tail) {
            break;
        }
    }

    // 当前无任务，但 producer 仍可能继续发布；
    // 直接进入下一轮轮询。
}
```

---

# 10. 时间步完整伪代码

```cpp
template <bool ReturnDense>
__global__ void persistent_snn_kernel(...) {
    cg::grid_group grid =
        cg::this_grid();

    const int global_tid =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    const int global_stride =
        gridDim.x * blockDim.x;

    const bool is_update_block =
        blockIdx.x < update_block_count;

    const int update_tid =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    const int update_stride =
        update_block_count *
        blockDim.x;

    const int lane =
        threadIdx.x & 31;

    const float decay =
        expf(-dt / tau_syn);

    const float reset_delta =
        v_threshold - v_reset;

    for (int t = 0;
         t < t_steps;
         ++t) {

        float* delta_read =
            (t & 1)
                ? recurrent_delta_1
                : recurrent_delta_0;

        float* delta_write =
            (t & 1)
                ? recurrent_delta_0
                : recurrent_delta_1;

        // ---------------------------------
        // Phase 0: 初始化时间步计数器
        // ---------------------------------

        if (global_tid == 0) {
            *queue_tail = 0;
            *queue_head = 0;
            *update_done_blocks = 0;
            *propagation_done_tasks = 0;
        }

        grid.sync();

        // ---------------------------------
        // Phase 1: 外部事件注入
        // 全 grid 参与
        // ---------------------------------

        const int event_start =
            event_offsets[t];

        const int event_end =
            event_offsets[t + 1];

        for (int event =
                 event_start + global_tid;
             event < event_end;
             event += global_stride) {

            const int pre =
                event_indices[event];

            if (pre >= 0 &&
                pre < n_neuron) {

                const float value =
                    has_event_values
                        ? event_values[event]
                        : 1.0f;

                atomicAdd(
                    input_current + pre,
                    value);
            }
        }

        grid.sync();

        // ---------------------------------
        // Phase 2A: UPDATE producer
        // ---------------------------------

        if (is_update_block) {
            for (int n = update_tid;
                 n < n_neuron;
                 n += update_stride) {

                const float recurrent =
                    psc[n] +
                    delta_read[n];

                delta_read[n] = 0.0f;

                const float current =
                    recurrent +
                    input_current[n];

                input_current[n] = 0.0f;

                const float v_pre =
                    v[n] +
                    dt * (
                        -(v[n] - v_reset) /
                            tau_mem +
                        current / c_m);

                const bool fired =
                    v_pre >= v_threshold;

                const float spike =
                    fired ? 1.0f : 0.0f;

                v[n] =
                    v_pre -
                    reset_delta * spike;

                psc[n] =
                    recurrent * decay;

                if constexpr (ReturnDense) {
                    dense_spikes[
                        t * n_neuron + n
                    ] = spike;
                }

                if (fired) {
                    const int row_start =
                        graph_indptr[n];

                    const int row_end =
                        graph_indptr[n + 1];

                    const int fanout =
                        row_end - row_start;

                    const int fragment_count =
                        (fanout +
                         kEdgesPerTask - 1) /
                        kEdgesPerTask;

                    if (fragment_count > 0) {
                        const int first_task =
                            atomicAdd(
                                queue_tail,
                                fragment_count);

                        for (int fragment = 0;
                             fragment <
                                 fragment_count;
                             ++fragment) {
                            spike_queue_neuron[
                                first_task +
                                fragment
                            ] = n;
                        }

                        __threadfence();

                        const uint32_t epoch =
                            static_cast<
                                uint32_t
                            >(t + 1);

                        for (int fragment = 0;
                             fragment <
                                 fragment_count;
                             ++fragment) {

                            const int task =
                                first_task +
                                fragment;

                            const uint32_t state =
                                (epoch << 16) |
                                static_cast<
                                    uint32_t
                                >(fragment + 1);

                            atomicExch(
                                spike_queue_state +
                                    task,
                                state);
                        }
                    }

                    if (return_events) {
                        const int rank =
                            atomicAdd(
                                event_counts + t,
                                1);

                        if (rank < n_neuron) {
                            event_indices_full[
                                t * n_neuron +
                                rank
                            ] = n;
                        }
                    }
                }
            }

            __syncthreads();

            if (threadIdx.x == 0) {
                __threadfence();

                atomicAdd(
                    update_done_blocks,
                    1);
            }
        }

        // ---------------------------------
        // Phase 2B: propagation consumer
        // 与 UPDATE 并发
        // ---------------------------------

        else {
            while (true) {
                int task = -1;

                if (lane == 0) {
                    while (true) {
                        const int head =
                            atomicAdd(
                                queue_head,
                                0);

                        const int tail =
                            atomicAdd(
                                queue_tail,
                                0);

                        if (head >= tail) {
                            break;
                        }

                        const int old =
                            atomicCAS(
                                queue_head,
                                head,
                                head + 1);

                        if (old == head) {
                            task = head;
                            break;
                        }
                    }
                }

                task =
                    __shfl_sync(
                        0xffffffffu,
                        task,
                        0);

                if (task >= 0) {
                    uint32_t state = 0;

                    if (lane == 0) {
                        const uint32_t epoch =
                            static_cast<
                                uint32_t
                            >(t + 1);

                        do {
                            state =
                                atomicAdd(
                                    spike_queue_state +
                                        task,
                                    0u);
                        } while (
                            (state >> 16) !=
                            epoch);
                    }

                    state =
                        __shfl_sync(
                            0xffffffffu,
                            state,
                            0);

                    const int fragment =
                        static_cast<int>(
                            state & 0xffffu
                        ) - 1;

                    const int neuron =
                        spike_queue_neuron[
                            task
                        ];

                    const int row_start =
                        graph_indptr[
                            neuron
                        ];

                    const int row_end =
                        graph_indptr[
                            neuron + 1
                        ];

                    const int edge_start =
                        row_start +
                        fragment *
                            kEdgesPerTask;

                    const int edge_end =
                        min(
                            edge_start +
                                kEdgesPerTask,
                            row_end);

                    for (int edge =
                             edge_start + lane;
                         edge < edge_end;
                         edge += 32) {

                        const int post =
                            graph_indices[
                                edge
                            ];

                        atomicAdd(
                            delta_write +
                                post,
                            graph_weight[
                                edge
                            ]);
                    }

                    __syncwarp();

                    if (lane == 0) {
                        __threadfence();

                        atomicAdd(
                            propagation_done_tasks,
                            1);
                    }

                    continue;
                }

                bool should_exit = false;

                if (lane == 0) {
                    const bool
                        producers_finished =
                            atomicAdd(
                                update_done_blocks,
                                0) ==
                            update_block_count;

                    if (producers_finished) {
                        const int tail =
                            atomicAdd(
                                queue_tail,
                                0);

                        const int done =
                            atomicAdd(
                                propagation_done_tasks,
                                0);

                        should_exit =
                            done >= tail;
                    }
                }

                should_exit =
                    __shfl_sync(
                        0xffffffffu,
                        should_exit,
                        0);

                if (should_exit) {
                    break;
                }
            }
        }

        // 保证 delta_write 已完全产生，
        // 下一时间步才可作为 delta_read 使用。
        grid.sync();
    }
}
```

---

# 11. Host 接口调整

## 11.1 删除参数

删除：

```cpp
int batch_size;

int* spike_queue_batch;
int* spike_queue_edge_start;
int* spike_queue_edge_end;

int* spike_count;
int* work_counter;
```

若上层接口暂时仍要求传入 `batch_size`，可以保留 host 参数并检查：

```cpp
if (batch_size != 1) {
    return error;
}
```

但不再传入 kernel。

## 11.2 新增参数

```cpp
float* recurrent_delta_0;
float* recurrent_delta_1;

int* spike_queue_neuron;
uint32_t* spike_queue_state;

int* queue_tail;
int* queue_head;
int* update_done_blocks;
int* propagation_done_tasks;

int update_block_count;
```

## 11.3 初始化

Kernel 首次启动前：

```cpp
cudaMemsetAsync(
    recurrent_delta_0,
    0,
    n_neuron * sizeof(float),
    stream);

cudaMemsetAsync(
    recurrent_delta_1,
    0,
    n_neuron * sizeof(float),
    stream);

cudaMemsetAsync(
    spike_queue_state,
    0,
    queue_capacity *
        sizeof(uint32_t),
    stream);
```

`spike_queue_neuron` 不需要初始化，因为只有 state epoch 匹配后才会读取。

---

# 12. 实施阶段

## 阶段 A：PSC 双缓冲，暂不重叠

先只验证状态拆分，不立刻进行 block 角色划分。

执行：

```text
全 grid UPDATE
→ grid.sync()
→ 全 grid propagation 写 delta_write
→ grid.sync()
```

需要确认：

* `dense_spikes` 与原版一致；
* event 输出一致；
* `v` 与原版一致；
* 下一时间步读取的 current 一致；
* 长时间运行没有 recurrent current 丢失。

这一步用来排除 PSC 语义错误。

## 阶段 B：任务描述符压缩

将队列改为：

```cpp
spike_queue_neuron
spike_queue_state
```

但仍维持串行阶段：

```text
UPDATE 发布全部任务
→ grid.sync()
→ propagation 消费
```

确认：

* fragment 恢复的 edge 范围正确；
* 每条活跃 fanout edge 恰好处理一次；
* 最后一个 fragment 不越过 `row_end`；
* 不同 fanout 长度下结果一致。

## 阶段 C：block 角色划分与并发

引入：

```cpp
update_block_count
```

实现：

```text
UPDATE blocks：
    更新神经元、发布任务

PROPAGATION blocks：
    并发消费任务
```

删除 UPDATE 后的 `grid.sync()`。

保留时间步末：

```cpp
grid.sync();
```

## 阶段 D：角色比例扫描

建议优先测试：

```text
update : propagation blocks

7 : 1
3 : 1
2 : 1
1 : 1
```

实际换算成 cooperative grid block 数，例如总共 32 blocks：

```text
28 + 4
24 + 8
20 + 12
16 + 16
```

应覆盖：

* 低 firing rate；
* 中 firing rate；
* 高 firing rate；
* 小规模网络；
* Flybrain 类大规模网络；
* 均匀 fanout；
* 重尾 fanout。

---

# 13. 正确性验证

## 13.1 单时间步测试

构造：

```text
少量 neuron
已知 CSR
已知初始 v/psc
手工控制发放 neuron
```

比较：

```text
原版 kernel
PSC 双缓冲串行版
block 流水化版
```

验证：

```cpp
dense_spikes
v
psc
delta_read / delta_write
event_indices
```

## 13.2 Fragment 边界测试

重点 fanout：

```text
0
1
31
32
1023
1024
1025
2048
2049
```

检查 fragment 数：

```text
0
1
1
1
1
1
2
2
3
```

并确认：

```text
不存在 edge 漏算
不存在 edge 重算
```

## 13.3 多时间步测试

至少测试：

```text
10
100
1000 timestep
```

观察误差是否随时间累积。

若 atomic 顺序变化造成浮点误差，只要求误差处于合理范围，不要求逐 bit 一致。

## 13.4 竞争条件检查

重点检查：

* UPDATE 是否只清空 `delta_read`；
* propagation 是否只写 `delta_write`；
* producer 是否先写 neuron 再发布 state；
* `update_done_blocks` 是否只在整个 block 完成后增加；
* `propagation_done_tasks` 是否只在 warp 完成所有 atomicAdd 后增加；
* 时间步结束是否依据 done task，而非 queue head。

---

# 15. 本轮暂不实现的内容

以下内容留到第一版结果明确后：

```text
UPDATE block 完成后转为 consumer helper
动态调整 UPDATE/PROPAGATION block 比例
每个 producer block 的局部队列
shared-memory spike staging
短 fanout 直接传播
queue overflow 处理
ready 自旋超时或错误处理
64-bit packed task descriptor
完全消除 timestep 末 grid.sync()
```

本轮只实现最小闭环：

```text
B=1
PSC 双缓冲
block 固定角色
neuron + fragment task
ready/epoch 发布协议
UPDATE–propagation 并发
时间步末 grid.sync
```

---

# 16. 最终施工顺序

```text
1. 强制 B=1，删除 kernel 内 batch 循环与 batch task 元数据

2. 增加 recurrent_delta 双缓冲
   修正 PSC 时间语义

3. 先实现串行双缓冲版
   验证 v、spike、current 结果

4. 将任务队列压缩为：
       queue_neuron
       queue_state(epoch + fragment)

5. 串行验证 fragment 恢复逻辑

6. 增加：
       queue_tail
       queue_head
       update_done_blocks
       propagation_done_tasks

7. 按 blockIdx 划分 UPDATE 和 PROPAGATION blocks

8. 实现 producer 发布协议：
       reserve
       write neuron
       threadfence
       publish state

9. 实现 consumer：
       CAS claim
       wait state
       decode fragment
       process edges
       mark done

10. 删除 UPDATE 后的 grid.sync()

11. 保留时间步末 grid.sync()

12. 扫描 block 角色比例并分析 overlap、tail 和资源争用
```
