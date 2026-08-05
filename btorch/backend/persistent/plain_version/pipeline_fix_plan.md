# Persistent SNN Pipeline 性能修正执行计划

## 1. 修正目标

保留当前总体结构：

```text
时间步初始化
→ 外部事件注入
→ grid.sync()

UPDATE blocks：
    更新 neuron
    spike 产生后立即发布 fragment task
    UPDATE 完成后转为 consumer

PROPAGATION blocks：
    从时间步开始持续消费 fragment task

全部 UPDATE 完成
且全部传播任务完成
→ grid.sync()
→ 下一时间步
```

本轮不取消 UPDATE block 的角色转换。

重点修正当前实现中的调度开销：

```text
consumer 高频轮询 queue_head / queue_tail
consumer 提前领取未发布 task 后自旋 ready
每 task 执行 device-wide threadfence
每个 warp 独立竞争全局 queue_head
host 侧对 epoch queue state 做不必要清零
```

---

# 2. 当前实现的主要性能问题

## 2.1 全局原子轮询

当前 consumer 在无任务时不断执行：

```cpp
atomicAdd(queue_head, 0);
atomicAdd(queue_tail, 0);
atomicAdd(update_done_blocks, 0);
atomicAdd(propagation_done_tasks, 0);
```

这些不是普通读取，而是全局 atomic read-modify-write。

大量 warp 同时访问同四个地址，会形成原子热点，并且拖慢 producer 对：

```cpp
atomicAdd(queue_tail, fragment_count);
atomicAdd(update_done_blocks, 1);
```

的真实更新。

## 2.2 每个 warp 独立争抢任务

一个 256-thread block 有 8 个 warp。

当前每个 warp 都独立执行：

```cpp
atomicCAS(queue_head, head, head + 1);
```

当任务数量较少时，一个 task 会对应多个失败 CAS。

UPDATE block 转为 consumer 后，consumer warp 数进一步增加，因此尾部会放大这个问题。

## 2.3 reserve 与 publish 没有区分

Producer 先执行：

```cpp
atomicAdd(queue_tail, fragment_count);
```

然后才写：

```cpp
spike_queue_neuron[task]
spike_queue_state[task]
```

因此 consumer 看到 `task < queue_tail` 并不意味着该 task 已发布。

当前 consumer 领取后在：

```cpp
spike_queue_state[task]
```

上持续原子自旋，进一步增加流量。

## 2.4 每 task 的完成协议过重

每个 fragment 完成后执行：

```cpp
__syncwarp();
__threadfence();
atomicAdd(propagation_done_tasks, 1);
```

其中 `__threadfence()` 是 device-wide memory fence。

每 1024 条 edge 执行一次，会显著增加传播任务固定成本。

---

# 3. 总体修改路线

分为两个层次：

## 第一层：先修复原子轮询风暴

保持：

```text
单一全局任务队列
UPDATE block 完成后转 consumer
ready state
per-task done counter
```

只改变读取、退避和任务领取方式。

目的是先确认 2000 ms 是否主要来自原子轮询。

## 第二层：按 block 批量领取任务

每个 block 只由一个调度线程访问全局 queue head。

一次领取多个 task，再通过 shared memory 分发给 block 内各 warp。

目的是恢复多个 warp 的传播并行度，同时减少全局原子竞争。

---

# 4. 阶段一：普通原子读取替代 atomicAdd(x, 0)

## 4.1 使用 acquire load

新增 CUDA atomic 支持：

```cpp
#include <cuda/atomic>
```

定义设备范围 atomic load：

```cpp
template <typename T>
__device__ __forceinline__
T device_load_acquire(T* ptr) {
    cuda::atomic_ref<T, cuda::thread_scope_device> ref(*ptr);
    return ref.load(cuda::memory_order_acquire);
}
```

对于 state：

```cpp
__device__ __forceinline__
uint32_t state_load_acquire(uint32_t* ptr) {
    cuda::atomic_ref<
        uint32_t,
        cuda::thread_scope_device
    > ref(*ptr);

    return ref.load(cuda::memory_order_acquire);
}
```

将所有纯读取：

```cpp
atomicAdd(queue_head, 0);
atomicAdd(queue_tail, 0);
atomicAdd(update_done_blocks, 0);
atomicAdd(propagation_done_tasks, 0);
atomicAdd(spike_queue_state + task, 0u);
```

替换为：

```cpp
device_load_acquire(queue_head);
device_load_acquire(queue_tail);
device_load_acquire(update_done_blocks);
device_load_acquire(propagation_done_tasks);
state_load_acquire(spike_queue_state + task);
```

这一步不改变控制逻辑，只消除无意义的 atomic RMW。

---

# 5. 阶段二：producer 使用 release 发布 state

当前 producer：

```cpp
spike_queue_neuron[task] = n;
__threadfence();
atomicExch(spike_queue_state + task, state);
```

改为 release store：

```cpp
spike_queue_neuron[task] = n;

cuda::atomic_ref<
    uint32_t,
    cuda::thread_scope_device
> state_ref(spike_queue_state[task]);

state_ref.store(
    state,
    cuda::memory_order_release);
```

consumer 使用 acquire load：

```cpp
uint32_t state;

do {
    state = state_load_acquire(
        spike_queue_state + task);
} while ((state >> 16) != expected_epoch);
```

release/acquire 保证：

```text
consumer 看到正确 epoch
→ 之前写入的 spike_queue_neuron 已可见
```

因此 producer 中每个 fired neuron 的：

```cpp
__threadfence();
```

可以删除。

---

# 6. 阶段三：删除 consumer 完成侧的 threadfence

当前：

```cpp
__syncwarp();

if (lane == 0) {
    __threadfence();
    atomicAdd(propagation_done_tasks, 1);
}
```

改为：

```cpp
__syncwarp();

if (lane == 0) {
    atomicAdd(propagation_done_tasks, 1);
}
```

各 lane 的：

```cpp
atomicAdd(delta_write + post, weight);
```

已经在抵达 `__syncwarp()` 前完成。

当前 timestep 最终还有：

```cpp
grid.sync();
```

下一 timestep 不会提前读取 `delta_write`。

因此第一版不再对每个 fragment 执行 device-wide fence。

---

# 7. 阶段四：给无任务轮询增加退避

consumer 需要区分三种情况：

```text
1. 有已发布 task
2. producer 尚未结束，但当前暂无 task
3. producer 已结束，且全部 task 已完成
```

对于情况 2，不再紧密轮询。

## 7.1 每 warp 退避状态

```cpp
int backoff_cycles = 32;
constexpr int kMaxBackoff = 1024;
```

领取到任务后重置：

```cpp
backoff_cycles = 32;
```

当前无任务但不能退出时：

```cpp
if (lane == 0) {
    __nanosleep(backoff_cycles);
    backoff_cycles =
        min(backoff_cycles << 1, kMaxBackoff);
}
```

warp 其他 lane 随 lane 0 一起进入下一次循环。

## 7.2 修改后的单 warp consumer 轮廓

```cpp
int backoff_cycles = 32;

while (true) {
    int task = -1;

    if (lane == 0) {
        const int head =
            device_load_acquire(queue_head);

        const int tail =
            device_load_acquire(queue_tail);

        if (head < tail) {
            if (atomicCAS(
                    queue_head,
                    head,
                    head + 1) == head) {
                task = head;
            }
        }
    }

    task = __shfl_sync(
        0xffffffffu,
        task,
        0);

    if (task >= 0) {
        backoff_cycles = 32;

        wait_and_process_task(task);
        continue;
    }

    bool should_exit = false;

    if (lane == 0) {
        const int update_done =
            device_load_acquire(
                update_done_blocks);

        const int tail =
            device_load_acquire(queue_tail);

        const int done =
            device_load_acquire(
                propagation_done_tasks);

        should_exit =
            update_done == update_block_count &&
            done >= tail;

        if (!should_exit) {
            __nanosleep(backoff_cycles);
            backoff_cycles =
                min(
                    backoff_cycles << 1,
                    kMaxBackoff);
        }
    }

    should_exit = __shfl_sync(
        0xffffffffu,
        should_exit,
        0);

    if (should_exit) {
        break;
    }
}
```

---

# 8. 阶段五：避免领取未发布 task 后长期自旋

仅改 acquire load 和退避后，consumer 仍可能：

```text
读取 queue_tail
→ 领取 task
→ state 尚未发布
→ 在该 task 上等待
```

这不会造成正确性错误，但会让一个 consumer warp 固定等待某个未准备好的 task。

## 8.1 简化修正：领取前检查 state

由于队列按连续槽 reserve，consumer 可以先检查当前 head 对应的 state：

```cpp
const int head =
    device_load_acquire(queue_head);

const int tail =
    device_load_acquire(queue_tail);

if (head < tail) {
    const uint32_t state =
        state_load_acquire(
            spike_queue_state + head);

    if ((state >> 16) == expected_epoch) {
        if (atomicCAS(
                queue_head,
                head,
                head + 1) == head) {
            task = head;
            claimed_state = state;
        }
    }
}
```

这样 consumer 只领取已发布的队首 task。

如果队首尚未发布：

```text
不领取
退避
下一轮再检查
```

由于同一 fired neuron 的 fragment 是顺序发布的，队首通常会很快 ready。

## 8.2 任务处理时复用 claimed state

```cpp
task = __shfl_sync(...);
claimed_state = __shfl_sync(...);

if (task >= 0) {
    const int fragment =
        static_cast<int>(
            claimed_state & kFragmentMask
        ) - 1;

    process_fragment(...);
}
```

这样删除领取后的 ready 自旋：

```cpp
do {
    state = ...
} while (...);
```

## 8.3 影响

这种方式会形成队首顺序发布：

```text
task head 未 ready
→ 后续 task 即使 ready 也暂时不能领取
```

但 producer 当前按照 task 顺序写 state，正常不会形成长时间洞。

第一版优先采用这种简单协议。

---

# 9. 阶段六：block 级批量任务领取

完成上述修复并确认运行时间恢复后，再把 per-warp claim 改成 per-block batch claim。

## 9.1 Shared memory

```cpp
constexpr int kWarpsPerBlock =
    kThreadsPerBlock / 32;

__shared__ int block_tasks[kWarpsPerBlock];
__shared__ uint32_t block_states[kWarpsPerBlock];
__shared__ int block_task_count;
```

每轮由 block 的 thread 0 负责检查队列。

## 9.2 只领取连续 ready task

不能直接：

```cpp
atomicAdd(queue_head, kWarpsPerBlock);
```

因为末尾 task 可能尚未发布。

调度线程先读取：

```cpp
head = load(queue_head);
tail = load(queue_tail);
```

然后从 `head` 开始检查最多 8 个连续 task：

```cpp
int ready_count = 0;

for (int i = 0;
     i < kWarpsPerBlock &&
     head + i < tail;
     ++i) {
    const uint32_t state =
        state_load_acquire(
            spike_queue_state + head + i);

    if ((state >> 16) != expected_epoch) {
        break;
    }

    block_states[i] = state;
    ++ready_count;
}
```

随后 CAS 一次领取整批：

```cpp
if (ready_count > 0 &&
    atomicCAS(
        queue_head,
        head,
        head + ready_count) == head) {

    for (int i = 0;
         i < ready_count;
         ++i) {
        block_tasks[i] = head + i;
    }

    block_task_count = ready_count;
} else {
    block_task_count = 0;
}
```

## 9.3 分发给 warp

```cpp
__syncthreads();

const int warp_id =
    threadIdx.x >> 5;

if (warp_id < block_task_count) {
    const int task =
        block_tasks[warp_id];

    const uint32_t state =
        block_states[warp_id];

    process_task(task, state);
}

__syncthreads();
```

每 block 每轮最多只产生：

```text
1 次 queue head CAS
若干普通 acquire load
```

而不是 8 个 warp 分别 CAS。

## 9.4 保留 UPDATE block 转 consumer

UPDATE block 完成后执行：

```cpp
__syncthreads();

if (threadIdx.x == 0) {
    atomicAdd(update_done_blocks, 1);
}

__syncthreads();
```

随后整个 block 进入相同的 block-consumer loop：

```cpp
run_block_consumer();
```

因此仍然保留 tail helper 行为。

---

# 10. block consumer 完整伪代码

```cpp
__shared__ int block_tasks[kWarpsPerBlock];
__shared__ uint32_t block_states[kWarpsPerBlock];
__shared__ int block_task_count;
__shared__ bool block_should_exit;
__shared__ int block_backoff;

if (threadIdx.x == 0) {
    block_backoff = 32;
}

__syncthreads();

while (true) {
    if (threadIdx.x == 0) {
        block_task_count = 0;
        block_should_exit = false;

        const int head =
            device_load_acquire(queue_head);

        const int tail =
            device_load_acquire(queue_tail);

        int ready_count = 0;

        for (int i = 0;
             i < kWarpsPerBlock &&
             head + i < tail;
             ++i) {
            const uint32_t state =
                state_load_acquire(
                    spike_queue_state +
                    head + i);

            if ((state >> 16) !=
                expected_epoch) {
                break;
            }

            block_states[i] = state;
            ++ready_count;
        }

        if (ready_count > 0) {
            const int old =
                atomicCAS(
                    queue_head,
                    head,
                    head + ready_count);

            if (old == head) {
                for (int i = 0;
                     i < ready_count;
                     ++i) {
                    block_tasks[i] =
                        head + i;
                }

                block_task_count =
                    ready_count;

                block_backoff = 32;
            }
        }

        if (block_task_count == 0) {
            const int update_done =
                device_load_acquire(
                    update_done_blocks);

            const int final_tail =
                device_load_acquire(
                    queue_tail);

            const int done =
                device_load_acquire(
                    propagation_done_tasks);

            block_should_exit =
                update_done ==
                    update_block_count &&
                done >= final_tail;

            if (!block_should_exit) {
                __nanosleep(
                    block_backoff);

                block_backoff =
                    min(
                        block_backoff << 1,
                        1024);
            }
        }
    }

    __syncthreads();

    if (block_should_exit) {
        break;
    }

    const int warp_id =
        threadIdx.x >> 5;

    const int lane =
        threadIdx.x & 31;

    if (warp_id < block_task_count) {
        const int task =
            block_tasks[warp_id];

        const uint32_t state =
            block_states[warp_id];

        const int fragment =
            static_cast<int>(
                state & kFragmentMask
            ) - 1;

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
                edge_start +
                    kEdgesPerTask,
                row_end);

        for (int edge =
                 edge_start + lane;
             edge < edge_end;
             edge += 32) {
            const int post =
                graph_indices[edge];

            atomicAdd(
                delta_write + post,
                graph_weight[edge]);
        }

        __syncwarp();

        if (lane == 0) {
            atomicAdd(
                propagation_done_tasks,
                1);
        }
    }

    __syncthreads();
}
```

---

# 11. 可进一步减少 done counter 竞争

初版继续每 task：

```cpp
atomicAdd(propagation_done_tasks, 1);
```

在 block batch claim 正确后，可以把同一批任务的完成数合并。

## 11.1 batch 完成聚合

每个 active warp 完成任务后：

```cpp
__syncwarp();
```

随后 block 同步：

```cpp
__syncthreads();
```

由 thread 0 一次增加：

```cpp
if (threadIdx.x == 0 &&
    block_task_count > 0) {
    atomicAdd(
        propagation_done_tasks,
        block_task_count);
}
```

这样每批最多 8 个 task，只产生一次全局完成原子。

修改后的尾部：

```cpp
if (warp_id < block_task_count) {
    process_task(...);
}

__syncthreads();

if (threadIdx.x == 0 &&
    block_task_count > 0) {
    atomicAdd(
        propagation_done_tasks,
        block_task_count);
}

__syncthreads();
```

这是推荐采用的最终版本。

---

# 12. host 侧修正

## 12.1 删除 queue state 全量清零

当前 host 将：

```cpp
spike_queue_edge_start
```

复用为：

```cpp
spike_queue_state
```

并在每次 forward 前调用：

```cpp
spike_queue_edge_start.zero_();
```

但 state 已编码 timestep epoch：

```text
高 16 bit = t + 1
低 16 bit = fragment + 1
```

consumer 只接受当前 epoch，因此无需每次 forward 清零整个 state queue。应删除该操作。

需要注意：

```text
不同 forward 调用都从 epoch=1 开始
```

如果 state 不清零，第一次 timestep 可能把上一次 forward 的 epoch=1 误认为当前 ready。

因此不能单纯删除 `zero_()` 而仍让 epoch 每次从 1 开始。

有两个方案。

### 方案 A：保留 forward 级 epoch base

host 维护递增：

```cpp
uint32_t forward_epoch_base;
```

kernel state：

```cpp
epoch =
    forward_epoch_base +
    t + 1;
```

consumer 使用相同 expected epoch。

这样 queue state 不需要清零。

### 方案 B：只清理实际使用范围

在 kernel 结束时保留最后一次使用过的最大 task 数，下一次 forward 只清理：

```text
[0, previous_max_tail)
```

但 persistent state 跨 forward 管理更复杂。

推荐采用方案 A。

## 12.2 delta buffer

当前每次 forward 分配并清零：

```cpp
recurrent_delta_0
recurrent_delta_1
```

暂时保留，避免一次修改过多。

后续再将其持久化到 Python/C++ operator state 中。

## 12.3 最后的 PSC 合并

当前 forward 结束后执行：

```cpp
psc_out.add_(last_delta);
```

暂时保留，以维持外部 PSC 表示兼容。

性能测试应分别记录：

```text
pipeline kernel 时间
psc add 时间
完整 forward 时间
```

---

# 13. 推荐施工顺序

## 第一步：最小性能修正

保持当前 per-warp queue：

```text
UPDATE block 完成后转 consumer
每个 warp 独立 claim
```

只修改：

```text
atomicAdd(x, 0)
→ acquire load

producer threadfence + atomicExch
→ release store

删除 consumer per-task threadfence

空队列加入 nanosleep backoff

领取前检查 queue_state epoch
```

目的：

```text
确认 2000 ms 是否由原子轮询和 ready 自旋造成
```

## 第二步：block 批量 claim

改为：

```text
thread 0 检查连续 ready tasks
一次 CAS 领取最多 8 个
shared memory 分发给 8 个 warp
```

目的：

```text
降低 queue_head 原子竞争
```

## 第三步：block 批量 done

由：

```text
每 task 一次 propagation_done_tasks atomic
```

改为：

```text
每 block 每轮一次 atomicAdd(done, ready_count)
```

目的：

```text
降低完成计数器竞争
```

## 第四步：forward epoch

增加：

```cpp
uint32_t epoch_base
```

删除全 queue `zero_()`。

## 第五步：扫描退避与角色比例

测试：

```text
backoff 初值：
32 / 64 / 128 cycles

backoff 上限：
256 / 512 / 1024 / 2048

UPDATE : PROPAGATION：
7:1
3:1
2:1
1:1
```

---

# 14. 必做诊断实验

## 实验 1：零 spike

所有 neuron 不发放。

预期：

```text
传播任务数 = 0
```

修改前若仍接近秒级，而加入 acquire load + backoff 后恢复到接近 UPDATE 时间，可确认轮询是主因。

## 实验 2：关闭 UPDATE helper

仅作为诊断开关：

```cpp
if (is_update_block) {
    update;
    mark_done;
} else {
    consume;
}
```

不是最终设计。

用于比较：

```text
固定角色
与
UPDATE 完成后转 consumer
```

若修复轮询后两者差距很小，说明 helper 不是核心问题。

## 实验 3：单 consumer block

只允许一个 propagation block 消费。

若时间大幅下降，说明之前主要是 queue 原子争用。

## 实验 4：逐项消融

依次测量：

```text
原实现

+ acquire load
+ ready-before-claim
+ backoff
+ 删除 task threadfence
+ block batch claim
+ block batch done
```

不要一次全部修改，否则无法知道主要收益来源。

---

# 15. 预期结果判断

修复后可能出现三类结果。

## 情况 A：2000 ms 恢复到几十 ms

说明原子轮询和 ready 自旋是主因。

随后 block batch claim 应进一步接近原始 12 ms。

## 情况 B：仍然数百或上千 ms

需要加入计数器统计：

```cpp
failed_claim_count
empty_poll_count
ready_not_published_count
processed_task_count
```

重点检查是否某个 task state 永远不匹配 epoch。

## 情况 C：恢复到约 15–25 ms，但仍慢于原版

说明工程错误已消除，剩余是架构成本：

```text
PSC 双缓冲
任务发布同步
固定资源划分
queue 管理
原子写回竞争
```

这时才进入真正的流水化收益分析。

---

# 16. 最终推荐结构

```text
UPDATE block
    更新 neuron
    reserve task slots
    写 neuron ID
    release-store task state
    UPDATE 完成
    update_done_blocks++
    转入 block consumer

Dedicated propagation block
    从时间步开始进入 block consumer

Block consumer
    thread 0：
        acquire-load head/tail
        检查连续 ready state
        一次 CAS 领取最多 8 task
        无 task 时指数退避

    shared memory：
        分发 task/state 给各 warp

    warp：
        根据 neuron + fragment 恢复 edge 范围
        执行 propagation

    block：
        一次 atomicAdd 汇总完成任务数

退出：
    update_done_blocks == update_block_count
    且 propagation_done_tasks == queue_tail

grid.sync()
```

核心原则是：

```text
保留 UPDATE 完成后转 consumer；
但 consumer 不再以每 warp 高频全局原子轮询的方式运行。
```
