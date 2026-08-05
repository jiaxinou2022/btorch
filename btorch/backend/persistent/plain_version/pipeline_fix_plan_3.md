# Persistent SNN Pipeline：Ticket Queue 改进执行计划

## 1. 改造目标

当前高活跃测试中：

```text
failed-claims = 172,350,368
```

说明大量 consumer warp 同时争抢同一个 `queue_head`：

```text
load head
→ 判断 head < tail
→ 多个 warp 对同一个 head 执行 CAS
→ 仅一个成功
→ 其余全部失败重试
```

本轮目标是将领取协议改为：

```text
atomicAdd 分配唯一 ticket
→ 每个 consumer warp 持有不同 task id
→ 等待该 task 被 reserve 并发布
→ 处理完成后领取下一个 ticket
```

从而恢复原始静态队列的核心优势：

```text
每次领取操作都产生唯一有效编号
不存在 failed CAS
```

同时适应流水化 producer 尚未完成、队列仍会增长的情况。

---

# 2. 保留与删除的结构

## 2.1 保留

保留当前：

```cpp
int* queue_tail;
int* update_done_blocks;
int* propagation_done_tasks;

int* spike_queue_neuron;
uint32_t* spike_queue_state;
```

含义：

```text
queue_tail
    producer 已 reserve 的 task 总数

update_done_blocks
    已完成 UPDATE 的 producer block 数

propagation_done_tasks
    已完整执行 propagation 的 task 数

spike_queue_neuron
    task 对应的 presynaptic neuron

spike_queue_state
    epoch + fragment id，且承担 ready 发布语义
```

继续保留：

* UPDATE block 完成后转为 consumer；
* producer 逐 fragment 发布；
* acquire/release 访问；
* PSC 双缓冲；
* timestep 末 `grid.sync()`。

## 2.2 删除或重命名

删除当前竞争式：

```cpp
int* queue_head;
```

或者将其重命名为：

```cpp
int* next_ticket;
```

含义变为：

```text
下一个尚未分配给 consumer 的逻辑 task 编号
```

consumer 不再执行：

```cpp
load queue_head
load queue_tail
atomicCAS(queue_head, head, head + 1)
```

而只执行：

```cpp
task = atomicAdd(next_ticket, 1);
```

---

# 3. Ticket Queue 的基本语义

设 consumer 领取：

```cpp
task = atomicAdd(next_ticket, 1);
```

该 task 可能处于三种状态。

## 3.1 task 已经存在且已发布

```text
task < queue_tail
且
state[task].epoch == current_epoch
```

直接处理。

## 3.2 task 尚未 reserve 或尚未发布

可能是：

```text
task >= 当前 queue_tail
```

或者：

```text
task < queue_tail
但 state[task] 尚未匹配当前 epoch
```

此时不能丢弃 ticket，也不能重新领取其他 ticket。

该 warp等待自己的 ticket：

```text
等待 queue_tail 增长
或
等待 state 发布
```

## 3.3 task 最终无效

当：

```text
update_done_blocks == update_block_count
```

说明不会再产生新任务。

此时若：

```text
task >= queue_tail
```

该 ticket 永远不会对应真实任务，consumer 退出。

---

# 4. 为什么 ticket 模式不会漏任务

假设最终：

```text
final_task_count = queue_tail
```

consumer ticket 依次由：

```cpp
atomicAdd(next_ticket, 1);
```

产生：

```text
0, 1, 2, ..., final_task_count - 1,
final_task_count, ...
```

其中：

```text
[0, final_task_count)
```

中的每个编号只会被一个 consumer warp领取一次。

因此：

* 不会重复处理；
* 不会漏掉 task；
* 不需要 CAS 重试；
* 超出最终范围的 ticket 在 producer 全部结束后退出。

---

# 5. 时间步初始化

每个 timestep 开始时：

```cpp
if (global_tid == 0) {
    *queue_tail = 0;
    *next_ticket = 0;
    *update_done_blocks = 0;
    *propagation_done_tasks = 0;
}

grid.sync();
```

`spike_queue_state` 继续使用 epoch，无需逐步清零。

---

# 6. Producer 逻辑

Producer 继续使用当前逐 fragment 发布实现。

```cpp
const int first_task =
    atomicAdd(queue_tail, fragment_count);

for (int fragment = 0;
     fragment < fragment_count;
     ++fragment) {

    const int task =
        first_task + fragment;

    spike_queue_neuron[task] =
        neuron;

    const uint32_t state =
        (expected_epoch << 16) |
        static_cast<uint32_t>(
            fragment + 1);

    state_store_release(
        spike_queue_state + task,
        state);
}
```

本轮不修改 producer 的任务生成方式。

---

# 7. Consumer 基础实现

## 7.1 每 warp 领取一个 ticket

```cpp
int task = -1;

if (lane == 0) {
    task = atomicAdd(
        next_ticket,
        1);
}

task = __shfl_sync(
    0xffffffffu,
    task,
    0);
```

每个 warp获得不同编号，不再存在 failed claim。

## 7.2 等待 task 生效

每个 warp本地维护退避：

```cpp
int backoff = 32;
constexpr int kMaxBackoff = 1024;
```

等待循环：

```cpp
uint32_t task_state = 0;
bool task_valid = false;
bool terminate = false;

while (true) {
    int tail = 0;
    int update_done = 0;

    if (lane == 0) {
        tail = load_acquire(
            queue_tail);

        if (task < tail) {
            task_state =
                state_load_acquire(
                    spike_queue_state +
                    task);

            task_valid =
                (task_state >> 16) ==
                expected_epoch;
        } else {
            update_done =
                load_acquire(
                    update_done_blocks);

            if (update_done ==
                    update_block_count) {
                terminate = true;
            }
        }

        if (!task_valid &&
            !terminate) {
            __nanosleep(backoff);

            backoff = min(
                backoff << 1,
                kMaxBackoff);
        }
    }

    task_state = __shfl_sync(
        0xffffffffu,
        task_state,
        0);

    task_valid = __shfl_sync(
        0xffffffffu,
        task_valid,
        0);

    terminate = __shfl_sync(
        0xffffffffu,
        terminate,
        0);

    if (task_valid ||
        terminate) {
        break;
    }
}
```

注意：

```text
task < tail
```

只表示该槽已经 reserve。

只有：

```text
state epoch 匹配
```

才表示 descriptor 已发布。

## 7.3 退出条件

当：

```text
update_done_blocks == update_block_count
```

且：

```text
task >= queue_tail
```

说明当前 ticket 永远无效。

此 warp可以退出 consumer loop：

```cpp
if (terminate) {
    break;
}
```

不需要再检查：

```cpp
propagation_done_tasks == queue_tail
```

作为单个 warp 的退出条件。

原因是该 warp拿到的 ticket 已经超过最终 task 数量，而所有较小 ticket 已经被其他 warp唯一领取。

不过 timestep 结束前仍需要整体确认全部有效 task 已完成。

---

# 8. Consumer 完整主循环伪代码

```cpp
const int lane =
    threadIdx.x & 31;

int backoff = 32;

while (true) {
    // --------------------------------
    // 1. 领取唯一 ticket
    // --------------------------------

    int task = -1;

    if (lane == 0) {
        task = atomicAdd(
            next_ticket,
            1);
    }

    task = __shfl_sync(
        0xffffffffu,
        task,
        0);

    // --------------------------------
    // 2. 等待 task 成为 ready，
    //    或确认其最终不存在
    // --------------------------------

    uint32_t state = 0;
    bool ready = false;
    bool invalid_final = false;

    while (true) {
        if (lane == 0) {
            const int tail =
                load_acquire(
                    queue_tail);

            if (task < tail) {
                state =
                    state_load_acquire(
                        spike_queue_state +
                        task);

                ready =
                    (state >> 16) ==
                    expected_epoch;
            } else {
                const int update_done =
                    load_acquire(
                        update_done_blocks);

                invalid_final =
                    update_done ==
                        update_block_count;
            }

            if (!ready &&
                !invalid_final) {
                __nanosleep(backoff);

                backoff =
                    min(
                        backoff << 1,
                        kMaxBackoff);
            }
        }

        state = __shfl_sync(
            0xffffffffu,
            state,
            0);

        ready = __shfl_sync(
            0xffffffffu,
            ready,
            0);

        invalid_final =
            __shfl_sync(
                0xffffffffu,
                invalid_final,
                0);

        if (ready ||
            invalid_final) {
            break;
        }
    }

    // --------------------------------
    // 3. ticket 超出最终任务范围
    // --------------------------------

    if (invalid_final) {
        break;
    }

    backoff = 32;

    // --------------------------------
    // 4. 解码并处理任务
    // --------------------------------

    const int fragment =
        static_cast<int>(
            state & 0xffffu
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

    // --------------------------------
    // 5. 标记任务完成
    // --------------------------------

    if (lane == 0) {
        atomicAdd(
            propagation_done_tasks,
            1);
    }
}
```

---

# 9. Timestep 结束协议

ticket 超出最终任务范围的 consumer 可以退出，但某些合法 task可能仍在处理中。

因此所有 block 进入时间步末 `grid.sync()` 前，必须保证：

```text
propagation_done_tasks == queue_tail
```

不能让某些 block提前进入 `grid.sync()`，而另一些 block仍在任务处理中吗？

可以，因为 cooperative `grid.sync()` 本身会等待所有 block到达。

合法 task的持有者只有在完成任务后，才会领取下一个 ticket并最终退出。

所以所有 consumer block能到达 `grid.sync()` 时，理论上所有已领取的合法 task都已完成。

但为了检测实现错误，推荐在 `grid.sync()` 前增加一个仅用于 debug 的检查：

```cpp
if (global_tid == 0) {
    assert(
        *propagation_done_tasks ==
        *queue_tail);
}
```

release 版本不需要额外等待循环。

更保守的第一版也可以让每个 block在退出 consumer 后检查：

```cpp
while (load_acquire(
           propagation_done_tasks) <
       load_acquire(queue_tail)) {
    __nanosleep(64);
}
```

但这会重新引入多 block轮询，不建议作为最终实现。

推荐依靠：

```text
ticket 唯一领取
+ 每个合法 ticket 完成后才能继续
+ 所有 consumer 最终到达 grid.sync
```

形成隐式完成保证。

---

# 10. 重要风险：consumer 过度领取未来 ticket

假设 consumer warp 数为 `W`，当前任务只有少量，但未来仍可能增加。

每个 warp都会立即领取一个 ticket：

```text
0 ... W - 1
```

如果当前：

```text
queue_tail << W
```

大量 warp会等待未来任务。

这不会产生 CAS 风暴，但仍可能产生：

* 多个 warp轮询 `queue_tail`；
* 很多最终无效 ticket；
* consumer 数量远高于任务生成速度；
* 高比例 idle warp。

因此本轮必须同时加入 consumer 数量控制。

---

# 11. 限制每 block 的 consumer warp 数

新增编译期或运行时参数：

```cpp
int consumer_warps_per_block;
```

启用条件：

```cpp
const int warp_in_block =
    threadIdx.x >> 5;

const bool active_consumer =
    warp_in_block <
        consumer_warps_per_block;
```

对于 dedicated propagation block：

```cpp
if (is_propagation_block &&
    active_consumer) {
    run_ticket_consumer();
}
```

对于完成 UPDATE 的 block：

```cpp
if (is_update_block) {
    run_update();

    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(
            update_done_blocks,
            1);
    }

    __syncthreads();

    if (active_consumer) {
        run_ticket_consumer();
    }
}
```

初始扫描：

```text
consumer_warps_per_block =
1, 2, 4, 8
```

推荐默认先使用：

```text
2 warp/block
```

---

# 12. 避免非 consumer warp 提前进入 grid.sync

若一个 block只有前两个 warp运行 consumer，其余 warp不能直接进入 `grid.sync()`，否则同一 block内线程执行路径不一致。

应当在 block 内先同步：

```cpp
if (active_consumer) {
    run_ticket_consumer();
}

__syncthreads();

grid.sync();
```

因为 `run_ticket_consumer()` 中不能包含 block barrier，所以未参与 consumer 的 warp可以停在最后一个 `__syncthreads()` 等待。

完整结构：

```cpp
if (is_update_block) {
    run_update();
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(
            update_done_blocks,
            1);
    }

    __syncthreads();
}

if (active_consumer) {
    run_ticket_consumer();
}

__syncthreads();
grid.sync();
```

Dedicated propagation block跳过 UPDATE，但同样进入 consumer。

---

# 13. 防止 UPDATE block 内非 consumer warp 长时间占用执行资源

同一 block 中只有部分 warp运行 consumer，其余 warp停在 block barrier。

这不会释放寄存器或 block资源，但能减少：

* queue ticket 原子；
* queue-tail 轮询；
* future-ticket 数量。

因此它主要是调度压力控制，而不是 occupancy 优化。

若 1–2 consumer warp/block 已能覆盖传播工作，应优先选择较少 warp。

---

# 14. 可选第二阶段：分块 ticket

在单 ticket模式正确且性能恢复后，可以测试：

```cpp
constexpr int kTicketChunk = 2;
```

领取：

```cpp
const int first_task =
    atomicAdd(
        next_ticket,
        kTicketChunk);
```

warp依次处理：

```cpp
for (int i = 0;
     i < kTicketChunk;
     ++i) {
    task = first_task + i;
    wait_and_process(task);
}
```

## 优点

* 每两个 task一次 ticket atomic；
* 降低 `next_ticket` 原子流量。

## 风险

如果第一个 task尚未产生，warp会等待，第二个 task也无法先处理。

因此测试顺序：

```text
chunk = 1
→ chunk = 2
→ chunk = 4
```

不应直接使用较大 chunk。

---

# 15. 新增调试计数

删除：

```text
failed_claims
```

因为 ticket模式中不再存在 CAS claim。

新增：

```cpp
unsigned long long* ticket_wait_tail;
unsigned long long* ticket_wait_ready;
unsigned long long* invalid_final_tickets;
unsigned long long* processed_tasks;
```

含义：

```text
ticket_wait_tail
    task >= 当前 queue_tail，等待未来 reserve

ticket_wait_ready
    task < queue_tail，但 state epoch 尚未匹配

invalid_final_tickets
    UPDATE 全部完成后，ticket 仍超出 final tail

processed_tasks
    实际完成的合法 task 数
```

统计位置：

```cpp
if (task >= tail &&
    updates_not_done) {
    debug_add(ticket_wait_tail);
}

if (task < tail &&
    state_not_ready) {
    debug_add(ticket_wait_ready);
}

if (task >= final_tail &&
    updates_done) {
    debug_add(invalid_final_tickets);
}

after_process:
    debug_add(processed_tasks);
```

应检查：

```text
processed_tasks == final queue_tail
```

---

# 16. 正确性检查

## 16.1 单时间步

比较原版与 ticket版：

```text
dense spike
event indices
v
psc
delta_write
```

## 16.2 极端任务数量

测试：

```text
0 task
1 task
consumer warp 数 - 1 task
consumer warp 数 task
consumer warp 数 + 1 task
大量 task
```

重点确认：

* 少任务时大量 future ticket能正确退出；
* 最后一个有效 task不会漏掉；
* `processed_tasks == queue_tail`；
* 不存在重复 task。

## 16.3 Producer 延迟发布

人为在 producer 中加入短暂延迟：

```cpp
if (fragment == 0) {
    __nanosleep(...);
}
```

确认 consumer：

* 能等待 `queue_tail`；
* 能等待 state ready；
* 不会误判为 invalid；
* producer 完成后才能最终退出。

---

# 17. 性能实验矩阵

固定高活跃工作负载，测试：

| 领取方式   | consumer warp/block | chunk |
| ------ | ------------------: | ----: |
| 当前 CAS |                   8 |     — |
| ticket |                   8 |     1 |
| ticket |                   4 |     1 |
| ticket |                   2 |     1 |
| ticket |                   1 |     1 |
| ticket |                   2 |     2 |
| ticket |                   2 |     4 |

重点记录：

```text
总时间
processed task 数
ticket_wait_tail
ticket_wait_ready
invalid_final_tickets
```

预期：

```text
failed-claims 从 1.7 亿降为 0
```

如果时间仍很高：

* `ticket_wait_tail` 很高：consumer 过多；
* `ticket_wait_ready` 很高：reserve/publish 洞仍严重；
* 两者都不高：传播 edge atomic 或 PSC 逻辑才是下一瓶颈。

---

# 18. Host 侧修改

原：

```cpp
queue_head
```

继续复用现有 tensor即可，只需语义改为：

```cpp
next_ticket
```

无需新增 host buffer。

初始化仍在 kernel 内：

```cpp
*next_ticket = 0;
```

调试计数器若启用，可增加：

```cpp
torch::zeros({4}, int64_options)
```

分别传入四类计数。

---

# 19. 推荐施工顺序

```text
1. 将 queue_head 重命名为 next_ticket

2. 删除所有：
       load head
       ready-before-CAS
       atomicCAS claim
       failed_claims 统计

3. consumer 每轮通过：
       atomicAdd(next_ticket, 1)
   领取唯一 ticket

4. 实现 ticket 等待状态机：
       等 queue_tail
       等 state ready
       producer 完成后判断最终越界

5. 保留 per-task propagation_done_tasks atomic

6. 增加：
       ticket_wait_tail
       ticket_wait_ready
       invalid_final_tickets
       processed_tasks

7. 限制 consumer warp/block
   先默认 2

8. 验证：
       processed_tasks == queue_tail

9. 扫描：
       consumer warp/block = 1,2,4,8

10. ticket=1 稳定后，再测试 chunk=2、4
```

---

# 20. 本轮不修改的内容

暂不修改：

```text
per-producer block queue
ready bitmap
允许跳过未 ready ticket
动态 consumer 数量
delta buffer 生命周期
psc 最后合并
fragment 大小
传播 atomic 聚合
```

只有 ticket模式仍因 `ticket_wait_ready` 出现明显性能问题时，才考虑将单全局队列拆成 per-producer queue。

---

# 21. 预期结果

最直接的预期变化：

```text
failed-claims：
172,350,368 → 0
```

高活跃耗时应明显下降。

如果 ticket + 8 warp/block 仍慢，而 ticket + 1/2 warp/block明显更快，说明：

```text
任务领取逻辑已修正，
剩余问题是 consumer 并发量过大。
```

如果 ticket模式下 `ticket_wait_ready` 极高，则说明：

```text
单全局 reserve queue 的乱序发布洞
成为下一主要瓶颈。
```

此时再进入：

```text
per-UPDATE-block 子队列
```

而不是继续调整全局 CAS。
