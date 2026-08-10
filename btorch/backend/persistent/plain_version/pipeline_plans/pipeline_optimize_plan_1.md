# Pipeline 第二阶段性能优化执行计划

## 0. 本轮目标

只实现以下三项：

```text
P1. 删除 propagation 完成路径中的纯冗余
P2. 拆分 dedicated propagation block 与 UPDATE-helper block 的 consumer warp 数
P3. 系统扫描 ticket chunk
```

暂不处理：

```text
delta buffer 持久化
psc_out 最终 fold
host 侧额外 kernel
per-producer queue
neuron UPDATE 顺序重排
fragment size
atomic writeback 聚合
```

这样能够较干净地判断：

```text
当前 14 ms - 12 ms
```

中有多少是 kernel 内调度策略造成的。

---

# 1. P1：删除 propagation_done_tasks 与 task 末尾同步

## 1.1 当前问题

ticket queue 已经满足：

```text
每个 task ticket 唯一分配
合法 task 必须处理完成后，该 warp 才继续领取 ticket
所有 consumer 最终退出后才进入 timestep 末 grid.sync()
```

因此当前：

```cpp
atomicAdd(propagation_done_tasks, 1);
```

不再参与正确性控制。

如果当前 kernel 中任务尾部仍类似：

```cpp
for (int edge = edge_start + lane;
     edge < edge_end;
     edge += 32) {
    atomicAdd(
        delta_write + graph_indices[edge],
        graph_weight[edge]);
}

__syncwarp();

if (lane == 0) {
    atomicAdd(
        propagation_done_tasks,
        1);
}
```

则两项均可删除。

## 1.2 修改后的传播尾部

直接变为：

```cpp
for (int edge = edge_start + lane;
     edge < edge_end;
     edge += 32) {

    const int post =
        graph_indices[edge];

    atomicAdd(
        delta_write + post,
        graph_weight[edge]);
}
```

然后 warp 直接处理其 chunk 中下一个 ticket。

## 1.3 为什么不需要 `__syncwarp()`

各 lane 的 edge loop 是 warp 内一致控制流：

```cpp
edge = edge_start + lane
edge += 32
```

所有 lane 完成自己的 atomic 操作后，自然继续后面的 warp-uniform ticket 控制。

原来的 `__syncwarp()` 主要用于：

```text
确保全部 lane 完成
→ lane0 才发布 done counter
```

done counter 删除后，不再需要这个发布点。

## 1.4 删除 kernel 参数

从：

```cpp
launch_persistent_snn_kernel(
    ...
    int* update_done_blocks,
    int* propagation_done_tasks,
    ...
);
```

改为：

```cpp
launch_persistent_snn_kernel(
    ...
    int* update_done_blocks,
    ...
);
```

kernel 内删除：

```cpp
*propagation_done_tasks = 0;
```

及所有引用。

## 1.5 Host 侧

当前 host：

```cpp
auto pipeline_done_counters =
    torch::zeros({2}, options_i);

launch(...,
    pipeline_done_counters.data_ptr<int>(),
    pipeline_done_counters.data_ptr<int>() + 1,
    ...
);
```

可以改成只保留一个 update counter：

```cpp
auto pipeline_update_done =
    torch::zeros({1}, options_i);

launch(...,
    pipeline_update_done.data_ptr<int>(),
    ...
);
```

不过如果想减少本轮 host 修改，可以继续复用已有 `pipeline_done_counters[0]`：

```cpp
launch(...,
    pipeline_done_counters.data_ptr<int>(),
    ...
);
```

第二个元素闲置即可。

第一轮建议采用后者，避免接口修改过大。

---

# 2. P1 正确性验收

增加 debug-only 计数：

```cpp
processed_tasks
```

task 完成后：

```cpp
#ifdef BTORCH_PIPELINE_DEBUG_COUNTERS
if (lane == 0) {
    atomicAdd(
        &debug_counters[kProcessedTasks],
        1ULL);
}
#endif
```

在测试中检查：

```text
processed_tasks == final queue_tail
```

注意 debug 模式下仍有一次 task-level atomic，只用于验证。

正式 benchmark 必须关闭 debug。

测试：

```text
零 spike
低活跃
正常活跃
高活跃
```

比较：

```text
dense spike
v
psc
event output
```

与修改前保持一致。

---

# 3. P2：拆分 dedicated consumer 和 helper consumer

## 3.1 当前问题

现在只有一个：

```cpp
consumer_warps_per_block
```

同时用于：

```text
A. 从 UPDATE 开始时就执行传播的 dedicated propagation block
B. UPDATE 结束后转成 consumer 的 UPDATE block
```

但两个角色需求不同。

Dedicated block 已经完全不参与 UPDATE，如果仅让其中：

```text
1 / 8 warp
```

运行 consumer，相当于大量常驻资源闲置。

Helper block 则不同。

它是在 UPDATE 已完成后临时参与尾部 drain：

```text
helper 数量很多
传播剩余任务通常已经不多
```

因此 helper 使用过多 warp 会重新制造：

```text
过多 future ticket
queue_tail 轮询
无效最终 ticket
```

所以拆成：

```cpp
int dedicated_consumer_warps;
int helper_consumer_warps;
```

---

# 4. Host 参数修改

当前：

```cpp
int pipeline_consumer_warps_per_block();
```

替换成两个函数。

## 4.1 Dedicated

```cpp
int pipeline_dedicated_consumer_warps() {
    const char* value =
        std::getenv(
            "BTORCH_PIPELINE_DEDICATED_WARPS");

    if (value == nullptr ||
        value[0] == '\0') {
        return 8;
    }

    // 支持 1 / 2 / 4 / 8
    ...
}
```

默认：

```text
8
```

## 4.2 Helper

```cpp
int pipeline_helper_consumer_warps() {
    const char* value =
        std::getenv(
            "BTORCH_PIPELINE_HELPER_WARPS");

    if (value == nullptr ||
        value[0] == '\0') {
        return 1;
    }

    ...
}
```

默认：

```text
1
```

host 调用：

```cpp
const int dedicated_consumer_warps =
    pipeline_dedicated_consumer_warps();

const int helper_consumer_warps =
    pipeline_helper_consumer_warps();

launch_persistent_snn_kernel(
    ...
    update_block_count,
    dedicated_consumer_warps,
    helper_consumer_warps,
    ticket_chunk,
    ...
);
```

---

# 5. Kernel 角色判断

已有：

```cpp
const bool is_update_block =
    blockIdx.x < update_block_count;
```

新增：

```cpp
const int warp_in_block =
    threadIdx.x >> 5;

const int consumer_warp_limit =
    is_update_block
        ? helper_consumer_warps
        : dedicated_consumer_warps;

const bool active_consumer =
    warp_in_block <
    consumer_warp_limit;
```

---

# 6. UPDATE block 路径

保持：

```cpp
if (is_update_block) {
    run_update_and_publish();

    __syncthreads();

    if (threadIdx.x == 0) {
        update_done_store_or_atomic();
    }

    __syncthreads();
}
```

然后只有：

```cpp
warp_in_block < helper_consumer_warps
```

的 warp 进入 ticket consumer。

```cpp
if (active_consumer) {
    run_ticket_consumer();
}
```

未参与 consumer 的 warp 停在最后的 block barrier：

```cpp
__syncthreads();
grid.sync();
```

---

# 7. Dedicated propagation block 路径

Dedicated block 跳过 UPDATE：

```cpp
if (!is_update_block) {
    // no update
}
```

随后：

```cpp
if (warp_in_block <
    dedicated_consumer_warps) {
    run_ticket_consumer();
}
```

默认设置：

```text
dedicated_consumer_warps = 8
```

即整 block 全部用于 propagation。

---

# 8. 推荐的总体控制结构

```cpp
for (int t = 0;
     t < t_steps;
     ++t) {

    initialize_timestep();

    grid.sync();

    inject_external_events();

    grid.sync();

    if (is_update_block) {
        // -------------------------
        // UPDATE + producer
        // -------------------------

        for (int n = update_tid;
             n < n_neuron;
             n += update_stride) {

            update_neuron(n);

            if (fired) {
                publish_fragments(n);
            }
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            update_done_release();
        }

        __syncthreads();
    }

    // -------------------------
    // Consumer
    //
    // dedicated block:
    //     many warps
    //
    // completed UPDATE block:
    //     few helper warps
    // -------------------------

    const int consumer_limit =
        is_update_block
            ? helper_consumer_warps
            : dedicated_consumer_warps;

    if (warp_in_block <
        consumer_limit) {

        run_ticket_consumer();
    }

    __syncthreads();

    grid.sync();
}
```

---

# 9. P2 第一轮测试矩阵

不要一次扫太大。

先固定：

```text
ticket chunk = 当前最佳值
```

测试：

```text
UPDATE:PROP = 3:1
```

四组：

| Dedicated | Helper |
| --------: | -----: |
|         1 |      1 |
|         4 |      1 |
|         8 |      1 |
|         8 |      2 |

目的：

```text
1,1
    当前行为附近基线

4,1 / 8,1
    测 dedicated block 利用率

8,2
    测 UPDATE 完成后增加 helper 是否有价值
```

如果：

```text
8,1 明显优于 1,1
```

说明之前的 dedicated block 浪费确实是主要成本之一。

如果：

```text
8,2 比 8,1 更慢
```

则保持 helper=1。

---

# 10. P2 第二轮：block ratio 扫描

固定第一轮最佳：

```text
dedicated warp = D*
helper warp = H*
```

扫：

```text
7:1
3:1
2:1
1:1
```

重点预期：

```text
7:1
```

可能更适合当前场景，因为：

```text
大多数 block 保留给 UPDATE
少量 propagation block 被完整利用
```

而不是当前：

```text
较多 propagation block
但每个只开极少 warp
```

这两个结构在实际资源利用上差别很大。

---

# 11. P3：系统调整 ticket chunk

## 11.1 当前逻辑

当前 host 默认：

```cpp
BTORCH_PIPELINE_TICKET_CHUNK = 4;
```

即：

```cpp
const int first_ticket =
    atomicAdd(
        next_ticket,
        ticket_chunk);
```

随后同一个 warp 连续处理：

```text
first
first+1
first+2
first+3
```

优点：

```text
减少 next_ticket atomic
```

缺点：

```text
降低动态均衡
future-ticket 等待会阻塞整个 chunk
```

---

# 12. P3 保持现有实现，只做参数扫描

第一轮不要改变 ticket queue 代码。

支持：

```text
chunk = 1
chunk = 2
chunk = 4
```

伪代码保持：

```cpp
while (true) {
    int ticket_base;

    if (lane == 0) {
        ticket_base =
            atomicAdd(
                next_ticket,
                ticket_chunk);
    }

    ticket_base =
        __shfl_sync(
            FULL_MASK,
            ticket_base,
            0);

    bool terminate = false;

    for (int offset = 0;
         offset < ticket_chunk;
         ++offset) {

        const int task =
            ticket_base + offset;

        TaskState result =
            wait_until_ready_or_final(
                task,
                expected_epoch);

        if (result.final_invalid) {
            terminate = true;
            break;
        }

        process_fragment(
            task,
            result.state);
    }

    if (terminate) {
        break;
    }
}
```

---

# 13. P3 测试逻辑

必须在完成 P2 后再扫。

因为：

```text
consumer warp 数
```

与：

```text
ticket chunk
```

直接影响同时持有的未来 ticket 数量。

假设：

```text
100 active consumer warps
```

则：

```text
chunk=1 → 最多约 100 ticket 在 flight
chunk=4 → 最多约 400 ticket 被提前占有
```

因此 chunk 不能脱离 consumer 数量单独判断。

---

# 14. 建议最终实验矩阵

先找到 P2 最佳配置。

假设结果暂定为：

```text
UPDATE:PROP = 7:1
dedicated = 8
helper = 1
```

再测试：

| Chunk | 说明          |
| ----: | ----------- |
|     1 | 最细粒度、最佳动态均衡 |
|     2 | 中间点         |
|     4 | 当前默认        |

重点记录：

```text
总 kernel 时间
ticket_wait_tail
ticket_wait_ready
invalid_final_tickets
processed_tasks
```

---

# 15. 对 chunk 结果的解释

## 若 chunk=1 最快

说明：

```text
任务负载不均
或
producer-consumer 时间差较大
```

细粒度调度价值大于 ticket atomic 开销。

保留：

```text
chunk=1
```

## 若 chunk=2 最快

说明：

```text
atomic ticket 成本已经可见
但 chunk=4 开始损害负载均衡
```

这通常是比较合理的折中。

## 若 chunk=4 最快

说明：

```text
queue 已经持续有充足 ready task
future-ticket 等待很小
ticket atomic 成本相对明显
```

保留 4。

---

# 16. 建议增加两个轻量 profiling 时间点

虽然本轮不进入完整 timing infrastructure，但可以在 debug 实验中记录：

```cpp
unsigned long long update_finish_clock;
unsigned long long pipeline_finish_clock;
```

例如最后一个 UPDATE block：

```cpp
if (threadIdx.x == 0) {
    const int done =
        atomicAdd(
            update_done_blocks,
            1);

    if (done ==
        update_block_count - 1) {
        update_finish_clock =
            clock64();
    }
}
```

时间步末前：

```cpp
if (global_tid == 0) {
    pipeline_finish_clock =
        clock64();
}
```

目的不是精确计时，而是判断：

```text
UPDATE finish → pipeline finish
```

尾巴有多长。

如果：

```text
7:1 + dedicated=8
```

下 propagation tail 很短，说明应继续把更多 block 给 UPDATE。

如果 tail 很长，则 propagation 资源不足。

---

# 17. P1–P3 推荐施工顺序

## Step 1：删除纯冗余

删除：

```text
propagation_done_tasks
task 尾部 atomicAdd(done)
对应 __syncwarp()
```

先 benchmark。

验收：

```text
结果正确
耗时 <= 当前 14 ms
```

---

## Step 2：拆分 dedicated/helper warp

新增：

```cpp
dedicated_consumer_warps
helper_consumer_warps
```

默认：

```text
8
1
```

先测试：

```text
3:1
D/H =
1/1
4/1
8/1
8/2
```

选最佳。

---

## Step 3：扫 block role ratio

固定最佳 D/H：

```text
7:1
3:1
2:1
1:1
```

找到 UPDATE/PROP block 比例最佳点。

---

## Step 4：扫 ticket chunk

固定最佳：

```text
role ratio
dedicated warps
helper warps
```

测试：

```text
1
2
4
```

---

# 18. 最终验收表

每个配置至少记录：

```text
kernel runtime
relative to naive
ticket_wait_tail
ticket_wait_ready
invalid_final_tickets
processed_tasks
```

最终整理类似：

| Role | D warp | H warp | Chunk |    Time | vs naive |
| ---- | -----: | -----: | ----: | ------: | -------: |
| 3:1  |      1 |      1 |     4 | 14.0 ms |    0.86× |
| 3:1  |      8 |      1 |     4 |     ... |      ... |
| 7:1  |      8 |      1 |     4 |     ... |      ... |
| 7:1  |      8 |      1 |     2 |     ... |      ... |
| 7:1  |      8 |      1 |     1 |     ... |      ... |

---

# 19. 这轮最关键的判断

如果前三项完成后：

```text
pipeline ≈ 12 ms
```

说明：

```text
流水化 overlap 基本覆盖了自身 kernel 协议成本
```

下一阶段才值得处理 host-side delta 和 PSC fold。

如果：

```text
pipeline < 12 ms
```

则已经证明去掉 UPDATE→propagation barrier 本身产生实际收益。

如果仍稳定：

```text
13–14 ms
```

且不同 role/chunk 扫描都无法进一步下降，则更可能说明剩余差距主要来自：

```text
PSC 双缓冲额外内存流量
+ pipeline 必需的 acquire/release/task-state 协议
```

此时继续微调 ticket queue 的收益预计已经有限，应转向 delta 状态表示和实际 overlap 测量。
