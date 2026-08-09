# Pipeline Binned 迁移执行计划

## 1. 目标

当前 pipeline 已解决严重的 queue 竞争问题，但仍比 naive 慢约 5–6%。

旧 binned 版本的关键优势不是单纯“多一个 bin”，而是：

```text
高 fanout：
    仍按大 fragment
    full warp / task

低 fanout：
    不按 1024-edge fragment 处理
    直接 queue neuron
    一个 warp 同时处理多个低 fanout neuron
```

本轮目标是把这一调度原则迁移到 pipeline，同时保持：

```text
UPDATE 与 propagation overlap
ticket queue
UPDATE block 完成后 helper
epoch/ready 发布
PSC 双缓冲
```

暂不引入：

```text
三级/四级 fanout bin
post-neuron reorder
shared-memory hash
atomic aggregation
TMA
```

---

# 3. 第一版只做两档

定义：

```cpp
constexpr int kHighFanoutThreshold = ...;
constexpr int kEdgesPerHighTask = 1024;
constexpr int kLowSubwarpSize = 8;
constexpr int kLowTasksPerWarp = 4;
```

先扫描 threshold：

```text
128
256
512
```

如果旧 binned 已经有经过验证的 threshold，可优先用旧值作为 baseline。

---

# 4. 两类 task 的语义

## HIGH task

保持当前 pipeline 逻辑：

```text
descriptor:
    neuron
    fragment id

execution:
    32-thread warp
    <= 1024 edges
```

边界恢复：

```cpp
row_start = indptr[neuron];
row_end   = indptr[neuron + 1];

edge_start =
    row_start +
    fragment * kEdgesPerHighTask;

edge_end =
    min(
        edge_start + kEdgesPerHighTask,
        row_end);
```

---

## LOW task

不再 fragment。

descriptor 只需要：

```text
neuron ID
```

一个 low task 就代表该 neuron 的完整 outgoing row。

执行：

```text
一个 warp
→ 4 个 8-thread subwarp
→ 同时处理 4 个 low neuron
```

这样 low fanout 不再消耗完整 warp/task。

---

# 5. Queue 布局

优先沿用旧 binned 的“两端队列”设计，避免重新分配两个完整 queue。

假设容量为：

```text
[0, queue_capacity)
```

HIGH 从前往后：

```text
0
1
2
...
```

LOW 从后往前：

```text
capacity-1
capacity-2
capacity-3
...
```

维护：

```cpp
int* high_tail;
int* low_tail;
```

其中：

```text
high task slot:
    slot = atomicAdd(high_tail, count)

low task slot:
    id = atomicAdd(low_tail, 1)
    slot = queue_capacity - 1 - id
```

两边增长，理论上不能相交。

---

# 6. Pipeline counter 设计

现有：

```text
queue_tail
next_ticket
```

改成：

```cpp
int* high_tail;
int* high_next_ticket;

int* low_tail;
int* low_next_ticket;
```

如果希望第一版尽量少改 host，可复用已有：

```text
spike_count[0] → high_tail
work_counter[0] → high_next_ticket
```

再从原有闲置 counter/tensor 中给 LOW 分配：

```text
spike_count[1] → low_tail
work_counter[1] → low_next_ticket
```

如果当前 pipeline tensor 只有一个元素，需要 host 将 counter size 扩到 2。

---

# 7. State / ready 布局

HIGH 与 LOW 都需要 pipeline ready publication。

继续复用：

```cpp
uint32_t* queue_state;
int* queue_neuron;
```

但 slot 位置不同。

HIGH：

```cpp
slot = high_task;
```

LOW：

```cpp
slot =
    queue_capacity - 1 - low_task;
```

state 仍编码：

```text
HIGH:
    epoch + fragment+1

LOW:
    epoch + 固定 low marker
```

建议 LOW 使用：

```cpp
constexpr uint32_t kLowTaskMarker = 1;
```

因为 low task 不需要 fragment。

不过如果 HIGH fragment 也会使用低 16 bit = 1，则无需区分，因为消费者本来就知道自己在 high queue 还是 low queue。

因此 LOW state 可以直接：

```cpp
(epoch << 16) | 1
```

---

# 8. Producer：UPDATE 后按 fanout 分类

当前：

```cpp
if (fired) {
    fragment_count =
        ceil(fanout / 1024);

    publish all fragments;
}
```

改为：

```cpp
if (fired) {
    const int row_start =
        graph_indptr[n];

    const int row_end =
        graph_indptr[n + 1];

    const int fanout =
        row_end - row_start;

    if (fanout >=
        high_fanout_threshold) {

        publish_high(n, fanout);

    } else {

        publish_low(n);
    }
}
```

---

# 9. HIGH producer

保持当前逐 fragment release publication。

```cpp
int fragment_count =
    (fanout +
     kEdgesPerHighTask - 1) /
    kEdgesPerHighTask;

int first =
    atomicAdd(
        high_tail,
        fragment_count);

for (int f = 0;
     f < fragment_count;
     ++f) {

    int slot =
        first + f;

    queue_neuron[slot] = n;

    uint32_t state =
        (epoch << 16) |
        (f + 1);

    state_store_release(
        queue_state + slot,
        state);
}
```

---

# 10. LOW producer

LOW 一个 fired neuron 只产生一个 queue item：

```cpp
int low_id =
    atomicAdd(low_tail, 1);

int slot =
    queue_capacity -
    1 -
    low_id;

queue_neuron[slot] = n;

uint32_t state =
    (epoch << 16) | 1;

state_store_release(
    queue_state + slot,
    state);
```

相比现有 pipeline：

```text
short neuron
→ 一个完整 warp task
```

变成：

```text
4 short neurons
→ 一个 warp execution group
```

---

# 11. Consumer 总体策略

第一版不要引入复杂优先级。

每个 active consumer warp：

```text
优先 HIGH
若 HIGH 暂时无 ready task
    尝试 LOW
若都没有
    backoff
```

原因：

```text
HIGH fragment 粒度大
更适合尽早覆盖 propagation latency

LOW 可以作为填空任务
提高 warp 利用率
```

伪代码：

```cpp
while (true) {

    if (try_process_high()) {
        backoff = MIN_BACKOFF;
        continue;
    }

    if (try_process_low_group()) {
        backoff = MIN_BACKOFF;
        continue;
    }

    if (all_updates_done &&
        high_exhausted &&
        low_exhausted) {
        break;
    }

    nanosleep(backoff);
}
```

---

# 12. HIGH ticket consumer

HIGH 保持当前 ticket 思路。

```cpp
int task =
    atomicAdd(
        high_next_ticket,
        ticket_chunk);
```

第一版建议：

```text
high ticket chunk = 1
```

避免同时把 binning 和 chunk batching 混在一起。

等待：

```cpp
high_tail
queue_state[task]
```

合法后 full warp 处理。

---

# 13. LOW ticket consumer：一次领取 4 个

LOW queue 应利用旧 binned 的核心优势。

lane 0：

```cpp
int base =
    atomicAdd(
        low_next_ticket,
        4);
```

broadcast：

```cpp
base =
    __shfl_sync(
        FULL_MASK,
        base,
        0);
```

warp 分成：

```cpp
int subgroup =
    lane >> 3;

int sublane =
    lane & 7;
```

每个 subgroup：

```cpp
int low_task =
    base + subgroup;
```

对应 slot：

```cpp
int slot =
    queue_capacity -
    1 -
    low_task;
```

---

# 14. LOW ready 判断

需要注意四个 low task 可能并非同时 ready。

不能让整个 warp 因一个未 ready task 卡住。

第一版可以让每个 subgroup 独立判断自己的 task。

subgroup leader：

```cpp
if (sublane == 0) {
    tail =
        load_acquire(low_tail);

    if (low_task < tail) {
        state =
            load_acquire(
                queue_state + slot);

        ready =
            epoch_matches(state);
    } else {
        final_invalid =
            updates_done;
    }
}
```

通过 subgroup shuffle：

```cpp
ready =
    __shfl_sync(
        subgroup_mask,
        ready,
        subgroup_first_lane);
```

然后：

```cpp
if (ready) {
    process_low_neuron();
}
```

---

# 15. LOW group 的一个关键问题：不能丢未 ready task

如果一次：

```text
low_next_ticket += 4
```

但其中：

```text
task 0 ready
task 1 not ready
task 2 ready
task 3 not ready
```

已经领取的 task 1/3 不能被丢掉。

因此第一版推荐使用更简单、更安全的协议：

```text
一次只领取“当前已 reserve 的连续 4 个 low task”
```

即 lane 0：

```cpp
int head =
    load(low_next_ticket);

int tail =
    load(low_tail);

int available =
    tail - head;

if (available >= 4) {
    base =
        atomicAdd(
            low_next_ticket,
            4);
}
```

然后四个 descriptor 各自等 ready。

由于 producer 是：

```text
reserve low_tail
→ immediately publish one neuron
```

LOW task reserve/publish 间隔很短。

第一版可以接受 subgroup 等 ready。

---

# 16. 更稳妥的 LOW 版本

如果担心四个 subgroup 各自 wait 形成 divergent spin，则改为：

```text
只有四个 task 都 ready
才批量领取
```

lane 0检查：

```cpp
head =
    load(low_next_ticket);

tail =
    load(low_tail);

if (head + 4 <= tail) {

    bool all_ready = true;

    for i in 0..3:
        slot =
            capacity - 1 -
            (head + i);

        if state epoch mismatch:
            all_ready = false;

    if (all_ready) {
        base =
            atomicAdd(
                low_next_ticket,
                4);
    }
}
```

这增加了 4 次 state load，但避免已领取 task 的等待状态复杂化。

对于第一版迁移，我更推荐这个方案：

```text
LOW 只批量领取连续 ready 的 4 task
```

因为更容易保证正确性。

---

# 17. LOW propagation

每个 8-thread subgroup：

```cpp
int neuron =
    queue_neuron[slot];

int row_start =
    graph_indptr[neuron];

int row_end =
    graph_indptr[neuron + 1];

for (int edge =
         row_start + sublane;
     edge < row_end;
     edge += 8) {

    const int post =
        graph_indices[edge];

    atomicAdd(
        delta_write + post,
        graph_weight[edge]);
}
```

没有 fragment decode。

---

# 18. HIGH/LOW 调度顺序第一版

推荐：

```text
HIGH → LOW → backoff
```

伪代码：

```cpp
while (true) {

    bool did_work = false;

    if (high_work_available()) {
        process_high();
        did_work = true;
    }

    if (!did_work &&
        low_group_available()) {
        process_low_group();
        did_work = true;
    }

    if (did_work) {
        backoff = MIN_BACKOFF;
        continue;
    }

    if (updates_done &&
        high_finished &&
        low_finished) {
        break;
    }

    nanosleep(backoff);

    backoff =
        min(
            backoff << 1,
            MAX_BACKOFF);
}
```

---

# 19. 为什么不应该只固定 HIGH-first

长期可能出现：

```text
HIGH queue 长期有任务
→ LOW queue 一直得不到处理
```

所以第一版需要一个很简单的公平机制。

例如：

```cpp
bool prefer_high = true;
```

每完成一次任务后翻转：

```cpp
prefer_high = !prefer_high;
```

调度：

```cpp
if (prefer_high) {
    try_high();
    if (!worked)
        try_low();
} else {
    try_low();
    if (!worked)
        try_high();
}
```

或者更偏高：

```text
HIGH
HIGH
LOW
```

第一版先用：

```text
1:1 alternating
```

便于分析。

---

# 20. UPDATE helper 与 dedicated consumer

保持当前最优结构：

```text
role ratio = 7:1
dedicated warp = 1
helper warp = 1
static waves = 0
ticket chunk high = 1
```

先不要同时调整这些参数。

这样能把性能变化尽可能归因于：

```text
task binning + execution width
```

---

# 21. Counter 初始化

每 timestep：

```cpp
if (global_tid == 0) {
    *high_tail = 0;
    *low_tail = 0;

    *high_next_ticket = 0;
    *low_next_ticket = 0;

    *update_done_blocks = 0;
}

grid.sync();
```

---

# 22. Queue capacity 检查

必须确保：

```text
high_tail + low_tail
<= queue_capacity
```

debug 版本可加：

```cpp
if (high_tail +
    low_tail >
    queue_capacity) {

    atomicExch(
        queue_overflow,
        1);
}
```

release 版本若现有 capacity 计算足够保守，可去掉检查。

---

# 23. Host queue capacity

当前 pipeline capacity 仍按照：

```text
N +
ceil(E / 128)
```

这实际上比 1024-edge high fragment 需要的容量保守很多。

Binned 迁移后：

```text
最大 low tasks <= N
最大 high tasks <= ceil(E / 1024)
```

所以：

```text
N + ceil(E/128)
```

依然安全。

第一版不修改 capacity 计算，避免引入额外问题。

后续再收紧。

---

# 24. Debug counter

新增：

```text
high_task_count
low_task_count

high_processed
low_processed

low_group_claims
low_partial_group_attempts
```

检查：

```text
high_processed == final high_tail
low_processed == final low_tail
```

还需要记录：

```text
low_group_claims
```

用于估算：

```text
low task / claim
```

是否接近 4。

---

# 25. 关键性能统计

每个 threshold 记录：

```text
Core kernel time
Pipeline UPDATE
Consumer startup
Overlap window
Propagation tail

high tasks
low tasks
high edges
low edges

high claims
low claims
```

最好增加：

```text
effective tasks per claim
```

例如：

```text
LOW:
    3.7 neuron / claim
```

---

# 26. 第一阶段实验：threshold 扫描

固定其他参数：

```text
7:1
dedicated=1
helper=1
high chunk=1
static=0
```

测试：

```text
threshold = 128
256
512
```

baseline：

```text
no binning
```

表：

| Threshold | High tasks | Low tasks | Claims | Tail | Kernel |
| --------: | ---------: | --------: | -----: | ---: | -----: |
|      none |            |           |        |      |        |
|       128 |            |           |        |      |        |
|       256 |            |           |        |      |        |
|       512 |            |           |        |      |        |

---

# 27. 结果判断

## 情况 A：Kernel 明显下降

例如：

```text
12.65 ms
→ 11.9 ms
```

说明 binned 迁移成功。

下一阶段再优化：

```text
subwarp size
high/low scheduling ratio
role ratio
```

---

## 情况 B：Task/claim 明显下降，但 kernel 不变

说明：

```text
scheduler 固定成本已经不是主瓶颈
```

此时看：

```text
propagation tail
UPDATE slowdown
```

若都不变，继续 binning 的意义有限，应转向 atomic writeback。

---

## 情况 C：Propagation tail 下降但 UPDATE slowdown 增加

说明 binning 增加了传播吞吐，但同时加剧了 overlap 资源竞争。

此时重新扫：

```text
7:1
更少 dedicated consumer
```

或者降低 LOW 的并发比例。

---

## 情况 D：Kernel 变慢

重点排查：

```text
LOW queue polling
4-state ready check
warp divergence
threshold 太高
subwarp 太窄
```

---

# 28. 第二阶段：subwarp size 扫描

只有两档 binned 已经有正收益后再做。

测试：

```text
subwarp = 4
→ 8 low tasks / warp

subwarp = 8
→ 4 low tasks / warp

subwarp = 16
→ 2 low tasks / warp
```

对应 threshold 应配套。

粗略：

```text
4 threads:
    fanout <= 32/64

8 threads:
    fanout <= 128/256

16 threads:
    fanout <= 256/512
```

第一版不要自动匹配，手动组合测试。

---

# 29. 第三阶段：HIGH/LOW 消费比例

如果 LOW queue 太容易占据 consumer，可测试：

```text
1 HIGH : 1 LOW
2 HIGH : 1 LOW
4 HIGH : 1 LOW
```

例如：

```cpp
int phase =
    local_schedule_counter++ % 3;

if (phase < 2)
    prefer_high = true;
else
    prefer_high = false;
```

目标是避免：

```text
为了提高 short-row lane utilization
反而延迟真正占 edge 大头的 high tasks
```

---

# 30. 推荐施工顺序

## Step 1

统计 fired fanout：

```text
neuron count
edge count
task count
```

确定 threshold 候选。

## Step 2

增加：

```text
high_tail
low_tail
high_next_ticket
low_next_ticket
```

并实现两端 queue。

## Step 3

Producer 分流：

```text
HIGH → 1024-edge fragments
LOW → neuron descriptor
```

## Step 4

先实现 HIGH consumer，确认与当前 pipeline 性能/正确性一致。

此时 LOW 暂时仍用 full warp。

目的是验证双队列本身没有工程错误。

## Step 5

实现 LOW：

```text
4 × 8-thread subwarp
```

一次领取四个连续 ready low task。

## Step 6

加入 HIGH/LOW alternating scheduler。

## Step 7

验证：

```text
high_processed == high_tail
low_processed == low_tail
spike/v/psc 完全正确
```

## Step 8

扫：

```text
threshold 128/256/512
```

## Step 9

若有收益，再扫：

```text
subwarp 4/8/16
```

## Step 10

最后重新扫 role ratio。

---

# 31. 第一版最重要的控制变量

为了能够判断 binned 本身有没有价值，第一版务必保持：

```text
role ratio = 当前最佳 7:1
dedicated warp = 1
helper warp = 1
ticket chunk = 1
static waves = 0
```

只改变：

```text
task formation
+
low-task execution width
```

不要同时重新做 queue chunk、静态 waves 或 warp role 优化。

---

# 32. 预期的最终结构

```text
                    fired neuron
                         |
                 +-------+-------+
                 |               |
             low fanout       high fanout
                 |               |
          one neuron/task    1024-edge fragments
                 |               |
                 v               v
        LOW queue(back)     HIGH queue(front)
                 |               |
                 +-------+-------+
                         |
                    consumers
                         |
             +-----------+-----------+
             |                       |
        LOW available           HIGH available
             |                       |
       8-thread × 4             full warp
             |                       |
             +-----------+-----------+
                         |
                   atomic writeback
```

这一版的核心目的不是把 scheduler 复杂化，而是把：

```text
task 粒度
```

重新和：

```text
实际 fanout / execution width
```

匹配起来。

如果旧 binned 的收益确实主要来自这一点，那么这是目前最有希望把 pipeline 从 12.6 ms 拉回甚至低于 naive 11.8 ms 的调度侧优化。
