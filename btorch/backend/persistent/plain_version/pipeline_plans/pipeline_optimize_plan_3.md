# Persistent SNN Pipeline：静态启动与资源平衡优化计划

## 1. 本轮目标

当前 1:1 配置下：

```text
Serial UPDATE        20.992 µs
Serial propagation   70.400 µs

Pipeline UPDATE      29.440 µs
Consumer startup      9.728 µs
Overlap window       14.848 µs
Propagation tail     62.976 µs
Pipeline total       92.160 µs
```

说明当前：

```text
UPDATE 额外成本：
29.440 - 20.992
= 8.448 µs

被隐藏的 propagation：
70.400 - 62.976
= 7.424 µs
```

当前资源交换略亏。

本轮优化目标：

```text
1. 缩短 consumer startup
2. 降低稳态 ticket 调度成本
3. 保持后半段动态负载均衡
4. 找到 UPDATE / PROP 更合理的资源配比
5. 判断 task 是否本来就发布得较晚
```

最终关注：

```text
T_pipeline
```

而不是单独最小化：

```text
T_update
```

---

# 2. 总体调度结构

将 propagation 生命周期拆成三个阶段：

```text
阶段 A：静态启动
    dedicated consumer warp
    直接处理预分配 task id

阶段 B：动态稳态
    dedicated consumer
    通过 ticket queue 继续领取任务

阶段 C：尾部 drain
    UPDATE block 完成后加入 helper
    与 dedicated consumer 一起动态领取剩余任务
```

结构：

```text
UPDATE producer
    |
    | publish task 0,1,2...
    v

Dedicated PROP:
    static wave 0
        warp0 -> task0
        warp1 -> task1
        ...
    |
    v
dynamic ticket queue
    |
    +----------------------+
                           |
UPDATE finishes            |
    |                      |
helper warps join ---------+
    |
dynamic drain
    |
grid.sync
```

---

# 3. 第一阶段：实现静态首轮 task assignment

## 3.1 目的

当前 consumer 从 first publish 到 first consume：

```text
9.728 µs
```

首轮静态 assignment 的目标是删除启动阶段的：

```text
atomicAdd(next_ticket)
```

以及其前后的动态调度等待。

Dedicated consumer 在进入 pipeline 时已经知道：

```text
自己应该等待哪个 task
```

---

# 4. Dedicated consumer 全局 rank

首先为 dedicated consumer warp 计算连续编号。

假设：

```cpp
const int warp_in_block =
    threadIdx.x >> 5;

const int lane =
    threadIdx.x & 31;

const int propagation_block_rank =
    blockIdx.x -
    update_block_count;
```

则：

```cpp
const int dedicated_warp_rank =
    propagation_block_rank *
        dedicated_consumer_warps +
    warp_in_block;
```

只对：

```cpp
!is_update_block &&
warp_in_block <
    dedicated_consumer_warps
```

有效。

Dedicated consumer 总数：

```cpp
const int propagation_block_count =
    gridDim.x - update_block_count;

const int total_dedicated_warps =
    propagation_block_count *
    dedicated_consumer_warps;
```

---

# 5. static_waves 参数

新增运行时参数：

```cpp
int static_waves;
```

支持：

```text
0
1
2
4
```

其中：

```text
0 = 当前纯动态 ticket 版本
```

建议环境变量：

```text
BTORCH_PIPELINE_STATIC_WAVES
```

默认第一轮设：

```text
1
```

---

# 6. 初始化 next_ticket

若使用 `static_waves`：

```cpp
const int static_task_count =
    total_dedicated_warps *
    static_waves;
```

时间步开始：

```cpp
if (global_tid == 0) {
    *queue_tail = 0;

    *next_ticket =
        static_task_count;

    *update_done_blocks = 0;
}

grid.sync();
```

即：

```text
[0, static_task_count)
```

这部分 ticket 不允许动态消费者再领取。

动态队列从：

```text
static_task_count
```

开始。

---

# 7. Dedicated consumer 静态阶段

Dedicated warp：

```cpp
if (!is_update_block &&
    warp_in_block <
        dedicated_consumer_warps) {

    for (int wave = 0;
         wave < static_waves;
         ++wave) {

        const int task =
            dedicated_warp_rank +
            wave *
                total_dedicated_warps;

        bool valid =
            wait_static_task_or_final(
                task,
                expected_epoch);

        if (!valid) {
            break;
        }

        process_fragment(task);
    }

    run_dynamic_consumer();
}
```

---

# 8. 静态 task 等待逻辑

静态 task 同样可能：

```text
尚未 reserve
尚未 ready
最终不存在
```

因此使用和 ticket queue 一样的等待状态机，但不执行 ticket atomic。

伪代码：

```cpp
bool valid = false;
bool final_invalid = false;
uint32_t state = 0;

int backoff = 32;

while (true) {
    if (lane == 0) {
        const int tail =
            load_acquire(queue_tail);

        if (task < tail) {
            state =
                state_load_acquire(
                    spike_queue_state +
                    task);

            valid =
                (state >> 16) ==
                expected_epoch;
        } else {
            const int update_done =
                load_acquire(
                    update_done_blocks);

            final_invalid =
                update_done ==
                    update_block_count;
        }

        if (!valid &&
            !final_invalid) {

            __nanosleep(backoff);

            backoff =
                min(
                    backoff << 1,
                    kMaxBackoff);
        }
    }

    valid =
        __shfl_sync(
            FULL_MASK,
            valid,
            0);

    final_invalid =
        __shfl_sync(
            FULL_MASK,
            final_invalid,
            0);

    state =
        __shfl_sync(
            FULL_MASK,
            state,
            0);

    if (valid ||
        final_invalid) {
        break;
    }
}
```

如果：

```cpp
final_invalid
```

则本 warp 后面的 static wave 也一定无效：

```cpp
break;
```

然后仍可进入动态队列。

不过如果 producer 已经全部结束，此时动态队列也会立即退出。

---

# 9. Helper block 不参与静态任务

UPDATE block 完成后：

```cpp
if (is_update_block) {
    run_update();

    __syncthreads();

    if (threadIdx.x == 0) {
        update_done_release();
    }

    __syncthreads();

    if (warp_in_block <
        helper_consumer_warps) {

        run_dynamic_consumer();
    }
}
```

Helper 不处理：

```text
task < static_task_count
```

因为这部分已经归 dedicated warp 所有。

---

# 10. 动态 consumer 保持现有 ticket queue

静态任务完成后：

```cpp
while (true) {
    int first_ticket = -1;

    if (lane == 0) {
        first_ticket =
            atomicAdd(
                next_ticket,
                ticket_chunk);
    }

    first_ticket =
        __shfl_sync(
            FULL_MASK,
            first_ticket,
            0);

    for (int offset = 0;
         offset < ticket_chunk;
         ++offset) {

        const int task =
            first_ticket + offset;

        auto result =
            wait_until_ready_or_final(
                task);

        if (result.final_invalid) {
            return;
        }

        process_fragment(
            task,
            result.state);
    }
}
```

本轮不修改 dynamic ticket queue 的基本协议。

---

# 11. 第二阶段：扫描 static waves

固定当前最佳：

```text
role ratio
dedicated warp
helper warp
ticket chunk
```

只扫：

```text
static_waves =
0
1
2
4
```

记录：

```text
Core kernel time
Consumer startup
Overlap window
Pipeline UPDATE
Propagation tail
```

---

# 12. static wave 结果判断

## 如果 wave=1 最快

说明：

```text
启动阶段动态 ticket 调度确实存在明显固定成本
```

而更多静态任务开始损害负载均衡。

保留：

```text
static_waves = 1
```

## 如果 wave=2/4 更快

说明：

```text
前期 task 足够规则
dynamic claim 本身成本仍然明显
```

可以进一步考虑更长静态前缀。

## 如果 wave=0 最快

说明：

```text
startup 主要不是 claim atomic
而是 task supply 本身过晚
```

这时直接进入 task publication 顺序优化。

---

# 13. 第三阶段：测量 task publication curve

需要判断 9.728 µs startup 来自：

```text
A. consumer 调度慢
还是
B. producer 前期根本没有足够任务
```

新增 debug-only publication 统计。

---

# 14. Publication milestone

推荐记录：

```text
first task
25% update progress
50% update progress
75% update progress
update finished
```

对应每个阶段的：

```text
queue_tail
```

而不是事后按照最终 task 数百分比触发，因为最终 task 数在 UPDATE 完成前未知。

---

# 15. 按 UPDATE block 完成度采样 queue_tail

已有：

```cpp
update_done_blocks
```

当某 block 完成 UPDATE：

```cpp
const int old =
    atomicAdd(
        update_done_blocks,
        1);

const int completed =
    old + 1;
```

debug 模式中判断：

```cpp
if (completed ==
    update_block_count / 4) {

    publication_stats[t][0] =
        load_acquire(queue_tail);
}

if (completed ==
    update_block_count / 2) {

    publication_stats[t][1] =
        load_acquire(queue_tail);
}

if (completed ==
    update_block_count * 3 / 4) {

    publication_stats[t][2] =
        load_acquire(queue_tail);
}

if (completed ==
    update_block_count) {

    publication_stats[t][3] =
        load_acquire(queue_tail);
}
```

最终：

```text
Q25
Q50
Q75
Q100
```

表示：

```text
UPDATE 完成 25% 时已有多少 task
UPDATE 完成 50% 时已有多少 task
...
```

---

# 16. Publication curve 分析

计算：

```text
Q25 / Q100
Q50 / Q100
Q75 / Q100
```

例如：

```text
0.05
0.18
0.52
1.00
```

说明任务明显后置。

如果：

```text
Q25 / Q100 很低
```

那么 dedicated consumer 前期闲置是 task supply 问题，不是 queue 问题。

如果：

```text
Q25 / Q100 已经很高
但 consumer startup 仍接近 10 µs
```

说明 scheduler startup 才是主要问题。

---

# 17. 第四阶段：重新扫 role ratio

在 static-wave 最优配置确定后，重新测试：

```text
7:1
3:1
2:1
1:1
```

这次不以：

```text
Pipeline UPDATE 越短越好
```

为标准。

---

# 18. 新的评价指标：resource exchange ratio

对每个配置计算：

```text
UPDATE slowdown
=
T_update_pipeline -
T_update_serial
```

```text
Propagation hidden
=
T_prop_serial -
T_prop_tail
```

定义：

```text
Exchange ratio
=
Propagation hidden /
UPDATE slowdown
```

当前 1:1：

```text
7.424 / 8.448
≈ 0.88
```

小于 1。

理想：

```text
Exchange ratio > 1
```

表示：

```text
为了 overlap 多付出的 UPDATE 时间
小于
被隐藏的 propagation 时间
```

---

# 19. 不要单独优化 exchange ratio

最终仍以：

```text
T_pipeline
```

最小为主。

因为一种配置可能：

```text
UPDATE slowdown = 1 µs
Propagation hidden = 2 µs
ratio = 2
```

但总收益只有 1 µs。

另一种：

```text
UPDATE slowdown = 10 µs
Propagation hidden = 25 µs
ratio = 2.5
```

总收益明显更大。

因此同时记录：

```text
Exchange ratio
Net overlap gain
Pipeline total
```

其中：

```text
Net overlap gain
=
Propagation hidden -
UPDATE slowdown
```

---

# 20. Role ratio 实验表

建议：

| Role | Update pipeline | Startup | Overlap |  Tail | Exchange ratio | Total |
| ---- | --------------: | ------: | ------: | ----: | -------------: | ----: |
| 7:1  |                 |         |         |       |                |       |
| 3:1  |                 |         |         |       |                |       |
| 2:1  |                 |         |         |       |                |       |
| 1:1  |           29.44 |    9.73 |   14.85 | 62.98 |           0.88 | 92.16 |

---

# 21. 第五阶段：寻找硬件效率平衡点

最佳配置不一定满足：

```text
UPDATE duration 最短
```

更合理的目标是：

```text
update_done
≈
pipeline_done
```

即：

```text
Propagation tail 尽可能接近 0
```

但不能以巨大 UPDATE slowdown 为代价。

理想时间线：

```text
UPDATE:
|--------------------------------------|

PROP:
     |---------------------------------|

                                      ^
                           approximately same end
```

---

# 22. 如果 7:1 最优

说明：

```text
少量 dedicated propagation block
已经足以覆盖较多传播工作

过多 propagation block
主要造成共享资源竞争
```

下一步应保持较少 dedicated block，并重点优化：

```text
传播 block 内 warp 利用率
task supply
atomic writeback
```

---

# 23. 如果 1:1 或 2:1 最优

说明：

```text
propagation 本身仍然需要大量执行资源
```

这时进一步降低 queue 调度成本和优化 propagation kernel 本身会更重要。

---

# 24. 第六阶段：若 task publication 明显后置，调整 UPDATE 顺序

只有 publication curve 明确显示：

```text
任务严重集中在 UPDATE 后半段
```

才做这一项。

第一版不要立刻按 firing probability 重排整个 neuron，因为会破坏：

```text
v
psc
input_current
```

的连续访问。

先做简单实验。

---

# 25. 简单 UPDATE 顺序实验

### 方案 A：原顺序

```cpp
n = logical_n;
```

### 方案 B：预计算 fanout 降序

host 预处理：

```cpp
update_order
```

kernel：

```cpp
const int logical_n =
    update_tid + k * update_stride;

const int n =
    update_order[logical_n];
```

观察：

```text
publication curve
consumer startup
UPDATE duration
pipeline duration
```

---

# 26. fanout 排序的目标

不是让 UPDATE 本身更均衡，而是：

```text
让会生成较多 propagation task 的 neuron
尽量更早完成 UPDATE
```

从而增加：

```text
early queue supply
```

但代价是：

```text
随机访问 v / psc / delta / input_current
```

因此只有在：

```text
startup / supply 问题明显
```

时才值得。

---

# 27. 更温和的顺序方案：分桶而不是全排序

如果全 fanout 排序严重损害 UPDATE locality，可按粗粒度分桶：

```text
high fanout
medium fanout
low fanout
```

例如：

```text
>=1024
256–1023
<256
```

UPDATE 顺序：

```text
每个 region 内保持原 neuron 顺序
但高 fanout bucket 略提前
```

这样既能：

```text
提前产生较多 task
```

又不会完全破坏连续访问。

---

# 28. 推荐施工顺序

## Step 1

实现：

```text
static_waves
```

以及：

```text
dedicated_warp_rank
static_task_count
next_ticket = static_task_count
```

---

## Step 2

测试：

```text
static_waves =
0 / 1 / 2 / 4
```

记录：

```text
startup
pipeline total
tail
```

---

## Step 3

增加 publication curve debug：

```text
Q25
Q50
Q75
Q100
```

---

## Step 4

分析：

```text
scheduler startup
vs
producer supply delay
```

---

## Step 5

固定最佳 static wave，重新扫：

```text
7:1
3:1
2:1
1:1
```

计算：

```text
UPDATE slowdown
Propagation hidden
Exchange ratio
Net overlap gain
Total
```

---

## Step 6

选择：

```text
总时间最短
+
硬件交换效率合理
```

的 role ratio。

---

## Step 7

只有 task publication 明显后置时：

```text
测试 UPDATE task 生成顺序
```

先 fanout bucket，再考虑完整排序。

---

# 29. 本轮暂不修改

暂不做：

```text
per-producer queue
ready bitmap
多级 scheduler
delta representation
atomic aggregation
TMA
warp specialization 内部重构
完全静态传播
```

当前距离 naive 只有约 5.7%，优先用最小改动找回这部分性能。

---

# 30. 最终希望得到的结构

```text
Timestep begins
      |
      v
UPDATE producers -------------------------------+
      |                                         |
      | first fragments                         |
      v                                         |
Dedicated PROP                                  |
      |                                         |
      | static first wave                       |
      | task_id = warp_rank                     |
      v                                         |
dynamic ticket queue                            |
      |                                         |
      |                                         |
UPDATE blocks finish ---------------------------+
      |
      v
helper warps join dynamic queue
      |
      v
PROP tail drain
      |
      v
grid.sync
```

最终评价不再是：

```text
UPDATE 有没有变慢
```

而是：

```text
是否用最少的共享资源竞争，
换取最多的 propagation 隐藏时间，
从而让 pipeline 总关键路径短于 naive。
```
