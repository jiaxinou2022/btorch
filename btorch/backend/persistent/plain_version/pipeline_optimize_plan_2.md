# Pipeline 第三阶段：Host 开销拆分与真实 Overlap 测量计划

## 1. 本阶段目标

回答两个问题：

```text
Q1. pipeline kernel 本身到底比 naive 快还是慢？
Q2. UPDATE 和 propagation 实际重叠了多少？
```

不要再只比较：

```text
naive forward = 12 ms
pipeline forward = 14 ms
```

因为 pipeline 路径当前额外包含：

```text
delta_0 分配 + 清零
delta_1 分配 + 清零
forward epoch 管理
偶发 queue state 清零
persistent pipeline kernel
psc_out.add_(last_delta)
```

而 naive 没有完全相同的 host-side 状态管理。

---

# 2. P4：拆分 kernel 与 host 侧状态管理开销

## 2.1 需要拆出的时间项

建议至少拆成：

```text
T_clone
T_delta_init
T_queue_state_clear
T_pipeline_kernel
T_psc_fold
T_total
```

其中真正关注：

```text
T_pipeline_kernel
T_delta_init
T_psc_fold
```

### 当前 pipeline host 路径

大致是：

```cpp
auto v_out = v.clone();
auto psc_out = psc.clone();

auto recurrent_delta_0 =
    torch::zeros_like(psc_out);

auto recurrent_delta_1 =
    torch::zeros_like(psc_out);

if (must_clear_queue_state) {
    spike_queue_edge_start.zero_();
}

launch_persistent_snn_kernel(...);

psc_out.add_(
    (t_steps & 1)
        ? recurrent_delta_1
        : recurrent_delta_0);
```

因此当前 14 ms 不是单纯 persistent kernel 时间。

---

# 3. P4-A：先加 benchmark-only 分段 CUDA Event

不要用 CPU `chrono` 测 GPU kernel，因为会混入异步提交误差。

在测试程序或 benchmark wrapper 中增加 CUDA Event：

```cpp
cudaEvent_t e0, e1, e2, e3, e4;

cudaEventCreate(&e0);
...
```

计时：

```cpp
cudaEventRecord(e0, stream);

// delta 初始化
allocate_or_zero_delta();

cudaEventRecord(e1, stream);

// persistent kernel
launch_persistent_snn_kernel(...);

cudaEventRecord(e2, stream);

// psc fold
psc_out.add_(last_delta);

cudaEventRecord(e3, stream);

cudaEventSynchronize(e3);
```

分别得到：

```cpp
float delta_ms;
float kernel_ms;
float fold_ms;

cudaEventElapsedTime(
    &delta_ms,
    e0,
    e1);

cudaEventElapsedTime(
    &kernel_ms,
    e1,
    e2);

cudaEventElapsedTime(
    &fold_ms,
    e2,
    e3);
```

同时完整 forward 外层继续保留原 benchmark。

---

# 4. P4-B：避免 allocator 噪声污染 delta 计时

`torch::zeros_like()` 同时包含：

```text
allocator
+
memset/zero kernel
```

因此最好拆成两个实验。

## 实验 A：保持现状

测：

```text
zeros_like + zero
```

代表当前真实 forward 成本。

## 实验 B：预分配 workspace

在 benchmark 外预先：

```cpp
auto delta0 =
    torch::empty_like(psc);

auto delta1 =
    torch::empty_like(psc);
```

每次迭代只：

```cpp
delta0.zero_();
delta1.zero_();
```

测纯清零成本。

如果：

```text
A 明显 > B
```

说明 allocator/缓存管理也是额外成本。

如果接近，则主要是显存写流量。

---

# 5. P4-C：建立 temporary pure-kernel benchmark path

为了准确比较 naive 与 pipeline，建议增加一个只用于 benchmark 的接口：

```text
persistent_snn_pipeline_kernel_only
```

它不负责：

```text
delta 创建
psc fold
event compact
额外状态转换
```

调用方提前提供：

```cpp
v_out
psc_out
delta0
delta1
input_current
queue buffers
```

函数只做：

```cpp
launch_persistent_snn_kernel(...);
```

这样比较：

```text
naive persistent kernel
vs
pipeline persistent kernel
```

才是 apples-to-apples。

---

# 6. P4-D：第一轮不要立即改变公开 API

当前先不重构正式 operator。

只在 benchmark 内维护：

```cpp
PipelineWorkspace {
    Tensor delta0;
    Tensor delta1;
}
```

伪代码：

```cpp
auto delta0 =
    torch::zeros_like(psc);

auto delta1 =
    torch::zeros_like(psc);

// warmup
for (...) {
    delta0.zero_();
    delta1.zero_();

    launch_pipeline_kernel(
        ...,
        delta0,
        delta1);
}

// timed
record(start);

launch_pipeline_kernel(...);

record(stop);
```

这样可以先知道：

```text
persistent delta workspace
```

能省多少。

---

# 7. P4-E：测试取消 `psc_out.add_()` 的 pure-kernel 模式

当前 pipeline 内部最终状态实际上是：

```text
psc_state
+
last_delta
```

host 为了兼容原 API 才执行：

```cpp
psc_out.add_(last_delta);
```

因此做两个 benchmark：

### Compatibility mode

```text
pipeline kernel
+
psc fold
```

### Split-state mode

```text
pipeline kernel only
```

不 fold。

只要下一次 kernel继续接受：

```text
psc
delta0
delta1
slot
```

split-state 本身是合法内部表示。

暂时不要求对外暴露，只测潜在收益。

---

# 8. P4 验收输出

建议生成：

| Component              | Naive | Pipeline |
| ---------------------- | ----: | -------: |
| Core persistent kernel |  x ms |     x ms |
| Delta init             |     — |     x ms |
| PSC fold               |     — |     x ms |
| Queue state clear      |     — |     x ms |
| Full forward           | 12 ms |    14 ms |

关键判断：

### 情况 A

```text
pipeline kernel < naive kernel
```

但 full forward 更慢。

说明：

> 流水化本身已经成功，剩余问题主要是状态表示和 host 接口。

### 情况 B

```text
pipeline kernel ≈ naive kernel
```

说明 overlap 大致只抵消了协议成本。

### 情况 C

```text
pipeline kernel > naive kernel
```

说明还需要继续分析 overlap 和资源竞争。

---

# 9. P5：测量真实 UPDATE–Propagation overlap

当前总时间无法告诉我们：

```text
传播到底什么时候开始
UPDATE 什么时候结束
最后一个 propagation 什么时候结束
```

需要在 kernel 内加入低开销 timestamp instrumentation。

---

# 10. P5-A：记录三个关键时间点

每个 timestep 记录：

```text
t_begin
t_update_done
t_pipeline_done
```

最好再增加：

```text
t_first_task_publish
t_first_task_consume
```

形成：

```text
t_begin
    ↓
first publish
    ↓
first consume
    ↓
UPDATE done
    ↓
last propagation done
```

---

# 11. 时间戳缓冲

新增 debug-only：

```cpp
unsigned long long* timing_stats;
```

每 timestep 例如 5 个字段：

```cpp
enum {
    TS_BEGIN = 0,
    TS_FIRST_PUBLISH = 1,
    TS_FIRST_CONSUME = 2,
    TS_UPDATE_DONE = 3,
    TS_PIPELINE_DONE = 4,
    TS_COLUMNS = 5
};
```

数组：

```text
[t_steps, 5]
```

---

# 12. P5-B：记录 timestep begin

外部事件同步结束、正式 UPDATE–propagation pipeline 开始前：

```cpp
grid.sync();

if (global_tid == 0) {
    timing_stats[
        t * TS_COLUMNS +
        TS_BEGIN] =
        clock64();
}
```

然后最好再：

```cpp
grid.sync();
```

但这会引入额外同步，污染 benchmark。

更推荐：

```text
只在 debug timing build 中启用
```

因此 timing build 不用于最终性能值，只用于结构归因。

---

# 13. P5-C：记录 first task publish

Producer 第一次发布当前 timestep task 时：

```cpp
const unsigned long long now =
    clock64();

atomicMin(
    &timing_stats[
        t * TS_COLUMNS +
        TS_FIRST_PUBLISH],
    now);
```

问题是 `atomicMin` 本身会有成本。

因此只让每个 producer 在其第一次 spike 时尝试：

```cpp
if (!local_published_any_task) {
    atomicMin(...);
    local_published_any_task = true;
}
```

更简单的是使用全局 flag：

```cpp
if (atomicCAS(
        first_publish_flag,
        0,
        1) == 0) {

    timing_stats[...] =
        clock64();
}
```

每 timestep初始化：

```cpp
*first_publish_flag = 0;
```

---

# 14. P5-D：记录 first task consume

consumer 第一次拿到 ready ticket 并准备执行：

```cpp
if (atomicCAS(
        first_consume_flag,
        0,
        1) == 0) {

    timing_stats[
        t * TS_COLUMNS +
        TS_FIRST_CONSUME] =
        clock64();
}
```

这可以测：

```text
publish → consume latency
```

如果很大，说明队列调度仍有启动延迟。

---

# 15. P5-E：记录最后一个 UPDATE block 完成

当前：

```cpp
if (threadIdx.x == 0) {
    int old =
        atomicAdd(
            update_done_blocks,
            1);
}
```

利用返回值：

```cpp
if (old ==
    update_block_count - 1) {

    timing_stats[
        t * TS_COLUMNS +
        TS_UPDATE_DONE] =
        clock64();
}
```

这个几乎没有额外同步。

---

# 16. P5-F：如何记录最后一个 propagation 完成

此前为了性能删除了：

```text
propagation_done_tasks
```

不要为了 timing 永久加回来。

只在 debug timing build 下临时恢复：

```cpp
#ifdef ENABLE_PIPELINE_TIMING
if (lane == 0) {
    const int old =
        atomicAdd(
            timing_done_tasks,
            1);

    const int final_tail =
        load_acquire(queue_tail);

    const int updates_done =
        load_acquire(
            update_done_blocks);

    if (updates_done ==
            update_block_count &&
        old + 1 ==
            final_tail) {

        timing_stats[
            t * TS_COLUMNS +
            TS_PIPELINE_DONE] =
            clock64();
    }
}
#endif
```

不过这里存在：

```text
最后 task 完成时，
final tail 是否已经固定
```

只有 UPDATE 已全部完成才能成立。

如果最后一个 propagation task 比 UPDATE 更早完成，则之后 producer 还可能继续增加 tail。

因此更稳妥的方法是：

```text
所有 consumer 最终退出前
由最后到达的 block/warp记录
```

---

# 17. 推荐的 pipeline done 时间戳方案

debug timing 模式重新增加：

```cpp
int* consumer_done_warps;
```

每个 active consumer warp 完全退出 ticket loop 后：

```cpp
if (lane == 0) {
    const int old =
        atomicAdd(
            consumer_done_warps,
            1);

    if (old ==
        total_consumer_warps - 1) {

        timing_stats[
            t * TS_COLUMNS +
            TS_PIPELINE_DONE] =
            clock64();
    }
}
```

但是 UPDATE block 是否转 consumer 导致 active consumer 数随阶段变化，需要正确计算总数。

更简单：

```text
用 block 完成计数
```

每个 block 中所有 active consumer warp退出后：

```cpp
__syncthreads();

if (threadIdx.x == 0) {
    const int old =
        atomicAdd(
            pipeline_done_blocks,
            1);

    if (old ==
        gridDim.x - 1) {

        timing_stats[
            ... TS_PIPELINE_DONE] =
            clock64();
    }
}

grid.sync();
```

timing build 中每 timestep增加一次 block atomic，足够轻量且容易保证语义。

---

# 18. P5 完整 timing 结构

```cpp
for (int t = 0;
     t < t_steps;
     ++t) {

    initialize();

    grid.sync();

    inject_events();

    grid.sync();

#ifdef ENABLE_PIPELINE_TIMING
    if (global_tid == 0) {
        timing_stats[
            t * TS_COLUMNS +
            TS_BEGIN] =
            clock64();
    }
#endif

    // --------------------
    // UPDATE producer
    // --------------------

    if (is_update_block) {
        for (...) {
            update();

            if (fired) {
#ifdef ENABLE_PIPELINE_TIMING
                if (first publish) {
                    record_first_publish();
                }
#endif
                publish_tasks();
            }
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            const int old =
                atomicAdd(
                    update_done_blocks,
                    1);

#ifdef ENABLE_PIPELINE_TIMING
            if (old ==
                update_block_count - 1) {

                timing_stats[
                    t * TS_COLUMNS +
                    TS_UPDATE_DONE] =
                    clock64();
            }
#endif
        }

        __syncthreads();
    }

    // --------------------
    // ticket consumer
    // --------------------

    if (active_consumer) {
        while (...) {
            wait_ticket();

#ifdef ENABLE_PIPELINE_TIMING
            record_first_consume_once();
#endif

            process_fragment();
        }
    }

    __syncthreads();

#ifdef ENABLE_PIPELINE_TIMING
    if (threadIdx.x == 0) {
        const int old =
            atomicAdd(
                pipeline_done_blocks,
                1);

        if (old ==
            gridDim.x - 1) {

            timing_stats[
                t * TS_COLUMNS +
                TS_PIPELINE_DONE] =
                clock64();
        }
    }
#endif

    grid.sync();
}
```

---

# 19. 需要计算的指标

拿到 clock 数据后计算：

## UPDATE duration

```text
T_update =
    update_done - begin
```

## propagation 启动延迟

```text
T_startup =
    first_consume - first_publish
```

## pipeline tail

```text
T_tail =
    pipeline_done - update_done
```

若：

```text
T_tail > 0
```

传播比 UPDATE 晚结束。

## overlap window

```text
T_overlap_window =
    update_done - first_consume
```

表示传播在 UPDATE 尚未完成时工作的时间。

---

# 20. 更关键的 overlap efficiency

另做两个 serial reference 实验。

## Reference A：只 UPDATE

暂时禁止 propagation consumer：

```text
T_update_only
```

## Reference B：原始 barrier propagation

保持：

```text
UPDATE
grid.sync
PROPAGATION
```

得到：

```text
T_update_serial
T_prop_serial
```

pipeline 得：

```text
T_pipeline
```

定义粗略 overlap gain：

```text
overlap_gain =
    T_update_serial
    + T_prop_serial
    - T_pipeline
```

定义 overlap efficiency：

```text
overlap_efficiency =
    overlap_gain /
    min(
        T_update_serial,
        T_prop_serial);
```

解释：

```text
≈ 1
    较短阶段几乎完全被隐藏

≈ 0
    基本没有真正重叠收益

< 0
    overlap 带来的资源竞争比隐藏收益更大
```

---

# 21. P5 角色比例诊断

对前三阶段找到的最佳配置附近测试：

```text
7:1
3:1
2:1
```

记录：

```text
T_update
T_tail
T_startup
T_pipeline
```

判断：

## 若 7:1

```text
T_update 小
T_tail 大
```

说明 propagation 资源不足。

## 若 1:1

```text
T_update 大
T_tail 小
```

说明 UPDATE 被过度削弱。

## 最佳点

应接近：

```text
T_update_done
≈
T_pipeline_done
```

即：

> UPDATE 和 propagation 差不多同时收尾。

这比仅凭总时间选角色比例更有解释力。

---

# 22. P4 + P5 推荐施工顺序

## Step 1

先做 host 侧分段计时：

```text
delta init
pipeline kernel
psc fold
full forward
```

不改 kernel 行为。

## Step 2

增加 workspace benchmark path：

```text
预分配 delta
只 zero
```

比较 allocator 与 zero 成本。

## Step 3

增加 pure-kernel benchmark：

```text
不计 delta init
不计 psc fold
```

得到真正：

```text
pipeline kernel vs naive kernel
```

## Step 4

编译独立：

```text
ENABLE_PIPELINE_TIMING
```

加入：

```text
begin
first publish
first consume
update done
pipeline done
```

## Step 5

对最佳 role/chunk 配置及邻近点做 timing。

## Step 6

综合判断下一阶段方向。

---

# 23. 下一阶段决策规则

### 若：

```text
pipeline kernel < naive kernel
full forward > naive
```

下一阶段优先：

```text
delta workspace 持久化
跨 forward 保留 split PSC
删除 psc fold
```

### 若：

```text
pipeline kernel > naive kernel
且 overlap efficiency 较高
```

说明 overlap 确实存在，但协议/双缓冲成本过高。

优先：

```text
减少 delta memory traffic
减少 ready/ticket 元数据
```

### 若：

```text
overlap efficiency 很低
```

优先处理：

```text
task 发布时间分布
consumer 启动延迟
角色比例
producer task 生成顺序
```

### 若：

```text
T_tail 很大
```

增加传播资源或改长 segment 调度。

### 若：

```text
T_update 明显变长
```

说明 propagation 与 UPDATE 存在明显资源竞争，应减少 dedicated blocks 或调整其 warp 数。
