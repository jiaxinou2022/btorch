**做一个专门的传统 RSNN runtime characterization 脚本，只回答“时间花在哪里，以及随 firing rate 怎么变”**。暂时不做 fanout、逐 timestep 分布、NCU 等扩展。实现上以 **PyTorch neuron update + cuSPARSE sparse propagation** 为核心，不接 CUDA Graph，这样既与最终 `cuSPARSE + CUDA Graph` baseline 保持算法一致，又能展示未经 launch 优化的传统 timestep based execution。

## 一、实验目标

最终希望得到两类数据：

1. **典型 firing rate 下的时间占比**

   ```text
   Total timestep time
   ├── Neuron update
   ├── Synaptic current computation (cuSPARSE)
   └── Execution overhead
   ```

   用于 Introduction 的 normalized stacked bar。

2. **不同 firing rate 下各部分绝对时间**

   ```text
   firing rate: 0.5% → 1% → 2% → 4% → 8% → 16%

   update time       ≈ relatively stable
   current time      ↑ with spike activity
   total time        ↑
   ```

   用于确认“current computation 是主要成本”这个结论不是单点偶然现象。

这里暂时**不单独精确拆 launch time**。CUDA Event 测 GPU phase，wall clock 测整个 recurrent loop，然后：

[
T_{\mathrm{overhead}}
=====================

## T_{\mathrm{wall}}

## T_{\mathrm{update,GPU}}

T_{\mathrm{current,GPU}}
]

论文中先叫 **execution overhead**，而不是 kernel launch overhead。后面如果需要，可以再用 Nsight Systems 证明其中大量来自 repeated launches。

---

# 二、Baseline 定义

保持非常朴素：

```text
for each timestep:
    neuron update
        ↓
    generate spike vector
        ↓
    cuSPARSE sparse current computation
        ↓
    next timestep
```

其中 neuron update 直接 PyTorch CUDA tensor operation，current computation 调你 benchmark 中已经使用的 cuSPARSE 路径。

最好直接复用最终 benchmark 的 sparse matrix representation 和 cuSPARSE adapter，避免：

```text
characterization:
    torch.sparse.mm

正式 baseline:
    cuSPARSE
```

导致论文前后算法不一致。

但**这里不 capture CUDA Graph**：

```text
Characterization baseline
= PyTorch update + cuSPARSE + ordinary launches

Final performance baseline
= PyTorch/cuSPARSE + CUDA Graph
```

Introduction 可以称前者：

> conventional phase separated GPU execution

而不是把它包装成最终 performance baseline。

---

# 三、建议文件结构

不用做复杂框架，一个文件基本够：

```text
characterize_naive_rsnn.py

├── load_network()
├── prepare_state()
├── generate_spike_control()
├── run_one_step()
├── benchmark_activity()
├── run_activity_sweep()
└── save_csv()
```

如果已有 benchmark provider，就直接 import，不要再写一套数据系统。

输出：

```text
results/
    naive_rsnn_breakdown.csv
```

画图可以单独：

```text
plot_rsnn_breakdown.py
```

---

# 四、核心执行路径

建议把 UPDATE 和 current computation 明确封装成两个 phase。

伪代码：

```python
def rsnn_step(state, W, spike_control=None):

    # --------------------------
    # Phase 1: neuron update
    # --------------------------
    state.v = decay * state.v + state.input_current

    spikes = state.v >= threshold

    state.v = torch.where(
        spikes,
        reset_value,
        state.v
    )

    # --------------------------
    # Phase 2: synaptic current
    # --------------------------
    current = cusparse_spmv_or_spmspv(
        W,
        spikes
    )

    state.input_current = current

    return state, spikes
```

这里有一个需要注意的问题：

如果你直接用

```python
spikes = (v >= threshold)
```

实际 firing rate 会由神经动力学决定，很难稳定 sweep。

所以 characterization 最好增加一个**可控 spike activity 模式**。

---

# 五、firing rate sweep 的实现

这一阶段目标是测 propagation workload 对 activity 的响应，所以不要让 neuron dynamics 干扰 firing rate。

可以有两个模式：

```python
mode = "natural"
mode = "controlled"
```

正式 sweep 使用 `controlled`。

## Controlled activity

每个 timestep 直接产生指定数量 active neurons：

```python
num_spikes = int(N * activity)
```

然后随机选：

```python
active_idx = permutation[:num_spikes]
```

构造 spike vector：

```python
spikes.zero_()
spikes[active_idx] = 1.0
```

伪代码：

```python
def generate_controlled_spikes(
    spike_tensor,
    activity,
    generator
):
    N = spike_tensor.numel()
    K = int(N * activity)

    spike_tensor.zero_()

    active_idx = torch.randperm(
        N,
        device="cuda",
        generator=generator
    )[:K]

    spike_tensor[active_idx] = 1.0

    return spike_tensor
```

不过这里有一个工程问题：

**不要把 `torch.randperm()` 的时间算进 runtime。**

所以最简单的方法是提前生成 spike sequence。

---

# 六、提前生成 sweep 的 spike sequence

benchmark 前：

```python
spike_sequences = {}
```

例如：

```python
activities = [
    0.005,
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
]
```

然后：

```python
for activity in activities:
    spike_sequences[activity] = generate_spike_sequence(
        N=N,
        T=T,
        activity=activity,
        seed=seed,
    )
```

shape：

```text
[T, N]
```

如果 N 太大，不想占大量显存，可以存 active index：

```text
[T, K]
```

更合理：

```python
def prepare_spike_indices(N, T, activity, seed):

    K = round(N * activity)

    indices = []

    for t in range(T):
        idx = random_sample_without_replacement(N, K)
        indices.append(idx)

    return indices
```

实际运行时：

```python
spikes.zero_()
spikes[indices[t]] = 1
```

不过这样又多了：

```text
zero_
scatter/index write
```

这个成本。

因此这里有两个选择。

### 推荐方案

**把 spike generation 算进 UPDATE。**

这是最符合传统 RSNN 的，因为真实执行里本来就需要 threshold/spike generation。

即：

```text
UPDATE =
    neuron state update
    +
    spike generation / construction
```

这样不需要过度纠结这一点。

---

# 七、最好使用“固定 spike pattern + 控制 activity”

为了实验可重复：

```python
seed = 42
```

每个 activity：

```text
same N
same W
same T
different number of active neurons
```

每个 repeat 也最好使用相同 spike sequence。

否则：

```text
repeat 1: 碰巧选很多 high fanout neuron
repeat 2: 碰巧选很多 low fanout neuron
```

会给 current timing 增加额外 variance。

所以：

```python
spike_sequence = prepare_once(...)
```

然后所有 repeats 重放。

这并不影响 activity characterization。

---

# 八、计时结构

这里建议同时使用：

* CUDA Events：phase GPU time
* `perf_counter()`：整个 timestep loop end to end time

不要每个 timestep synchronize。

否则会人为制造：

```python
UPDATE
cudaSynchronize
PROP
cudaSynchronize
```

把 baseline 改坏。

正确做法是用事件累计一整段。

## 简化实现

每个 repeat 建：

```python
update_start
update_end
prop_start
prop_end
total_start
total_end
```

但一个 event pair 没办法跨多个交替 phase 分别累计。

所以最简单且实验干净的方法是：

### 方案 A：每 timestep event

```python
for t:
    update_start[t].record()
    update(...)
    update_end[t].record()

    prop_start[t].record()
    propagate(...)
    prop_end[t].record()
```

loop 后只同步一次：

```python
torch.cuda.synchronize()
```

然后：

```python
sum(
    start.elapsed_time(end)
    for ...
)
```

如果 T=1000，event 数量很多但 characterization 可以接受。

---

## 更简单的推荐方式

用一个小的 timing helper：

```python
class PhaseTimer:
    def __init__(self, T):
        self.update_start = [Event() for _ in range(T)]
        self.update_end = [Event() for _ in range(T)]

        self.current_start = [Event() for _ in range(T)]
        self.current_end = [Event() for _ in range(T)]
```

执行：

```python
wall_start = perf_counter()

for t in range(T):

    timer.update_start[t].record()

    spikes = neuron_update(...)

    timer.update_end[t].record()

    timer.current_start[t].record()

    current = cusparse(...)

    timer.current_end[t].record()

torch.cuda.synchronize()

wall_end = perf_counter()
```

计算：

```python
update_ms = sum(
    s.elapsed_time(e)
    for s, e in zip(
        timer.update_start,
        timer.update_end,
    )
)

current_ms = sum(
    s.elapsed_time(e)
    for s, e in zip(
        timer.current_start,
        timer.current_end,
    )
)

wall_ms = (wall_end - wall_start) * 1000

overhead_ms = max(
    0,
    wall_ms - update_ms - current_ms
)
```

然后统一除以 T：

```python
update_us_per_step = update_ms * 1000 / T
current_us_per_step = current_ms * 1000 / T
overhead_us_per_step = overhead_ms * 1000 / T
```

---

# 九、但最好再保留一个纯 total GPU timer

因为：

```text
wall - phase GPU
```

里面会混入 Python。

所以增加：

```python
gpu_total_start.record()

for t:
    ...

gpu_total_end.record()
```

这样你会有三种量：

```text
wall_total
gpu_total
sum_phase_gpu
```

于是：

```text
GPU inter-phase gap
    = gpu_total
    - update_gpu
    - current_gpu

Host / framework overhead
    = wall_total
    - gpu_total
```

虽然 Introduction 最后完全可以把两者合起来：

```text
Execution overhead
```

但数据本身最好保留。

伪代码：

```python
gpu_total_start.record()
wall_start = perf_counter()

for t in range(T):

    update_start[t].record()
    ...
    update_end[t].record()

    current_start[t].record()
    ...
    current_end[t].record()

gpu_total_end.record()

torch.cuda.synchronize()
wall_end = perf_counter()
```

得到：

```python
gpu_gap_ms = (
    gpu_total_ms
    - update_ms
    - current_ms
)

host_overhead_ms = (
    wall_ms
    - gpu_total_ms
)
```

最终：

```python
execution_overhead_ms = (
    gpu_gap_ms
    + host_overhead_ms
)
```

这个划分已经足够用了。

---

# 十、Warmup 和 repeat

建议：

```python
WARMUP = 100
T = 500~1000
REPEAT = 10
```

如果单 step 很短，T 至少 1000。

流程：

```python
for activity in ACTIVITIES:

    prepare_spike_sequence(activity)

    # warmup
    for _ in range(WARMUP):
        run_step(...)

    synchronize()

    results = []

    for repeat in range(REPEAT):
        result = benchmark(...)
        results.append(result)

    aggregate(results)
```

这里不要：

```text
warmup each timestep + synchronize each timestep
```

---

# 十一、总 benchmark 伪代码

完整一些可以写成：

```python
ACTIVITIES = [
    0.005,
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
]

def benchmark_activity(
    network,
    activity,
    spike_sequence,
    T,
    repeats,
):

    results = []

    # --------------------------------
    # Warmup
    # --------------------------------

    state = init_state(network)

    for t in range(WARMUP):
        spikes = spike_sequence[t % len(spike_sequence)]

        neuron_update(
            state,
            spikes,
        )

        state.current = cusparse_propagate(
            network.W,
            spikes,
        )

    torch.cuda.synchronize()

    # --------------------------------
    # Measurement
    # --------------------------------

    for r in range(repeats):

        reset_state(state)

        timer = PhaseTimer(T)

        gpu_total_start = Event()
        gpu_total_end = Event()

        torch.cuda.synchronize()

        wall_start = perf_counter()
        gpu_total_start.record()

        for t in range(T):

            # UPDATE
            timer.update_start[t].record()

            spikes = neuron_update_controlled(
                state,
                spike_sequence[t]
            )

            timer.update_end[t].record()

            # CURRENT COMPUTATION
            timer.current_start[t].record()

            state.current = cusparse_propagate(
                network.W,
                spikes
            )

            timer.current_end[t].record()

        gpu_total_end.record()

        torch.cuda.synchronize()
        wall_end = perf_counter()

        update_ms = timer.sum_update()
        current_ms = timer.sum_current()

        gpu_total_ms = gpu_total_start.elapsed_time(
            gpu_total_end
        )

        wall_ms = (
            wall_end - wall_start
        ) * 1000

        gpu_gap_ms = max(
            0,
            gpu_total_ms
            - update_ms
            - current_ms
        )

        host_overhead_ms = max(
            0,
            wall_ms - gpu_total_ms
        )

        results.append({
            "activity": activity,

            "update_us":
                update_ms * 1000 / T,

            "current_us":
                current_ms * 1000 / T,

            "gpu_gap_us":
                gpu_gap_ms * 1000 / T,

            "host_overhead_us":
                host_overhead_ms * 1000 / T,

            "execution_overhead_us":
                (
                    gpu_gap_ms
                    + host_overhead_ms
                ) * 1000 / T,

            "gpu_total_us":
                gpu_total_ms * 1000 / T,

            "wall_us":
                wall_ms * 1000 / T,
        })

    return results
```

最外层：

```python
network = load_network(...)

all_results = []

for activity in ACTIVITIES:

    spike_sequence = prepare_spike_sequence(
        N=network.num_neurons,
        T=T,
        activity=activity,
        seed=42,
    )

    results = benchmark_activity(
        network,
        activity,
        spike_sequence,
        T,
        repeats,
    )

    all_results.extend(results)

save_csv(all_results)
```

---

# 十二、CSV 最少保留这些列

不用现在加很多 metadata：

```text
dataset
N
nnz
activity
repeat

update_us
current_us

gpu_gap_us
host_overhead_us
execution_overhead_us

gpu_total_us
wall_us
```

另外建议免费附送：

```text
actual_spike_rate
```

检查 sweep 有没有生成错。

就够了。

---

# 十三、第一版图怎么画

### 图 A：Introduction runtime breakdown

先选一个 representative activity，比如之后 benchmark 中使用的 typical activity。

例如：

```text
activity = 1%
```

画：

```text
Neuron update
Synaptic current computation
Execution overhead
```

normalized：

```python
update_pct = update / wall
current_pct = current / wall
overhead_pct = overhead / wall
```

**这一张用于 Introduction。**

如果有多个 dataset，就每个 dataset 一根 stacked bar；如果目前只有 FlyBrain，一根也可以先检查结果，论文最终再补数据集。

---

### 图 B：activity sweep

横轴：

```text
Firing rate (%)
```

纵轴：

```text
Time per timestep (μs)
```

三组值：

```text
Update
Synaptic current computation
Execution overhead
```

这里我更倾向于**stacked bar**而不是 line，因为你同时想表达：

```text
total runtime
+
runtime composition
```

例如：

```text
0.5%  █████████████
1%    ████████████████
2%    ███████████████████
4%    █████████████████████████
8%    ███████████████████████████████
```

每根里面分：

```text
update | current | overhead
```

这样一张图就能同时回答：

1. propagation 是否占大头；
2. propagation 是否随 activity 增长；
3. update 是否相对稳定；
4. low activity 下 overhead 是否更显著。

---

# 十四、实现时最需要防的几个工程错误

这一阶段我会重点检查下面 5 个问题：

1. **cuSPARSE backend 必须与最终 baseline 一致。**
   不要 characterization 用 `torch.sparse.mm`，性能 benchmark 又换另一套 adapter。

2. **不要每 phase synchronize。**
   CUDA Event record 不要求你立即 synchronize，最后统一同步。

3. **activity generation 不要混入 propagation。**
   如果要计入，放进 UPDATE。

4. **不要在 timed loop 中重新申请 sparse matrix / descriptor / workspace。**
   cuSPARSE descriptor、workspace、matrix storage 都提前准备。

5. **所有 activity 共用相同 network。**
   唯一改变的是 spike activity，不要顺手改变 connectivity。

---

## 最终施工顺序

可以直接按下面四步做：

**Step 1：复用现有 cuSPARSE baseline。**
抽出 `neuron_update()` 和 `cusparse_propagate()`，保证 descriptor、workspace、matrix 全部预创建。

**Step 2：加入 controlled firing rate。**
提前生成 0.5%、1%、2%、4%、8%、16% 六组固定 spike sequence。

**Step 3：加入 phase timer。**
同时获取 `update GPU time`、`current GPU time`、`GPU total` 和 `wall total`，输出 CSV。

**Step 4：生成两张图。**
一张 representative firing rate 的 normalized breakdown，另一张 firing rate sweep 的 absolute stacked runtime。

做到这里就可以停。**CUDA Graph 暂时不要接入 characterization 脚本**。后面只需要在论文里说明最终 baseline 使用 CUDA Graph 优化重复 launch，而这一组分析采用未捕获的 conventional execution 来展示原始 RSNN timestep execution structure；如果审稿阶段需要更严谨，再补一根 `cuSPARSE + CUDA Graph` 柱即可。这样不会让这个本来很简单的实验膨胀成新的 benchmark 框架。
