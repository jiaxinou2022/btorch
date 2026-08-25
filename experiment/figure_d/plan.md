测试数据集采用flybrain，在5090上测试，可进入服务器调试，是micromamba的ml-py312环境 zhanghan@162.105.95.95，私钥如果缺失，可到本地win环境中寻找。运行在micromamba的ml-py312，可用GPU

可以把这张图迁移成一张很合适的 **“不同 firing/activity regime 下，各种 SpMSpV / RSNN execution strategy 的性能适用区间”** 图。这里最关键的是：**横轴应该是真实测得的 spike rate，而不是输入强度参数本身**；否则不同 dataset 的 recurrent dynamics 不同，同一个 `weight_scale/input_scale` 实际产生的 activity 可能完全不同。

你现在的 benchmark 本身已经比较适合作为基础：它让各 provider 跑同一套 recurrent LIF + ExponentialPSC dynamics，并且 CSV 中已经保留 raw latency samples 和 `speedup_vs_cusparse_direct_cudagraph`，因此不需要重新设计整个 benchmark harness。  VDHA 本身也是按 input sparsity 1%、5%、10%、20% 做性能 characterization，而且论文明确指出 cuSPARSE 的工作量基本不随 input sparsity 减少，而 column-driven SpMSpV 更直接随 active NNZ 变化，这恰好可以成为这张 RSNN 图的解释基础。

---

# 一、这张图具体回答什么问题

我建议不要把它定义成单纯的“benchmark 大图”，而定义成：

> **How does runtime efficiency change with neuronal activity across different sparse propagation strategies?**

也就是回答两个问题：

1. **低 firing rate 时，event-driven / SpMSpV 方法究竟有多大优势？**
2. **随着 firing rate 增大，各方法的 crossover point 在哪里？**

尤其能体现你自己的 persistent 方法：

* `cuSPARSE + CUDA Graph`

  * row-major / matrix-driven
  * 基本不吃 spike sparsity 的红利
* `TileSpMSpV`

  * sparse-vector driven
  * 利用 active input
  * 但 tile metadata / bitmap / task overhead 存在
* `GlobalAtomic`

  * 极简 vector-driven baseline
  * 可以把它理解成“没有 sophisticated aggregation 的下限”
* `VDHA + CUDA Graph`

  * sophisticated SpMSpV
  * hash aggregation + load balance
* `Persistent + similarity + hash`

  * RSNN-specific
  * persistent execution + block/task organization + similarity ordering + local hash aggregation

因此这张图其实不是“谁最快”那么简单，而是在展示：

[
\text{activity sparsity}
\rightarrow
\text{active work}
\rightarrow
\text{aggregation/locality/load balance}
\rightarrow
\text{runtime}
]

---

# 二、实验矩阵

我建议首先固定 **B=1**。

这是非常重要的，因为你的任务本质上是 SpMSpV，而且 VDHA 当前 benchmark capability 本身也是 batch-size=1。之前 benchmark 也已经把 VDHA 声明成 batch-one provider。

## 2.1 Provider

主图只放这 5 个：

| Figure label | benchmark provider          | 定位                           |
| ------------ | --------------------------- | ---------------------------- |
| cuSPARSE     | `cusparse_direct_cudagraph` | matrix-driven baseline       |
| TileSpMSpV   | `tilespmspv`                | tiled SpMSpV                 |
| GlobalAtomic | `globalatomic`              | basic vector-driven          |
| VDHA         | `vdha_cudagraph`            | SOTA SpMSpV                  |
| Ours         | `persistent_simhash`        | similarity + hash persistent |

其中最后一个就是：

```text
persistent_snn_update_linger_kernel.cu
+ similarity ordering
+ hash table
```

图里不要直接叫：

> persistent_snn_update_linger_simhash

论文 legend 应该最终换成框架名，例如：

> **PaceRSNN**

---

# 三、activity sweep 怎么设计

## 主 sweep

建议：

```text
0.5%, 1%, 2%, 5%, 10%, 20%
```

即：

```python
TARGET_RATES = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
```

如果实验成本不高，可以增加：

```text
40%
```

作为 extreme dense regime：

```python
TARGET_RATES = [
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
    0.40,
]
```

### 为什么不是等间隔

因为 SNN 最有意义的区域恰恰是：

```text
0 ~ 10%
```

如果做：

```text
5, 10, 15, 20, 25, 30%
```

会把最重要的低活动区间压掉。

VDHA 自己是：

```text
1%, 5%, 10%, 20%
```

所以你这套：

```text
0.5%, 1%, 2%, 5%, 10%, 20%
```

相当于进一步强化 RSNN low-activity regime，同时与 VDHA 的实验范围高度重叠。

---

# 四、不要直接把 `input_scale` 当横轴

这里我建议专门修改 benchmark。

比如：

```bash
--activity-target 0.01
```

实际上内部可能通过调整 input strength 达成。

但 CSV 必须同时保存：

```text
target_activity
input_scale
measured_activity
```

主图横轴用：

```text
measured_activity
```

而不是：

```text
target_activity
```

更不能用：

```text
weight_scale
```

---

# 五、如何控制 firing rate

这里有两个方案，我建议采用 **方案 A 做正文**。

## 方案 A：保留真实 recurrent RSNN dynamics

对每个：

```text
dataset × target_rate
```

先运行一个 calibration stage。

例如：

```python
def calibrate_input_scale(
    matrix,
    case,
    target_rate,
):
    lo = INPUT_SCALE_MIN
    hi = INPUT_SCALE_MAX

    for _ in range(12):
        scale = (lo + hi) / 2

        x_seq = make_input_sequence(
            case,
            scale=scale,
            seed=ACTIVITY_SEED,
        )

        result = run_reference_rsnn(
            x_seq,
            matrix,
            case,
        )

        rate = result.spikes.float().mean().item()

        if rate < target_rate:
            lo = scale
        else:
            hi = scale

    return scale, rate
```

得到：

```text
target = 5%
realized = 4.87%
input_scale = 0.XXX
```

然后所有 provider **完全共享这个 x_seq**。

这是最符合你论文 RSNN story 的实验。

---

# 六、这里有一个非常重要的 benchmark 陷阱

你不能：

```text
provider A:
自己跑一次 RSNN → 得到 spike trace A

provider B:
自己跑一次 RSNN → 得到 spike trace B
```

然后发现：

```text
A activity = 4.7%
B activity = 5.3%
```

还拿来直接比较。

尤其浮点误差经过 threshold 后可能影响后续 recurrent activity。

应该是：

### Calibration

```text
reference implementation
        ↓
确定 input sequence
        ↓
得到 reference activity
```

然后：

```text
所有 provider
共享 matrix
共享 x_seq
共享 initial state
共享 T
```

benchmark 当前本来就在强调：

> same recurrent equations, same zero state and fixed input

所以这里只需要把 activity-control 加到输入生成阶段即可。

---

# 七、Dataset 维度怎么安排

你说“不同的图对应不同 dataset”，这个思路是对的。

我建议最终布局类似：

```text
          Dataset A            Dataset B            Dataset C
      FlyBrain/FlyWire        Mice Column          Synthetic
```

如果已经有更多真实 dataset：

```text
(a) FlyBrain
(b) Mouse cortical
(c) Dataset 3
(d) Dataset 4
```

每个 subplot 内横轴统一：

```text
0.5   1   2   5   10   20 (%)
```

统一 y-axis。

### 特别重要

**不要每个 dataset 单独 auto-scale y-axis。**

否则：

```text
Dataset A: 0.8–2×
Dataset B: 0.5–8×
```

视觉上看起来一样激烈。

最好：

```python
sharey=True
```

或者至少 major figure 共用上界。

---

# 八、Synthetic dataset 不要只做一种

如果这一张图允许四个 panel，我会更推荐：

```text
(a) FlyBrain
(b) Mouse
(c) Uniform Synthetic
(d) Heavy-tailed Synthetic
```

因为这恰好区分两个正交维度：

### activity

横轴控制：

[
r = \frac{N_{\text{spike}}}{NT}
]

### connectivity irregularity

dataset 控制：

```text
uniform fanout
vs
heavy-tailed fanout
```

这样你的 similarity/hash/block balancing 是否有效，会变得非常容易解释。

例如：

```text
Uniform:
GlobalAtomic 其实已经不错

Heavy-tail:
GlobalAtomic deteriorates
VDHA / Ours 更明显获益
```

这会比四个随便找来的 biological datasets 更有 mechanism evidence。

---

# 九、运行参数建议

正文版本建议：

```text
B = 1
T = 256
warmup = 10
repeat = 30
```

如果 dataset 很大：

```text
T = 128
warmup = 10
repeat = 20
```

也足够。

当前 benchmark 本身已经保存：

```text
latency_samples_ms
latency_ms
```

以及基于 provider 的 speedup 字段，因此数据结构可以直接扩展。

---

# 十、每一个实验点实际执行流程

比如：

```text
Dataset = FlyBrain
Target activity = 2%
```

完整流程：

```text
1. load FlyBrain
2. construct recurrent CSR
3. calibrate input_scale
4. generate fixed x_seq
5. run reference
6. measure actual firing rate

7. prepare cuSPARSE
8. prepare TileSpMSpV
9. prepare GlobalAtomic
10. prepare VDHA
11. prepare Persistent-SimHash

12. correctness check

13. warmup
14. repeat timing

15. save all raw samples
```

特别是：

> preparation / preprocessing 必须全部在 timing scope 外。

例如：

* CSR → CSC
* TileSpMSpV preprocessing
* similarity reorder
* block generation
* hash metadata
* VDHA handle construction
* CUDA Graph capture

都不能计入 steady-state latency。

你的 VDHA wrapper 本来就是 prepare handle 后，再在 graph 内调用 device compute，这个方向是正确的。

---

# 十一、persistent similarity reorder 的公平性

这里最好不要：

```text
为了 1% firing rate 排一次
为了 2% 再排一次
为了 5% 再排一次
```

如果 similarity reorder 是基于**静态 connectivity**：

很好。

```text
preprocess once per dataset
reuse across every activity
```

这样最强。

如果 similarity reorder 是基于 spike trace：

那就需要特别小心，因为容易形成 oracle。

最好定义为：

```text
connectivity-only preprocessing
```

例如：

```text
postsynaptic-set similarity
```

随后：

```text
same ordering for all activity levels
```

这样论文 claim 会干净很多。

---

# 十二、CSV 建议新增字段

现有数据不要覆盖。

建议新实验单独：

```text
benchmark/results/activity_sweep/
```

每行至少：

```text
gpu
dataset

provider

n_neuron
graph_synapses
average_fanout

target_activity
measured_activity
input_scale

t_steps
batch_size

warmup
repeat
seed

similarity_enabled
hash_enabled

preprocess_ms
latency_median_ms
latency_mean_ms
latency_std_ms
latency_p25_ms
latency_p75_ms

speedup_vs_cusparse

spike_mismatch_rate
status
```

### 特别值得存

再存：

```text
active_neurons_per_step_mean
active_neurons_per_step_std
active_edges_per_step_mean
active_edges_per_step_std
```

因为 firing rate 本身并不能完全代表工作量。

真正 PROP 工作量更接近：

[
W_t=
\sum_{i\in S_t}
\mathrm{fanout}(i)
]

所以可以记录：

[
\overline W
===========

\frac1T
\sum_t
\sum_{i\in S_t}
d_i
]

这之后会非常有用。

---

# 十三、为什么一定要记录 active edges

比如两个 workload：

```text
A:
5% neurons spike
但是 spike 的主要是 low-fanout neurons

B:
5% neurons spike
但是 spike 的主要是 hub neurons
```

二者虽然：

[
r=5%
]

但 PROP workload 完全不同。

所以正文 x-axis 依然可以是：

> Firing Rate (%)

但在 characterization 或 supplement 里最好再给：

> Active Synapses / timestep

这可以防止 reviewer 问：

> Does firing rate accurately represent sparse propagation workload?

---

# 十四、制图：我反而不建议完全照搬原图的 boxplot

原图 boxplot 很适合：

```text
一个 x position
= 大量 benchmark matrices
```

箱线代表 dataset distribution。

而你的：

```text
一个 dataset
一个 activity
一个 provider
```

本身只有一个 workload。

所以如果把 30 次 timing samples 直接画 boxplot，信息价值不大——GPU timing variance 通常远小于算法之间的差距。

## 更推荐这样

```text
               Firing Rate (%)

Speedup
  ^
4 |                         Ours ─●
  |                    ●────
3 |               ●────       VDHA
  |          ●────
2 |     ●────
  |             TileSpMSpV
1 |---------- cuSPARSE ----------------
  |       GlobalAtomic
0 +------------------------------------->
      0.5   1   2   5   10   20
```

每种算法：

* 一个固定颜色
* 一个 marker
* 连线
* error bar = timing IQR / std

同时：

```text
y = 1
```

画一条红色虚线。

这样最能看到 **crossover**。

---

# 十五、主图建议

最终可以做成：

```text
GPU: RTX 5090

  FlyBrain             Mouse              Uniform             Heavy-tail

4 ─                    ─                  ─                   ─
3 ─  ●                 ─                  ─                   ─
2 ─     ●              ─                  ─                   ─
1 ───────────────      ─────────────      ─────────────       ─────────────
0 ─                    ─                  ─                   ─
   .5 1 2 5 10 20       .5 1 2 5...       .5 1 2 5...        .5 1 2 5...

                  Firing Rate (%)
           Speedup over cuSPARSE + CUDA Graph
```

Legend 放最顶部：

```text
cuSPARSE | GlobalAtomic | TileSpMSpV | VDHA | PaceRSNN
```

---

# 十六、cuSPARSE 是否还需要画一条 curve？

由于：

[
Speedup_{\text{cuSPARSE}}
=========================

\frac{T_\text{cuSPARSE}}{T_\text{cuSPARSE}}
=1
]

所以实际上无需再画 cuSPARSE marker。

直接：

```text
y = 1 dashed horizontal baseline
```

label：

```text
cuSPARSE + CUDA Graph
```

这样少一条曲线，画面更清楚。

于是只有四条真正的 method curve：

```text
GlobalAtomic
TileSpMSpV
VDHA
PaceRSNN
```

这个比原图更适合你的数据。

---

# 十七、speedup 的计算

严格按照相同：

```text
dataset
activity
seed
```

下匹配。

即：

[
S_{m,d,r}
=========

\frac{
T_{\mathrm{cuSPARSE},d,r}
}{
T_{m,d,r}
}
]

不要：

```python
global_cusparse_average / method_latency
```

必须：

```python
baseline = df[
    (df.dataset == dataset) &
    (df.activity_id == activity_id) &
    (df.provider == "cusparse_direct_cudagraph")
]
```

---

# 十八、这里还有一个很值得额外计算的量：effective speedup

你目前这张图展示 full-RSNN latency，而 provider 的优化对象主要是 propagation。

所以最好 CSV 同时保存：

```text
full_rsnn_latency
propagation_latency   # 如果可 instrument
```

最终正文只画：

> end-to-end RSNN speedup

但 analysis 时可以知道：

```text
SpMSpV kernel:
2.8× faster

End-to-end:
1.6× faster
```

这样才能做 Amdahl 分析。

否则如果最后 Ours 只有：

```text
1.2×
```

你不知道究竟是：

```text
PROP 优化没效果
```

还是：

```text
PROP 已经快很多，但 UPDATE 占比上升
```

---

# 十九、推荐增加一个独立 sweep driver

我不建议把几十个新参数全塞进：

```text
benchmark_rsnn_cudagraph_compare.py
```

最好：

```text
benchmark/
├── benchmark_rsnn_cudagraph_compare.py
├── benchmark_activity_sweep.py      # new
└── plot_activity_sweep.py           # new
```

其中 sweep driver 调现有 benchmark API。

伪代码：

```python
DATASETS = [
    "flybrain",
    "mice_column_v1",
    "uniform",
    "heavy_tail",
]

ACTIVITIES = [
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
]

PROVIDERS = [
    "cusparse_direct_cudagraph",
    "tilespmspv",
    "globalatomic",
    "vdha_cudagraph",
    "persistent_simhash",
]

for dataset in DATASETS:
    matrix = load_dataset(dataset)

    prepare_dataset_specific_metadata(matrix)

    for target_rate in ACTIVITIES:

        input_scale = calibrate_activity(
            matrix,
            target_rate,
        )

        x_seq = make_fixed_input(
            scale=input_scale,
            seed=0,
        )

        measured_rate = measure_reference_activity(
            matrix,
            x_seq,
        )

        for provider in PROVIDERS:

            runner = prepare_provider(
                provider,
                matrix,
                x_seq,
            )

            correctness = verify(runner)

            samples = benchmark(
                runner,
                warmup=10,
                repeat=30,
            )

            save_row(
                dataset=dataset,
                provider=provider,
                target_activity=target_rate,
                measured_activity=measured_rate,
                input_scale=input_scale,
                samples=samples,
                correctness=correctness,
            )
```

---

# 二十、最好做 3 个 seed，而不是靠 30 个 timing repeat

这里要区分两种 variance。

### Timing variance

```text
同一个 workload 跑 30 次
```

测的是 GPU runtime noise。

### Workload variance

```text
seed 0
seed 1
seed 2
```

测的是不同 spike population 的差异。

后者其实更重要。

所以正式版：

```text
seed = 0, 1, 2
repeat = 20
```

最终每个点：

```text
median across three workload seeds
```

error bar：

```text
min/max
```

或者：

```text
standard deviation across seeds
```

而不是把 60 次 GPU timing 当成 60 个 independent samples。

---

# 二十一、因此最终完整实验量

如果：

```text
4 datasets
× 6 activity levels
× 5 providers
× 3 seeds
```

就是：

[
4\times6\times5\times3=360
]

个 benchmark cases。

每 case：

```text
10 warmup
20 timing samples
```

完全可控。

---

# 二十二、第一轮不要直接跑全部

建议按三个阶段。

## Stage 1 — plumbing validation

只跑：

```text
FlyBrain

activity:
1%, 5%, 20%

provider:
cuSPARSE
GlobalAtomic
VDHA
Ours
```

共：

```text
12 cases
```

检查：

```text
target activity 是否准
correctness 是否通过
speedup calculation 是否正确
persistent 是否真正启用了 similarity/hash
```

---

## Stage 2 — mechanism sanity check

只跑 FlyBrain 全 sweep：

```text
0.5%
1%
2%
5%
10%
20%
```

期待的趋势至少应该合理：

### cuSPARSE

normalized：

```text
1 → 1 → 1 → 1
```

### GlobalAtomic / Tile / VDHA / Ours

随着 firing rate 增大：

```text
speedup 大体下降
```

但不一定严格单调。

尤其 VDHA 论文自己就显示 vector-driven 方法对 sparsity 的 scaling 和 cuSPARSE 很不同。

如果你得到：

```text
1%: 0.8×
5%: 2.0×
20%: 4.0×
```

就先别急着跑全 dataset，优先检查 benchmark。

---

## Stage 3 — full paper figure

再跑：

```text
4 dataset
×
6 activity
×
5 provider
×
3 seed
```

---

# 二十三、我预期最有价值的结果不一定是“Ours 全程最快”

实际上比较漂亮、也更 believable 的图可能是：

```text
                   activity

GlobalAtomic       ───────\
TileSpMSpV        ─────────\
VDHA              ───────────\
Ours             ─────────────\
cuSPARSE         ----------------
                1×

         low                   high
```

例如：

```text
0.5%       Ours 3.2×
1%         Ours 2.9×
2%         Ours 2.5×
5%         Ours 2.0×
10%        Ours 1.6×
20%        Ours 1.2×
```

而 high activity 最后接近甚至落后 cuSPARSE。

这种结果其实非常适合 RSNN：

> PaceRSNN specifically targets the sparse-activity regime characteristic of event-driven neural simulation.

比“所有 sparsity 下一律 5× faster”可信得多。

---

# 二十四、这张图之后可以直接支持一个很强的分析

如果出现：

```text
low firing rate:
Ours > VDHA > Tile > GlobalAtomic >> cuSPARSE

higher firing rate:
Ours ≈ VDHA ≈ cuSPARSE
```

可以拆成：

### `cuSPARSE → GlobalAtomic`

证明：

> skipping inactive presynaptic neurons matters.

### `GlobalAtomic → TileSpMSpV / VDHA`

证明：

> naive vector-driven execution is insufficient because of irregularity and accumulation overhead.

### `VDHA → Ours`

证明：

> generic SpMSpV optimizations still leave RSNN-specific temporal execution and locality opportunities.

这正好把这张 benchmark 图和你后面的 **block / similarity / hash ablation** 接起来。

---

# 二十五、我会采用的最终 figure specification

**Figure X: Performance across neuronal activity levels.**

横轴：

```text
Average Firing Rate (%)
0.5   1   2   5   10   20
```

纵轴：

```text
Speedup over cuSPARSE + CUDA Graph
```

subplot：

```text
(a) FlyBrain
(b) microns_mm3
(c) Schmidt et al. macaque multi-area model
(d) Uniform(N = 131072 neurons
mean fanout = 128
nnz ≈ 16.8 M
B = 1
T = 256
dtype = FP32
self-loop = disabled
duplicate edges = disabled)
```

curve：

```text
GlobalAtomic
TileSpMSpV
VDHA
PaceRSNN
```

baseline：

```text
red dashed y=1
cuSPARSE + CUDA Graph
```

每个 point：

```text
median of 3 workload seeds
```

error bar：

```text
seed-level min/max
```

底层每个 workload：

```text
20 timing repetitions
```

并且正文 caption 或 setup 中明确：

```text
B = 1
T = 128/256
FP32
preprocessing excluded
CUDA Graph capture excluded
same connectivity
same input trace
same initial state
measured rather than requested firing rate
```

这会比直接一比一照抄你给的 Sputnik/SparTA 那种 grouped boxplot 更适合 RSNN：**原图在展示一大组 matrix instances 的分布，你这里真正值得展示的是随 activity 连续变化的 crossover curve。**
