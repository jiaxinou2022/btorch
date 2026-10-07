可以。这个阶段不要急着实现新的重排算法，先把问题变成一个**统计假设验证实验**：

> **H1：fanin similarity 能否预测 neuron 的同时发放（co-spiking）？**
> **H2：如果能，利用这种相关性形成的 neuron group，是否比随机 / fanout-only group 更容易产生可利用的 fanout aggregation？**
> **H3：这种关系是否在不同网络结构、不同发放率下稳定存在，而不是某个数据集的偶然现象？**

你当前 `reorder.py` 已经能够做 physical permutation，并已有 `dominant_post_blocks()` 和 fanout-based similarity ordering，因此这一轮最好只新增**分析脚本**，暂时不改 kernel，也不真正重排运行。

---

# 一、整个实验分成三层

建议严格按下面三层做，因为每层回答一个不同问题。

```text
Layer 1: Structural hypothesis
fanin similarity
      ↓
actual co-spiking ?
```

如果这一层不成立，后面 fanin-aware reorder 基本没必要做。

```text
Layer 2: Execution opportunity
co-spiking + fanout similarity
      ↓
potential block aggregation ?
```

判断即使 fanin 能预测活动，这些共同活动是否真的对你的 hash/block aggregation 有帮助。

```text
Layer 3: Block-level feasibility
fanin-aware grouping
      ↓
32-neuron BlockTask quality
```

这一层才模拟未来真正的 preprocessing algorithm。

先不测 kernel runtime，先看理论机会量。

---

# 二、数据集怎么选

最重要的是覆盖**不同网络结构**，而不仅仅是不同规模。

我建议至少四类。

| 类别                                     | 目的                          |
| -------------------------------------- | --------------------------- |
| Uniform / Erdos-Renyi synthetic        | 对照组，没有强 community structure |
| clustered / community synthetic        | 检验结构相关性强时 fanin 是否有效        |
| spatial / distance-dependent synthetic | 模拟局部神经连接                    |
| real SNN connectivity dataset          | 判断真实网络中是否存在现象               |

如果目前已有 FlyBrain，那么它可以承担 real network。

Synthetic 不需要很多组，一开始：

```text
Uniform
Community
Spatial
FlyBrain
```

四类已经足够。

再在每个 dataset 上扫：

```text
network firing rate:
0.1%
0.5%
1%
2%
5%
10%
```

不一定必须精确这些值，如果模型 firing rate 无法直接控制，可以记录**实际测得 firing rate**作为 x 轴。

---

# 三、一定要区分两个 similarity

不要一开始只用你当前 `dominant_post_block` 那种 coarse metric。

先计算相对“真实”的结构 similarity，判断假设本身。

## Fanin similarity

定义：

[
F^{in}_i = {j \mid j\rightarrow i}
]

最基础用 Jaccard：

[
S_{in}(i,j)
===========

\frac{|F^{in}_i\cap F^{in}_j|}
{|F^{in}_i\cup F^{in}_j|}
]

然后后续可以再测试 weighted version。

## Fanout similarity

[
F^{out}_i = {j\mid i\rightarrow j}
]

[
S_{out}(i,j)
============

\frac{|F^{out}_i\cap F^{out}_j|}
{|F^{out}_i\cup F^{out}_j|}
]

---

# 四、Co-spiking 也不要只用一个指标

如果只是：

```python
num_same_spike / T
```

对于 firing rate 很低的 SNN 会很容易接近 0。

推荐至少记录两个。

## 1. Joint firing probability

[
P_{11}(i,j)
===========

\frac{1}{T}
\sum_t s_i(t)s_j(t)
]

表示实际同时 spike 的频率。

这是与 kernel workload 最直接相关的指标。

---

## 2. Jaccard co-activation

定义 spike timestep set：

[
T_i = {t:s_i(t)=1}
]

则：

[
S_{spike}
=========

\frac{|T_i\cap T_j|}
{|T_i\cup T_j|}
]

这样能减少 firing rate 差异带来的影响。

---

## 3. 一个 normalized correlation

可以再加 cosine：

[
C_{spike}
=========

\frac{\sum_t s_i(t)s_j(t)}
{\sqrt{
\sum_t s_i(t)
\sum_t s_j(t)
}}
]

这个很适合低活动率 binary vectors。

最终建议主要报告：

```text
Joint firing probability       → execution relevance
Spike Jaccard / cosine         → correlation relevance
```

---

# 五、Experiment 1：验证最根本假设

## 问题

> Fanin similarity 越高，neuron 是否越容易同时 spike？

这是整个研究的 gatekeeper。

---

## 实验方法

不要算所有：

[
N^2
]

pair。

对于 140k neuron 这种规模完全不现实。

采取**分层随机采样 pair**。

例如每个 dataset 随机采：

```text
1M neuron pairs
```

如果太重，先：

```text
100k
```

也可以。

对于每一对：

```text
i, j

fanin_similarity
fanout_similarity
joint_spike_probability
spike_jaccard
degree_i
degree_j
```

---

## 然后按 fanin similarity 分 bin

例如：

```text
[0, .01)
[.01, .05)
[.05, .1)
[.1, .2)
[.2, .4)
[.4, .6)
[.6, .8)
[.8, 1]
```

对每个 bin 统计：

```text
mean joint firing probability
median
P25 / P75
sample count
```

最终图：

```text
x = fanin similarity
y = normalized co-spike probability
```

最好再 normalization：

[
Lift(i,j)
=========

\frac{P_{11}(i,j)}
{p_i p_j}
]

其中：

[
p_i=P(s_i=1)
]

这样：

```text
Lift = 1
```

表示独立 firing。

```text
Lift > 1
```

表示正相关。

这个指标尤其重要，因为：

> firing rate 越高，P11 本身自然越高。

---

# 六、Experiment 1 的核心结果应该长这样

最理想：

```text
fanin similarity

0.0   → lift 1.02
0.1   → lift 1.18
0.2   → lift 1.45
0.4   → lift 2.1
0.6   → lift 3.0
```

那么说明：

> topology 本身确实携带 dynamic activity locality。

如果是：

```text
0.0   → 1.01
0.6   → 1.03
```

那基本可以停止这个方向。

---

# 七、同时一定要测：fanin similarity 和 fanout similarity 是否本来就相关

这是第二个很重要的控制实验。

对刚才采样的 pair 直接计算：

[
Corr(S_{in},S_{out})
]

同时做二维 heatmap：

```text
x = fanin similarity
y = fanout similarity
density = pair count
```

因为如果：

[
S_{in}\approx S_{out}
]

那么现有 fanout reorder 已经可能间接完成 activity clustering。

你真正需要的是：

> fanin 有没有提供 **fanout 之外的额外信息**？

因此更关键的统计其实是：

[
Corr(S_{in},S_{spike}\mid S_{out})
]

不用一开始做特别复杂的 partial correlation。

可以简单固定 fanout similarity：

```text
Fanout similarity ∈ [0.2,0.3]
```

然后比较：

```text
low fanin similarity
vs
high fanin similarity
```

的 co-spike rate。

例如：

| Fanout sim | Fanin sim | Co-spike lift |
| ---------- | --------: | ------------: |
| 0.2–0.3    |     <0.05 |           1.1 |
| 0.2–0.3    |   0.1–0.2 |           1.4 |
| 0.2–0.3    |      >0.4 |           2.3 |

如果还能明显增长，就证明 fanin 是独立有价值的信息。

---

# 八、Experiment 2：直接验证“aggregation opportunity”

Experiment 1 只说明：

```text
similar fanin
→ simultaneous spikes
```

但这不一定能加速。

真正加速要求：

```text
simultaneously firing
AND
similar fanout
```

因此定义一个非常重要的 pair metric：

[
A(i,j)
======

P_{11}(i,j)
\times
|F^{out}_i\cap F^{out}_j|
]

或者 normalized：

[
A_{norm}(i,j)
=============

P_{11}(i,j)
\times
S_{out}(i,j)
]

这个可以叫：

**expected aggregation opportunity**

实验：

比较：

```text
random pair
fanout-similar pair
fanin-similar pair
fanin+fanout similar pair
```

的：

```text
E[A(i,j)]
```

如果结果：

```text
random            1.0x
fanin             1.8x
fanout             2.2x
fanin + fanout     4.7x
```

就说明两级方法真的有意义。

---

# 九、Experiment 3：模拟真正的 32-neuron BlockTask

这一层最重要，因为 pair-level correlation 最终不一定转换成 block-level benefit。

模拟四种 grouping。

### Baseline A — Identity

```text
original neuron order
```

### Baseline B — Random

```text
random permutation
```

### Baseline C — Fanout-only

复用你现在的 similarity logic。

### Candidate D — Fanin → Fanout

暂时不用复杂算法。

做一个简单 heuristic：

```text
global sort by fanin signature
local sort by fanout signature
```

只为了评估机会。

---

# 十、每个 BlockTask 要统计什么

假设：

```text
B = 32 neurons
```

对于每个 timestep (t)：

[
A_B(t)
======

{i\in B:s_i(t)=1}
]

---

## Metric 1：Active occupancy

[
Occ(B,t)=|A_B(t)|
]

只统计：

```text
Occ > 0
```

的 block。

然后：

[
E[Occ\mid Occ>0]
]

这直接回答：

> fanin grouping 是否使 firing 从“散落”变成“成组”？

例如：

```text
identity       1.3
fanout-only    1.5
fanin-fanout   3.8
```

就是非常强的信号。

---

# 十一、Metric 2：Active block count

每 timestep：

```text
number of blocks containing ≥1 spike
```

如果同样 spike 数量下：

```text
identity:     1000 active blocks
fanin sort:    600 active blocks
```

说明 spike 更集中。

定义：

[
BlockCompression
================

\frac{N_{spikes}}
{N_{active\ blocks}}
]

本质上就是：

> 一次 block task 平均能够承载多少 active neurons。

这正是你 blockization 的潜在收益来源之一。

---

# 十二、Metric 3：Raw edge contributions

一个 block 内 active neuron：

[
Raw(B,t)
========

\sum_{i\in A_B(t)} degree(i)
]

表示原本需要处理多少 synaptic updates。

---

# 十三、Metric 4：Unique destinations

把所有 active neurons 的 fanout union：

[
Unique(B,t)
===========

\left|
\bigcup_{i\in A_B(t)}
F^{out}_i
\right|
]

理论最理想 global atomic 数量接近这个量。

因此：

[
AggregationRatio
================

1-\frac{Unique}{Raw}
]

例如：

```text
raw updates = 100
unique destination = 60

aggregation ratio = 40%
```

这个指标非常适合直接对应你的 shared hash。

---

# 十四、Metric 5：真实 post-block locality

由于你当前 hash / block 结构可能不是 exact target dedup，最好再算一个硬件相关版本。

比如：

```python
post_block = post_id // 32
```

统计：

```text
unique post blocks touched
```

以及：

[
PostBlockReuse
==============

\frac{RawEdges}
{UniquePostBlocks}
]

因为你的当前 `dominant_post_blocks()` 本身就是基于这种 32-neuron post bucket 构造的。

---

# 十五、最关键的是画一个因果链图

每种 dataset 最终可以得到：

```text
              fanin similarity
                     ↓
              co-spike lift
                     ↓
          active neurons / block
                     ↓
           destination overlap
                     ↓
       potential global atomic saving
```

而不是直接：

```text
new reorder faster
```

这会让后续 Method motivation 非常扎实。

---

# 十六、推荐代码结构

先建：

```text
benchmark/reorder_analysis/
```

例如：

```text
reorder_analysis/
├── extract_spikes.py
├── topology_stats.py
├── pair_analysis.py
├── block_analysis.py
├── grouping.py
└── plot_reorder_analysis.py
```

或者嫌复杂的话，第一版一个：

```text
benchmark/analyze_reorder_similarity.py
```

完全够。

---

# 十七、第一阶段伪代码：准备 graph

当前 CSR 是：

```text
row = pre neuron
indices = post neuron
```

所以 fanout 已经直接存在。

fanin 需要构造 transpose adjacency。

```python
def build_fanin(graph):
    N = graph.shape[0]

    # CSR(pre -> post)
    out_indptr = graph.indptr
    out_indices = graph.indices

    # construct CSC / transpose CSR
    in_lists = [[] for _ in range(N)]

    for pre in range(N):
        for e in range(out_indptr[pre], out_indptr[pre + 1]):
            post = out_indices[e]
            in_lists[post].append(pre)

    return in_lists
```

实际实现不要 Python list，最好：

```text
torch / scipy csr transpose
```

例如 conceptually：

```python
A = scipy.sparse.csr_matrix(...)
A_T = A.transpose().tocsr()
```

---

# 十八、Pair sampling

避免 (O(N^2))。

```python
def sample_pairs(N, num_pairs, seed):
    rng = Random(seed)

    pairs = []

    while len(pairs) < num_pairs:
        i = rng.randrange(N)
        j = rng.randrange(N)

        if i == j:
            continue

        pairs.append((i, j))

    return pairs
```

但纯 random pair 有个问题：

对于稀疏 graph：

```text
绝大多数 Jaccard = 0
```

所以建议一半随机，一半 positive candidates。

```python
pairs = []

# 50% random
pairs += random_pairs(N, M // 2)

# 50% topology-near candidates
for i in sampled_neurons:
    candidates = neurons_sharing_some_fanin(i)
    sample several j from candidates
```

最后统计时进行分 bin，不把 candidate sampling 当成总体概率估计即可。

---

# 十九、计算 exact Jaccard

如果邻接表已经 sorted：

```python
def jaccard_sorted(a, b):
    i = 0
    j = 0
    intersection = 0

    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            intersection += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1

    union = len(a) + len(b) - intersection

    if union == 0:
        return 0.0

    return intersection / union
```

第一版 Python / CPU offline 分析即可。

不要急着 GPU 化。

---

# 二十、Spike trace

假设运行：

```text
T = 1000–10000 timesteps
```

存成：

```python
spikes[T, N]
```

binary。

如果太大，可以用：

```text
uint8
```

甚至 bitset。

例如：

```python
spike_times[i] = sorted timestep list
```

对于低 firing rate 反而更省。

```python
spike_times = [
    np.flatnonzero(spikes[:, i])
    for i in range(N)
]
```

然后：

```python
def spike_jaccard(ti, tj):
    intersection = intersect_count(ti, tj)
    union = len(ti) + len(tj) - intersection

    return intersection / union if union else 0
```

---

# 二十一、Co-spike lift

这是 Experiment 1 最重要的统计。

```python
def co_spike_stats(spikes_i, spikes_j, T):

    count_i = spikes_i.sum()
    count_j = spikes_j.sum()

    joint = (spikes_i & spikes_j).sum()

    p_i = count_i / T
    p_j = count_j / T
    p_11 = joint / T

    expected = p_i * p_j

    lift = (
        p_11 / expected
        if expected > 0
        else np.nan
    )

    return {
        "p_i": p_i,
        "p_j": p_j,
        "p_joint": p_11,
        "lift": lift,
    }
```

---

# 二十二、Experiment 1 主循环

```python
for dataset in datasets:

    graph = load_graph(dataset)

    fanout = build_fanout(graph)
    fanin = build_fanin(graph)

    for condition in firing_conditions:

        spikes = run_snn(
            graph,
            condition=condition,
            timesteps=T,
        )

        pairs = sample_pairs_stratified(
            graph,
            num_pairs=M,
        )

        records = []

        for i, j in pairs:

            sin = jaccard(
                fanin[i],
                fanin[j],
            )

            sout = jaccard(
                fanout[i],
                fanout[j],
            )

            spike_stats = co_spike_stats(
                spikes[:, i],
                spikes[:, j],
                T,
            )

            records.append({
                "dataset": dataset,
                "condition": condition,
                "i": i,
                "j": j,

                "fanin_sim": sin,
                "fanout_sim": sout,

                **spike_stats,
            })

        save_csv(records)
```

---

# 二十三、然后做 binned statistics

```python
fanin_bins = [
    0.0,
    0.01,
    0.05,
    0.1,
    0.2,
    0.4,
    0.6,
    0.8,
    1.01,
]

df["fanin_bin"] = pd.cut(
    df.fanin_sim,
    bins=fanin_bins,
)

summary = (
    df.groupby("fanin_bin")
      .agg(
          mean_lift=("lift", "mean"),
          median_lift=("lift", "median"),
          mean_joint=("p_joint", "mean"),
          mean_fanout=("fanout_sim", "mean"),
          count=("lift", "count"),
      )
)
```

---

# 二十四、Experiment 2：Expected aggregation opportunity

```python
df["aggregation_opportunity"] = (
    df["p_joint"]
    * df["fanout_sim"]
)
```

也可以 exact：

```python
shared_targets = intersection_count(
    fanout[i],
    fanout[j],
)

opportunity = p_joint * shared_targets
```

两个都记录。

然后：

```text
fanin similarity bin
        ↓
mean expected aggregation opportunity
```

如果随着 fanin similarity 继续上升，就是很好的信号。

---

# 二十五、Block grouping 第一版不要复杂

只做一个简单 oracle-like prototype。

## Fanout-only

可以复用当前 permutation。

例如当前已有：

```python
ReorderConfig(
    mode="global_similarity",
)
```

或者：

```python
global_cost_similarity
```

生成 order。

---

## Fanin-only

做一个类似：

```python
fanin_signature[i]
```

第一版甚至可以用：

```text
dominant input block
```

---

## Fanin → Fanout

例如：

```python
def two_level_order(
    fanin_signature,
    fanout_signature,
):

    neurons = list(range(N))

    neurons.sort(
        key=lambda i: (
            fanin_signature[i],
            fanout_signature[i],
            i,
        )
    )

    return neurons
```

注意：

这里暂时不是最终 algorithm。

只是回答：

> 如果把 fanin locality 纳入 physical layout，block metric 有没有改善？

---

# 二十六、fanin signature 怎么做最简单

直接仿照现在的 `dominant_post_blocks()`。

当前代码是找：

```text
一个 CSR row 最多 edge 落在哪个 post block
```

对于 fanin：

```text
转置 graph 后
寻找 neuron 的输入主要来自哪个 pre block
```

伪代码：

```python
def dominant_fanin_block(
    transpose_graph,
    block_size=32,
):

    dominant = [-1] * N

    for neuron in range(N):

        inputs = transpose_graph.row(neuron)

        histogram = Counter()

        for pre in inputs:
            block = pre // block_size
            histogram[block] += 1

        if histogram:
            dominant[neuron] = argmax(histogram)

    return dominant
```

这样可以快速得到：

```text
fanin dominant block
fanout dominant block
```

进行第一轮实验。

---

# 二十七、但推荐同时做 top-2 signature

因为 top-1 太粗。

例如：

```python
fanin_signature[i] = (
    dominant_1,
    dominant_2,
)
```

可以同时测试：

```text
top-1 fanin
top-2 fanin
exact Jaccard oracle
```

这样你能知道：

> 简单 preprocessing approximation 会损失多少 structural information。

---

# 二十八、Block analysis 伪代码

```python
def analyze_order(
    order,
    fanout,
    spikes,
    block_size=32,
):

    blocks = [
        order[i:i + block_size]
        for i in range(0, len(order), block_size)
    ]

    metrics = []

    for t in range(T):

        spike_mask = spikes[t]

        for block_id, block in enumerate(blocks):

            active = [
                n for n in block
                if spike_mask[n]
            ]

            if not active:
                continue

            raw_edges = 0
            destinations = set()
            post_blocks = set()

            for neuron in active:

                targets = fanout[neuron]

                raw_edges += len(targets)
                destinations.update(targets)

                for post in targets:
                    post_blocks.add(
                        post // POST_BLOCK_SIZE
                    )

            unique_dest = len(destinations)

            aggregation_ratio = (
                1.0 - unique_dest / raw_edges
                if raw_edges
                else 0.0
            )

            metrics.append({
                "t": t,
                "block": block_id,

                "active_neurons": len(active),

                "raw_edges": raw_edges,
                "unique_destinations": unique_dest,

                "aggregation_ratio":
                    aggregation_ratio,

                "unique_post_blocks":
                    len(post_blocks),
            })

    return metrics
```

---

# 二十九、注意：AggregationRatio 最好按贡献量加权

不要简单：

```text
所有 active block 的 ratio 求平均
```

因为：

```text
block A: 2 edges
block B: 10000 edges
```

不应该权重一样。

全局版本：

[
R =
1-
\frac{\sum_{B,t}Unique(B,t)}
{\sum_{B,t}Raw(B,t)}
]

代码：

```python
global_ratio = (
    1
    - df.unique_destinations.sum()
      / df.raw_edges.sum()
)
```

这个才接近 theoretical atomic reduction。

---

# 三十、再加一个特别重要的指标：Realized fanout overlap

因为：

```text
静态 fanout overlap 很高
```

不代表它被真正利用。

定义：

[
StaticOverlap(B)
]

与：

[
RealizedOverlap(B,t)
]

前者：

```text
32 neuron 全部参与
```

后者：

```text
只有本 timestep firing neurons
```

比较：

[
Utilization
===========

\frac{RealizedAggregation}
{PotentialAggregation}
]

这能够非常直接说明 fanin-aware reorder 到底是在解决什么。

当前 fanout-only 排序可能：

```text
Potential aggregation: 50%
Realized aggregation:    8%
```

fanin+fanout：

```text
Potential aggregation: 42%
Realized aggregation:   27%
```

即使它静态 fanout similarity 略低，反而更有价值。

这会是一个非常有说服力的结果。

---

# 三十一、我建议最终输出 5 张分析图

初期不需要论文级制图，先 exploratory plot。

### Figure 1

```text
Fanin similarity
vs
Co-spike lift
```

四个 dataset 四条线。

判断 H1。

---

### Figure 2

```text
Fanin similarity
vs
Fanout similarity
```

判断两个 topology signal 是否冗余。

可以用 heatmap。

---

### Figure 3

```text
Fanin similarity
vs
Expected aggregation opportunity
```

即：

[
P_{11}\times FanoutOverlap
]

这是连接 topology 和 kernel 的关键图。

---

### Figure 4

四种 layout：

```text
Identity
Random
Fanout-only
Fanin→Fanout
```

比较：

```text
average active neurons per active BlockTask
```

---

### Figure 5

同样四种 layout：

```text
Potential global atomic reduction
```

也就是 weighted：

[
1-\frac{\sum UniqueDest}
{\sum RawUpdates}
]

这张最接近最终性能。

---

# 三十二、最好再扫 firing rate

因为这可能是一个非常关键的条件。

很可能在：

```text
firing rate = 0.1%
```

时，即使 correlation 很强：

```text
32-neuron block
```

也很少同时出现 >1 spike。

那么无法聚合。

反而：

```text
1%-5%
```

可能开始出现明显收益。

所以画：

```text
x = firing rate

y1 = active neuron / active block
y2 = aggregation ratio
```

分别比较：

```text
fanout-only
fanin→fanout
```

这可能直接告诉你算法的 applicable regime。

---

# 三十三、另外控制 degree

这是必须注意的一个 confounder。

比如高 fanin degree neuron：

* 两两 Jaccard 更容易非零；
* firing rate 可能更高；
* 甚至 fanout degree 也可能更高。

那么你可能误判：

```text
fanin similarity
→ co-spiking
```

实际上是：

```text
high degree
→ both
```

所以 pair record 一定保存：

```python
fanin_degree_i
fanin_degree_j

fanout_degree_i
fanout_degree_j

firing_rate_i
firing_rate_j
```

然后至少做一个简单控制：

在类似 degree bucket 内比较。

例如：

```text
fanin degree = 32–64
```

内部：

```text
low similarity
vs
high similarity
```

如果仍然有 co-spike lift，结论才比较稳。

---

# 三十四、再控制 spatial / community effects

如果数据集本身存在 neuron type / brain region / population labels，最好记录：

```text
same population?
same region?
same neuron type?
```

因为也可能：

```text
same region
   ↓
similar fanin

same region
   ↓
similar activity
```

那么 fanin 只是 proxy。

不过这个不妨碍 optimization 有效，只影响论文解释：

不要说：

> fanin similarity causes co-spiking

而说：

> fanin similarity provides an inexpensive structural proxy for co-activation.

更加严谨。

---

# 三十五、建议实验顺序

不要一开始全数据集全 sweep。

### Phase A：10分钟级 sanity test

选：

```text
small synthetic
```

跑：

```text
T = 1000
pairs = 100k
```

看：

```text
fanin similarity vs spike lift
```

分析代码有没有问题。

---

### Phase B：验证 hypothesis

四类 dataset：

```text
Uniform
Community
Spatial
FlyBrain
```

每个：

```text
100k–1M pairs
T = 5k–10k
```

得到 Figure 1–3。

到这里做一次判断：

> fanin 到底值不值得继续。

---

### Phase C：block-level prototype

只在有明显 H1 信号的数据集做：

```text
Identity
Random
Fanout
Fanin
Fanin→Fanout
```

得到：

```text
active occupancy
active block count
aggregation ratio
```

---

### Phase D：rate sweep

再扫：

```text
activity rate
```

建立适用范围。

---

### Phase E：才决定是否写真正 preprocessing algorithm

如果看到：

```text
fanin similarity ↑
        ↓
co-spike lift ↑
        ↓
active occupancy ↑
        ↓
aggregation ratio ↑
```

才去设计：

```text
hierarchical clustering
joint score
MinHash
top-k signatures
greedy block packing
```

否则及时停止。

---

# 三十六、可以设置几个 go/no-go 标准

避免看到一点点 correlation 就继续深挖。

例如不是硬阈值，但可以作为判断依据：

### H1

高 fanin similarity group 的：

```text
co-spike lift
```

相比 random 至少明显 >1，并且多个 dataset 重复。

比如：

```text
1.5x+
```

才算比较有意思。

---

### H2

fanin-aware layout：

```text
active neurons / active block
```

相比 fanout-only 有：

```text
>10–20%
```

提升。

---

### H3

weighted potential aggregation：

```text
1 - unique / raw
```

相比 fanout-only：

```text
有稳定提升
```

而不只是在一个数据集。

如果 H1 成立、H2 成立但 H3 不成立：

> fanin clustering 有 activity locality，但破坏 fanout locality 太严重。

此时就需要 joint objective，而不是 strict two-level sort。

这个结果本身也非常有指导意义。

---

# 三十七、完整执行流程伪代码

```python
def run_analysis(dataset, firing_condition):

    # ------------------------------------------------
    # 1. Graph
    # ------------------------------------------------

    graph = load_graph(dataset)

    fanout = csr_rows(graph)
    fanin = csr_rows(transpose(graph))

    N = graph.num_neurons


    # ------------------------------------------------
    # 2. Dynamic spike trace
    # ------------------------------------------------

    spikes = simulate(
        graph,
        condition=firing_condition,
        timesteps=T,
    )

    firing_rate = spikes.mean(axis=0)


    # ------------------------------------------------
    # 3. Pair-level statistics
    # ------------------------------------------------

    pairs = stratified_sample_pairs(
        fanin,
        num_pairs=M,
    )

    pair_stats = []

    for i, j in pairs:

        fanin_sim = jaccard(
            fanin[i],
            fanin[j],
        )

        fanout_sim = jaccard(
            fanout[i],
            fanout[j],
        )

        p_joint = mean(
            spikes[:, i]
            & spikes[:, j]
        )

        p_independent = (
            firing_rate[i]
            * firing_rate[j]
        )

        lift = (
            p_joint / p_independent
            if p_independent > 0
            else NaN
        )

        shared_targets = intersection_size(
            fanout[i],
            fanout[j],
        )

        aggregation_opportunity = (
            p_joint
            * shared_targets
        )

        pair_stats.append({
            "fanin_sim": fanin_sim,
            "fanout_sim": fanout_sim,

            "p_joint": p_joint,
            "lift": lift,

            "shared_targets":
                shared_targets,

            "aggregation_opportunity":
                aggregation_opportunity,

            "fanin_degree_i":
                len(fanin[i]),

            "fanin_degree_j":
                len(fanin[j]),

            "fanout_degree_i":
                len(fanout[i]),

            "fanout_degree_j":
                len(fanout[j]),
        })


    # ------------------------------------------------
    # 4. Construct candidate layouts
    # ------------------------------------------------

    orders = {
        "identity":
            identity_order(N),

        "random":
            random_order(N),

        "fanout":
            build_existing_fanout_order(
                graph,
            ),

        "fanin":
            build_fanin_order(
                fanin,
            ),

        "fanin_fanout":
            build_two_level_order(
                fanin,
                fanout,
            ),
    }


    # ------------------------------------------------
    # 5. Block-level simulation
    # ------------------------------------------------

    block_stats = {}

    for name, order in orders.items():

        block_stats[name] = analyze_blocks(
            order=order,
            fanout=fanout,
            spikes=spikes,
            block_size=32,
        )


    # ------------------------------------------------
    # 6. Aggregate
    # ------------------------------------------------

    summary = {}

    for name, stats in block_stats.items():

        summary[name] = {

            "mean_active_occupancy":
                mean_active_neurons(stats),

            "active_blocks_per_step":
                mean_active_blocks(stats),

            "weighted_aggregation_ratio":
                1
                - sum_unique_targets(stats)
                  / sum_raw_edges(stats),

            "post_block_reuse":
                compute_post_block_reuse(
                    stats
                ),
        }


    save(
        pair_stats,
        block_stats,
        summary,
    )
```

---

# 三十八、这一轮最值得优先关注的不是 runtime，而是三个数字

如果想进一步压缩成最小实验，我甚至建议第一轮就只做：

[
\boxed{
Corr(S_{fanin}, CoSpikeLift)
}
]

↓

[
\boxed{
ActiveNeuronsPerActiveBlock
}
]

↓

[
\boxed{
1-\frac{UniqueDestination}
{RawContribution}
}
]

对应：

```text
Does topology predict activity?
             ↓
Can activity be packed?
             ↓
Does packing actually create aggregation?
```

这三步一旦连起来，才说明新的 preprocessing 值得做。

而且后续无论最终得到的是“两级排序”、joint-score greedy，还是 spike-profile-driven reorder，这组实验都不会浪费——它们本身就是后续论文 Motivation / Characterization 部分最需要的证据。
