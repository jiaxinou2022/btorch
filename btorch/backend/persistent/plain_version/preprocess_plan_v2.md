可以。基于上一轮结果，这一版测试的核心不再是“证明 fanin 能不能预测 spike”，而是更直接地问：

> **在保持现有 fanout 聚合优势的前提下，能否利用真实类脑网络中那一小部分额外 activity locality，进一步提高 BlockTask 内的有效共发放和写回聚合？**

上一轮 FlyBrain 上已经看到：fanin 与 activity 有弱但稳定的关系，但 fanin/fanout 高度相关（Spearman ≈ 0.87），严格的 `global fanin → local fanout` 只能带来约 0.5–5.3% occupancy gain，而且 aggregation gain 不稳定。 所以下一轮应把 **fanout 保留为主目标，fanin/activity 作为 secondary refinement**。

---

# 1. 这一版整体测试框架

测试两条路线：

```text
A. Hierarchical heuristic
   global fanout
        ↓
   local fanin refinement
```

以及：

```text
B. Joint objective
   Score =
   FanoutOverlap × ActivityAffinity
```

其中 B 再分成：

```text
Structural:
ActivityAffinity ← fanin similarity

Oracle:
ActivityAffinity ← measured co-spiking
```

因此最终建议至少比较：

```text
Identity
Fanout-only                    baseline
Fanout → local Fanin           cheap heuristic
Fanout × FaninAffinity         joint structural
Fanout × SpikeAffinity         oracle / upper bound
```

最后一个不是一定要作为正式算法，而是回答：

> **如果 activity information 本身足够准确，这个优化方向理论上还有多大空间？**

---

# 2. firing rate 改成 Hz，并采用 population posterior statistic

这一点建议直接统一掉之前的 `input_rate` 表述。

假设模拟：

* neuron 数：(N)
* timestep 数：(T)
* timestep 大小：(\Delta t) 秒
* 总 spike 数：

[
S=\sum_{t=1}^{T}\sum_{i=1}^{N}s_i(t)
]

总模拟时间：

[
D=T\Delta t
]

则整个网络的 **population mean firing rate**：

[
\boxed{
FR_{\mathrm{Hz}}
================

\frac{S}{N D}
}
]

单位：

[
spikes/neuron/second = Hz
]

例如：

```python
total_spikes = spikes.sum()

duration_s = num_steps * dt_seconds

firing_rate_hz = (
    total_spikes
    / num_neurons
    / duration_s
)
```

这就是后验统计：

> 先实际运行网络，再根据整个 simulation window 内产生的总 spike 数量算平均 firing rate。

以后图和报告都不要再用：

```text
input_rate = 0.001
```

作为主要横轴。

而是：

```text
Measured mean firing rate (Hz)
```

input strength / input rate 只作为控制参数记录在 metadata 中。

---

# 3. 最好同时保存三个 activity statistic

主结果用：

### Population mean firing rate

[
FR=
\frac{\sum_i S_i}{N D}
]

单位 Hz。

同时保存：

### Active-neuron mean firing rate

只统计至少 spike 一次的 neuron：

[
FR_{active}
===========

\frac{\sum_i S_i}
{N_{active}D}
]

以及：

### Silent fraction

[
P_{silent}
==========

\frac{N-N_{active}}{N}
]

因为 FlyBrain 上上一轮 silent fraction 非常高，最低 condition 甚至约 58%。

两个网络都可能：

```text
population FR = 5 Hz
```

但：

```text
Network A:
100% neuron ≈ 5 Hz

Network B:
10% neuron ≈ 50 Hz
90% silent
```

它们对 BlockTask 的行为完全不同。

因此主表至少应该有：

```text
mean_FR_Hz
active_FR_Hz
silent_fraction
```

---

# 4. firing-rate sweep 怎么设计

不要直接规定“input rate”。

应该寻找一组输入参数，使最终 posterior firing rate 覆盖你关心的区间。

例如目标：

```text
~0.5 Hz
~1 Hz
~2 Hz
~5 Hz
~10 Hz
~20 Hz
~50 Hz
```

不要求精确命中，只要最终画图按照实际测出的 Hz 排序。

例如：

```text
target        measured

0.5 Hz   →    0.63 Hz
1 Hz     →    1.14 Hz
2 Hz     →    1.86 Hz
5 Hz     →    5.37 Hz
10 Hz    →   11.1 Hz
```

最终横轴：

```text
0.63  1.14  1.86  5.37  11.1 Hz
```

而不是 target rate。

---

# 5. 如果输入参数和 firing rate 映射未知，可以先 calibration

伪代码：

```python
TARGET_RATES_HZ = [
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
]

def calibrate_conditions(graph):

    candidate_inputs = generate_input_conditions()

    results = []

    for cond in candidate_inputs:

        spikes = simulate(
            graph,
            input_condition=cond,
            timesteps=CALIBRATION_STEPS,
        )

        fr_hz = compute_population_fr_hz(
            spikes,
            dt_seconds,
        )

        results.append(
            (cond, fr_hz)
        )

    selected = []

    for target in TARGET_RATES_HZ:

        cond, measured = min(
            results,
            key=lambda x:
                abs(x[1] - target)
        )

        selected.append(cond)

    return unique(selected)
```

真正实验再用较长 trace 重跑。

---

# 6. Algorithm A：Global Fanout → Local Fanin

这是下一轮最应该先做的版本。

你现在已有：

```text
global_similarity
global_cost_similarity
global_similarity_cost
```

以及 `dominant_post_blocks()`。

所以第一阶段完全沿用当前最好的 fanout ordering：

```python
base_order = build_fanout_order(graph)
```

然后只允许在一个小窗口内重新排序。

---

# 7. 为什么用 local refinement

目标是保持：

```text
global fanout locality
```

只在它已经认为“差不多可以交换”的 neuron 之间，用 fanin 判断：

> 哪些 neuron 更值得进入同一个 32-neuron BlockTask？

因此：

```text
Global fanout structure
────────────────────────────────────>

|       fanout cluster A       |
| N7 N3 N9 N1 N6 N8 N2 N5 ... |
              ↓
       local fanin reorder

| N7 N9 N3 N8 | N1 N6 N5 N2 |
    block 0          block 1
```

而不是跨很远的位置把一个 fanin-similar neuron 拉回来。

---

# 8. 第一版最简单的 local refinement

扫：

```text
local_window =
32
64
128
256
```

其中：

* 32：基本只是 block 内重排，不会改变成员，理论上 aggregation ratio 基本不变；
* 64：允许相邻两个 BlockTask 交换；
* 128：四个 BlockTask；
* 256：八个 BlockTask。

真正有意义的应该从 64 开始。

---

# 9. Fanin signature

第一版不要上 exact all-pair clustering。

可以复用和当前 fanout 相同思想：

```text
dominant fanin block
```

更好一点：

```text
top-2 fanin blocks
```

例如：

```python
fanin_signature[i] = (
    dominant_pre_block_0,
    dominant_pre_block_1,
)
```

然后 local stable sort：

```python
def refine_by_fanin(
    fanout_order,
    fanin_signature,
    window_size,
):

    refined = []

    for start in range(
        0,
        len(fanout_order),
        window_size,
    ):

        window = fanout_order[
            start:start + window_size
        ]

        window = sorted(
            window,
            key=lambda neuron: (
                fanin_signature[neuron],
            ),
        )

        refined.extend(window)

    return refined
```

这是最便宜的 sanity version。

---

# 10. 更合理的 local refinement：直接面向 32-neuron BlockTask

单纯 sort signature 仍然不一定得到最好的 32-neuron group。

推荐再做一个 greedy 版本。

对于 fanout window：

```text
64 / 128 neurons
```

每次构造 32-neuron block。

```python
def pack_by_fanin(
    window,
    fanin,
    block_size=32,
):

    remaining = set(window)
    result = []

    while remaining:

        seed = choose_seed(remaining)

        block = [seed]
        remaining.remove(seed)

        while (
            len(block) < block_size
            and remaining
        ):

            best = max(
                remaining,
                key=lambda candidate:
                    fanin_affinity_to_block(
                        candidate,
                        block,
                        fanin,
                    )
            )

            block.append(best)
            remaining.remove(best)

        result.extend(block)

    return result
```

---

# 11. block affinity 不需要 pairwise mean

更适合用“与整个 block 的共有输入”。

例如：

[
Affinity_{in}(j,B)
==================

\frac{1}{|B|}
\sum_{i\in B}
Jaccard(In_j,In_i)
]

第一版：

```python
def fanin_affinity_to_block(
    candidate,
    block,
    fanin,
):
    score = 0.0

    for neuron in block:
        score += jaccard(
            fanin[candidate],
            fanin[neuron],
        )

    return score / len(block)
```

窗口只有 64/128，所以 offline preprocessing 成本可以接受。

---

# 12. Algorithm A 完整伪代码

```python
def build_fanout_then_fanin_order(
    graph,
    *,
    local_window_size,
    neuron_block_size=32,
):

    # ----------------------------------
    # Stage 1: existing fanout ordering
    # ----------------------------------

    base_perm = build_neuron_permutation(
        graph,
        ReorderConfig(
            mode="global_cost_similarity",
            neuron_block_size=32,
        ),
    )

    base_order = (
        base_perm.new_to_old
        .cpu()
        .tolist()
    )


    # ----------------------------------
    # Stage 2: construct fanin graph
    # ----------------------------------

    fanin = transpose_to_csr(graph)


    # ----------------------------------
    # Stage 3: local refinement
    # ----------------------------------

    final_order = []

    for start in range(
        0,
        len(base_order),
        local_window_size,
    ):

        window = base_order[
            start:start + local_window_size
        ]

        refined_window = pack_by_fanin(
            window,
            fanin,
            block_size=neuron_block_size,
        )

        final_order.extend(
            refined_window
        )

    return final_order
```

这里最重要的约束就是：

[
\boxed{
\text{任何 neuron 都不能跨 fanout local window}
}
]

从而限制 fanout locality 的损失。

---

# 13. Algorithm B：联合 score

第二条路线直接回答：

> 不做人工 primary/secondary hierarchy，可不可以直接最大化 expected useful aggregation？

理论上真正的 pair-level usefulness 是：

[
P(s_i,s_j)
\times
Overlap(Out_i,Out_j)
]

于是 structural approximation：

[
\boxed{
Score(i,j)
==========

FanoutOverlap(i,j)
\times
ActivityAffinity(i,j)
}
]

---

# 14. 但不要直接用 `fanout_jaccard × fanin_jaccard`

这里有个实际问题。

大量 neuron：

```text
fanin_jaccard = 0
```

那么整个 score：

[
0
]

fanout 信息会完全丢失。

因此我更推荐：

[
\boxed{
Score(i,j)
==========

OutOverlap(i,j)
\times
(1+\lambda A(i,j))
}
]

其中：

[
A(i,j)\in[0,1]
]

这样：

```text
λ = 0
```

严格退化成：

```text
fanout-only
```

很好做 ablation。

---

# 15. structural ActivityAffinity

第一版：

[
A_{struct}(i,j)
===============

Jaccard(Fanin_i,Fanin_j)
]

所以：

[
Score_{struct}
==============

OutOverlap
\times
(1+\lambda Jaccard_{in})
]

λ 扫：

```text
0
0.1
0.25
0.5
1
2
```

我更关注：

```text
0.1–0.5
```

因为上一轮已经证明 fanin 是弱 secondary signal，而不是主信号。

---

# 16. FanoutOverlap 最好不要直接用 Jaccard

你的实际 kernel 关注的是：

> 能减少多少 global atomic。

因此：

[
|Out_i\cap Out_j|
]

比：

[
Jaccard(Out_i,Out_j)
]

更直接。

例如：

```text
A:
degree 1000
B:
degree 1000
shared = 100

C:
degree 10
D:
degree 10
shared = 5
```

Jaccard 可能认为 C/D 更相似。

但实际可减少的 global atomic：

```text
A/B → up to 100
C/D → up to 5
```

因此推荐：

[
O(i,j)=|Out_i\cap Out_j|
]

或者 normalized execution score：

[
O(i,j)
======

\frac{|Out_i\cap Out_j|}
{\min(deg_i,deg_j)}
]

两个都保存，但 packing 先试 raw overlap。

---

# 17. Joint greedy block packing

对于每个 fanout coarse window：

```python
def joint_pack(
    window,
    fanin,
    fanout,
    lambda_activity,
    block_size=32,
):

    remaining = set(window)
    result = []

    while remaining:

        seed = choose_seed_by_degree(
            remaining,
            fanout,
        )

        block = [seed]
        remaining.remove(seed)

        while (
            len(block) < block_size
            and remaining
        ):

            best = None
            best_score = -inf

            for candidate in remaining:

                score = joint_block_score(
                    candidate,
                    block,
                    fanin,
                    fanout,
                    lambda_activity,
                )

                if score > best_score:
                    best = candidate
                    best_score = score

            block.append(best)
            remaining.remove(best)

        result.extend(block)

    return result
```

---

# 18. Candidate-to-block joint score

推荐直接看它加入这个 block 后能和已有成员共享多少 destination：

[
OutOverlap(j,B)
===============

\sum_{i\in B}
|Out_j\cap Out_i|
]

activity affinity：

[
A(j,B)
======

\frac1{|B|}
\sum_{i\in B}
A(j,i)
]

则：

[
\boxed{
Score(j,B)
==========

OutOverlap(j,B)
\left[
1+\lambda A(j,B)
\right]
}
]

伪代码：

```python
def joint_block_score(
    candidate,
    block,
    fanin,
    fanout,
    lam,
):

    out_score = 0
    activity_score = 0.0

    for neuron in block:

        out_score += intersection_size(
            fanout[candidate],
            fanout[neuron],
        )

        activity_score += jaccard(
            fanin[candidate],
            fanin[neuron],
        )

    activity_score /= len(block)

    return (
        out_score
        * (1.0 + lam * activity_score)
    )
```

---

# 19. 更进一步：直接算新增 global destination，更贴近 hash

pairwise shared target 还是近似。

你真正想最小化：

[
\left|
Out_j
\setminus
\bigcup_{i\in B}Out_i
\right|
]

即：

> candidate 加入 block 后会新增多少 unique destination？

定义：

[
Reuse(j,B)
==========

## |Out_j|

|Out_j\setminus U_B|
]

其中：

[
U_B=\bigcup_{i\in B}Out_i
]

其实：

[
Reuse(j,B)
==========

|Out_j\cap U_B|
]

那么 score：

[
\boxed{
Score(j,B)
==========

|Out_j\cap U_B|
(1+\lambda A(j,B))
}
]

这个比 pairwise overlap 更接近你 shared hash 的真实执行。

伪代码：

```python
def joint_block_score(
    candidate,
    block,
    block_destinations,
    fanin,
    fanout,
    lam,
):

    reuse = intersection_size(
        fanout[candidate],
        block_destinations,
    )

    affinity = mean(
        jaccard(
            fanin[candidate],
            fanin[n],
        )
        for n in block
    )

    return (
        reuse
        * (1.0 + lam * affinity)
    )
```

我建议最终用这个。

---

# 20. Oracle：用实际 co-spiking 替代 fanin

如果已经有 calibration spike trace：

对 neuron (i,j)：

[
P_{ij}
======

\frac{
\sum_t s_i(t)s_j(t)
}{
T
}
]

但直接 joint probability 会被 firing rate 支配。

可以使用 spike Jaccard：

[
A_{spike}(i,j)
==============

\frac{
|\mathcal{T}_i\cap\mathcal{T}_j|
}{
|\mathcal{T}_i\cup\mathcal{T}_j|
}
]

或者 cosine：

[
A_{spike}
=========

\frac{S_i\cdot S_j}
{\sqrt{n_i n_j}}
]

然后：

```python
Score =
    fanout_reuse
    * (
        1
        + lam * spike_affinity
      )
```

---

# 21. 为什么 Oracle 很重要

你可能得到三种结果。

### Case 1

```text
fanout-only
    <
fanin affinity
    ≈
spike oracle
```

说明：

> fanin 已经是足够好的 cheap structural proxy。

非常理想。

---

### Case 2

```text
fanout-only
    ≈
fanin
    <<
spike oracle
```

说明：

> activity locality 很有价值，但 fanin 太弱。

那后续可以考虑：

```text
profiling-guided reorder
```

而不是 topology-only reorder。

---

### Case 3

```text
fanout-only
≈ fanin
≈ oracle
```

说明：

> 现有 fanout ordering 已经几乎吃满这个优化空间。

那就及时停止 preprocessing 方向。

这也是非常有价值的结果。

---

# 22. 实验的核心指标

不要再重点看 pair correlation。

这轮最终主要看三个 execution-level metric。

## ① Occupancy gain

[
Occ
===

E[
N_{\mathrm{active}}
\mid
BlockTask\ active
]
]

相对于 fanout-only：

[
Gain_{occ}
==========

\frac{
Occ_{candidate}
}{
Occ_{fanout}
}-1
]

---

## ② Weighted aggregation ratio

和上一轮一致：

[
R_{agg}
=======

1-
\frac{
\sum UniqueDest
}{
\sum RawUpdates
}
]

这是最关键的 structural metric。

---

## ③ Estimated global atomic count

甚至直接报告：

[
AtomicEstimate
==============

\sum UniqueDest
]

normalization：

[
\boxed{
AtomicReduction
===============

1-
\frac{
AtomicEstimate_{candidate}
}{
AtomicEstimate_{fanout}
}
}
]

这个比：

```text
aggregation ratio +0.01
```

更容易理解。

---

# 23. 再增加一个 tradeoff metric

因为现在就是 fanout 与 activity 的 tradeoff。

定义：

### Static fanout quality

假设所有 block neuron 都 active：

[
R_{static}
==========

1-
\frac{
|\cup_i Out_i|
}{
\sum_i|Out_i|
}
]

### Realized aggregation

根据真实 spike trace：

[
R_{real}
]

然后看：

```text
candidate        static       realized

fanout-only       high          X
local-fanin      slightly ↓     ↑ ?
joint             maybe ↓       ↑ ?
```

你真正希望出现：

```text
static fanout quality:
-0.5%

realized aggregation:
+3%
```

说明牺牲了一点理论 locality，却获得了更好的动态 utilization。

---

# 24. local window sweep

第一条路线建议：

```text
window:
32
64
128
256
512
```

但：

```text
32
```

只是 sanity baseline。

真正关注：

```text
64
128
256
```

预期存在 U-shape：

```text
window 太小
→ activity freedom 不够

window 太大
→ fanout locality 被破坏
```

因此可能：

```text
128
```

左右最好。

不要预设答案。

---

# 25. λ sweep

Joint score：

```text
lambda:

0
0.1
0.25
0.5
1
2
```

画：

```text
x = lambda

left y:
atomic reduction

right y:
occupancy gain
```

或者 Pareto：

```text
x = static fanout aggregation

y = realized aggregation
```

这样可以直接看到：

> activity signal 加多少开始伤害 fanout。

---

# 26. firing rate sweep

主横轴正式改为：

```text
Population mean firing rate (Hz)
```

比如最后实际得到：

```text
0.7 Hz
1.3 Hz
2.4 Hz
5.1 Hz
9.8 Hz
18.6 Hz
```

则所有表格按这些真实值展示。

推荐图：

```text
x = Mean firing rate (Hz)

y = Atomic reduction vs fanout-only
```

几条线：

```text
local-fanin-64
local-fanin-128
joint-fanin-best
oracle-spike
```

这张最终可能直接进论文。

---

# 27. firing rate 计算伪代码

如果 simulator 输出：

```python
spikes[T, N]
```

：

```python
def firing_statistics(
    spikes,
    dt_ms,
):

    T, N = spikes.shape

    duration_s = (
        T * dt_ms / 1000.0
    )

    spike_count_per_neuron = (
        spikes.sum(dim=0)
    )

    total_spikes = (
        spike_count_per_neuron.sum()
    )

    population_mean_hz = (
        total_spikes
        / N
        / duration_s
    )

    active_mask = (
        spike_count_per_neuron > 0
    )

    active_neurons = (
        active_mask.sum()
    )

    if active_neurons > 0:

        active_mean_hz = (
            total_spikes
            / active_neurons
            / duration_s
        )

    else:
        active_mean_hz = 0

    silent_fraction = (
        1
        - active_neurons / N
    )

    return {
        "population_mean_hz":
            population_mean_hz,

        "active_mean_hz":
            active_mean_hz,

        "silent_fraction":
            silent_fraction,

        "total_spikes":
            total_spikes,
    }
```

---

# 28. 完整实验主循环

```python
for dataset in datasets:

    graph = load_graph(dataset)

    # ---------------------------------
    # calibration
    # ---------------------------------

    conditions = calibrate_conditions(
        graph,
        target_rates_hz=[
            0.5,
            1,
            2,
            5,
            10,
            20,
        ],
    )


    # ---------------------------------
    # topology
    # ---------------------------------

    fanout = csr_rows(graph)

    fanin = csr_rows(
        transpose(graph)
    )


    # ---------------------------------
    # baseline order
    # ---------------------------------

    fanout_order = (
        build_existing_fanout_order(
            graph
        )
    )


    for condition in conditions:

        # -----------------------------
        # run spike trace
        # -----------------------------

        spikes = simulate(
            graph,
            condition,
            timesteps=T,
        )

        fr_stats = firing_statistics(
            spikes,
            dt_ms=DT_MS,
        )

        fr_hz = (
            fr_stats[
                "population_mean_hz"
            ]
        )


        # -----------------------------
        # candidate A:
        # global fanout -> local fanin
        # -----------------------------

        for window in [
            32,
            64,
            128,
            256,
            512,
        ]:

            order = (
                refine_fanout_order_by_fanin(
                    fanout_order,
                    fanin,
                    window_size=window,
                    block_size=32,
                )
            )

            metrics = analyze_blocks(
                order,
                fanout,
                spikes,
            )

            save_result(
                dataset=dataset,
                firing_rate_hz=fr_hz,
                method=f"local_fanin_{window}",
                metrics=metrics,
            )


        # -----------------------------
        # candidate B:
        # joint structural score
        # -----------------------------

        for lam in [
            0,
            0.1,
            0.25,
            0.5,
            1,
            2,
        ]:

            order = joint_pack_order(
                graph,
                fanin,
                fanout,
                lambda_activity=lam,
                activity_mode="fanin",
            )

            metrics = analyze_blocks(
                order,
                fanout,
                spikes,
            )

            save_result(
                dataset=dataset,
                firing_rate_hz=fr_hz,
                method=f"joint_fanin_{lam}",
                metrics=metrics,
            )


        # -----------------------------
        # oracle
        # -----------------------------

        spike_affinity = (
            build_spike_affinity(
                spikes
            )
        )

        for lam in [
            0.1,
            0.25,
            0.5,
            1,
        ]:

            order = joint_pack_order(
                graph,
                fanin,
                fanout,
                lambda_activity=lam,
                activity_mode="spike",
                activity_affinity=
                    spike_affinity,
            )

            metrics = analyze_blocks(
                order,
                fanout,
                spikes,
            )

            save_result(
                dataset=dataset,
                firing_rate_hz=fr_hz,
                method=f"oracle_{lam}",
                metrics=metrics,
            )
```

---

# 29. Oracle 有一个实验设计细节：不能拿同一 trace 排序又评估

否则会 information leakage。

应该：

```text
calibration trace
        ↓
learn activity affinity
        ↓
generate permutation

evaluation trace
        ↓
measure occupancy / aggregation
```

例如：

```python
profile_spikes = simulate(
    graph,
    seed=SEED_A,
    ...
)

evaluation_spikes = simulate(
    graph,
    seed=SEED_B,
    ...
)
```

这样才能回答：

> co-activity pattern 是否具有跨时间窗口稳定性？

否则 oracle 只是“看答案重新排序”。

---

# 30. FlyBrain 应该成为这一版实验的第一优先级

上一轮 synthetic 几乎没有 activity locality，而 FlyBrain 明显表现不同。

因此不需要再让四类 dataset 平均分配实验精力。

建议：

### Primary

```text
FlyBrain / FlyWire
full Hz sweep
all windows
all λ
oracle
```

### Controls

```text
uniform
community
spatial
```

只做：

```text
2–3 representative Hz
fanout
best local-fanin
best joint-fanin
```

作用只是证明：

> optimization 是针对 biological-connectome structure，而不是普遍的 graph sorting trick。

这反而会让论文 story 更集中。

---

# 31. 最终 decision tree

这轮之后基本可以决定 preprocessing 方向到底值不值得保留。

### 如果：

[
local\ fanin > fanout
]

而且基本不损失 static fanout locality：

> 直接采用 `global fanout → local fanin refinement`。

它简单、cheap、解释也漂亮。

---

### 如果：

[
joint\ fanin > local\ fanin
]

：

> 使用 joint expected-aggregation objective。

可以成为更正式的方法。

---

### 如果：

[
oracle \gg fanin > fanout
]

：

> activity locality 有价值，但 fanin proxy 太弱，考虑 profile-guided preprocessing。

---

### 如果：

[
oracle \approx fanout
]

：

> 停止这个方向。

因为说明 activity-aware reorder 的 theoretical headroom 本身就小。

---

## 我最推荐的实施顺序

先不要同时写两个复杂版本。第一轮只实现：

```text
1. posterior population firing rate (Hz)
2. global fanout → local fanin
   window = 64 / 128 / 256
3. joint score:
   ReuseOut × (1 + λ FaninAffinity)
   λ = 0 / .25 / .5 / 1
4. spike-profile oracle
```

而且 joint score 最好使用：

[
\boxed{
Score(j,B)
==========

|Out_j\cap U_B|
\left(
1+\lambda A(j,B)
\right)
}
]

而不是纯 `Jaccard_out × Jaccard_in`。它直接对应你的 kernel 目标：**candidate 加入当前 BlockTask 后，能够复用多少已有 destination，同时它又有多大概率和 block 内 neuron 一起被激活。**

这样这一轮实验就从“排序 heuristic 对比”升级成了一个相当清晰的 **expected realized aggregation** 优化问题。
