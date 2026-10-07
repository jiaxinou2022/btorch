可以，下一步最合理的不是继续“想一个更复杂的排序”，而是把候选 fanout preprocessing 做成一个**分层筛选实验**：先用便宜的结构指标淘汰，再用真实 kernel 决胜。上一轮已经说明，真正值得继续的是纯 fanout packing，而不是 fanin/activity 分支；FlyBrain 上 `lambda=0` 的 block-aware fanout greedy 已经给出约 3.9%–4.7% 的 estimated atomic reduction。

我建议最终比较 5 类算法，但分两阶段，不要一上来全跑真实 kernel。

---

## 一、候选算法

第一类是当前 baseline：

**A. Dominant-post-block sort**

也就是现有 coarse similarity：

[
key_i=\operatorname{argmax}_b h_i[b]
]

其中 (h_i[b]) 是 neuron (i) 指向 post block (b) 的 edge 数。这个作为原始 fanout preprocessing baseline。

第二类是更丰富但仍然廉价的一维排序：

**B. Top-k fanout signature**

保留 top-2 / top-4 dominant post blocks：

[
Sig_i=(b_1,c_1,b_2,c_2,\ldots)
]

然后 lexicographic sort。它回答：

> 现有方法效果有限，是不是仅仅因为 dominant block 丢掉了太多 fanout 信息？

第三类是压缩 sketch：

**C. Bit-sketch / SimHash coarse sort**

把整个 fanout 压成 64/128-bit signature，再按 signature 排序或分 bucket。

它回答：

> 能否用很便宜的固定长度表示，保留比 top-k 更多的全局 fanout locality？

第四类是已经有积极结果的：

**D. Block-aware Union Greedy**

直接构建每个 32-neuron BlockTask：

[
Reuse(j,B)=|F_j\cap U_B|
]

其中：

[
U_B=\bigcup_{i\in B}F_i
]

每次选择 reuse 最大的 neuron。

这是现在最重要的 candidate。

第五类是其增强版：

**E. Contention-weighted Union Greedy**

[
Score(j,B)=
\sum_{p\in F_j\cap U_B}
w_p
]

例如：

[
w_p=\log(1+\deg_{in}(p))
]

让高 fanin、潜在 atomic hotspot 的 destination 优先被 block 内合并。

---

# 二、实验问题不要只问“哪个 aggregation ratio 最大”

最终要同时回答四个问题：

[
\boxed{
\text{quality},\quad
\text{kernel speed},\quad
\text{load balance},\quad
\text{preprocessing cost}
}
]

否则可能得到一个：

```text
atomic 少 6%
preprocessing 慢 100×
kernel 只快 0.2%
```

这种没有工程价值的算法。

所以最终选择不是单指标 winner，而是 Pareto comparison。

---

# 三、Phase 1：结构级筛选

这一阶段不跑完整 kernel benchmark，只用已有 graph + spike trace 做结构估计。

重点放 FlyBrain，synthetic 只做 control。

建议方法：

```text
identity
dominant
top2
top4
simhash64
simhash128
union_greedy
weighted_union_greedy
```

---

## 1. Static aggregation potential

对于每个最终 BlockTask (B)：

[
Raw(B)=\sum_{i\in B}|F_i|
]

[
Unique(B)=\left|\bigcup_{i\in B}F_i\right|
]

定义：

[
R_{static}
==========

1-
\frac{\sum_B Unique(B)}
{\sum_B Raw(B)}
]

这个指标衡量：

> 如果 block 内 neuron 都 active，理论可合并掉多少 destination。

---

## 2. Realized aggregation

使用真实 spike trace。

在 timestep (t)：

[
A_{B,t}
=======

{i\in B:s_i(t)=1}
]

然后：

[
Raw_{B,t}
=========

\sum_{i\in A_{B,t}}|F_i|
]

[
Unique_{B,t}
============

\left|
\bigcup_{i\in A_{B,t}}F_i
\right|
]

全局：

[
R_{real}
========

1-
\frac{\sum_{B,t}Unique_{B,t}}
{\sum_{B,t}Raw_{B,t}}
]

这是更接近 shared hash 的指标。

---

## 3. Estimated atomic reduction

相对于现有 fanout baseline：

[
AtomicReduction
===============

1-
\frac{
\sum Unique_{candidate}
}{
\sum Unique_{baseline}
}
]

上一轮真正有价值的就是这个指标。

---

## 4. Load balance

每个 block 的 edge workload：

[
Load(B)=\sum_{i\in B}|F_i|
]

记录：

```text
mean
std
CV
P95
P99
max
```

尤其看：

[
CV=
\frac{\sigma_{Load}}{\mu_{Load}}
]

因为 union greedy 可能把高 degree neuron 聚到一起。

---

## 5. Preprocessing cost

记录：

```text
preprocess_ms
temporary_memory_MB
```

最好也记录复杂度性质：

```text
dominant       O(E)
top-k          O(E)
simhash        O(E × sketch_bits-ish)
union greedy   O(candidate comparisons)
```

---

# 四、Phase 1 的淘汰标准

不要所有方法都进 kernel。

例如可以规定：

候选必须满足至少：

[
AtomicReduction > 1%
]

并且：

[
LoadCV
]

不能恶化超过比如 10–15%。

同时 preprocessing cost 不能离谱。

最后只留下 2–3 个进入真实 kernel。

预期很可能是：

```text
dominant baseline
union greedy
weighted union greedy
```

如果 top-k / sketch 很接近 greedy，则也值得留下，因为 preprocessing 更便宜。

---

# 五、Top-k signature 的具体实现

先建立 post-block histogram。

```python
def build_post_block_histogram(
    fanout,
    post_block_size=32,
):
    signatures = []

    for neuron in range(N):

        hist = {}

        for post in fanout[neuron]:
            block = post // post_block_size
            hist[block] = hist.get(block, 0) + 1

        signatures.append(hist)

    return signatures
```

取 top-k：

```python
def topk_signature(hist, k=4):
    items = sorted(
        hist.items(),
        key=lambda x: (-x[1], x[0])
    )

    items = items[:k]

    sig = []

    for block, count in items:
        sig.extend([
            block,
            count,
        ])

    while len(sig) < 2 * k:
        sig.extend([-1, 0])

    return tuple(sig)
```

排序：

```python
order = sorted(
    range(N),
    key=lambda i: (
        topk_signature(
            hist[i],
            k=4,
        ),
        degree_bucket[i],
        i,
    )
)
```

这里建议分别测：

```text
top2
top4
top8
```

但没必要更多。

---

# 六、SimHash / bit-sketch 版本

如果想测试“压缩成一个数字”是否值得：

```python
def build_bit_sketch(
    targets,
    bits=128,
):

    sketch = 0

    for post in targets:

        h = hash_post(post)

        bit = h % bits

        sketch |= (
            1 << bit
        )

    return sketch
```

这其实更像 Bloom-style sketch，不是真正 SimHash，但对于 set overlap 更直接。

排序可以先简单：

```python
order = sorted(
    range(N),
    key=lambda i: sketch[i],
)
```

但这个排序只是粗 baseline。

更合理的是：

```text
sketch
→ bucket
→ local union greedy
```

所以我建议其实直接测两种：

```text
sketch_sort
sketch_then_greedy
```

---

# 七、Sketch → local greedy

这可能是最终比较有工程价值的版本。

```python
def build_sketch_guided_order(
    fanout,
    sketch,
    coarse_window=256,
    block_size=32,
):

    coarse_order = sorted(
        range(N),
        key=lambda i: sketch[i],
    )

    final_order = []

    for start in range(
        0,
        N,
        coarse_window,
    ):

        window = coarse_order[
            start:
            start + coarse_window
        ]

        packed = union_greedy_pack(
            window,
            fanout,
            block_size,
        )

        final_order.extend(
            packed
        )

    return final_order
```

这样可以比较：

```text
global exact greedy
vs
sketch-local greedy
```

如果收益接近，而预处理快很多，那么后者更值得采用。

---

# 八、Union Greedy

核心：

```python
def union_greedy_pack(
    neurons,
    fanout,
    block_size=32,
):

    remaining = set(neurons)
    order = []

    while remaining:

        seed = choose_seed(
            remaining,
            fanout,
        )

        block = [seed]
        remaining.remove(seed)

        union_targets = set(
            fanout[seed]
        )

        while (
            len(block) < block_size
            and remaining
        ):

            best = None
            best_score = -1

            for candidate in remaining:

                reuse = intersection_size(
                    fanout[candidate],
                    union_targets,
                )

                if reuse > best_score:
                    best_score = reuse
                    best = candidate

            block.append(best)
            remaining.remove(best)

            union_targets.update(
                fanout[best]
            )

        order.extend(block)

    return order
```

但全局这样做可能太贵。

实际测试建议至少两个版本：

```text
global greedy           quality upper-ish bound
windowed greedy         practical
```

例如：

```text
window = 128
256
512
1024
```

---

# 九、Seed selection 也应该纳入小型 ablation

因为 greedy 很依赖 seed。

只测三种就够：

### first

```python
seed = first remaining
```

### max-degree

```python
seed = argmax(
    degree[i]
)
```

### max-contention

定义：

[
C_i=
\sum_{p\in F_i}
\log(1+\deg_{in}(p))
]

然后：

```python
seed = argmax(
    contention_score[i]
)
```

不用把 seed 当主研究方向，只找一个稳定版本。

---

# 十、Weighted Union Greedy

先计算：

```python
post_weight[p] = log1p(
    fanin_degree[p]
)
```

然后：

```python
def weighted_reuse(
    candidate,
    union_targets,
    fanout,
    post_weight,
):

    score = 0.0

    for post in fanout[candidate]:

        if post in union_targets:
            score += (
                post_weight[post]
            )

    return score
```

于是：

```python
best = argmax(
    weighted_reuse(...)
)
```

这里建议测三种 weight：

[
w_p=1
]

baseline union；

[
w_p=\log(1+d_p)
]

moderate contention；

[
w_p=\sqrt{d_p}
]

更强 contention emphasis。

不要一开始用：

[
w=d
]

因为 FlyBrain degree tail 很重，很容易被极端节点支配。

---

# 十一、一个特别值得加的候选：Normalized reuse

纯：

[
|F_j\cap U_B|
]

天然偏好长 row。

可以测试：

[
Score=
\frac{|F_j\cap U_B|}
{|F_j|}
]

表示 candidate 自己有多少比例能复用。

但它又可能过度偏好短 row。

所以更稳的是：

[
\boxed{
Score=
\frac{|F_j\cap U_B|}
{|F_j|^\alpha}
}
]

扫：

```text
alpha = 0
0.5
1
```

其中：

```text
alpha=0
```

就是当前 union greedy。

这个值得放进 Phase 1，因为它直接检测：

> 当前收益是不是单纯被高-degree neuron 主导。

---

# 十二、因此 Phase 1 我建议实际跑这些

不是十几二十个，而是：

| ID | 方法                               |
| -- | -------------------------------- |
| A  | dominant-block baseline          |
| B  | top4 signature                   |
| C  | bit-sketch sort                  |
| D  | sketch + local union greedy      |
| E  | raw union greedy                 |
| F  | normalized union greedy α=0.5    |
| G  | weighted union greedy log-degree |

就 7 个。

其中真正复杂的只有 D–G。

---

# 十三、Phase 1 主循环伪代码

```python
METHODS = [
    "dominant",
    "top4",
    "sketch_sort",
    "sketch_local_greedy",
    "union_greedy",
    "normalized_union",
    "weighted_union",
]

for dataset in datasets:

    graph = load_graph(dataset)

    fanout = build_fanout(graph)
    fanin_degree = (
        compute_fanin_degree(graph)
    )

    spikes = load_or_run_trace(
        dataset
    )

    baseline_order = (
        build_dominant_order(
            graph
        )
    )

    baseline_metrics = (
        analyze_order(
            baseline_order,
            fanout,
            spikes,
        )
    )

    for method in METHODS:

        start = timer()

        order = build_order(
            method,
            graph=graph,
            fanout=fanout,
            fanin_degree=fanin_degree,
        )

        preprocess_ms = (
            timer() - start
        )

        metrics = analyze_order(
            order,
            fanout,
            spikes,
        )

        save({
            "dataset": dataset,
            "method": method,

            "preprocess_ms":
                preprocess_ms,

            "static_aggregation":
                metrics.static_aggregation,

            "realized_aggregation":
                metrics.realized_aggregation,

            "atomic_reduction":
                compare_atomic(
                    metrics,
                    baseline_metrics,
                ),

            "load_cv":
                metrics.block_load_cv,

            "load_p99":
                metrics.block_load_p99,
        })
```

---

# 十四、Phase 1 最关键的一张表

最终先得到：

| Method        | Atomic ↓ | Static agg ↑ | Load CV | Preprocess |
| ------------- | -------: | -----------: | ------: | ---------: |
| dominant      | baseline |     baseline |   1.00× |         1× |
| top4          |        ? |            ? |       ? |          ? |
| sketch        |        ? |            ? |       ? |          ? |
| sketch+greedy |        ? |            ? |       ? |          ? |
| union         |      ~4% |      +7.6 pp |       ? |          ? |
| normalized    |        ? |            ? |       ? |          ? |
| weighted      |        ? |            ? |       ? |          ? |

然后筛掉明显差的。

---

# 十五、Phase 2：真实 kernel benchmark

只让 Phase 1 最好的 2–3 个 candidate 进入。

例如假设最后是：

```text
dominant
union
weighted_union
sketch_local_greedy
```

真实测：

```text
persistent + block + hash
```

而且以 FlyBrain 为主。

firing rate：

```text
0.5
1
2
5
10
20
50 Hz
```

以实际 posterior population mean Hz 为横轴。

---

# 十六、真实 kernel 要测哪些东西

主指标：

[
Time/step
]

和：

[
Speedup=
T_{baseline}/T_{candidate}
]

然后至少再记录：

```text
hash flush count / global atomic count
BlockTask count
edges processed
```

如果方便 profiler：

```text
Long Scoreboard
L2 throughput
DRAM throughput
atomic-related stalls
SM utilization
```

---

# 十七、一定要检查 physical reorder 对 memory locality 的副作用

一个新的 order 可能：

```text
aggregation ↑
```

但：

```text
CSR row locality ↓
cache behavior ↓
```

所以至少比较：

```text
L2 hit rate
DRAM bytes
Long Scoreboard
```

如果：

```text
atomic -5%
runtime +2%
```

你就知道收益被读侧损失吃掉了。

---

# 十八、kernel benchmark 伪代码

```python
for rate_target in TARGET_RATES_HZ:

    workload = prepare_flybrain(
        target_rate_hz=rate_target
    )

    measured_hz = measure_rate(
        workload
    )

    for method in selected_methods:

        perm = preprocessing_orders[
            method
        ]

        reordered = apply_permutation(
            workload,
            perm
        )

        warmup(
            reordered,
            kernel="persistent_hash",
        )

        times = []

        for repeat in range(REPEATS):

            elapsed = benchmark_kernel(
                reordered,
                kernel="persistent_hash",
            )

            times.append(elapsed)

        save({
            "method": method,
            "firing_rate_hz":
                measured_hz,

            "median_us":
                median(times),

            "p25":
                percentile(times, 25),

            "p75":
                percentile(times, 75),
        })
```

---

# 十九、最终算法选择标准

我建议按这个优先级决定，而不是只看谁快一点。

### 第一优先：真实 kernel speedup

必须是真收益。

比如：

[

> 1%-2%
> ]

而且稳定。

### 第二优先：跨 firing-rate 稳定性

不要只在：

```text
50 Hz
```

有效。

最好在 FlyBrain 的主要工作区间都有收益。

### 第三优先：preprocessing cost

如果：

```text
union greedy:
+2.5% kernel
5 sec preprocessing

sketch greedy:
+2.3% kernel
80 ms preprocessing
```

那显然后者更工程化。

### 第四优先：鲁棒性

不同 FlyBrain 子图 / seed /规模不应完全崩。

---

# 二十、最终 decision tree

可以非常机械地决定。

```text
                Phase 1
                   │
          atomic reduction >1% ?
             /             \
           no               yes
           │                 │
         淘汰          load balance OK?
                          /       \
                        no         yes
                        │           │
                      淘汰      Phase 2 kernel
                                    │
                              runtime improves?
                                /        \
                              no          yes
                              │            │
                            淘汰       preprocess cost?
                                         /       \
                                      huge       reasonable
                                       │             │
                                 compare cheaper    adopt
                                   approximation
```

---

# 二十一、我对候选的预期排序

基于目前数据，我会把优先级排成：

### 1. Raw union greedy

最重要，因为已经有 positive evidence。

### 2. Contention-weighted union greedy

最值得作为增强版本。

### 3. Sketch-guided local union greedy

最值得作为工程优化版本。

### 4. Normalized union greedy

主要用于判断 degree bias。

### 5. Top-k signature

作为 cheap baseline。

### 6. Pure sketch sort

更多是验证“单数字排序到底够不够”。

---

## 最终建议的实验主线

不要把实验写成：

> “比较很多不同 similarity 方法”。

而应该围绕一个更清楚的问题：

> **fanout representation 要保留多少信息，以及 preprocessing objective 要多直接地贴近 BlockTask 的 destination reuse，才能获得最好性能？**

于是候选自然形成：

[
\text{dominant scalar}
\rightarrow
\text{richer signature}
\rightarrow
\text{compressed sketch}
\rightarrow
\text{block-aware union}
\rightarrow
\text{contention-aware union}
]

这是一个很干净的 refinement ladder。

最终真正采用哪个算法，则由：

[
\boxed{
\text{real kernel speedup}
+
\text{preprocessing efficiency}
+
\text{load-balance robustness}
}
]

共同决定，而不是看结构指标单独拍板。
