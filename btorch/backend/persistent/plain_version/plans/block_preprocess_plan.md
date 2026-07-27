# Persistent SNN 物理重排实现计划

## 1. 目标与基本原则

当前 block kernel 固定以连续 32 个 neuron 组成一个 spike block：

```text
block 0: neuron [0, 31]
block 1: neuron [32, 63]
...
```

因此，重排的核心不是在运行时增加新的映射或调度逻辑，而是在进入 persistent kernel 前，将 neuron 状态和 CSR 矩阵都物理变换到新的编号空间：

```text
原始编号空间
    ↓ 输入重排
重排编号空间
    ↓ persistent 推理
重排编号空间输出
    ↓ 逆重排
原始编号空间
```

预处理保存两个映射：

```cpp
new_to_old[new_id] = old_id;
old_to_new[old_id] = new_id;
```

其中：

```cpp
old_to_new[new_to_old[i]] == i
```

物理重排包括：

1. CSR 的 presynaptic row 重排；
2. CSR 中 postsynaptic column index 同步重映射；
3. neuron 状态、输入电流和输出张量进入 kernel 前重排；
4. persistent 运行结束后按照逆映射恢复原编号顺序。

最终 persistent kernel 内部不再访问映射表，仍直接使用：

```cpp
psc[b * n_neuron + reordered_post]
```

从而避免每条突触写回时增加一次映射读取。

---

# 2. 数据表示

## 2.1 映射方向

建议明确采用：

```cpp
new_to_old[new_id]  // 新位置原本是谁
old_to_new[old_id]  // 原位置现在去了哪里
```

输入重排：

```cpp
reordered[new_id] = original[new_to_old[new_id]];
```

输出还原：

```cpp
original[old_id] = reordered[old_to_new[old_id]];
```

CSR row 重排时使用 `new_to_old`，column 重映射时使用 `old_to_new`。

---

## 2.2 需要重排的静态数据

```text
graph_indptr
graph_indices
graph_weight
fanout / degree
其他与 neuron 编号绑定的静态参数
```

如果每个 neuron 有独立参数，例如：

```text
threshold
decay
reset
tau
```

也需要按相同映射重排。

---

## 2.3 需要在推理前后重排的动态数据

进入 persistent 前：

```text
initial membrane state
initial PSC/current
external input
其他按 neuron 排列的状态
```

运行结束后：

```text
membrane output
spike output
PSC/current output
其他需要暴露给外部的状态
```

若一次 persistent launch 内执行较长时间窗口，则输入和输出重排成本只发生各一次，容易摊薄。

若外部输入按每个时间步不断传入，应优先在生成输入时直接写入重排编号，而不是每步额外执行 permutation kernel。

---

# 3. 统一实验框架

所有重排方案最终只需要输出：

```cpp
std::vector<int> new_to_old;
std::vector<int> old_to_new;
```

后续物理 CSR 重排、状态重排和 persistent kernel完全共用。

建议接口：

```cpp
enum class ReorderMode {
    Identity,
    GlobalCost,
    LocalCost,
    GlobalSimilarity,
    LocalSimilarity,
    GlobalCostSimilarity,
    LocalCostSimilarity
};

struct ReorderConfig {
    ReorderMode mode;

    int neuron_block_size = 32;

    // 成本分桶
    int fanout_bucket_count;
    std::vector<int> fanout_boundaries;

    // 极长行
    int extreme_fanout_threshold;

    // 相似性
    int post_block_size;
    int similarity_candidates;
};
```

统一流程：

```cpp
Permutation build_permutation(
    const CSRGraph& graph,
    const ReorderConfig& config);

CSRGraph reorder_graph(
    const CSRGraph& old_graph,
    const Permutation& permutation);
```

这样可以快速替换重排策略而不修改后续代码。

---

# 4. 物理重排的总体流程

## 4.1 CPU 参考实现

第一版先实现 CPU 版本，保证正确性，并作为 CUDA 版本的参考结果。

```cpp
Permutation build_permutation_cpu(graph, config);

new_indptr, new_indices, new_weights =
    reorder_csr_cpu(graph, permutation);

run_reference_test();
```

CPU 版本不要求快，但必须便于调试和验证。

---

## 4.2 GPU 高效实现

CUDA 重排分为三个阶段：

```text
阶段 A：计算排序/分组 key，并生成 permutation
阶段 B：计算新 CSR row length 与 indptr
阶段 C：并行搬运 CSR edge，并重映射 post index
```

建议尽量使用 CUB：

```text
cub::DeviceRadixSort
cub::DeviceScan
cub::DeviceHistogram
```

避免手写通用排序。

---

# 5. CSR 物理重排

假设已有：

```cpp
new_to_old[new_row]
old_to_new[old_row]
```

## 5.1 计算新 row length

```cpp
__global__ void compute_reordered_row_lengths(
    const int* old_indptr,
    const int* new_to_old,
    int* new_row_lengths,
    int n)
{
    int new_row = blockIdx.x * blockDim.x + threadIdx.x;

    if (new_row >= n)
        return;

    int old_row = new_to_old[new_row];

    new_row_lengths[new_row] =
        old_indptr[old_row + 1] - old_indptr[old_row];
}
```

随后进行 exclusive scan：

```cpp
new_indptr = exclusive_scan(new_row_lengths);
```

---

## 5.2 并行搬运边

每个 CUDA block 或 warp负责一个新 CSR row。

```cpp
__global__ void reorder_csr_rows(
    const int* old_indptr,
    const int* old_indices,
    const float* old_weights,

    const int* new_to_old,
    const int* old_to_new,

    const int* new_indptr,
    int* new_indices,
    float* new_weights,

    int n)
{
    int new_row = blockIdx.x;

    if (new_row >= n)
        return;

    int old_row = new_to_old[new_row];

    int old_begin = old_indptr[old_row];
    int old_end   = old_indptr[old_row + 1];
    int new_begin = new_indptr[new_row];

    for (int local_edge = threadIdx.x;
         old_begin + local_edge < old_end;
         local_edge += blockDim.x)
    {
        int old_edge = old_begin + local_edge;
        int new_edge = new_begin + local_edge;

        int old_post = old_indices[old_edge];
        int new_post = old_to_new[old_post];

        new_indices[new_edge] = new_post;
        new_weights[new_edge] = old_weights[old_edge];
    }
}
```

这里同时完成：

1. presynaptic row 的物理移动；
2. postsynaptic column 的编号转换。

因此 persistent 内部不需要做：

```cpp
post = old_to_new[graph_indices[edge]];
```

---

## 5.3 是否需要重新排列 row 内 edge

第一版不需要。

只需保持原 row 内 edge 顺序：

```text
old row edge order
→ 搬到 new row
→ post id 转换
```

后续若要进一步做 post 局部排序或 segmented reduce，再单独增加实验版本。

---

# 6. 动态状态的输入重排与输出还原

对于形状：

```text
[batch, neuron]
```

使用二维 permutation kernel。

## 6.1 输入重排

```cpp
template<typename T>
__global__ void reorder_state_forward(
    const T* original,
    T* reordered,
    const int* new_to_old,
    int batch_size,
    int n_neuron)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * n_neuron;

    if (idx >= total)
        return;

    int b = idx / n_neuron;
    int new_id = idx % n_neuron;
    int old_id = new_to_old[new_id];

    reordered[b * n_neuron + new_id] =
        original[b * n_neuron + old_id];
}
```

---

## 6.2 输出还原

```cpp
template<typename T>
__global__ void reorder_state_backward(
    const T* reordered,
    T* original,
    const int* old_to_new,
    int batch_size,
    int n_neuron)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * n_neuron;

    if (idx >= total)
        return;

    int b = idx / n_neuron;
    int old_id = idx % n_neuron;
    int new_id = old_to_new[old_id];

    original[b * n_neuron + old_id] =
        reordered[b * n_neuron + new_id];
}
```

---

## 6.3 避免无谓拷贝

如果调用方允许内部状态长期保持重排顺序，则不需要每次 persistent launch 都来回转换所有状态。

推荐生命周期：

```text
模型初始化：
    构建映射
    物理重排CSR
    物理重排静态参数

一个推理窗口开始：
    输入重排一次

persistent执行多个时间步：
    所有状态保持重排编号

窗口结束：
    仅需要暴露的输出逆重排一次
```

如果连续多次调用 persistent，可继续保存重排状态，只转换外部输入和最终输出。

---

# 7. 方案一：成本重排

## 7.1 目标

使每个固定32-neuron block中的 fanout分布更加可控，减少某些block同时包含多条大行产生的长尾。

但这里存在两种不同目标，需要都测试。

### 成本聚合

将fanout相近的neuron放在一起。

优点：

* block内部row长度相似；
* warp内路径更一致；
* prefix/run处理时分歧较小；
* 后续可以按block类型专门化。

风险：

* 多条大fanout行被聚到同一block；
* block总edge数可能增大；
* 最后形成重block尾部。

### 成本交错

将大行与小行搭配，使每个block总fanout更均衡。

优点：

* task总成本更均匀；
* tail更小。

风险：

* block内部row长度差异增加；
* warp执行路径和lane利用率可能变差。

因此“成本重排”至少要比较：

```text
Cost-Similar：相似成本聚合
Cost-Balanced：大小成本交错
```

---

## 7.2 极长行后置

首先将：

```cpp
fanout >= extreme_fanout_threshold
```

的行放在全部普通行之后。

```text
普通行
极长行
```

由于极长行本来就走独立segment路径，集中放置可以：

* 简化长行区间判断；
* 让普通block不被极端degree污染；
* 提高普通block的degree稳定性；
* 使长行CSR在物理地址上连续。

---

## 7.3 全局成本相似重排

最简单方式是按fanout排序。

```cpp
for neuron i:
    key[i] = fanout[i];

stable_sort(neurons, key ascending);

normal neurons first;
extreme neurons last;
```

伪代码：

```cpp
vector<int> normal;
vector<int> extreme;

for i in [0, n):
    if fanout[i] >= threshold:
        extreme.push_back(i);
    else:
        normal.push_back(i);

sort(normal, by fanout ascending);
sort(extreme, by fanout ascending);

new_to_old = concat(normal, extreme);
```

GPU实现可以使用：

```text
key = (is_extreme, fanout)
value = old_id
CUB radix sort
```

---

## 7.4 全局成本分桶重排

为了降低完整排序成本，也可以只做近似桶。

例如：

```text
bucket 0: fanout [0, 4]
bucket 1: fanout [5, 8]
bucket 2: fanout [9, 16]
bucket 3: fanout [17, 32]
bucket 4: fanout [33, 64]
bucket 5: fanout [65, 128]
bucket 6: fanout [129, 255]
bucket 7: fanout >= 256
```

GPU伪代码：

```cpp
__global__ void assign_cost_bucket(
    const int* fanout,
    int* bucket_id,
    int* old_id,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
        return;

    bucket_id[i] = classify_fanout(fanout[i]);
    old_id[i] = i;
}

// stable sort by bucket_id
cub::DeviceRadixSort::SortPairs(
    bucket_id,
    sorted_bucket_id,
    old_id,
    new_to_old);
```

桶内保留原始顺序，预处理速度较快。

---

## 7.5 成本均衡交错方案

将普通行先按fanout排序，然后从两端交替取元素：

```text
largest, smallest, second largest, second smallest, ...
```

再每32个组成固定block。

```cpp
sort(normal, fanout ascending);

left = 0;
right = normal.size() - 1;

while left <= right:
    for block_slot in [0, 32):
        if block_slot is even:
            take normal[right--];
        else:
            take normal[left++];
```

更稳妥的版本是在每个block中轮流放高、中、低fanout。

该方案不改变固定block边界，只改变哪些neuron进入同一物理block。

---

# 8. 方案二：相似性重排

## 8.1 简化后的目标

当前暂不做复杂图聚类，也不根据动态活动统计优化。

对每个 presynaptic neuron，统计其突触主要落在哪一个固定 post block：

```text
post block = post_id / 32
```

定义：

[
dominant_block(i)
=================

\arg\max_b
|{j\in N(i): \lfloor j/32\rfloor=b}|
]

然后按照 `dominant_block` 聚合presynaptic neuron。

直觉是：

* 连接主要落入同一post区域的pre neuron被放入相邻物理block；
* 当它们同时发放时，post写入地址更集中；
* block内reduce、cache locality和atomic合并潜力提高。

---

## 8.2 计算dominant post block

第一版CPU伪代码：

```cpp
for pre in [0, n):
    unordered_map<int, int> counts;

    for edge in row(pre):
        post = indices[edge];
        post_block = post / 32;
        counts[post_block]++;

    dominant_block[pre] =
        argmax_block(counts);

    dominant_count[pre] =
        max_count(counts);
```

但GPU上不能为每行使用通用哈希表。

---

## 8.3 CUDA简化实现

考虑到post block数量可能较大，建议采用以下两种路径之一。

### 路径A：采样近似

每行最多采样固定数量edge，例如32或64条。

```cpp
__global__ void estimate_dominant_post_block(
    const int* indptr,
    const int* indices,
    int* dominant_block,
    int* dominant_count,
    int n,
    int sample_count)
{
    int row = blockIdx.x;

    int begin = indptr[row];
    int end   = indptr[row + 1];
    int degree = end - begin;

    // shared table，仅统计采样的post block
    init_small_table();

    for sample assigned to thread:
        int edge = choose_uniform_sample(
            begin, end, sample_count);

        int post_block = indices[edge] / 32;

        insert_or_increment(post_block);

    reduce_max_count();

    if threadIdx.x == 0:
        dominant_block[row] = best_block;
        dominant_count[row] = best_count;
}
```

该方法适合快速试验，预处理开销低。

### 路径B：排序后run length统计

生成所有：

```text
(pre_id, post_block)
```

对pair排序，再run-length encode统计每个pre对应的最大post block频数。

该方法精确，但临时内存和排序成本较高，更适合作为离线预处理。

---

## 8.4 全局相似性重排

排序key：

```cpp
key = (
    is_extreme,
    dominant_post_block,
    fanout_bucket,
    old_id
);
```

其中是否加入 `fanout_bucket` 取决于具体实验。

纯相似性版本：

```text
normal rows按dominant_post_block排序
extreme rows放到最后
```

伪代码：

```cpp
for neuron i:
    key[i].extreme = fanout[i] >= threshold;
    key[i].dominant = dominant_post_block[i];
    key[i].old_id = i;

stable_sort(neurons, key);
```

---

# 9. 成本约束下的相似性聚合

纯相似性可能把许多高fanout neuron聚到同一block，制造更严重的block尾部。

因此更推荐的主方案是：

```text
第一关键字：是否极长
第二关键字：fanout粗桶
第三关键字：dominant post block
```

即：

```cpp
key = (
    is_extreme,
    fanout_bucket,
    dominant_post_block,
    old_id
);
```

这会使：

* 同一个物理block中的fanout大致相似；
* 在相近fanout范围内，优先聚合相似post区域；
* 极长行统一移到末尾。

另一种排序次序也需要比较：

```cpp
key = (
    is_extreme,
    dominant_post_block,
    fanout_bucket,
    old_id
);
```

两者区别：

```text
Cost → Similarity：
优先执行稳定，次要提高post局部性

Similarity → Cost：
优先提高reduce潜力，次要控制成本
```

建议两者都跑实验，而不是预先判断。

---

# 10. 局部重排

## 10.1 定义

只允许 neuron 在一个局部窗口内交换。

例如：

```text
local window = 256 / 512 / 1024 neurons
```

每个窗口独立排序：

```text
[0, W)
[W, 2W)
[2W, 3W)
...
```

窗口之间保持原始顺序。

---

## 10.2 优点

* 重排幅度小；
* 更容易保留原数据和活动的局部结构；
* permutation更接近恒等映射；
* 对外部输入、标签或区域语义影响更小；
* 预处理可以高度并行；
* 每个CUDA block可以独立处理一个窗口；
* 更适合在线或快速初始化。

---

## 10.3 缺点

* 无法聚合远距离但高度相似的neuron；
* 极长行可能仍散落在多个窗口；
* 全局成本均衡能力有限；
* 相似性收益上限低于全局重排。

---

## 10.4 局部成本重排

```cpp
for each window:
    stable_sort(
        neurons in window,
        key = (is_extreme, fanout_bucket));
```

GPU上可以使用：

* 每窗口block内bitonic sort；
* segmented radix sort；
* 先生成全局复合key，key中保留window id。

统一全局radix sort的key：

```cpp
key = (
    window_id,
    is_extreme,
    fanout_bucket,
    old_local_position
);
```

因为 `window_id` 是最高关键字，所以不会跨窗口移动。

---

## 10.5 局部相似性重排

```cpp
key = (
    window_id,
    is_extreme,
    dominant_post_block,
    fanout_bucket,
    old_local_position
);
```

或：

```cpp
key = (
    window_id,
    is_extreme,
    fanout_bucket,
    dominant_post_block,
    old_local_position
);
```

这样仍可以统一用CUB radix sort，而不必为每个窗口单独启动排序。

---

# 11. 全局重排

## 11.1 优点

* 最大化成本分组效果；
* 极长行可以真正全部集中在尾部；
* 最大化dominant post block聚合；
* 更可能提升block内post重复率；
* 更容易观察算法上限。

## 11.2 缺点

* permutation距离原始顺序较远；
* CSR和所有state完全重排；
* 若存在多层或多图映射，管理复杂度更高；
* 可能破坏原有活动局部性；
* 更容易将多个高活跃或高fanout neuron集中在一起；
* 排序和预处理成本更高。

因此全局重排更适合：

* 模型静态；
* 图重复使用很多次；
* 推理窗口较长；
* 用于测试理论上限。

局部重排则适合成本更敏感或结构需要保守的场景。

---

# 12. CUDA生成permutation的推荐实现

## 12.1 生成属性

```cpp
__global__ void compute_reorder_attributes(
    const int* indptr,
    const int* dominant_post_block,
    uint64_t* keys,
    int* values,
    int n,
    ReorderConfig config)
{
    int old_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (old_id >= n)
        return;

    int degree =
        indptr[old_id + 1] - indptr[old_id];

    int extreme =
        degree >= config.extreme_fanout_threshold;

    int cost_bucket =
        classify_fanout(degree);

    int similarity_bucket =
        dominant_post_block[old_id];

    int window_id =
        config.is_local
            ? old_id / config.local_window_size
            : 0;

    uint64_t key =
        pack_key(
            window_id,
            extreme,
            cost_bucket,
            similarity_bucket,
            old_id);

    keys[old_id] = key;
    values[old_id] = old_id;
}
```

根据实验方案改变 `pack_key` 的字段顺序。

---

## 12.2 排序

```cpp
cub::DeviceRadixSort::SortPairs(
    temp_storage,
    temp_bytes,
    keys_in,
    keys_out,
    values_in,
    new_to_old,
    n);
```

---

## 12.3 构建逆映射

```cpp
__global__ void invert_permutation(
    const int* new_to_old,
    int* old_to_new,
    int n)
{
    int new_id =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (new_id >= n)
        return;

    int old_id = new_to_old[new_id];
    old_to_new[old_id] = new_id;
}
```

---

# 13. 正确性验证

每个方案必须通过以下验证。

## 13.1 permutation合法性

检查：

```text
new_to_old包含0到n-1且无重复
old_to_new是其逆排列
```

GPU上可用标记数组检查，CPU上也可直接排序验证。

---

## 13.2 CSR语义一致

对任意原始边：

```text
old_pre → old_post, weight
```

重排后必须存在：

```text
old_to_new[old_pre]
    →
old_to_new[old_post],
weight
```

随机抽样或全量检查。

---

## 13.3 推理输出一致

比较：

```text
原始图 + 原始kernel
```

与：

```text
重排输入
+ 重排图
+ persistent
+ 输出逆重排
```

检查：

* spike一致；
* membrane误差；
* PSC误差；
* 多时间步累计误差。

由于 atomic累加顺序变化，浮点结果可能存在微小差异，应使用合理容差，而不是要求bitwise一致。

---

# 14. 实验矩阵

建议先控制方案数量，避免组合爆炸。

## 第一轮：只测成本与范围

```text
Baseline
Global-Cost-Similar
Global-Cost-Balanced
Local-Cost-Similar, W=256
Local-Cost-Similar, W=1024
```

目标：确认重排能否降低block task方差和barrier。

---

## 第二轮：加入相似性

```text
Global-Similarity
Local-Similarity, W=256
Local-Similarity, W=1024
Global-Cost→Similarity
Global-Similarity→Cost
Local-Cost→Similarity
```

目标：确认相似性是否提高：

```text
block内edges/unique_post
L1/L2局部性
atomic冲突或reduce潜力
```

---

## 第三轮：选择最佳方案后配合reduce

只对最好的2–3种重排启用：

```text
无reduce
轻量tile reduce
小hash reduce
```

避免在所有重排方案上都测试hash。

---

# 15. 需要记录的统计量

## 15.1 静态统计

每个物理32-neuron block记录：

```text
fanout sum
fanout max
fanout mean
fanout variance
extreme row count
dominant post block分布
结构post重复率
```

结构重复率可定义为：

[
R_{\text{struct}}
=================

\frac{\sum_i fanout_i}
{|\bigcup_i post(i)|}
]

---

## 15.2 动态统计

结合实际spike mask记录：

```text
active row count
active edge count
active edge P50/P90/P99/max
active unique post count
active edges / active unique posts
run count
run path比例
prefix path比例
```

因为静态连接相似不一定对应动态同时发放，所以最终判断必须以动态统计为准。

---

## 15.3 NCU指标

重点比较：

```text
gpu duration
eligible warps per scheduler
issue active
long scoreboard
barrier
average active lanes
L1 hit rate
L2 hit rate
atomic sectors / requests
```

预期：

### 成本重排成功

```text
barrier下降
P99 task edge下降
eligible warp提高
```

### 相似性重排成功

```text
active edges / unique post提高
atomic reduce潜力提高
cache locality改善
```

---

# 16. 重排开销评估

总收益不能只看persistent kernel时间，还要区分：

```text
一次性模型预处理成本
每个推理窗口输入/输出permutation成本
persistent运行时间
```

定义：

[
T_{\text{total}}
================

T_{\text{preprocess}}/N_{\text{reuse}}
+
T_{\text{permute-in}}
+
T_{\text{persistent}}
+
T_{\text{permute-out}}
]

其中图的物理重排通常是一次性的，可以被大量推理摊薄。

需要报告break-even次数：

[
N_{\text{break-even}}
=====================

\frac{T_{\text{preprocess}}}
{T_{\text{baseline}}-
T_{\text{reordered}}-
T_{\text{runtime permutation}}}
]

如果模型长期重复推理，一次性预处理成本可以接受；如果图频繁变化，则局部桶排序更有吸引力。

---

# 17. 推荐施工顺序

## 第一步：建立统一permutation框架

实现：

```text
new_to_old
old_to_new
输入重排
输出逆重排
CSR物理重排
正确性测试
```

先使用identity permutation验证完整路径。

---

## 第二步：实现CPU成本重排

包括：

```text
全局fanout排序
全局fanout分桶
局部fanout分桶
极长行后置
成本交错
```

用于快速判断哪些策略值得搬到CUDA。

---

## 第三步：CUDA化CSR和状态重排

优先CUDA化：

```text
row length
exclusive scan
CSR edge搬运
输入输出permutation
```

排序本身使用CUB。

---

## 第四步：实现dominant post block

先做CPU精确版本或GPU采样版本。

验证：

```text
dominant block集中度
重排后结构重复率
动态重复率
```

只有统计显示重复率提高，才继续优化GPU精确计算。

---

## 第五步：比较局部与全局方案

优先比较：

```text
Global Cost
Local Cost W=256/1024
Global Cost→Similarity
Local Cost→Similarity
```

选出2–3个候选。

---

## 第六步：与block reduce组合

只有重排后动态重复率明显提高时，才重新启用或改造reduce。

---

# 18. 最小可行版本

为了尽快得到第一轮结果，建议MVP只包含：

```text
1. old_to_new / new_to_old
2. CPU生成permutation
3. CUDA重排CSR
4. CUDA输入/输出permutation
5. 四个策略：
   - Identity
   - Global fanout bucket
   - Local fanout bucket
   - Global dominant-post bucket
6. 极长行统一放到尾部
```

第一版不做：

```text
按成本上限动态切block
复杂图聚类
精确Jaccard相似度
row内edge排序
运行时映射
动态自适应重排
```

保持 persistent kernel完全不变。

---

# 19. 最终推荐的默认候选

最值得首先测试的主方案是：

```text
Local Cost→Similarity
```

具体key：

```cpp
(
    window_id,
    is_extreme,
    fanout_bucket,
    dominant_post_block,
    original_position
)
```

建议窗口：

```text
W = 256
W = 1024
```

原因是它在三个目标之间较平衡：

1. fanout桶降低同一block中的执行差异；
2. dominant post block提高局部写回相似性；
3. 局部窗口避免全局重排破坏活动和结构局部性；
4. 可以直接用一次全局radix sort实现；
5. 不修改固定32-neuron block kernel。

同时将以下方案作为上限对照：

```text
Global Cost→Similarity
```

如果全局版本明显更快，说明远距离聚合收益重要；如果局部版本接近全局，则优先采用局部版本，因为预处理和数据扰动更小。

---

# 20. 预期结果与决策标准

## 保留某个重排方案的最低条件

它至少应满足以下之一：

```text
persistent时间下降超过5%
barrier显著下降
block active-edge P99显著下降
动态post重复率明显提高
```

若仅静态相似性提高，但运行时延迟没有改善，则不应继续增加复杂度。

## 相似性重排是否值得继续

只有当：

[
R_{\text{dynamic,new}}

>

R_{\text{dynamic,old}}
]

且提升足以支持reduce时才继续。

经验上，如果：

```text
edges / unique_post
```

只从1.05提升到1.10，意义很小。

若能提升到1.3以上，值得测试轻量reduce；若接近1.5或更高，block内reduce可能成为主要收益来源。

## 全局与局部的选择

如果：

```text
Global性能优势 < 3%
```

优先局部方案。

如果：

```text
Global性能优势明显 > 5–10%
```

再考虑接受全局重排的更高预处理成本和更强数据扰动。
