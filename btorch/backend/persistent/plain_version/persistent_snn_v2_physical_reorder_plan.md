# Persistent SNN V2：Neuron 与 CSR 物理重排计划

## 1. 目标

第二版只修改数据布局：

```text
预处理：
    重排 neuron
    重建 CSR rows
    重写 CSR post indices

运行时：
    kernel 继续使用连续 neuron block
    kernel 主体基本不改

输出：
    将 spike / state 从新编号映射回原编号
```

目标是让同一个 32-neuron block 内的 rows：

- fanout 更接近；
- post 访问区域更集中；
- 更适合现有 warp-cooperative atomic；
- 不增加 kernel 内动态调度成本。

---

## 2. 编号映射

```cpp
new_to_old[new_id] = old_id;
old_to_new[old_id] = new_id;
```

约定：

- kernel 内部统一使用 `new_id`；
- 外部 API 仍使用 `old_id`；
- permutation 在 graph 初始化时生成一次。

---

## 3. 需要重排的内容

同步处理：

```text
v
psc
其他逐 neuron 状态
CSR row
CSR post index
external input neuron id
输出 spike/state
```

标量参数不需要重排。

---

## 4. CSR 重建

```cpp
rebuild_csr():

    new_indptr[0] = 0

    for new_pre in [0, N):

        old_pre =
            new_to_old[new_pre]

        row_len =
            old_indptr[old_pre + 1]
            - old_indptr[old_pre]

        new_indptr[new_pre + 1] =
            new_indptr[new_pre]
            + row_len


    for new_pre in [0, N):

        old_pre =
            new_to_old[new_pre]

        old_begin =
            old_indptr[old_pre]

        old_end =
            old_indptr[old_pre + 1]

        new_begin =
            new_indptr[new_pre]

        for offset in [0, old_end-old_begin):

            old_edge =
                old_begin + offset

            new_edge =
                new_begin + offset

            old_post =
                old_indices[old_edge]

            new_indices[new_edge] =
                old_to_new[old_post]

            new_weights[new_edge] =
                old_weights[old_edge]
```

第一版先保留 row 内 edge 顺序。

后续可测试：

```text
每条 row 内按 new_post 排序
```

---

## 5. State、输入与输出

### State

```cpp
for new_id in [0, N):

    old_id =
        new_to_old[new_id]

    new_v[new_id] =
        old_v[old_id]

    new_psc[new_id] =
        old_psc[old_id]
```

如果状态从零初始化，直接按新布局分配。

跨 window 时状态一直保留在新编号空间，不要反复来回转换。

### Sparse input

```cpp
new_input_id =
    old_to_new[old_input_id]
```

尽量在输入预处理阶段转换一次。

### Dense input

```cpp
new_input[..., new_id] =
    old_input[
        ...,
        new_to_old[new_id]
    ]
```

### Event output

```cpp
old_neuron =
    new_to_old[new_neuron]
```

### Dense output / final state

```cpp
old_output[..., old_id] =
    new_output[
        ...,
        old_to_new[old_id]
    ]
```

只在真正返回用户结果时还原。

---

# 6. 重排策略

## 方案 A：全局 fanout 排序

排序键：

```text
fanout ascending
```

优点：

- 最简单；
- block 内 row length 更一致；
- 可作为重排 baseline。

缺点：

- 不保证 post locality；
- 可能破坏原始局部性。

---

## 方案 B：局部 fanout 排序

先按旧编号划 region：

```text
REGION_SIZE = 128 / 256 / 512
```

只在 region 内排序：

```cpp
for region:
    sort neurons by fanout
```

优点：

- 保留部分原始局部性；
- 工程简单；
- 通常比全局排序更稳。

---

## 方案 C：按主要 post tile 排序

将 post 空间切 tile：

```text
POST_TILE_SIZE = 64 / 128 / 256
```

每条 row 统计：

```cpp
primary_tile(pre) =
    argmax_tile edge_count(pre, tile)
```

排序键：

```text
(primary_tile, fanout)
```

优点：

- 同 block rows 更可能访问相近 post；
- 可能改善 cache locality；
- 可能提高 cooperative atomic 的局部性。

缺点：

- 只保留主 tile，信息较粗。

---

## 方案 D：Top-2 tile signature

每条 row 记录：

```text
top1_tile
top2_tile
fanout
```

排序键：

```text
(top1_tile, top2_tile, fanout)
```

优点：

- 比完整 histogram 聚类简单；
- 比只看 primary tile 更准确；
- 适合作为主要 post-aware 方法。

---

## 方案 E：局部 histogram greedy packing

每条 row 构造 post-tile histogram：

```text
hist[post_tile] = edge count
```

相似度可使用：

```text
weighted overlap =
    sum min(hist_i, hist_j)
    /
    sum max(hist_i, hist_j)
```

在每个 region 内：

```cpp
while unassigned neurons remain:

    choose seed

    block = {seed}

    repeatedly add neuron
        most similar to block profile

    until block size == 32
```

优点：

- 直接针对 32-neuron block；
- 能显式提高 block 内 post overlap。

缺点：

- 预处理更复杂；
- 需要限制候选范围。

只建议在局部 region 内使用。

---

## 方案 F：图划分 / 聚类

可作为上限实验：

```text
METIS-style partition
spectral clustering
community detection
```

目标 cluster size：

```text
32 或 64
```

不建议优先实现，成本过高。

---

# 7. 推荐实验矩阵

```text
R0: 不重排

R1: 全局 fanout 排序

R2: 局部 fanout 排序
    region = 128 / 256 / 512

R3: 全局 primary-post-tile 排序
    tile = 64 / 128 / 256

R4: 局部 primary-tile + fanout
    region = 256
    tile = 128

R5: 局部 top2-tile signature

R6: 局部 histogram greedy packing
```

优先实现：

```text
R2 → R4 → R5
```

---

# 8. 推荐第一实现

```text
REGION_SIZE = 256
POST_TILE_SIZE = 128

排序键：
    primary_post_tile
    fanout
```

步骤：

```cpp
for each neuron:

    degree =
        indptr[n+1] - indptr[n]

    count edges by post tile

    primary_tile =
        tile with maximum count


for each 256-neuron region:

    sort by:
        primary_tile
        fanout


concatenate all regions

build new_to_old
build old_to_new
rebuild CSR
```

该方案同时考虑：

- 保留部分原始局部性；
- 提高 post locality；
- 控制 block 内 fanout 差异。

---

# 9. Row 内 edge 排序实验

对最佳 neuron permutation 测试：

```text
E0: 保持原 edge 顺序
E1: 每条 row 按 new_post 排序
```

E1 可能改善：

- 相邻 post 地址；
- cache line locality；
- cooperative atomic 的地址集中度。

需要接受少量浮点累加顺序变化，并继续使用现有误差容限。

---

# 10. 工程结构

建议新增：

```cpp
GraphReorderPlan {
    new_to_old
    old_to_new
    reordered_indptr
    reordered_indices
    reordered_weights
}
```

接口：

```cpp
plan =
    build_reorder_plan(
        graph,
        method,
        region_size,
        post_tile_size
    )
```

运行：

```cpp
run_persistent(
    plan.reordered_graph,
    reordered_state,
    reordered_input
)
```

输出：

```cpp
restore_output(
    output,
    plan.new_to_old
)
```

排序与 CSR 重建逻辑不要放进 persistent kernel wrapper 的热路径。

---

# 11. 正确性检查

必须验证：

```text
old_to_new 与 new_to_old 互逆
nnz 不变
每条 row degree 不变
weight 不变
每条旧边都映射为对应新边
forward 误差 < 1e-5 或 1e-4
```

建议先用随机 permutation 测试整套映射和 CSR 重建，确认正确后再接入具体重排算法。

---

# 12. Benchmark

## 预处理

```text
permutation time
CSR rebuild time
row-sort time
额外内存
```

## Kernel

```text
cell update time
recurrent fanout time
total kernel time
L2 hit rate
Long Scoreboard
atomic throughput
warp execution efficiency
```

## 布局质量

每个 32-neuron block 统计：

```text
fanout variance
unique post tiles
dominant tile ratio
unique posts / total edges
平均 post 距离
```

## End-to-end

```text
input remap time
kernel time
output restore time
total inference time
```

---

# 13. 决策标准

保留一个方案需要满足：

```text
kernel 时间稳定下降
加上输出还原后仍有收益
显存增长可接受
预处理可被多时间步摊销
多个 spike rate / batch / graph 上有效
```

如果：

```text
recurrent 加速
但 cell update 变慢
```

说明排列改善了 CSR，却破坏了 state locality。

如果：

```text
kernel 加速
但 output restore 抵消收益
```

应让 state 长期保持新编号，只在最终输出时转换。

---

# 14. 实施顺序

## Step 1：Permutation 基础框架

实现：

```text
new_to_old
old_to_new
CSR rebuild
state reorder
output restore
```

先用随机 permutation 做正确性测试。

## Step 2：Fanout baseline

实现：

```text
全局 fanout
局部 fanout
```

## Step 3：Post-tile 排序

实现：

```text
primary tile
primary tile + fanout
```

## Step 4：Row 内 post 排序

比较：

```text
保持原顺序
按 new_post 排序
```

## Step 5：更强相似性聚合

仅在简单 post-aware 排序已有收益后实现：

```text
top2 tile signature
局部 histogram greedy packing
```

---

# 15. 最小可运行版本

第一版重排只需要：

```text
1. 计算每条 row 的 fanout 和 primary_post_tile

2. 每 256 个 neuron 内按：
   primary_post_tile
   fanout
   排序

3. 构造 permutation

4. 重建 CSR row 和 post index

5. state / input 转到新编号

6. persistent kernel 不修改

7. 输出 spike id 转回旧编号
```

该版本足以验证：

> 物理重排能否通过改善 block 内 post locality 和 fanout 同质性，为现有 spike-block kernel 带来额外收益。
