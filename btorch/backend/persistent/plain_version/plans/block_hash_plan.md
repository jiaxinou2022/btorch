# Persistent Block Hash Aggregation 实施计划

## 1. 目标

当前V4使用：

```text
32-neuron spike block
→ 按active-edge budget切成多个logical task
→ 每个logical task由一个warp独立处理
```

推荐配置为：

```text
block edge budget = 256
long segment size = 512
```

现有32-edge tile reduce的问题是聚合范围太小：

```text
logical task 256 edges
    ├── tile 0: 32 edges
    ├── tile 1: 32 edges
    ...
    └── tile 7: 32 edges
```

若相同post分别出现在不同tile中，`match_any`完全无法合并。

下一版目标是：

```text
在更大的edge范围内建立warp-private或CTA-private hash
→ 同一post的weight局部累加
→ 最后每个unique post只执行一次global atomic
```

重点扫描以下聚合尺度：

```text
32 edges       现有tile基线
64 edges
128 edges
256 edges      一个logical task
512 edges      两个logical tasks或较大task
original block 一个32-neuron spike block的全部active edges
```

由于 `match_any` 已确认固定开销过高，后续reduce路径不再使用它。

---

# 2. 第一阶段：先统计各尺度的理论收益

在正式实现hash前，先增加stats build，对每个原始spike block记录不同尺度下的unique post数量。

## 2.1 统计目标

对于一个原始BlockTask的logical edge stream：

```text
e0, e1, e2, ..., eN-1
```

分别按以下窗口切分：

```text
G = 32, 64, 128, 256, 512, full block
```

每个窗口统计：

```text
edge_count
unique_post_count
duplicate_count = edge_count - unique_post_count
```

定义理论atomic比例：

[
R_G
===

\frac{\sum_{\text{windows}}N_{\text{unique post}}}
{\sum_{\text{windows}}N_{\text{edges}}}
]

理论可减少atomic比例：

[
S_G = 1-R_G
]

## 2.2 必须输出的结果

```text
aggregation scale
covered edges
unique posts
ideal global atomics
ideal atomic reduction
P50/P90/P99 edges/unique-post ratio
high-duplicate task coverage
```

特别统计：

```text
ratio >= 1.1 的edge占比
ratio >= 1.25 的edge占比
ratio >= 1.5 的edge占比
ratio >= 2.0 的edge占比
```

## 2.3 决策标准

如果：

```text
256-edge尺度理想atomic减少 < 3%
```

则warp-private logical-task hash大概率不值得实现。

如果：

```text
256-edge尺度减少 >= 8%–10%
```

值得测试warp-private hash。

如果：

```text
256-edge减少很小
但full-block减少明显
```

则重复主要跨logical task，需要研究原始BlockTask级协作，而不是单task hash。

如果：

```text
full-block仍不足5%
```

则当前相似性算法对实际post集合重合仍不够，应优先改聚合算法，而不是继续hash工程。

---

# 3. Hash聚合的基本设计

## 3.1 第一版使用warp-private shared hash

每个warp处理一个logical task，并拥有独立hash区域：

```text
hash_keys[warp_in_cta][HASH_SIZE]
hash_values[warp_in_cta][HASH_SIZE]
hash_tags[warp_in_cta][HASH_SIZE]
```

每条edge执行：

```text
post → hash slot
slot为空：
    插入post和weight
slot相同post：
    累加weight
probe失败：
    fallback到global atomic
```

task结束后，warp遍历hash表，将每个有效entry写回一次。

## 3.2 不使用固定全表清空

原有hash每task初始化整张表，固定成本过高。

下一版使用epoch/tag：

```cpp
hash_tags[slot] == current_epoch
```

表示slot属于当前task。

新task只增加epoch，不需要清空所有key/value。

需要处理epoch溢出，第一版可使用32-bit epoch，实际测试中不会溢出。

---

# 4. Warp-private hash伪代码

## 4.1 数据结构

```cpp
template<int HashSize>
struct WarpHashStorage {
    int* keys;
    float* values;
    uint32_t* tags;
};
```

shared memory布局：

```cpp
__shared__ int hash_keys[WARPS_PER_BLOCK][HASH_SIZE];
__shared__ float hash_values[WARPS_PER_BLOCK][HASH_SIZE];
__shared__ uint32_t hash_tags[WARPS_PER_BLOCK][HASH_SIZE];
__shared__ uint32_t warp_epochs[WARPS_PER_BLOCK];
```

每个warp只访问自己的行，不需要CTA-wide barrier。

---

## 4.2 插入和累加

```cpp
template<int HashSize, int MaxProbe>
__device__ bool warp_hash_accumulate(
    int post,
    float weight,
    int* keys,
    float* values,
    uint32_t* tags,
    uint32_t epoch)
{
    uint32_t slot = hash(post) & (HashSize - 1);

    #pragma unroll
    for (int probe = 0; probe < MaxProbe; ++probe) {
        uint32_t current_tag =
            atomicAdd(&tags[slot], 0);

        if (current_tag != epoch) {
            uint32_t old_tag =
                atomicCAS(
                    &tags[slot],
                    current_tag,
                    epoch);

            if (old_tag == current_tag) {
                keys[slot] = post;
                values[slot] = weight;
                return true;
            }
        }

        if (tags[slot] == epoch
            && keys[slot] == post)
        {
            atomicAdd(
                &values[slot],
                weight);
            return true;
        }

        slot =
            (slot + 1) & (HashSize - 1);
    }

    return false;
}
```

这里需要注意发布顺序：

```text
tag生效
→ key/value写入
```

可能让其他lane看到tag已更新但key尚未写完。

更稳妥的第一版应将key作为占位状态，而不是单独先更新tag：

```text
key slot:
    EMPTY
    LOCKED
    valid post
```

epoch用于避免全表清空，但插入时仍需要一个临时锁状态。

---

## 4.3 更稳健的key/tag方案

每个slot保存一个64-bit packed key：

```text
高32位：epoch
低32位：post
```

空slot是：

```text
epoch != current_epoch
```

插入使用一次64-bit CAS：

```cpp
uint64_t expected =
    packed_keys[slot];

uint64_t desired =
    pack(epoch, post);

if (epoch_of(expected) != epoch) {
    old = atomicCAS(
        &packed_keys[slot],
        expected,
        desired);

    if (old == expected) {
        values[slot] = weight;
        return true;
    }
}
```

但仍存在value初始化与并发累加顺序问题。

最简单且正确的第一版可采用：

```text
1. 一个warp先分批把本轮edge插入key
2. __syncwarp
3. 再累加value
```

或者使用三状态slot：

```text
EMPTY_FOR_EPOCH
INITIALIZING
READY
```

---

# 5. 推荐的两阶段hash插入

为避免复杂发布竞争，建议第一版按32-edge批次执行两阶段操作，但聚合状态跨所有批次保留。

## 阶段A：建立key

每个lane尝试寻找或插入post：

```cpp
slot = find_or_claim_key(post);
edge_slot[lane] = slot;
```

插入新key时：

```cpp
atomicCAS(key, EMPTY, post)
```

同一warp内多个lane可能竞争，但最终都会找到相同post对应的slot。

随后：

```cpp
__syncwarp();
```

## 阶段B：累加value

```cpp
atomicAdd(
    &hash_values[slot],
    weight);
```

然后处理下一批32条edge。

整个logical task完成后才flush。

---

## 5.1 Hash初始化

对于warp-private hash，第一版可先使用lane协作初始化，而不是立即实现epoch。

```cpp
for (slot = lane;
     slot < HASH_SIZE;
     slot += 32)
{
    keys[slot] = EMPTY;
    values[slot] = 0;
}

__syncwarp();
```

虽然有固定成本，但更容易验证正确性和hash收益。

在确认hash有潜在收益后，再加入epoch/tag优化。

这样可以避免一次同时引入：

* hash聚合逻辑；
* epoch发布协议；
  -复杂状态竞争；

导致难以调试。

---

# 6. 基础warp hash伪代码

```cpp
template<int HashSize, int MaxProbe>
__device__ void process_logical_task_hash(
    LogicalTask task,
    ...)
{
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    int* keys =
        &shared_hash_keys[warp][0];

    float* values =
        &shared_hash_values[warp][0];

    // --------------------------------
    // 1. 初始化
    // --------------------------------

    for (int slot = lane;
         slot < HashSize;
         slot += 32)
    {
        keys[slot] = EMPTY_KEY;
        values[slot] = 0.0f;
    }

    __syncwarp();

    // --------------------------------
    // 2. 建立row prefix
    // --------------------------------

    build_row_prefix(task);

    // --------------------------------
    // 3. 遍历logical task的所有edge
    // --------------------------------

    for (int logical_base = task.begin;
         logical_base < task.end;
         logical_base += 32)
    {
        int logical_edge =
            logical_base + lane;

        bool valid =
            logical_edge < task.end;

        int post = -1;
        float weight = 0.0f;

        if (valid) {
            int physical_edge =
                map_logical_to_physical(
                    logical_edge);

            post =
                graph_indices[physical_edge];

            weight =
                graph_weight[physical_edge];
        }

        int slot = -1;
        bool inserted = false;

        if (valid) {
            uint32_t probe_slot =
                hash(post) & (HashSize - 1);

            #pragma unroll
            for (int probe = 0;
                 probe < MaxProbe;
                 ++probe)
            {
                int previous =
                    atomicCAS(
                        &keys[probe_slot],
                        EMPTY_KEY,
                        post);

                if (previous == EMPTY_KEY
                    || previous == post)
                {
                    slot = probe_slot;
                    inserted = true;
                    break;
                }

                probe_slot =
                    (probe_slot + 1)
                    & (HashSize - 1);
            }
        }

        __syncwarp();

        if (valid && inserted) {
            atomicAdd(
                &values[slot],
                weight);
        }

        if (valid && !inserted) {
            // fallback
            atomicAdd(
                &psc[batch * n_neuron + post],
                weight);
        }

        __syncwarp();
    }

    // --------------------------------
    // 4. Flush
    // --------------------------------

    for (int slot = lane;
         slot < HashSize;
         slot += 32)
    {
        int post = keys[slot];

        if (post != EMPTY_KEY) {
            atomicAdd(
                &psc[batch * n_neuron + post],
                values[slot]);
        }
    }
}
```

这版不是最终高性能实现，但逻辑清晰，适合作为hash收益验证基线。

---

# 7. 聚合尺度的候选设计

## 7.1 尺度A：一个logical task

当前推荐budget为256：

```text
一个warp
一个256-edge logical task
一张warp-private hash
```

优点：

* 与现有调度完全兼容；
* 不需CTA协作；
* task成本有上界；
* 不引入新的跨warp同步；
* 最适合第一版。

缺点：

* 无法合并跨logical task重复；
* hash容量可能接近unique post数量。

推荐作为首个实现。

---

## 7.2 尺度B：两个连续logical tasks

一个warp一次领取同一原始BlockTask的两个连续segment：

```text
segment k
segment k+1
```

共聚合最多512 edges。

descriptor需要识别：

```text
是否存在同block的下一segment
```

或者UPDATE阶段直接生成chunk descriptor：

```text
logical_begin
logical_end <= 512
```

优点：

* 捕获更多跨tile和跨segment重复；
* prefix只构建一次；
* 减少task claim和metadata开销。

缺点：

* 单warp工作时间变长；
  -可能重新产生尾部；
  -需要更大hash；
  -对于batch 4可能更差。

该尺度应在256-edge hash有收益后再试。

---

## 7.3 尺度C：完整原始BlockTask

一个warp领取原始32-neuron spike block，然后内部处理全部active edges：

```text
不再按256独立调度
但内部仍可每256边做进度切片
```

优点：

* 最大化同block重复聚合；
* prefix只构建一次；
  -最大程度减少global atomic。

缺点：

* 重新引入原始BlockTask长尾；
* 可能有数千edge；
  -hash容量难以覆盖全部unique post；
  -一个warp长时间占用；
  -与V4 edge-budget的核心收益冲突。

所以不建议直接回到“一warp处理完整block”。

更合理的是让一个CTA领取原始block，多个warp分别处理segment，但共享一个CTA hash。

---

## 7.4 尺度D：CTA级完整BlockTask hash

一个CTA处理一个原始BlockTask：

```text
warp 0处理segment 0
warp 1处理segment 1
...
所有warp写同一CTA hash
最后CTA flush
```

优点：

* 保留edge-budget并行；
* 捕获跨logical task重复；
  -可使用更大shared hash；
  -多个warp共同隐藏L2延迟。

缺点：

* CTA-wide shared atomic冲突；
  -需要`__syncthreads()`；
  -重新引入CTA barrier；
  -BlockTask大小差异会造成CTA尾部；
  -任务数量减少；
  -设计复杂度高。

当前NCU刚刚将barrier stall从35.87%降至18.06%，因此不应立即引入CTA-wide hash。

CTA方案只作为后续上限实验，不应第一轮施工。

---

# 8. Hash尺寸扫描

HashSize应与聚合edge尺度共同扫描。

建议第一轮：

```text
aggregation edges: 128 / 256
hash size:         64 / 128 / 256 / 512
max probe:         4 / 8 / 16
```

但不需要全组合。

## 推荐组合

```text
128 edges:
    hash 128
    hash 256

256 edges:
    hash 128
    hash 256
    hash 512
```

HashSize最好是2的幂。

## Load factor

定义：

[
L=
\frac{N_{\text{unique post}}}
{\text{HashSize}}
]

建议控制在：

```text
L <= 0.5–0.7
```

若256-edge task平均有220个unique post，则：

```text
hash 256过满
hash 512更合适
```

但每warp 512 slots代价很高：

```text
keys:   512 × 4 B = 2 KB
values: 512 × 4 B = 2 KB
合计:   4 KB / warp
```

8 warp/CTA即32 KB shared memory，尚可能接受，但会影响occupancy。

如果加入tag，shared消耗进一步增加。

---

# 9. Shared memory与occupancy评估

每warp hash内存：

[
M_{\text{warp}}
===============

H
\times
(\text{sizeof key}+\text{sizeof value})
]

若key和value都是4字节：

| HashSize | 每warp | 8 warp/CTA |
| -------: | ----: | ---------: |
|       64 | 512 B |       4 KB |
|      128 |  1 KB |       8 KB |
|      256 |  2 KB |      16 KB |
|      512 |  4 KB |      32 KB |

还需加现有shared prefix、metadata和其他buffer。

第一轮优先测试：

```text
HashSize = 128 / 256
```

避免一开始将occupancy压得过低。

必须记录：

```text
active blocks/SM
achieved occupancy
eligible warps
long scoreboard
barrier
```

Hash减少global atomic，但若occupancy从83%明显降到30%–40%，可能得不偿失。

---

# 10. 自适应启用策略

不能对所有task无条件hash。

## 10.1 静态hint

预处理为每个32-neuron block计算：

```text
static edges / static unique posts
```

保存：

```cpp
block_hash_hint[block_id]
```

但静态值可能高估，因为本时间步只有部分row发放。

## 10.2 动态hint

UPDATE阶段已知：

```text
spike_mask
active rows
active edges
```

可以为每个原始block保存轻量级类别：

```text
active edge count
similarity-group id
```

但不能在UPDATE阶段精确统计unique post，否则成本太高。

第一版可使用：

```cpp
enable_hash =
    reordered_similarity_mode
    && task_edges >= HashMinEdges;
```

第二版再用预处理hint。

## 10.3 Fallback策略

若hash probe失败率高，直接global atomic：

```cpp
if (!hash_insert(...)) {
    atomicAdd(global_psc, weight);
}
```

统计：

```text
probe failures
fallback atomics
average probes
occupied slots
flush atomics
```

若fallback过高，说明hash过小或task重复不足。

---

# 11. 实验顺序

## 阶段0：理论上界统计

统计：

```text
32 / 64 / 128 / 256 / 512 / full-block
```

的unique post比例。

选择真正有明显atomic reduction潜力的尺度。

---

## 阶段1：128-edge warp hash

固定：

```text
reorder = global similarity
block budget = 256
long segment = 512
```

但每个logical task内部每128 edges单独建表。

测试：

```text
hash 128
hash 256
direct atomic baseline
```

目的：

* 验证hash框架；
  -控制shared和初始化成本；
  -判断跨4个tile的重复是否足够。

---

## 阶段2：256-edge logical-task hash

一张表覆盖完整logical task。

测试：

```text
hash 128
hash 256
hash 512
probe 4/8
```

重点比较：

```text
input edges
unique slots
global atomics
fallback atomics
hash initialization cost
kernel time
```

---

## 阶段3：初始化优化

只有基础hash减少global atomic并接近或超过baseline时，再实现：

```text
epoch/tag清空
```

或者使用used-slot list：

```text
插入新key时记录slot
flush时只遍历used slots
下一task只清空used slots
```

used-slot list可能比epoch更简单。

### Used-slot方案

插入新key成功时：

```cpp
position =
    atomicAdd(local_used_count, 1);

used_slots[position] =
    slot;
```

task结束只遍历：

```text
used_slots[0:used_count]
```

而不是完整HashSize。

清理时也只清理used slot。

这对unique post远小于HashSize的任务尤其有效。

---

## 阶段4：聚合尺度扩大

只有256-edge hash明显正收益时，测试：

```text
512-edge warp chunk
```

一个warp领取同一原始block的两个连续logical segments，并共用一张表。

若出现明显尾部或occupancy下降，则停止扩大。

---

## 阶段5：决定是否做CTA hash

只有同时满足：

```text
full-block理想atomic reduction显著高于256-edge
且
重复主要跨logical segment
且
256-edge hash已证明reduce本身可赚钱
```

才考虑CTA级共享hash。

否则不要引入CTA barrier。

---

# 12. 推荐的第一版实现

第一版控制变量：

```text
aggregation scale = 256 edges
HashSize = 256
MaxProbe = 8
warp-private
shared memory
full initialization
full table flush
fallback global atomic
```

保留两个路径：

```cpp
if constexpr (kHashMode == Off) {
    direct_atomic();
} else {
    warp_hash_aggregate();
}
```

伪代码总流程：

```cpp
task = claim_logical_task();

build_row_prefix(task);

initialize_warp_hash();

for each edge in logical task:
    post, weight = load_edge();

    if hash_insert_or_accumulate(post, weight) fails:
        atomicAdd(global_psc[post], weight);

flush_hash_to_global();
```

这版不追求最终性能，目标是验证：

```text
256-edge尺度的真实atomic减少量
以及减少量能否覆盖hash成本
```

---

# 13. 验收指标

## Hash功能有效

至少应看到：

```text
global atomics / input edges
显著低于1
```

建议最低目标：

```text
< 0.90
```

即减少10%以上global atomic。

若只有：

```text
0.97–0.99
```

通常不值得继续优化复杂hash。

## Hash性能有效

应看到：

```text
Long Scoreboard下降
L2 atomic sectors下降
issue active提高
kernel时间下降
```

如果global atomic下降但kernel变慢，检查：

* shared atomic冲突；
  -初始化/flush；
* probe长度；
* occupancy；
* barrier；
* fallback。

## 相似性重排有效

比较：

```text
identity + hash
global similarity + hash
```

如果similarity的atomic下降显著更多，并最终更快，才能证明：

```text
相似性聚合
→ hash可合并性提高
→ 性能收益
```

如果二者hash收益接近，则当前重排主要改善任务结构，而不是post重合。

---

# 14. 停止条件

满足任一条件即可停止hash方向：

```text
256-edge理想atomic减少 < 5%
```

或：

```text
实际global atomic减少 < 5%
```

或：

```text
global atomic减少 > 10%
但经过初始化/occupancy优化后仍慢 > 3%
```

或：

```text
相似性重排没有比identity产生更多hash压缩
```

此时应转向：

```text
软件流水
warp specialization
加载与atomic重叠
更好的任务领取
```

而不是继续扩大hash。

---

# 15. 最终推荐施工路线

```text
1. 统计不同aggregation scale的unique-post上界
2. 实现256-edge warp-private hash基线
3. 扫HashSize 128/256/512
4. 扫aggregation scale 128/256
5. 实现used-slot清理/flush
6. 比较identity与global similarity
7. 若256-edge有效，再测试512-edge
8. 只有跨task重复非常高时才考虑CTA hash
```

第一目标不是立刻追求最佳hash，而是回答：

> **当前global similarity产生的重复，究竟是否足以在256-edge logical task内减少至少约10%的global atomic。**
