这个改法可以比较直接地嵌进去，而且不需要动现有 producer/consumer queue 主体。

当前 UPDATE 的核心是：

```cpp
for (int n = update_tid; n < n_neuron; n += update_stride) {
    const float recurrent = psc[n] + delta_read[n];
    ...
    v[n] = ...;
    psc[n] = ...;
    ...
}
```

也就是所有 UPDATE CTA 用 grid-stride 方式遍历全体 neuron；而 consumer 部分已经天然支持“UPDATE block 完成后转 helper”。所以这次施工的核心是：

> **只重构 UPDATE 的 neuron ownership 和 state access，不改 propagation queue。**
>
> 将 UPDATE block 一开始静态绑定到连续 neuron 区间；每 timestep 先把这一整段 state 批量搬入 dynamic shared memory，在 shared/register 中完成 UPDATE，再集中写回。写回完成后，这块 shared scratch 可以留给后续 propagation aggregation 使用。当前版本暂时只验证 UPDATE staging 本身，不马上实现 hash reuse。

下面给具体计划。

---

# 一、第一阶段目标：先只验证 static neuron partition + shared staging

本轮不要同时加：

* shared hash aggregation；
* 新 queue；
* TMA；
* cp.async；
* propagation 重构。

先回答一个问题：

> **把 UPDATE 的 dense global-memory traffic 压缩到“开头 preload + 结尾 writeback”后，pipeline 下 UPDATE slowdown 和 propagation hidden 是否改善？**

当前 propagation、binning、ticket consumer 全部原样保留。

---

# 二、先改 neuron ownership

当前代码：

```cpp
const int update_tid =
    blockIdx.x * blockDim.x + threadIdx.x;

const int update_stride =
    update_block_count * blockDim.x;
```

然后：

```cpp
for (int n = update_tid;
     n < n_neuron;
     n += update_stride)
```

改为 block-level contiguous partition。

## 2.1 静态分区

```cpp
const int update_block_rank =
    blockIdx.x;

const int neurons_per_update_block =
    (n_neuron +
     update_block_count - 1) /
    update_block_count;

const int neuron_begin =
    update_block_rank *
    neurons_per_update_block;

const int neuron_end =
    min(
        neuron_begin +
        neurons_per_update_block,
        n_neuron);

const int local_neuron_count =
    max(0,
        neuron_end - neuron_begin);
```

于是：

```text
CTA 0 -> [0, K)
CTA 1 -> [K, 2K)
CTA 2 -> [2K, 3K)
...
```

不需要任何 UPDATE work counter。

---

# 三、先计算 shared-memory footprint

当前 UPDATE 每 neuron 用到：

```text
v
psc
delta_read
input_current
```

但是没有必要全部驻留。

最朴素版本可以先存：

```text
sh_v
sh_recurrent
sh_current
```

因为：

```cpp
recurrent =
    psc + delta_read;

current =
    recurrent + input_current;
```

`delta_read` 和 `input_current` 在 preload 时就已经消费完。

这样 shared footprint：

```text
12 bytes / neuron
```

第一版建议就是这个布局。

```cpp
extern __shared__ unsigned char smem[];

float* sh_v =
    reinterpret_cast<float*>(smem);

float* sh_recurrent =
    sh_v + neurons_per_update_block;

float* sh_current =
    sh_recurrent +
    neurons_per_update_block;
```

需要：

```cpp
shared_bytes =
    3 *
    neurons_per_update_block *
    sizeof(float);
```

---

# 四、为什么第一版不建议只存 `v + recurrent`

理论上 `current` 也可以即时计算后留 register。

但一个 thread 可能负责多个 neuron：

```cpp
local =
    threadIdx.x,
    threadIdx.x + blockDim.x,
    ...
```

如果想让所有 global reads 完成后再进入 compute phase，就必须把每个 neuron 的 current 暂存下来。

否则：

```cpp
read current
→ immediately compute
```

又会把 load 和 compute 混回来。

因此为了真正做成：

```text
LOAD ALL
→ COMPUTE ALL
→ STORE ALL
```

第一版存：

```text
v
recurrent
current
```

最干净。

---

# 五、UPDATE Phase A：整段 preload

替换当前 UPDATE loop 的前半部分。

```cpp
if (is_update_block) {

    for (int local = threadIdx.x;
         local < local_neuron_count;
         local += blockDim.x) {

        const int n =
            neuron_begin + local;

        const float recurrent =
            psc[n] +
            delta_read[n];

        const float current =
            recurrent +
            input_current[n];

        sh_v[local] =
            v[n];

        sh_recurrent[local] =
            recurrent;

        sh_current[local] =
            current;

        // dense input state consumed
        delta_read[n] = 0.0f;
        input_current[n] = 0.0f;
    }

    __syncthreads();
```

注意这里仍然有：

```text
load:
v
psc
delta
input

store:
delta=0
input=0
```

所以 preload phase 本身是 memory-heavy。

但 preload 之后，直到最终 writeback 前，dense neuron state不再访问 global。

---

# 六、UPDATE Phase B：纯 shared/register compute

```cpp
for (int local = threadIdx.x;
     local < local_neuron_count;
     local += blockDim.x) {

    const int n =
        neuron_begin + local;

    const float v_old =
        sh_v[local];

    const float recurrent =
        sh_recurrent[local];

    const float current =
        sh_current[local];

    const float v_pre =
        v_old +
        dt * (
            -(v_old - v_reset) /
                tau_mem +
            current / c_m);

    const bool fired =
        v_pre >= v_threshold;

    const float spike =
        fired ? 1.0f : 0.0f;

    sh_v[local] =
        v_pre -
        reset_delta * spike;

    sh_recurrent[local] =
        recurrent * decay;

    ...
}
```

这里 `sh_current` 后面已经没用了。

---

# 七、Spike publication 仍放在 compute phase

不要为了集中 writeback 把 spike publication 拖到最后。

当前你已经知道 task publication 是前置的，所以应继续保持：

```cpp
if (fired) {
    publish task immediately;
}
```

也就是说 Phase B：

```text
shared neuron compute
+
sparse global task publish
```

而不是完全 global-free。

这是正确的，因为 pipeline 的价值依赖 propagation 尽早拿到工作。

伪代码：

```cpp
if (fired) {
    const int row_start =
        graph_indptr[n];

    const int row_end =
        graph_indptr[n + 1];

    const int fanout =
        row_end - row_start;

    publish_current_logic(
        n,
        row_start,
        row_end,
        fanout,
        expected_epoch);
}
```

这里直接复用你当前：

```cpp
high_task
low_task
queue_tail
state_store_release
```

整段代码，不改。

---

# 八、dense spike 输出也留在 compute phase

当前：

```cpp
if constexpr (ReturnDense) {
    dense_spikes[...] = spike;
}
```

这个 global store 也会打破“完全 memory-free”。

第一版先保持正确性。

但测试时最好用：

```text
return_dense = false
```

或者分别比较：

```text
dense off
dense on
```

这样能更清楚看到 neuron-state staging本身的效果。

如果你的 production benchmark必须 dense，则最终当然还要算进去。

---

# 九、UPDATE Phase C：集中 writeback

compute 全部结束后：

```cpp
__syncthreads();
```

然后：

```cpp
for (int local = threadIdx.x;
     local < local_neuron_count;
     local += blockDim.x) {

    const int n =
        neuron_begin + local;

    v[n] =
        sh_v[local];

    psc[n] =
        sh_recurrent[local];
}
```

这样 UPDATE dense state traffic 的宏观形态就是：

```text
preload
██████

compute + publish
      █████████████

writeback
                   ████
```

而 propagation dedicated consumers 从最早 task出现以后一直持续执行。

---

# 十、UPDATE block 完成后 shared scratch 可复用

写回完成：

```cpp
__syncthreads();
```

然后：

```cpp
if (threadIdx.x == 0) {
    atomicAdd(
        update_done_blocks,
        1);
}

__syncthreads();
```

此时：

```text
sh_v
sh_recurrent
sh_current
```

都已经 dead。

所以逻辑上：

```cpp
HashEntry* helper_hash =
    reinterpret_cast<HashEntry*>(
        smem);
```

是安全的。

本轮先不要真的实现 hash，只保留这个结构。

以后 Phase 2 可以：

```text
same shared scratch

UPDATE:
[v | recurrent | current]

↓

helper PROP:
[hash table]
```

---

# 十一、当前 kernel 最大的工程改动：dynamic shared launch

现在 launcher 是：

```cpp
cudaLaunchCooperativeKernel(
    kernel,
    grid_dim,
    block_dim,
    args,
    0,
    stream);
```

第四个参数目前是：

```text
0
```

要改成：

```cpp
const size_t update_shared_bytes =
    static_cast<size_t>(
        neurons_per_update_block) *
    3 *
    sizeof(float);
```

但 launcher 当前不知道：

```text
neurons_per_update_block
```

所以 host/launcher 侧需要根据：

```cpp
n_neuron
update_block_count
```

计算：

```cpp
const int neurons_per_update_block =
    (n_neuron +
     update_block_count - 1) /
    update_block_count;
```

然后：

```cpp
const size_t shared_bytes =
    neurons_per_update_block *
    3 *
    sizeof(float);
```

launch：

```cpp
cudaLaunchCooperativeKernel(
    kernel,
    grid_dim,
    block_dim,
    args,
    shared_bytes,
    stream);
```

---

# 十二、必须同步修改 occupancy 计算

这是最容易漏掉的工程问题。

当前：

```cpp
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &active_blocks,
    kernel,
    block_dim,
    0);
```

仍然用：

```text
dynamic smem = 0
```

如果 kernel实际用了 32KB dynamic smem，这里计算出的 cooperative grid会严重错误。

因此必须改成：

```cpp
int persistent_snn_max_active_blocks_per_sm(
    int block_dim,
    bool return_dense,
    bool return_events,
    size_t dynamic_smem_bytes);
```

然后：

```cpp
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &active_blocks,
    kernel,
    block_dim,
    dynamic_smem_bytes);
```

---

# 十三、这里会出现一个 chicken-and-egg 问题

因为：

```text
update_block_count
依赖 grid_dim

shared_bytes
依赖 update_block_count

grid_dim
又依赖 occupancy(shared_bytes)
```

所以不能再直接：

```text
先 occupancy
→ grid_dim
→ update_block_count
```

需要改成迭代或候选搜索。

这是这次改造里最重要的 host-side设计点。

---

# 十四、推荐的解决方法：host 侧搜索可行 grid

不要搞复杂 fixed-point。

直接枚举 resident block 数即可。

假设 GPU：

```text
SM_count = S
```

候选：

```cpp
for (int blocks_per_sm =
         max_blocks_without_smem;
     blocks_per_sm >= 1;
     --blocks_per_sm) {

    grid_dim =
        blocks_per_sm *
        sm_count;

    update_blocks =
        pipeline_update_block_count(
            grid_dim);

    neurons_per_block =
        ceil_div(
            n_neuron,
            update_blocks);

    shared_bytes =
        neurons_per_block *
        bytes_per_neuron;

    actual_active =
        occupancy(
            block_dim,
            shared_bytes);

    if (actual_active >=
        blocks_per_sm) {

        accept;
        break;
    }
}
```

这样能得到一个自洽：

```text
grid_dim
update_block_count
neurons_per_update_block
shared_bytes
```

组合。

---

# 十五、还要加 shared-memory 上限保护

建议增加环境变量：

```text
BTORCH_PIPELINE_UPDATE_SMEM_KB
```

例如默认：

```text
32 KB
```

或者：

```text
48 KB
```

不要直接让：

```text
N / update_blocks
```

决定任意大的 shared。

检查：

```cpp
required_shared_bytes
<=
update_smem_limit
```

若超出，则：

```text
当前配置不可用
```

或降低 neurons per CTA 的设计。

---

# 十六、不过“所有 neuron 必须一开始分配完”意味着一个硬约束

你已经明确要求：

> 每个 CTA 从一开始分到连续 neuron，而且所有 neuron从一开始分配给所有 UPDATE CTA。

那就意味着：

```text
update_block_count
*
max_neurons_per_cta
>=
n_neuron
```

如果不满足：

```text
该 role ratio / occupancy 配置无效
```

不能临时让 CTA 后续再领第二 tile，否则破坏这个设计。

所以 host 侧候选合法条件：

```cpp
const int neurons_per_block =
    ceil_div(
        n_neuron,
        update_blocks);

const size_t shared_bytes =
    neurons_per_block *
    bytes_per_neuron;

if (shared_bytes >
    max_shared_budget) {
    reject;
}
```

---

# 十七、第一版 bytes_per_neuron 建议固定 12 B

也就是：

```text
v          4
recurrent  4
current    4
------------
12 B
```

不要一开始追极限压缩。

测试稳定后再考虑：

### 版本 A

```text
12 B/neuron
```

### 版本 B

尝试把 current 留 register / 重算：

```text
8 B/neuron
```

但 8 B 版本可能重新引入访存相位不纯或 register pressure。

所以第二阶段再做。

---

# 十八、正确性上最需要注意：delta 清零必须在 preload 完成

当前：

```cpp
recurrent =
    psc[n] +
    delta_read[n];

delta_read[n] = 0;
```

这个语义必须保留。

因为 propagation 当前 timestep写的是：

```cpp
delta_write
```

不会碰：

```cpp
delta_read
```

所以 preload 时集中读取并清零是安全的。

这恰好是双缓冲设计帮了你。

---

# 十九、input_current 也是同样逻辑

外部事件注入后已有：

```cpp
grid.sync();
```

所以 UPDATE preload 时：

```cpp
current =
    recurrent +
    input_current[n];

input_current[n] = 0;
```

也是安全的。

之后 propagation不会碰 `input_current`。

---

# 二十、不要给 preload/writeback 加 grid.sync

这一点非常重要。

只用：

```cpp
__syncthreads();
```

不要让所有 UPDATE CTA 同步：

```text
全 grid preload
→ grid sync
→ compute
→ grid sync
→ all writeback
```

否则：

* 所有 CTA 同时打 global memory；
* dedicated propagation 也被强制等；
* pipeline直接被破坏。

正确结构是各 CTA 独立：

```text
CTA0:
LOAD → COMPUTE → STORE

CTA1:
   LOAD → COMPUTE → STORE

CTA2:
      LOAD → COMPUTE → STORE
```

自然错相。

---

# 二十一、第一轮 benchmark 设计

先只比较两版：

```text
A. current grid-stride UPDATE
B. static shared-resident UPDATE
```

固定：

```text
role ratio = 7:1
dedicated warp = 1
helper warp = 1
ticket chunk = 1
static waves = 0
```

因为这是你当前最终默认配置。此前 7:1 是当前 role scan 中的稳定选择。

---

# 二十二、必须记录的性能指标

至少：

| Metric          | Baseline | Shared staging |
| --------------- | -------: | -------------: |
| Serial UPDATE   |          |                |
| Pipeline UPDATE |          |                |
| UPDATE slowdown |          |                |
| PROP tail       |          |                |
| PROP hidden     |          |                |
| Exchange ratio  |          |                |
| Pipeline total  |          |                |
| Core kernel     |          |                |
| Blocks/SM       |          |                |
| Shared/CTA      |          |                |

最重要不是：

```text
Serial UPDATE 是否更快
```

而是：

```text
Pipeline UPDATE slowdown 是否下降
```

以及：

```text
PROP hidden 是否增加
```

---

# 二十三、预期的三种结果

### 情况 A：最理想

```text
Serial UPDATE:
20.99 → 22 us

Pipeline UPDATE:
25.6 → 23 us

PROP tail:
68.6 → 60 us
```

这说明：

> staging 单独略有成本，但成功降低了 concurrent interference。

这是这条路线真正想要的结果。

---

### 情况 B：UPDATE 单独和 pipeline 都变慢

例如：

```text
Serial UPDATE:
21 → 28

Pipeline UPDATE:
25.6 → 31

PROP tail:
基本不变
```

说明：

> 当前 LIF compute window 太短，shared staging / barrier 成本大于时间解耦收益。

这时不要继续堆 shared residency。

---

### 情况 C：UPDATE slowdown降低，但 total没明显改善

例如：

```text
UPDATE slowdown:
4.6 → 1.5

PROP hidden:
1.8 → 2.0
```

这说明：

> UPDATE interference确实被削弱了，但 propagation本身仍然被 RED/L2限制。

这时候就非常适合再叠加：

```text
shared hash aggregation
```

因为 staging 已经证明资源正交性改善，只差 propagation backend。

---

# 二十四、第二阶段才做 shared scratch reuse

如果第一阶段成立，再扩展：

```cpp
extern __shared__ unsigned char smem[];
```

UPDATE 完成后：

```cpp
HashEntry* hash =
    reinterpret_cast<HashEntry*>(
        smem);
```

然后 helper consumer 使用：

```cpp
process_fragment_with_hash(
    ...,
    hash);
```

shared footprint计算：

```cpp
shared_bytes =
    max(
        update_state_bytes,
        propagation_hash_bytes);
```

而不是：

```cpp
update_state_bytes +
propagation_hash_bytes
```

这时才真正把两个优化合并。

---

# 二十五、代码结构建议

当前 kernel已经很长，建议不要把三阶段全部塞在主循环里。

虽然之前你对 device function overhead比较谨慎，但 `__forceinline__` 后不会产生实际 call overhead。

可以拆：

```cpp
__device__ __forceinline__
void preload_update_state(...);

__device__ __forceinline__
void compute_update_state_and_publish(...);

__device__ __forceinline__
void writeback_update_state(...);
```

如果你仍然希望避免函数拆分，至少用明确 section 注释：

```cpp
// --------------------------------------------------
// UPDATE PHASE A: PRELOAD STATIC NEURON PARTITION
// --------------------------------------------------

// --------------------------------------------------
// UPDATE PHASE B: SHARED-RESIDENT COMPUTE + PUBLISH
// --------------------------------------------------

// --------------------------------------------------
// UPDATE PHASE C: CONCENTRATED WRITEBACK
// --------------------------------------------------
```

方便后面 NCU 对应 SASS/source。

---

# 二十六、完整伪代码

最终第一版 kernel 主体可以变成：

```cpp
extern __shared__
unsigned char shared_storage[];

float* sh_v =
    reinterpret_cast<float*>(
        shared_storage);

float* sh_recurrent =
    sh_v +
    neurons_per_update_block;

float* sh_current =
    sh_recurrent +
    neurons_per_update_block;

for (int t = 0;
     t < t_steps;
     ++t) {

    reset_queue();

    grid.sync();

    inject_external_events();

    grid.sync();

    if (is_update_block) {

        // =========================================
        // Static ownership
        // =========================================

        const int begin =
            blockIdx.x *
            neurons_per_update_block;

        const int end =
            min(
                begin +
                neurons_per_update_block,
                n_neuron);

        const int count =
            max(0, end - begin);

        // =========================================
        // Phase A: global preload
        // =========================================

        for (int local =
                 threadIdx.x;
             local < count;
             local += blockDim.x) {

            const int n =
                begin + local;

            const float recurrent =
                psc[n] +
                delta_read[n];

            const float current =
                recurrent +
                input_current[n];

            sh_v[local] =
                v[n];

            sh_recurrent[local] =
                recurrent;

            sh_current[local] =
                current;

            delta_read[n] =
                0.0f;

            input_current[n] =
                0.0f;
        }

        __syncthreads();

        // =========================================
        // Phase B: shared/register UPDATE
        //          sparse task publishing remains
        // =========================================

        for (int local =
                 threadIdx.x;
             local < count;
             local += blockDim.x) {

            const int n =
                begin + local;

            float v_old =
                sh_v[local];

            float recurrent =
                sh_recurrent[local];

            float current =
                sh_current[local];

            float v_pre =
                update_voltage(
                    v_old,
                    current);

            bool fired =
                v_pre >=
                v_threshold;

            sh_v[local] =
                fired
                    ? v_pre -
                        reset_delta
                    : v_pre;

            sh_recurrent[local] =
                recurrent *
                decay;

            if constexpr (
                ReturnDense) {
                dense_spikes[
                    t * n_neuron +
                    n] =
                    fired ? 1.0f : 0.0f;
            }

            if (fired) {
                publish_existing_pipeline_tasks(
                    n,
                    expected_epoch,
                    ...);
            }
        }

        __syncthreads();

        // =========================================
        // Phase C: concentrated global writeback
        // =========================================

        for (int local =
                 threadIdx.x;
             local < count;
             local += blockDim.x) {

            const int n =
                begin + local;

            v[n] =
                sh_v[local];

            psc[n] =
                sh_recurrent[local];
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            atomicAdd(
                update_done_blocks,
                1);
        }

        __syncthreads();

        // shared_storage is dead as neuron state
        // and may later become helper hash scratch.
    }

    // =============================================
    // Existing propagation consumer
    // unchanged in first implementation
    // =============================================

    if (active_consumer) {
        run_existing_consumer();
    }

    __syncthreads();
    grid.sync();
}
```

---

## 推荐实际施工顺序

1. **只改静态连续 neuron ownership，不用 shared。**
   确认结果完全正确，并看单纯 ownership 是否影响性能。

2. **加入 12 B/neuron dynamic shared preload/compute/writeback。**
   同步修改 cooperative occupancy 计算。

3. **重新测 serial UPDATE 与 pipeline interference。**

4. **扫 shared budget / neurons-per-CTA。**
   建议至少看 `8/16/24/32/48 KB` 对应的 occupancy 和性能，而不是只测一个点。

5. 若 staging 能降低 concurrent interference，再实现：

   ```text
   shared state → shared hash reuse
   ```

这个顺序能很好地区分：到底是**连续 ownership**有效，还是**访存相位化**有效，还是最后必须和 propagation aggregation 组合后才有效。
