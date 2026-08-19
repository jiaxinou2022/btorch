# UPDATE 参数跨时间步 Shared-Memory 常驻优化执行计划

## 1. 优化目标

在现有 `persistent_snn_spike_block_kernel` 基础上，对 UPDATE 阶段做一次尽量局部的收尾优化：

1. 将前 `update_block_count` 个 CTA 固定为 UPDATE CTA；
2. 将全部 neuron 按 **连续大块**静态分配给这些 CTA；
3. 每个 UPDATE CTA 在 kernel 开始时一次性将自己负责 neuron 的可常驻状态加载到 shared memory；
4. 整个 `t_steps` 循环中保持这些状态驻留，不再每 timestep 从 global memory 重新读取/写回；
5. kernel 结束后统一写回 global memory；
6. propagation、spike-block task queue、atomic accumulation 等现有路径保持不变。

现有 host 端已经按 cooperative occupancy 计算 persistent grid，并将整个时间循环放在一次 kernel launch 内，因此具备跨 timestep 保持 CTA-local shared state 的基本条件。

---

# 2. 总体设计

## 2.1 UPDATE CTA 静态划分

直接使用连续大块划分，不要求按 32-neuron block 对齐：

```cpp
const bool is_update_block =
    blockIdx.x < update_block_count;

// 上取整，每个 UPDATE CTA 固定负责一段连续 neuron
const int neurons_per_update_block =
    (n_neuron + update_block_count - 1) / update_block_count;

const int neuron_begin =
    blockIdx.x * neurons_per_update_block;

const int neuron_end =
    min(neuron_begin + neurons_per_update_block,
        n_neuron);
```

逻辑结构：

```text
UPDATE CTA 0:
[0, K)

UPDATE CTA 1:
[K, 2K)

UPDATE CTA 2:
[2K, 3K)

...

UPDATE CTA i:
[iK, min((i+1)K, N))
```

其中：

```text
K = ceil(N / update_block_count)
```

目前实测约：

```text
~234 neurons / UPDATE CTA
```

而 CTA 为 256 threads，因此当前场景下基本可以做到：

```text
1 thread ↔ 1 neuron
```

或者最多极少数配置下每 thread 多处理一次。

---

# 3. 本轮应该常驻哪些数据

第一版只建议真正常驻：

```text
v
```

后续如果有收益，再尝试 CSR metadata。

## 可以常驻

```cpp
v[n]
```

原因：

* 仅 UPDATE 阶段修改；
* neuron ownership 固定；
* propagation 不会修改 `v`；
* 每 timestep 都会访问；
* temporal reuse 明确。

因此生命周期可以变成：

```text
global v
   │
   │ kernel prologue
   ▼
shared_v
   │
   ├── UPDATE t0
   ├── UPDATE t1
   ├── UPDATE t2
   ├── ...
   └── UPDATE tT
   │
   │ kernel epilogue
   ▼
global v
```

---

## 暂时不能常驻

### `psc`

不能直接长期驻留：

```cpp
shared_psc[local_n]
```

因为 propagation 中其他 CTA 会：

```cpp
atomicAdd(&psc[post], weight);
```

直接修改 global `psc`。

如果 owner CTA 保留另一份 shared copy：

```text
shared psc != global psc
```

下一 timestep 就会丢失 propagation 新产生的电流。

所以：

```cpp
psc[n]
```

继续保持 global memory。

---

### `input_current`

同理，每 timestep 的输入事件路径会更新它，而且它本身不存在值得跨 timestep 保存的状态：

```cpp
current = input_current[n];
input_current[n] = 0;
```

因此保持 global。

---

# 4. 第一阶段：只实现连续静态 neuron ownership

这一阶段先**不要引入 shared memory**。

目的：

> 单独验证新的 neuron 静态分配是否正确，以及它本身带来的性能变化。

---

## 4.1 定义 UPDATE CTA

在 kernel 开头：

```cpp
const bool is_update_block =
    blockIdx.x < update_block_count;
```

注意只有：

```cpp
blockIdx.x < update_block_count
```

的 CTA 才拥有 neuron。

非 UPDATE CTA 不应该使用后面的：

```cpp
neuron_begin
neuron_end
```

参与 UPDATE。

---

## 4.2 计算静态连续区间

推荐：

```cpp
const int neurons_per_update_block =
    (n_neuron + update_block_count - 1) /
    update_block_count;

const int neuron_begin =
    blockIdx.x * neurons_per_update_block;

const int neuron_end =
    min(neuron_begin + neurons_per_update_block,
        n_neuron);
```

实际使用时放在：

```cpp
if (is_update_block)
```

语义下即可。

---

## 4.3 替换原 UPDATE grid-stride

原始形式类似：

```cpp
for (int n = global_tid;
     n < n_neuron;
     n += global_stride) {

    update_neuron(n);
}
```

修改为：

```cpp
if (is_update_block) {

    for (int n = neuron_begin + threadIdx.x;
         n < neuron_end;
         n += blockDim.x) {

        update_neuron(n);
    }
}
```

由于目前：

```text
neurons_per_update_block ≈ 234
blockDim = 256
```

绝大多数情况下实际上就是：

```cpp
if (is_update_block) {

    const int n =
        neuron_begin + threadIdx.x;

    if (n < neuron_end) {
        update_neuron(n);
    }
}
```

但建议先保留 loop：

```cpp
n += blockDim.x
```

避免以后 `update_block_count` 改变后出现隐性错误。

---

# 5. 第二阶段：将 `v` 常驻 Shared Memory

确认静态 ownership 正确后，再增加：

```cpp
shared_v
```

---

## 5.1 Shared-memory buffer

按照当前约 234 neuron / CTA，可以直接预留：

```cpp
constexpr int kMaxUpdateNeuronsPerBlock = 256;

__shared__ float shared_v[
    kMaxUpdateNeuronsPerBlock];
```

资源占用：

```text
256 × 4 B
= 1024 B / CTA
```

约 1 KB。

暂时无需设计动态 shared memory。

---

# 6. Kernel Prologue：初始化 Resident State

在：

```cpp
for (int t = 0; t < t_steps; ++t)
```

之前完成一次 global → shared。

---

## 6.1 Ownership

```cpp
const bool is_update_block =
    blockIdx.x < update_block_count;

const int neurons_per_update_block =
    (n_neuron + update_block_count - 1) /
    update_block_count;

const int neuron_begin =
    blockIdx.x *
    neurons_per_update_block;

const int neuron_end =
    min(neuron_begin +
            neurons_per_update_block,
        n_neuron);
```

对于非 UPDATE CTA，不访问这些 neuron 状态。

---

## 6.2 Shared 初始化

```cpp
if (is_update_block) {

    for (int n =
             neuron_begin +
             threadIdx.x;
         n < neuron_end;
         n += blockDim.x) {

        const int local_n =
            n - neuron_begin;

        shared_v[local_n] =
            v[n];
    }
}
```

如果当前实际数据布局仍然保留 `(B,N)`，则索引沿用原 kernel 的：

```cpp
const int cell =
    batch_idx * n_neuron + n;

shared_v[local_n] =
    v[cell];
```

如果当前生产 benchmark 已固定 `B=1`，第一版可以直接按现有 B=1 fast path 实现，不需要为了这次小优化重新设计多 batch resident layout。

---

# 7. Timestep UPDATE：改用 Shared `v`

原 UPDATE：

```cpp
float voltage =
    v[cell];

const float current =
    psc[cell] +
    input_current[cell];

...

v[cell] =
    new_voltage;
```

修改成：

```cpp
const int local_n =
    n - neuron_begin;

float voltage =
    shared_v[local_n];

const float current =
    psc[cell] +
    input_current[cell];

input_current[cell] =
    0.0f;

const float v_pre =
    voltage +
    dt * (
        -(voltage - v_reset) /
            tau_mem
        + current / c_m
    );

const bool fired =
    v_pre >= v_threshold;

voltage =
    v_pre -
    reset_delta *
    static_cast<float>(fired);

shared_v[local_n] =
    voltage;

psc[cell] *= decay;
```

其余 UPDATE 内容完全保持原样。

例如：

```cpp
dense_spikes[...]

event output

spike descriptor generation

task queue generation
```

不要在本轮一起重构。

---

# 8. Spike-block 逻辑保持现状

连续大块 ownership 不需要为了 shared residency 去修改当前 propagation 描述符机制。

原则是：

```text
修改 neuron 的 UPDATE ownership
            │
            ▼
修改 v 的存储位置
            │
            ▼
spike 是否发生的计算结果
            │
            ▼
继续进入原 spike-block task generation
```

即：

```cpp
const bool fired = ...;

// 后面仍执行当前已有逻辑
if (fired) {
    ...
}
```

这样可以最大限度隔离变量。

---

# 9. UPDATE → PROP 同步保持不变

当前 persistent kernel 的核心 producer-consumer 关系仍是：

```text
UPDATE
  ↓
produce spike/tasks
  ↓
synchronization
  ↓
PROP
  ↓
atomicAdd(psc)
  ↓
synchronization
  ↓
next timestep UPDATE
```

此次优化不改变：

```text
grid.sync()
task counters
queue reset
work counter
propagation
```

尤其 `psc` 必须继续依赖 timestep 尾部同步保证：

```text
PROP(t)
完成 psc accumulation
        ↓
UPDATE(t+1)
读取新的 psc
```

---

# 10. Kernel Epilogue：一次性写回 `v`

`t_steps` 全部完成后：

```cpp
if (is_update_block) {

    for (int n =
             neuron_begin +
             threadIdx.x;
         n < neuron_end;
         n += blockDim.x) {

        const int local_n =
            n - neuron_begin;

        v[n] =
            shared_v[local_n];
    }
}
```

如果存在 batch offset：

```cpp
v[cell] =
    shared_v[local_n];
```

于是原本：

```text
T 次 global load
+
T 次 global store
```

变成：

```text
1 次 global load
+
1 次 global store
```

---

# 11. 完整伪代码

```cpp
__global__
void persistent_snn_spike_block_kernel(...) {

    namespace cg =
        cooperative_groups;

    cg::grid_group grid =
        cg::this_grid();

    constexpr int
        kMaxUpdateNeuronsPerBlock =
            256;

    __shared__ float
        shared_v[
            kMaxUpdateNeuronsPerBlock];


    // =====================================
    // 1. Static UPDATE CTA ownership
    // =====================================

    const bool is_update_block =
        blockIdx.x <
        update_block_count;

    const int neurons_per_update_block =
        (n_neuron +
         update_block_count - 1) /
        update_block_count;

    const int neuron_begin =
        blockIdx.x *
        neurons_per_update_block;

    const int neuron_end =
        min(
            neuron_begin +
            neurons_per_update_block,
            n_neuron);


    // =====================================
    // 2. Kernel prologue:
    //    global v -> shared v
    // =====================================

    if (is_update_block) {

        for (int n =
                 neuron_begin +
                 threadIdx.x;
             n < neuron_end;
             n += blockDim.x) {

            const int local_n =
                n -
                neuron_begin;

            shared_v[local_n] =
                v[n];
        }
    }

    __syncthreads();


    // =====================================
    // 3. Persistent timestep loop
    // =====================================

    for (int t = 0;
         t < t_steps;
         ++t) {

        // ---------------------------------
        // UPDATE
        // ---------------------------------

        if (is_update_block) {

            for (int n =
                     neuron_begin +
                     threadIdx.x;
                 n < neuron_end;
                 n += blockDim.x) {

                const int local_n =
                    n -
                    neuron_begin;

                float voltage =
                    shared_v[
                        local_n];

                const float current =
                    psc[n] +
                    input_current[n];

                input_current[n] =
                    0.0f;

                const float v_pre =
                    voltage +
                    dt * (
                        -(voltage -
                          v_reset) /
                            tau_mem
                        + current /
                            c_m);

                const bool fired =
                    v_pre >=
                    v_threshold;

                voltage =
                    v_pre -
                    reset_delta *
                    float(fired);

                shared_v[local_n] =
                    voltage;

                psc[n] *= decay;


                // -------------------------
                // 原 spike generation
                // -------------------------

                if (return_dense) {
                    dense_spikes[
                        ...] =
                        float(fired);
                }

                if (fired) {

                    // 保持当前
                    // spike-block descriptor
                    // / task queue
                    // generation
                    ...
                }
            }
        }


        // ---------------------------------
        // 原 UPDATE/PROP synchronization
        // ---------------------------------

        ...
        grid.sync();


        // ---------------------------------
        // PROP
        // ---------------------------------

        process_existing_spike_tasks();

        // arbitrary CTA:
        //
        // atomicAdd(
        //     &psc[post],
        //     weight
        // );

        ...


        // ---------------------------------
        // timestep boundary
        // ---------------------------------

        grid.sync();
    }


    // =====================================
    // 4. Kernel epilogue:
    //    shared v -> global v
    // =====================================

    if (is_update_block) {

        for (int n =
                 neuron_begin +
                 threadIdx.x;
             n < neuron_end;
             n += blockDim.x) {

            const int local_n =
                n -
                neuron_begin;

            v[n] =
                shared_v[
                    local_n];
        }
    }
}
```

---

# 12. 一个实现细节：不要让非 UPDATE CTA 越界使用 ownership

按照：

```cpp
const int neuron_begin =
    blockIdx.x * neurons_per_update_block;
```

当：

```text
blockIdx.x >= update_block_count
```

时：

```text
neuron_begin >= N
```

通常不会造成实际错误，只要所有使用都被：

```cpp
if (is_update_block)
```

保护。

推荐保持：

```cpp
const bool is_update_block = ...;

const int neuron_begin = ...;
const int neuron_end = min(...);

if (is_update_block) {
    ...
}
```

不要出现：

```cpp
shared_v[
    neuron_end - neuron_begin
]
```

之类在 guard 外使用 ownership 结果的代码。

---

# 13. Shared-memory Capacity Guard

虽然目前：

```text
~234 < 256
```

仍然建议显式保护。

## Debug kernel 侧

```cpp
assert(
    neurons_per_update_block <=
    kMaxUpdateNeuronsPerBlock);
```

更推荐 host 侧判断：

```cpp
const int neurons_per_update_block =
    (n_neuron +
     update_block_count - 1) /
    update_block_count;

TORCH_CHECK(
    neurons_per_update_block <= 256,
    "Too many neurons per UPDATE CTA.");
```

这样一旦未来：

```text
grid_dim
update_block_count
N
occupancy
```

变化，不会静默 shared-memory 越界。

---

# 14. 第三阶段：可选的 CSR Metadata Residency

只有 `shared_v` 已经验证出收益后，再考虑这一项。

每个 neuron 每 timestep 都会读取：

```cpp
graph_indptr[n]
graph_indptr[n + 1]
```

而图结构完全静态。

可以增加：

```cpp
__shared__ int
    shared_edge_start[256];

__shared__ int
    shared_degree[256];
```

初始化一次：

```cpp
if (is_update_block) {

    for (int n =
             neuron_begin +
             threadIdx.x;
         n < neuron_end;
         n += blockDim.x) {

        const int local_n =
            n - neuron_begin;

        const int begin =
            graph_indptr[n];

        const int end =
            graph_indptr[n + 1];

        shared_edge_start[
            local_n] =
            begin;

        shared_degree[
            local_n] =
            end - begin;
    }
}
```

每 timestep：

```cpp
const int edge_begin =
    shared_edge_start[
        local_n];

const int edge_end =
    edge_begin +
    shared_degree[
        local_n];
```

替代：

```cpp
const int edge_begin =
    graph_indptr[n];

const int edge_end =
    graph_indptr[n + 1];
```

---

# 15. 为什么 CSR Cache 要单独做

不要直接实现：

```text
static ownership
+
shared_v
+
shared CSR
```

否则性能变化无法归因。

而且：

```text
graph_indptr
```

本身：

* 数据量较小；
* read-only；
* temporal locality 强；
* 很可能已经被 L2/cache 较好地捕获。

因此它的收益不如 `v` residency 明确。

建议严格按照：

```text
A. Static ownership

B. Static ownership
   + resident v

C. Static ownership
   + resident v
   + resident CSR
```

逐步测。

---

# 16. 暂时不要进一步缓存的内容

本次明确不做：

```text
shared psc
shared input_current
shared spike queue
shared propagation accumulation
destination-owned propagation
```

尤其不要为了 `psc` residency 修改现有 atomic propagation。

否则优化会迅速从：

> UPDATE state residency

扩大成：

> 重构 PROP 数据流和 ownership

这与当前“收尾阶段的小优化”目标不匹配。

---

# 17. Correctness 测试

## Test 1：单 timestep

```text
T = 1
```

验证：

```text
global v
→ shared
→ UPDATE
→ global
```

行为正确。

比较：

```python
torch.testing.assert_close(
    v_opt,
    v_ref
)

torch.testing.assert_close(
    psc_opt,
    psc_ref
)
```

如果返回 dense：

```python
torch.testing.assert_close(
    dense_opt,
    dense_ref
)
```

---

## Test 2：多 timestep

至少：

```text
T = 10
T = 100
benchmark 默认 T
```

这是本优化最关键的正确性测试。

因为真正需要验证的是：

```text
UPDATE(t)
   ↓
shared_v 保存
   ↓
UPDATE(t+1)
```

而不是单纯 staging。

---

## Test 3：不同 N

至少覆盖：

```text
N % update_block_count == 0

N % update_block_count != 0

neurons_per_update_block < 256

neurons_per_update_block ≈ 256
```

尤其检查最后 CTA：

```cpp
const int neuron_end =
    min(
        neuron_begin +
        neurons_per_update_block,
        n_neuron);
```

---

## Test 4：Spike 输出

确认：

```text
spike count
dense spike
event indices
task count
```

均与 baseline 一致。

因为虽然本次没有主动修改 spike-block generation，但 neuron traversal 改变以后，task enqueue 顺序可能发生变化。

因此：

* 数值结果必须一致；
* task 的物理排列不必强求完全 byte-wise 一致；
* 最终 propagation/output 必须一致。

---

# 18. Performance 实验

建议最终形成下面的消融：

| Version  | 连续静态 ownership | `v` 常驻 | CSR 常驻 |
| -------- | -------------: | -----: | -----: |
| Baseline |              × |      × |      × |
| A        |              ✓ |      × |      × |
| B        |              ✓ |      ✓ |      × |
| C        |              ✓ |      ✓ |      ✓ |

---

## 重点记录

### Production

```text
μs / step
total forward
```

### Instrumented

```text
UPDATE duration
overlap window
PROP tail
pipeline duration
```

### Resource

```text
registers / thread

static shared memory / CTA

active CTA / SM

cooperative grid size

update_block_count

neurons_per_update_block
```

现有 host 侧的 cooperative grid 会依据 spike-block kernel occupancy 计算，因此加入 static shared memory 后必须重新确认 active blocks per SM 是否变化。

---

# 19. 预期实验结果

## A：连续静态 ownership

根据此前类似实验，可以期待：

```text
小幅改善
```

原因主要来自：

```text
连续访问
+
固定 ownership
+
更稳定的 CTA 工作集
```

---

## B：Resident `v`

主要应该观察：

```text
UPDATE time ↓
```

因为每 neuron 每 timestep 少掉：

```text
global load(v)
global store(v)
```

总 global transaction 从近似：

```text
2 × N × T
```

下降到：

```text
2 × N
```

当然 UPDATE 中仍然存在：

```text
psc load/store
input_current load/store
spike/task operations
```

所以不要期待 UPDATE 时间按比例大幅下降。

---

## Overall

PROP 仍然是主要耗时，因此：

```text
overall 0.x% ~ low-single-digit %
```

都属于合理结果。

这个优化的价值更多在于：

```text
改动小
+
语义清晰
+
真正利用 persistent kernel 的跨 timestep 生命周期
```

---

# 20. 推荐的实际施工顺序

## Step 1 — Static ownership

只实现：

```cpp
const bool is_update_block =
    blockIdx.x <
    update_block_count;

const int neurons_per_update_block =
    (n_neuron +
     update_block_count - 1) /
    update_block_count;

const int neuron_begin =
    blockIdx.x *
    neurons_per_update_block;

const int neuron_end =
    min(
        neuron_begin +
        neurons_per_update_block,
        n_neuron);
```

UPDATE 改为：

```cpp
for (int n =
         neuron_begin +
         threadIdx.x;
     n < neuron_end;
     n += blockDim.x)
```

跑 correctness + benchmark。

---

## Step 2 — Resident `v`

加入：

```cpp
__shared__
float shared_v[256];
```

实现：

```text
prologue:
global v → shared_v

timestep:
shared_v → reg
        → UPDATE
        → shared_v

epilogue:
shared_v → global v
```

再次测速。

---

## Step 3 — Capacity / occupancy 检查

确认：

```text
neurons_per_update_block <= 256
```

以及：

```text
active CTA / SM
```

没有因为新增 shared memory 发生不利变化。

---

## Step 4 — Optional CSR residency

仅在 B 已有正收益后，实现：

```cpp
shared_edge_start[256]
shared_degree[256]
```

如果没有明显额外收益，直接回退，不保留。

---

## Step 5 — 最终保留最小有效版本

最终判断：

```text
Static ownership
    │
    ├── 有收益 → 保留
    │
    ▼
Resident v
    │
    ├── 有收益 → 保留
    │
    ▼
Resident CSR
    │
    └── 只有明确额外收益才保留
```

---

# 21. 最终目标结构

```cpp
__global__
void kernel(...) {

    __shared__
    float shared_v[256];

    // --------------------------------
    // Static continuous ownership
    // --------------------------------

    const bool is_update_block =
        blockIdx.x <
        update_block_count;

    const int neurons_per_update_block =
        (n_neuron +
         update_block_count - 1) /
        update_block_count;

    const int neuron_begin =
        blockIdx.x *
        neurons_per_update_block;

    const int neuron_end =
        min(
            neuron_begin +
            neurons_per_update_block,
            n_neuron);


    // --------------------------------
    // Load persistent state once
    // --------------------------------

    if (is_update_block) {

        for (int n =
                 neuron_begin +
                 threadIdx.x;
             n < neuron_end;
             n += blockDim.x) {

            shared_v[
                n - neuron_begin] =
                v[n];
        }
    }

    __syncthreads();


    // --------------------------------
    // Persistent temporal loop
    // --------------------------------

    for (int t = 0;
         t < t_steps;
         ++t) {

        if (is_update_block) {

            for (int n =
                     neuron_begin +
                     threadIdx.x;
                 n < neuron_end;
                 n += blockDim.x) {

                const int local_n =
                    n - neuron_begin;

                float voltage =
                    shared_v[
                        local_n];

                // global per-step inputs
                const float current =
                    psc[n] +
                    input_current[n];

                input_current[n] =
                    0.0f;

                // neuron UPDATE
                voltage =
                    update_voltage(
                        voltage,
                        current,
                        ...);

                shared_v[
                    local_n] =
                    voltage;

                psc[n] *= decay;

                // original spike-block
                // task generation
                ...
            }
        }

        // Existing synchronization
        ...

        // Existing propagation
        ...

        grid.sync();
    }


    // --------------------------------
    // Flush persistent state once
    // --------------------------------

    if (is_update_block) {

        for (int n =
                 neuron_begin +
                 threadIdx.x;
             n < neuron_end;
             n += blockDim.x) {

            v[n] =
                shared_v[
                    n -
                    neuron_begin];
        }
    }
}
```

本轮最值得坚持的原则就是：**连续大块静态分 neuron，但只把真正具有 CTA-exclusive ownership 的 `v` 做跨 timestep residency；其余数据流尽量不动。** 这样既能充分利用 persistent kernel，又不会为了一个收尾优化重新牵动 propagation 架构。
