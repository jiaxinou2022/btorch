## 核验结论

仍有几个问题需要并入第三阶段开头修正。其中最重要的不是主 benchmark，而是新建的 operator microbenchmark 目前还不能可靠地区分 native 与 adapted 性能。

---

# 一、第二阶段实现核验

## 1. Runner 与 reset：通过

公共接口现在为：

```python
@dataclass
class BenchmarkRunner:
    run_fn
    reset_fn
    metadata
    reset_policy
```

persistent runner 设置：

```python
reset_policy="before_sample"
```

计时代码在每个 sample 前调用：

```python
_reset_before_sample(fn)
start.record()
fn()
end.record()
```

且 reset 在 CUDA Event 起点之前，因此：

* reset 不计入 kernel latency；
* 每次 persistent 推理从同一状态开始；
* warmup、correctness 和 timing 生命周期被分开；
* 可以通过 `continuous_state` 切换成连续状态语义。

这一部分实现正确。

### 小问题

`before_phase` 目前实际上依赖计时函数开头统一调用 `_reset_runner()`，行为正确，但语义可以再明确一些。建议第三阶段补一个统一生命周期函数，避免 GPU、host、operator 三套计时代码以后再次分叉。

---

## 5. Metadata：主体通过

`PreparedMetadata` 已覆盖：

* logical/physical shape；
* layout；
* index/value/compute dtype；
* workspace；
* persistent storage；
* execution class；
* timing scope；
* host control；
* D2H/H2D；
* graph capture；
* transform mode；
* spike representation；
* allocator delta。

相比第一阶段按 provider 名称推断，当前 metadata 确实主要来自 prepare 后的实际 buffer。

但有两个未完成点。

### 问题 A：padding 只有工具函数，没有接入 provider prepare

现在有：

```python
pad_csr_neurons(...)
PaddingInfo
ProviderCapability.n_alignment
```

但在 `bench_case()` 中，原始 `matrix` 仍直接传给 provider，没有：

```python
physical_matrix, padding_info = pad_csr_neurons(...)
```

所以目前：

```text
physical_shape == logical_shape
padding_ratio == 1.0
```

基本是固定值，不代表 padding 系统已经真正完成。

这不阻碍进入第三阶段，但第三阶段开始时应该补上。

### 问题 B：`persistent_bytes` 和 `workspace_bytes` 有重复计数风险

例如某些 provider：

* `workspace_bytes` 已统计 adapter buffers；
* `persistent_bytes` 又包含 adapter buffers。

这并非一定错误，只要定义为：

```text
workspace_bytes：其中属于工作区的部分
persistent_bytes：prepare 后总常驻内存
```

但字段说明应明确，否则读者可能把两者相加，导致重复统计。

建议改名或补充：

```text
workspace_bytes
total_prepared_bytes
```

---

## 6. 内存 allocation audit：可作为提示，不能作为严格证明

当前方法比较：

```python
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
```

前后差值。

这可以发现明显的持久分配，但不能可靠捕捉：

* 分配后又释放的瞬时 allocation；
* CUDA Graph pool 内部复用；
* native library 内部的 `cudaMallocAsync`；
* allocator reserved 不变时的 block reuse；
* C++ 扩展中绕过 PyTorch allocator 的分配。

因此：

```text
run_allocated_bytes == 0
```

只能解释为：

> PyTorch allocator 未观察到净增长。

不能解释为：

> timed path 保证完全没有 allocation。

第三阶段应把它保留为 metadata，同时增加一次 profiler 审计，而不是每次 benchmark 都启用 profiler。

---

# 二、当前必须修正的问题：operator microbenchmark

`benchmark_sparse_operator_adapters.py` 已经建立了独立实验框架，但目前的 `native`、`adapted` 和 `layout_only` 三种模式定义并不成立。

## 1. native 与 adapted 实际上基本相同

当前：

```python
spikes_nb = spikes_bn.view(T, N, 1)

def native_run():
    operator(spikes_nb[t].view(1, N))

def adapted_run():
    operator(spikes_bn[t])
```

两者最终传入 `operator()` 的 shape 都是：

```text
[1, N]
```

因此对于 Sputnik：

```python
rhs = spikes.view(N, 1)
```

两条路径完全相同。

对于 VDHA：

```python
spikes.reshape(-1)
```

两条路径也完全相同。

所以当前输出中的：

```text
mode=native
mode=adapted
```

并不代表两个不同路径。

### 修正原则

真正的 native runner 应绕过公共 adapter，直接调用底层 native API，并直接传入 native-layout buffer。

例如 Sputnik：

```python
native_run:
    rhs_nb 已预先准备为 [N,1]
    直接调用 cbn_sputnik_compute_device

adapted_run:
    输入为公共 [1,N]
    执行 adapter 的 view/pack
    再调用 cbn_sputnik_compute_device
```

为此不能只返回一个闭包 `operator()`，而应返回结构化对象。

```python
@dataclass
class PreparedOperator:
    run_native: Callable
    run_adapted: Callable
    run_transform_only: Callable
    output: Tensor
    metadata: OperatorMetadata
    release: Callable
```

---

## 2. layout-only 当前测到的主要是 Python 循环

当前：

```python
def layout_run():
    for timestep in range(T):
        spikes_bn[timestep].view(N, 1)
```

`view()` 不产生 CUDA kernel。

而计时使用 CUDA Event，因此 GPU 时间理论上接近零。测到的值主要是：

* event 分辨率；
* 可能的 stream/event 固定开销；
* 并不是 Python 循环时间，因为 CUDA Event 不记录 CPU Python 执行。

因此这一项没有实际分析价值。

### 修正方式

对于 zero-copy view：

```text
layout_only_latency = 0
layout_transform_mode = zero_copy_view
```

无需单独用 CUDA Event 测。

只有真实 GPU pack kernel 才测 `layout_only`：

```python
def layout_run():
    pack_bn_to_nb_kernel(src, dst)
```

如果想测 CPU adapter 调用开销，则要使用 wall-clock，而不是 CUDA Event，并与 GPU execution 分开记录。

---

## 3. 当前 activity sweep 对 Sputnik 和 VDHA dense 基本不会改变工作量

`make_spike_trace()` 会改变零元素比例，但：

* Sputnik 是 sparse-matrix × dense-vector；
* VDHA 当前也是 dense-input 动态接口。

如果 kernel 没有基于 spike 值跳过零元素，它们的工作量可能与 activity 基本无关。

因此这个 sweep 能测到的主要是：

* 数值内容变化是否影响 kernel；
* 而不是 SpMSpV 活跃边数量变化。

第三阶段中应保留 activity sweep，但加入真正的 event-driven provider：

* persistent recurrent-only；
* prespan；
* spike-list VDHA，若未来可实现；
* custom active-row propagation。

否则“activity sweep”图容易被误读。

---

## 4. 每个 repeat 都单独 `end.synchronize()`，测的是 isolated latency

当前：

```python
for repeat:
    start.record()
    run()
    end.record()
    end.synchronize()
```

这没有错误，但测的是：

```text
每个完整 T-step trace 独立同步后的 GPU elapsed time
```

应再增加一种 throughput 模式：

```python
record all start/end
run all repeats
synchronize once
```

从而区分：

* isolated trace latency；
* steady-state queued throughput。

主 RSNN benchmark 已使用后一种方式，两者目前并不完全一致。

---

# 三、第三阶段总体目标

第三阶段应从“修 adapter”转向“建立可控、可解释的 workload 系统”。

核心目标是把两个问题彻底拆开：

## 问题一：算子本身对什么 workload 敏感？

固定 spike trace，只测：

[
y_t = W s_t
]

控制：

* 实际 activity；
* active-edge 数量；
* fanout 分布；
* post collision；
* active neuron 与高 fanout neuron 的相关性；
* 图规模。

## 问题二：完整 RSNN 中实际产生了什么 workload？

运行闭环 LIF，记录：

* 实际 spike rate；
* 每 timestep active neuron；
* active edges；
* fanout cost；
* collision proxy；
* burstiness；
* 长尾任务数量。

最终性能图不再只以 `event_rate` 为横轴，而使用实际测得的 workload 指标。

---

# 四、第三阶段文件架构

建议新增三个文件，并精简现有主 benchmark。

```text
benchmark/
├── benchmark_data.py
├── benchmark_workload_stats.py
├── benchmark_sparse_operator.py
├── benchmark_rsnn_cudagraph_compare.py
├── provider_common.py
└── sota_rsnn_cudagraph.py
```

## `benchmark_data.py`

负责：

* 真实数据集加载；
* synthetic graph；
* spike trace；
* workload manifest；
* padding；
* 缓存与复现。

## `benchmark_workload_stats.py`

负责：

* 从 CSR 和 spike trace 计算 workload 指标；
* closed-loop RSNN activity 统计；
* 输出每 timestep 统计和 summary。

## `benchmark_sparse_operator.py`

替换当前不完整的 adapter microbenchmark，负责：

* operator-only；
* native/adapted；
* activity sweep；
* graph distribution sweep；
* correctness。

主 RSNN benchmark只负责：

* full-window；
* 不再承担数据生成细节；
* 读取统一的 `BenchmarkWorkload`。

---

# 五、第三阶段 Step 0：修正 operator adapter contract

这是第三阶段施工前置项。

## 新接口

```python
@dataclass
class OperatorResult:
    output: torch.Tensor


@dataclass
class PreparedOperator:
    native_run: Callable[[], None] | None
    adapted_run: Callable[[], None]
    transform_run: Callable[[], None] | None

    native_output: torch.Tensor | None
    adapted_output: torch.Tensor

    metadata: PreparedMetadata
    release_fn: Callable[[], None]
```

## Sputnik 伪代码

```python
def prepare_sputnik_operator(weight, spike_trace_bn):
    T, B, N = spike_trace_bn.shape
    assert B == 1

    spike_trace_nb = spike_trace_bn.view(T, N, 1)
    output_native = empty(T, N, 1)
    output_adapted = empty(T, N, 1)

    native_handle = prepare_sputnik(weight)
    adapted_handle = prepare_sputnik(weight)

    def native_run():
        for t in range(T):
            launch_sputnik(
                native_handle,
                spike_trace_nb[t],
                output_native[t],
                stream,
            )

    def adapted_run():
        for t in range(T):
            rhs_nb = spike_trace_bn[t].view(N, 1)
            launch_sputnik(
                adapted_handle,
                rhs_nb,
                output_adapted[t],
                stream,
            )

    return PreparedOperator(
        native_run=native_run,
        adapted_run=adapted_run,
        transform_run=None,  # zero-copy，不单独计 GPU 时间
        ...
    )
```

## VDHA dense 伪代码

由于 native input 本身就是 dense vector：

```python
native_run == adapted_run
```

因此不要虚构两条曲线。

```python
PreparedOperator(
    native_run=None,
    adapted_run=vdha_dense_run,
    transform_run=None,
    metadata.native_available=False,
)
```

CSV 中：

```text
mode = adapted_dense
native_status = not_distinct_from_adapter
```

---

# 六、第三阶段 Step 1：统一 workload 数据结构

## 数据结构

```python
@dataclass
class GraphStats:
    n: int
    nnz: int
    mean_fanout: float
    max_fanout: int
    fanout_std: float
    fanout_cv: float
    fanout_gini: float

    p50_fanout: float
    p90_fanout: float
    p95_fanout: float
    p99_fanout: float

    unique_posts_ratio: float
    graph_type: str
    seed: int


@dataclass
class SpikeStats:
    requested_activity: float
    measured_activity: float

    mean_active_neurons: float
    p95_active_neurons: float
    max_active_neurons: int

    mean_active_edges: float
    p95_active_edges: float
    max_active_edges: int

    temporal_cv: float
    burst_factor: float


@dataclass
class BenchmarkWorkload:
    matrix: CSR
    spike_trace: torch.Tensor | None
    input_trace: torch.Tensor | None

    graph_stats: GraphStats
    spike_stats: SpikeStats | None

    logical_n: int
    physical_n: int
    batch_size: int
    t_steps: int

    mode: Literal["operator", "closed_loop"]
    workload_id: str
```

---

# 七、第三阶段 Step 2：可控 synthetic graph

不能只使用 `N + fanout` 的 uniform graph。至少实现四类。

## 1. Uniform graph

用于基础规模测试。

```python
for pre in neurons:
    posts = random_sample(N, fanout)
```

## 2. Power-law fanout

用于测试重尾和 long-segment。

```python
degrees = sample_truncated_powerlaw(
    alpha,
    min_degree,
    max_degree,
)

for pre, degree in enumerate(degrees):
    posts = random_sample(N, degree)
```

控制参数：

```text
alpha
max_fanout
heavy_row_fraction
```

## 3. Post-hotspot graph

用于控制 atomic collision。

```python
hot_posts = choose_posts(hotspot_count)

for each edge:
    with probability hotspot_probability:
        post = sample(hot_posts)
    else:
        post = sample(all_posts)
```

控制：

```text
hotspot_count
hotspot_probability
```

## 4. Clustered graph

用于测试 blockization 和组内 post overlap。

```python
assign neurons to communities

for pre:
    with probability intra_cluster_ratio:
        post from same community
    else:
        post from another community
```

控制：

```text
community_count
intra_cluster_ratio
```

## 图生成统一伪代码

```python
def generate_graph(spec, device):
    if spec.kind == "uniform":
        edges = generate_uniform(spec)
    elif spec.kind == "powerlaw":
        edges = generate_powerlaw(spec)
    elif spec.kind == "hotspot":
        edges = generate_hotspot(spec)
    elif spec.kind == "clustered":
        edges = generate_clustered(spec)

    csr = edges_to_source_csr(edges)
    stats = summarize_graph(csr)

    return csr, stats
```

---

# 八、第三阶段 Step 3：可控 spike trace

至少实现四种 spike pattern。

## 1. IID uniform

```python
spike[t,n] ~ Bernoulli(activity)
```

## 2. Fanout-correlated

高 fanout neuron 更容易发放：

```python
prob[n] ∝ fanout[n] ** correlation_strength
normalize mean(prob) to target_activity
```

这能控制 active-edge 数量，而不仅仅是 active-neuron 数量。

## 3. Temporally correlated

模拟持续活跃和 burst：

```python
active[t] =
    keep previous active neurons with probability persistence
    + sample new neurons
```

## 4. Cluster burst

某段时间集中激活同一社区：

```python
cluster = select_cluster(t)
spikes[t, cluster] ~ Bernoulli(high_rate)
spikes[t, others] ~ Bernoulli(low_rate)
```

## 统一伪代码

```python
def generate_spike_trace(graph, spec):
    if spec.kind == "iid":
        spikes = iid_trace(spec)
    elif spec.kind == "fanout_correlated":
        spikes = fanout_correlated_trace(graph, spec)
    elif spec.kind == "temporal":
        spikes = temporal_trace(spec)
    elif spec.kind == "cluster_burst":
        spikes = cluster_burst_trace(graph, spec)

    stats = summarize_spike_workload(graph, spikes)
    return spikes, stats
```

---

# 九、第三阶段 Step 4：计算真正有解释力的 workload 指标

对于 source-oriented CSR：

```python
fanout = indptr[1:] - indptr[:-1]
```

每个 timestep：

```python
active = nonzero(spikes[t])
active_edges = sum(fanout[active])
```

还需要估计 post collision。

## 精确 collision 指标

中小图上：

```python
posts = concatenate(
    csr.indices[indptr[pre]:indptr[pre+1]]
    for pre in active
)

unique_posts = unique(posts).numel()
collision_edges = active_edges - unique_posts
collision_ratio = collision_edges / active_edges
```

## 大图近似

避免每 timestep 大量 `unique`：

```python
sample active neurons
sample their posts
estimate duplicate ratio
```

或者离线 CPU 计算，不计入 benchmark。

## 统计伪代码

```python
def summarize_spike_workload(matrix, spikes):
    fanout = csr_row_lengths(matrix)

    active_counts = []
    active_edges = []
    collision_ratios = []

    for t in range(T):
        active = nonzero(spikes[t])

        edge_count = fanout[active].sum()
        collision = estimate_collision(matrix, active)

        active_counts.append(len(active))
        active_edges.append(edge_count)
        collision_ratios.append(collision)

    return SpikeStats(
        measured_activity=mean(active_counts) / N,
        mean_active_edges=mean(active_edges),
        p95_active_edges=percentile(active_edges, 95),
        temporal_cv=std(active_counts) / mean(active_counts),
        ...
    )
```

---

# 十、第三阶段 Step 5：闭环 RSNN activity 校准

完整 RSNN 中，`event_rate` 不等于 spike rate。

需要增加 target-firing-rate calibration。

## 目标

用户指定：

```text
target_activity = 0.1%, 0.3%, 1%, 3%, 10%
```

系统调整：

```text
input_amplitude
或 input_event_rate
```

使实际 firing rate接近目标。

## 二分校准伪代码

```python
def calibrate_input(case, matrix, target_activity):
    low = min_amplitude
    high = max_amplitude

    best = None

    for _ in range(max_iterations):
        amplitude = (low + high) / 2

        x_seq = generate_input(
            event_rate=fixed_event_rate,
            amplitude=amplitude,
        )

        spikes = run_reference_rsnn(
            warmup_steps + measurement_steps,
        )

        measured = spikes[measurement_window].mean()

        if best is None or abs(measured - target) < best.error:
            best = CalibrationResult(amplitude, measured)

        if measured < target:
            low = amplitude
        else:
            high = amplitude

    return best
```

### 注意

闭环网络可能存在：

* 非单调；
* 突然爆发；
* 静默—失稳转变。

因此不能只保留最后一次二分结果，还应记录：

```text
calibration_status
target_activity
measured_activity
activity_error
stable
```

如果输出不稳定，就标记 workload 不可用，而不是强行纳入 sweep。

---

# 十一、第三阶段 Step 6：operator-only benchmark

## 实验矩阵

第一版不要铺得太大，建议：

### 图类型

```text
uniform
powerlaw
hotspot
clustered
FlyBrain
```

### activity

```text
0.1%
0.3%
1%
3%
10%
```

### 规模

```text
N = 4K, 16K, 64K
```

真实 FlyBrain 使用原始规模。

### provider

```text
torch CSR
cuSPARSE SpMV
Sputnik
VDHA dense-input
persistent recurrent-only
prespan/reference event-driven
```

其中后两个如果当前还没有 recurrent-only entry point，可先预留接口，再做完整 RSNN sweep。

## 主循环

```python
for graph_spec in graph_specs:
    workload_graph = prepare_graph(graph_spec)

    for spike_spec in spike_specs:
        workload = prepare_operator_workload(
            workload_graph,
            spike_spec,
        )

        reference = run_reference_operator(workload)

        for provider in providers:
            prepared = provider.prepare_operator(workload)

            validate(prepared, reference)

            samples = time_operator(
                prepared.adapted_run,
            )

            write_row(
                graph_stats=workload.graph_stats,
                spike_stats=workload.spike_stats,
                provider_metadata=prepared.metadata,
                latency=samples,
            )
```

---

# 十二、第三阶段 Step 7：完整 RSNN benchmark

完整 RSNN 只使用已经校准过的 closed-loop workload。

## 每个结果必须记录

```text
requested_activity
measured_activity
mean_active_neurons
mean_active_edges
p95_active_edges
max_active_edges
mean_collision_ratio
temporal_cv
```

## 主循环

```python
for dataset in datasets:
    graph = load_graph(dataset)

    for target_activity in targets:
        calibration = calibrate_input(
            graph,
            target_activity,
        )

        workload = build_closed_loop_workload(
            graph,
            calibration,
        )

        for provider in full_rsnn_providers:
            result = benchmark_provider(
                provider,
                workload,
            )

            write_row(
                calibration=calibration,
                workload_stats=result.spike_stats,
                provider=result.provider_metadata,
                latency=result.samples,
            )
```

---

# 十三、第三阶段 Step 8：真正接入 padding

当前 padding 函数应在 provider prepare 前应用。

## 伪代码

```python
def prepare_physical_case(provider, matrix, x_seq, case):
    capability = PROVIDER_CAPABILITIES[provider]

    padded_matrix, padding = pad_csr_neurons(
        matrix,
        capability.n_alignment,
    )

    if padding.physical_n == padding.logical_n:
        return matrix, x_seq, case, padding

    physical_x = zeros(
        T,
        B,
        padding.physical_n,
        device=x_seq.device,
    )
    physical_x[..., :padding.logical_n].copy_(x_seq)

    physical_case = replace(
        case,
        n_neuron=padding.physical_n,
    )

    return (
        padded_matrix,
        physical_x,
        physical_case,
        padding,
    )
```

结果 correctness 时裁剪：

```python
def crop_result(result, logical_n):
    return RSNNResult(
        spikes=result.spikes[..., :logical_n],
        v=result.v[..., :logical_n],
        psc=result.psc[..., :logical_n],
    )
```

metadata：

```python
metadata.logical_shape = original_shape
metadata.physical_shape = padded_shape
metadata.padding_ratio = physical_n / logical_n
```

仍然禁止主实验自动 pad batch。

---

# 十四、第三阶段 Step 9：一次性工程审计

不建议每次正式 benchmark 都运行 profiler，而是在每个 provider 完成后运行一次 audit case。

## Audit case

```text
N = 4096
T = 32
B = 1
activity = 1%
```

## 检查项

使用 PyTorch profiler 或 Nsight Systems 检查：

```text
cudaMalloc
cudaFree
cudaMemcpy D2H
cudaMemcpy H2D
aten::empty
aten::contiguous
stream synchronization
unexpected default-stream launch
```

## 结果记录

```python
@dataclass
class AuditResult:
    has_cuda_malloc: bool
    has_d2h: bool
    has_h2d: bool
    has_internal_sync: bool
    unexpected_alloc_ops: list[str]
```

正式 benchmark metadata 中读取 audit manifest：

```json
{
  "sputnik_cudagraph": {
    "audited": true,
    "cuda_malloc_in_run": false,
    "d2h_in_run": false
  }
}
```

这样比 allocator 净增长更可信。

---

# 十五、第三阶段验收标准

完成以下条件后，第三阶段才算通过。

## 数据系统

* 图生成与 benchmark 主逻辑分离；
* 至少支持 uniform、power-law、hotspot、clustered；
* spike trace 至少支持 IID、fanout-correlated、temporal burst；
* 所有 workload 有固定 seed 和 workload ID；
* workload manifest 可保存并复用。

## operator benchmark

* native 与 adapted 不再是假定名称，而是真正不同入口；
* zero-copy transform 不再伪装成 GPU kernel latency；
* activity、active edges、collision 都写入 CSV；
* operator-only 与 full-RSNN 完全分离；
* correctness 使用固定 trace。

## full RSNN

* 记录实际 firing rate；
* 使用 target activity calibration；
* 报告 active-edge workload；
* 不再仅以 input event rate解释性能；
* 同一 workload 下所有 provider 使用相同输入和初始状态。

## padding

* physical workload 真正进入 provider；
* correctness 裁剪到 logical neurons；
* CSV 中 physical shape 与 padding ratio 是实测值；
* 不进行 batch padding。

## 工程可信度

* 每个 device provider 完成一次 profiler audit；
* allocation metadata 的定义清晰；
* `workspace_bytes` 与 `total_prepared_bytes` 不重复解释；
* 不同 timing scope 不计算 speedup。

---

## 推荐施工优先级

第三阶段建议按这个顺序推进：

1. 修正 operator benchmark 的 native/adapted 定义；
2. 建立 `BenchmarkWorkload` 和 graph/spike stats；
3. 实现 uniform、power-law、hotspot 三种 synthetic graph；
4. 实现 IID、fanout-correlated、temporal 三种 spike trace；
5. 建立 operator-only sweep；
6. 给 closed-loop RSNN 增加实际 firing-rate 与 active-edge 统计；
7. 增加 target activity calibration；
8. 真正接入 N-padding；
9. 做一次 provider profiler audit；
10. 最后再扩展完整数据集和绘图。

