第二阶段
第二阶段目标：修正算子适配，消除非算子开销干扰
第二阶段不应马上大规模重写所有 SOTA 后端，而应先回答两个问题：
当前每个算子的计时中混入了哪些布局转换、分配、压缩或 CPU–GPU 通信？
哪些干扰可以在 Python adapter 层消除，哪些必须修改 C++/CUDA 接口？
核心原则是：
保持 B=1 RSNN workload 不变，为各算子提供最合理的合法调用方式；不能支持的算子不通过改变 workload 强行加入。
 
一、第二阶段最终应交付什么
建议第二阶段完成后，benchmark 中的 provider 分成三类。
A. 原生 GPU-resident provider
包括：
cuSPARSE direct；
persistent；
Sputnik；
VDHA，前提是接口经过修正。
要求：
输入在 GPU
输出在 GPU
无 timed allocation
无 D2H/H2D
使用当前 CUDA stream
workspace 预分配
每次执行的布局转换明确


这些 provider 可以进入统一的 GPU execution 表。
B. GPU-resident，但存在必要适配成本
例如：
算子要求 NB，RSNN 状态使用 BN；
算子要求 sparse spike list；
算子需要 padding；
算子需要额外格式转换。
这类应同时测：
native operator
adapted operator
full RSNN window


C. host-controlled fallback
包括当前的：
MH-SpGEMM；
DTC-SpMM；
FlashSparse。
保留公开 wrapper 结果，但只进入：
public-wrapper end-to-end


不能与 GPU execution 结果直接排名。
 
二、施工顺序
Step 0：补完第一阶段遗留的 reset 语义
当前 persistent runner 虽然实现了：
def reset():
    state.v.copy_(initial_v)
    state.psc.copy_(initial_psc)
    ...


但 time_samples_ms() 只在整个 warmup 阶段和整个 timing 阶段前各 reset 一次：
_reset_runner(fn)
for _ in range(warmup):
    fn()


_reset_runner(fn)
for repeat:
    fn()


因此多个 timed repeat 之间仍是连续状态，而非每次从相同初始状态运行。
这需要在第二阶段开始前修掉，否则后面的 adapter 优化结果仍可能受 activity trajectory 变化影响。
建议增加 reset policy
class BenchmarkRunner:
    reset_policy: Literal[
        "stateless",
        "before_phase",
        "before_sample",
    ]


PyTorch forward：stateless
CUDA Graph full-window：通常 stateless
persistent mutable state：before_sample
连续仿真实验：before_phase
伪代码
def run_warmup(runner, warmup):
    runner.reset()


    for _ in range(warmup):
        if runner.reset_policy == "before_sample":
            runner.reset()
        runner.run()


    synchronize()



def time_gpu_samples(runner, repeat):
    samples = []


    for _ in range(repeat):
        if runner.reset_policy == "before_sample":
            runner.reset()


        start.record()
        runner.run()
        end.record()


        samples.append((start, end))


    synchronize()
    return elapsed_times(samples)


注意：如果 reset() 本身不应计入 GPU execution，则必须在 start.record() 之前调用。
 
三、统一 provider prepare 接口
当前 metadata 主要由：
provider_measurement_metadata(provider)


按 provider 名称静态推断。这只能表达预期，不能表达实际准备出的 buffer、padding 和转换。
例如目前：
padding_ratio = 1.0
workspace_bytes = 0


对大量 provider 只是默认值，而不是实际测得值。
第二阶段应将 metadata 从“名称推断”改为“prepare 结果返回”。
建议结构
@dataclass
class PreparedMetadata:
    logical_shape: tuple[int, ...]
    physical_shape: tuple[int, ...]
    input_layout: str
    operator_layout: str


    index_dtype: str
    value_dtype: str
    compute_dtype: str


    workspace_bytes: int
    persistent_bytes: int
    padding_ratio: float


    uses_host_control: bool
    uses_internal_sync: bool
    uses_d2h: bool
    uses_h2d: bool
    allocates_during_run: bool
    graph_capturable: bool


    layout_transform_mode: str
    spike_representation: str


@dataclass
class BenchmarkRunner:
    run_fn: Callable
    reset_fn: Callable


    metadata: PreparedMetadata
    reset_policy: str


prepare 伪代码
class ProviderAdapter:
    def supports(self, workload) -> SupportResult:
        ...


    def prepare(self, workload) -> BenchmarkRunner:
        logical_shape = workload.shape
        physical_shape = align_shape(logical_shape)


        buffers = allocate_buffers(physical_shape)
        workspace = allocate_workspace(...)
        native_handle = create_plan(...)


        metadata = PreparedMetadata(
            logical_shape=logical_shape,
            physical_shape=physical_shape,
            padding_ratio=numel(physical_shape) / numel(logical_shape),
            ...
        )


        return BenchmarkRunner(
            run_fn=run,
            reset_fn=reset,
            metadata=metadata,
        )


 
四、Sputnik 适配修正
当前 Sputnik 每个 timestep 执行：
rhs = spikes.transpose(0, 1).contiguous()
output = torch.empty(...)
sputnik_compute_device(...)
return output.transpose(0, 1)


这会带来两个问题：
每步创建输出 tensor；
.transpose().contiguous() 可能产生真实 copy。
但主任务固定 B=1，此时 [1, N] 和 [N, 1] 的元素存储顺序完全相同，可以直接用 view，不需要转置复制。
第二阶段目标
B=1 路径
rhs = spikes.view(N, 1)
output_nb = preallocated_output
sputnik_compute_device(rhs, output_nb)
return output_nb.view(1, N)


整个过程中：
不分配；
不 copy；
只改变 view；
保持 device-resident。
泛化路径
虽然主 benchmark 是 B=1，adapter 最好仍支持：
B=1：zero-copy view
B>1：显式 pack kernel 或 native BN 支持


不能继续依赖 timestep 内 .contiguous()。
伪代码
def prepare_sputnik(weight, case):
    handle = sputnik_prepare(...)


    output_nb = torch.empty(
        (N, B),
        device="cuda",
        dtype=torch.float32,
    )


    if B == 1:
        def run_matmul(spikes_bn):
            rhs_nb = spikes_bn.view(N, 1)


            sputnik_compute_device(
                handle,
                rhs_nb.data_ptr(),
                output_nb.data_ptr(),
                current_stream(),
            )


            return output_nb.view(1, N)


        transform_mode = "zero_copy_view"


    else:
        rhs_nb = torch.empty((N, B), device="cuda")


        def run_matmul(spikes_bn):
            pack_bn_to_nb(spikes_bn, rhs_nb)
            sputnik_compute_device(...)
            return output_nb.transpose(0, 1)


        transform_mode = "gpu_pack_kernel"


    return runner(
        workspace=[rhs_nb?, output_nb],
        layout_transform_mode=transform_mode,
        allocates_during_run=False,
    )


验证实验
对 Sputnik 增加三个 microbenchmark：
sputnik_native_matmul
sputnik_layout_only
sputnik_adapted_matmul


分别测：
native = time(sputnik(rhs_native))
layout = time(pack_or_view)
adapted = time(pack + sputnik)


判断 slowdown 是否来自 kernel 本身还是 layout adapter。
 
五、VDHA 适配修正
当前 VDHA 使用：
cbn_vdha_prepare_dense(...)
cbn_vdha_compute_dense_device(...)


输入是完整 dense spike vector。
这虽然是 device-resident，但未必体现 VDHA 的核心 SpMSpV 路径。第二阶段应先核查底层 C ABI，而不是直接假定 dense 接口就是最终版本。
分两步施工
5.1 先保留 dense-input 路径，但消除分配
当前每步：
output = torch.empty(...)


应改为 prepare 时预分配：
output = torch.empty(N, device=device)


运行时只传 pointer。
def run_dense(spikes):
    vdha_compute_dense_device(
        handle,
        spikes.data_ptr(),
        output.data_ptr(),
        stream,
    )
    return output.view(1, N)


标记：
provider = vdha_dense_input
spike_representation = dense


5.2 调查并实现 sparse-vector 路径
目标接口：
active_indices
active_values 或隐含值 1
active_count


理想 CUDA API：
vdha_compute_sparse_device(
    handle,
    active_indices,
    active_count,
    output,
    stream
);


RSNN UPDATE 阶段应直接生成 spike list，而不是：
dense z
→ 扫描/压缩
→ sparse list


如果暂时只能由 dense spike 转 sparse，则要把压缩成本明确拆出来。
伪代码
def prepare_vdha_sparse(weight, case):
    handle = vdha_prepare(...)


    active_indices = torch.empty(N, dtype=int32, device="cuda")
    active_count = torch.zeros(1, dtype=int32, device="cuda")
    output = torch.empty(N, device="cuda")


    def run(spikes_dense):
        dense_to_indices(
            spikes_dense,
            active_indices,
            active_count,
        )


        vdha_compute_sparse_device(
            handle,
            active_indices,
            active_count,
            output,
            current_stream(),
        )


        return output.view(1, N)


    metadata.spike_representation = "dense_to_sparse_indices"
    metadata.layout_transform_mode = "gpu_compaction"


后续若统一 UPDATE kernel 能直接输出 spike list：
def run_from_native_spike_list(spike_list):
    vdha_compute_sparse_device(
        handle,
        spike_list.indices,
        spike_list.count,
        output,
        stream,
    )


需要输出的对比
VDHA dense-input
VDHA sparse-input，包括压缩
VDHA sparse-input，不包括压缩


这样才能知道：
VDHA kernel 自身性能；
dense-to-sparse 转换成本；
与 persistent 原生 spike queue 的差异。
 
六、统一 timed allocation 检查
第二阶段应明确保证 GPU-resident provider 在 timed run() 内不进行：
torch.empty
torch.zeros
clone
contiguous
to(...)
cpu()
numpy()


实现方式
在 prepare 时分配全部 buffer：
@dataclass
class ProviderBuffers:
    input_native: Tensor | None
    output_native: Tensor
    recurrent: Tensor
    spike_indices: Tensor | None
    spike_count: Tensor | None
    workspace: tuple[Tensor, ...]


运行时只操作已有 tensor。
简单审计方法
先进行代码静态检查，再增加运行时 allocation 观察。
伪代码
before_alloc = torch.cuda.memory_allocated()
before_reserved = torch.cuda.memory_reserved()


for _ in range(check_repeat):
    runner.run()


synchronize()


after_alloc = torch.cuda.memory_allocated()
after_reserved = torch.cuda.memory_reserved()


metadata.run_allocated_bytes = after_alloc - before_alloc
metadata.run_reserved_delta = after_reserved - before_reserved


这不能捕获 allocator 内所有瞬时复用，但能作为初筛。
更严格可通过 profiler 检查：
aten::empty
aten::contiguous
cudaMalloc
cudaFree
cudaMemcpy


验收条件：
GPU-resident provider:
    run_allocated_bytes == 0
    无 cudaMalloc/cudaFree
    无 D2H/H2D


 
七、内存布局和地址对齐审计
当前代码已经有 input_layout 和 required_layout，但 need_layout_transform 只是字符串比较，无法说明实际是否发生 copy。
第二阶段应记录实际布局。
Tensor 审计函数
def inspect_tensor(tensor, required_alignment):
    return {
        "shape": tuple(tensor.shape),
        "stride": tuple(tensor.stride()),
        "contiguous": tensor.is_contiguous(),
        "storage_offset": tensor.storage_offset(),
        "address_alignment": tensor.data_ptr() % required_alignment,
        "dtype": str(tensor.dtype),
    }


prepare 时检查
def assert_native_buffer(tensor, spec):
    if spec.require_contiguous:
        assert tensor.is_contiguous()


    if tensor.data_ptr() % spec.address_alignment != 0:
        raise ProviderNotApplicable(
            "buffer address does not satisfy alignment"
        )


    if not stride_matches(tensor, spec.layout):
        raise ProviderNotApplicable(
            "unsupported physical stride"
        )


CSV 中不需要存所有 tensor 细节，但建议保存：
input_contiguous
input_address_mod
index_dtype
value_dtype
physical_n
physical_batch
layout_transform_mode
layout_transform_in_timing


 
八、padding 架构施工
当前 ProviderCapability.n_alignment 已经存在，但只是拒绝非对齐 workload。更合理的是：
算子允许通过 zero-padding 执行时，adapter 应自动 pad，并承担 padding 成本。
数据结构
@dataclass
class PhysicalWorkload:
    logical_n: int
    physical_n: int


    logical_batch: int
    physical_batch: int


    logical_nnz: int
    physical_nnz: int


    valid_neuron_mask: Tensor


CSR padding
对于方阵：
physical_n = ceil_div(logical_n, n_alignment) * n_alignment


不需要人为增加 nnz，只需扩展 shape 和 row pointer：
new_indptr = zeros(physical_n + 1)


new_indptr[:logical_n + 1] = original_indptr
new_indptr[logical_n + 1:] = original_nnz


额外 neuron：
没有入边；
没有出边；
输入为零；
输出被切回 logical shape。
伪代码
def pad_csr(matrix, n_alignment):
    n_logical = matrix.shape[0]
    n_physical = align_up(n_logical, n_alignment)


    if n_physical == n_logical:
        return matrix, PaddingInfo.identity()


    padded_indptr = torch.empty(
        n_physical + 1,
        dtype=matrix.indptr.dtype,
        device=matrix.indptr.device,
    )


    padded_indptr[:n_logical + 1] = matrix.indptr
    padded_indptr[n_logical + 1:] = matrix.nnz


    padded = CSR(
        indptr=padded_indptr,
        indices=matrix.indices,
        data=matrix.data,
        shape=(n_physical, n_physical),
    )


    return padded, PaddingInfo(
        logical_n=n_logical,
        physical_n=n_physical,
        padding_ratio=n_physical / n_logical,
    )


结果裁剪
def logical_result(result, logical_n):
    return RSNNResult(
        spikes=result.spikes[..., :logical_n],
        v=result.v[..., :logical_n],
        psc=result.psc[..., :logical_n],
    )


注意事项
不要为了适配 B=8/16/32 的算子自动 pad batch，因为这会虚构额外 RSNN 实例，并改变任务计算量。
因此：
N padding：通常允许
B padding：主 B=1 benchmark 中禁止


除非 tensor-core 算子能够在 mask 后证明只计算一个逻辑向量，但即使如此，物理执行仍是更宽 SpMM，应单独分类，而不是主结果。
 
九、host-controlled provider 的处理
当前 MH-SpGEMM、DTC-SpMM、FlashSparse 每步可能包含 CPU 数组转换、wrapper 同步和 H2D/D2H。
第二阶段不建议立刻重写三套 CUDA backend。先把它们正式隔离到 public-wrapper 路径。
增加 execution class
ExecutionClass = Literal[
    "device_native",
    "device_adapted",
    "host_wrapper",
]


分类
persistent                 device_native
cuSPARSE direct            device_native
Sputnik B=1 view           device_adapted
VDHA dense                 device_adapted
MH-SpGEMM current wrapper  host_wrapper
DTC current wrapper        host_wrapper
FlashSparse current wrapper host_wrapper


CSV 和 speedup
只允许：
def comparable(a, b):
    return (
        a.execution_class == b.execution_class
        and a.timing_scope == b.timing_scope
        and a.logical_batch == b.logical_batch
        and a.logical_n == b.logical_n
    )


host wrapper 表可以保留绝对 latency，但不与 persistent 输出默认 speedup。
后续 raw CUDA adapter 优先级
若要继续实现：
DTC-SpMM；
FlashSparse；
MH-SpGEMM。
但由于主任务为 B=1，DTC/FlashSparse 即使做出 device adapter，也大概率仍然不适用。因此第二阶段不值得优先投入大量时间。
 
十、增加 operator microbenchmark，隔离适配成本
第二阶段最好新增一个较小的文件：
benchmark_sparse_operator_adapters.py


不运行完整 LIF，只测：
recurrent = W @ spike


这样可以快速定位 adapter 的工程问题，而不会受到 threshold trajectory 影响。
输入
spikes = generate_fixed_spike_trace(
    T=128,
    N=N,
    activity=activity,
    seed=seed,
)


每个 provider 提供三个 runner
native_runner
adapted_runner
full_rsnn_runner


例如 Sputnik：
native:
    输入已经是 NB


adapted:
    输入是公共 BN
    包括必要布局适配


full:
    LIF + adapted recurrent


VDHA：
native sparse:
    输入是 active_indices


adapted sparse:
    dense spike → compact → VDHA


dense:
    dense spike → VDHA dense API


伪代码
for activity in activities:
    spike_trace = make_spikes(activity)


    for provider in providers:
        runner = provider.prepare_operator(
            matrix,
            spike_trace,
        )


        correctness = compare_operator_output(
            runner,
            reference,
        )


        native_samples = time_runner(
            runner.native,
        )


        adapted_samples = time_runner(
            runner.adapted,
        )


        write_row(
            provider=provider,
            activity=activity,
            native_ms=median(native_samples),
            adapted_ms=median(adapted_samples),
            adaptation_ms=adapted_ms - native_ms,
        )


建议 activity 先只扫：
0.1%, 1%, 5%, 10%


第二阶段重点是核验 adapter，不是立即做完整论文实验。
 
十一、建议的文件改动
1. benchmark_rsnn_cudagraph_compare.py
负责：
workload；
provider 调度；
correctness；
full-window timing；
CSV 输出。
需要修改：
BenchmarkRunner metadata
reset policy
prepare 返回实际 metadata
execution_class
physical/logical shape
比较资格判断


2. sota_rsnn_cudagraph.py
负责具体 adapter。
重点修改：
Sputnik 预分配 output
Sputnik B=1 zero-copy view
VDHA 预分配 output
VDHA dense/sparse 路径区分
统一 native handle 生命周期
返回 BenchmarkRunner，而非裸 callable


3. 新增 benchmark_sparse_operator_adapters.py
负责：
固定 spike trace
native/adapted operator timing
layout/compaction 成本
activity sweep


4. 可选新增 provider_common.py
如果主文件继续膨胀，可以把这些提取出来：
BenchmarkRunner
PreparedMetadata
ProviderCapability
PaddingInfo
SupportResult


 
十二、完整施工伪代码
Provider prepare
def prepare_provider(provider, workload):
    support = provider.supports(workload)


    if not support.supported:
        raise ProviderNotApplicable(support.reason)


    physical = prepare_physical_workload(
        workload,
        n_alignment=provider.n_alignment,
        allow_n_padding=True,
        allow_batch_padding=False,
    )


    buffers = provider.allocate_buffers(physical)
    handle = provider.create_handle(
        matrix=physical.matrix,
        buffers=buffers,
    )


    runner = BenchmarkRunner(
        run_fn=lambda: provider.run(
            handle,
            physical,
            buffers,
        ),
        reset_fn=lambda: provider.reset(buffers),
        reset_policy=provider.reset_policy,
        metadata=PreparedMetadata(
            logical_shape=workload.shape,
            physical_shape=physical.shape,
            padding_ratio=physical.padding_ratio,
            workspace_bytes=count_bytes(buffers.workspace),
            execution_class=provider.execution_class,
            layout_transform_mode=provider.layout_transform_mode,
            allocates_during_run=False,
        ),
    )


    audit_runner(runner)
    return runner


Benchmark 生命周期
def benchmark_runner(runner, reference):
    # correctness 与 timing 使用独立生命周期
    runner.reset()
    result = runner.run()
    synchronize()


    logical_result = crop_to_logical_shape(
        result,
        runner.metadata.logical_shape,
    )


    correctness = compare(
        logical_result,
        reference,
    )


    # warmup
    for _ in range(warmup):
        if runner.reset_policy == "before_sample":
            runner.reset()
        runner.run()


    synchronize()


    # timing
    samples = []
    for _ in range(repeat):
        if runner.reset_policy == "before_sample":
            runner.reset()


        start.record()
        runner.run()
        end.record()


        samples.append((start, end))


    synchronize()


    return summarize(samples, correctness)


比较资格
def comparison_group(row):
    return (
        row["dataset"],
        row["logical_n"],
        row["logical_batch"],
        row["t_steps"],
        row["timing_scope"],
        row["execution_class"],
        row["state_semantics"],
    )


for group in group_rows(rows, key=comparison_group):
    baseline = select_baseline(group)


    for row in group:
        row["speedup"] = baseline.latency / row.latency


 
十三、阶段验收标准
第二阶段完成，不以“所有 SOTA 都运行成功”为标准，而以 benchmark 是否可解释为标准。
必须满足
persistent 每个独立 latency sample 从相同状态开始；
Sputnik timed path 中无 torch.empty() 和 .contiguous()；
B=1 Sputnik 使用 zero-copy view；
VDHA output 预分配；
VDHA 明确标记 dense 或 sparse spike 输入；
GPU execution provider 中无 D2H/H2D；
host wrapper 与 GPU execution 不互算 speedup；
CSV 记录 logical/physical shape；
CSV 记录实际 padding ratio；
CSV 记录 execution class；
CSV 记录 layout transform 是否在 timing 内；
provider metadata 来自实际 prepare，而不是只按名字推断；
至少完成一组 operator-only adapter 验证。
可暂缓
为 DTC-SpMM 编写 raw CUDA adapter；
为 FlashSparse 编写 raw CUDA adapter；
为 MH-SpGEMM 编写 device-only 接口；
完整 activity sweep；
完整 synthetic workload 系统；
统一 fused UPDATE kernel。
这些更适合进入第三阶段或后续扩展。
 
十四、推荐的实际执行清单
第二阶段可以按下面顺序施工：
任务 1：runner 生命周期
增加 reset_policy；
修正 per-sample reset；
correctness 与 timing 独立 reset；
加连续仿真模式开关，但默认关闭。
任务 2：Sputnik
output 预分配；
B=1 zero-copy view；
移除 run 内 .contiguous()；
记录 workspace/layout；
做 native 与 adapted microbenchmark。
任务 3：VDHA
output 预分配；
标记 dense-input；
核查 sparse C ABI；
能实现则加入 sparse-input；
不能实现则明确保留 dense-adapted 标签。
任务 4：metadata 和 padding
PreparedMetadata；
logical/physical shape；
N-padding；
地址和 stride 审计；
禁止主实验进行 B-padding。
任务 5：host wrapper 隔离
增加 execution_class；
调整 speedup group；
DTC、FlashSparse、MH-SpGEMM 只输出 public-wrapper E2E；
去除默认主排名。
任务 6：operator adapter 实验
固定 spike trace；
参考输出；
native/adapted/full 三层计时；
先测试 4 个 activity 点；
确认 adapter 成本占比。
完成这一步之后，benchmark 才适合进入第三阶段：独立数据生成、指定 spike activity、算子级与完整 RSNN 计时分离，以及统一 UPDATE 路径。
