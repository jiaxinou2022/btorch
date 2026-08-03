# Benchmark 第一阶段修正执行计划

## 总目标

第一阶段不追求增加新的 baseline，而是**修复 benchmark 的实验语义，使后续优化和论文实验有可信基础**。

核心原则：

> 先固定 RSNN 推理任务，再让不同算子适配任务；不要为了适配某个算子改变任务定义。

第一阶段完成后，应达到：

* 所有主要 provider 在 **同一 B=1 RSNN workload** 下比较；
* 所有 timing 都明确属于 GPU-only 或 E2E；
* persistent 与其他方法从同一初始状态运行；
* correctness 与 timing 不互相污染；
* 不同 provider 的 UPDATE / recurrent 部分差异被明确记录。

---

# Phase 0：建立 benchmark 当前问题清单与修改分支

## 目标

避免直接修改导致 benchmark 语义变化不可追踪。

建立：

```
benchmark/
├── benchmark_rsnn_cudagraph.py      # 当前版本备份
├── benchmark_rsnn_v2.py              # 第一阶段修正版
├── provider/
│   ├── base.py
│   ├── torch_provider.py
│   ├── persistent_provider.py
│   └── sota_provider.py
```

记录当前版本：

```
commit:
baseline_before_fix

issues:
- batch mismatch
- timing mismatch
- persistent state contamination
- correctness contamination
- provider capability missing
```

---

# Phase 1：删除自动 B=32 逻辑，恢复统一 B=1

## 当前问题

目前：

```python
SOTA_FIXED_BATCH_SIZE = {
    "vdha_cudagraph":1,
    "mh_spgemm_eager":32,
    "dtc_spmm_eager":32,
    "flashsparse_eager":32
}
```

导致 benchmark 自动生成多个任务。

---

## 修改目标

主 benchmark：

```text
B = 1
```

所有 provider：

```
N neurons
T timesteps
B=1
same CSR
same input
same initial state
```

---

## 删除

删除：

```python
plan_provider_batches()
```

以及：

```python
--auto-sota-batches
```

---

## 新逻辑

```python
case = BenchCase(
    n_neuron=N,
    batch_size=1,
    t_steps=T,
    ...
)


for provider in providers:

    if not provider.supports(case):
        result = {
            "status":"not_applicable"
        }
    else:
        benchmark(provider)
```

---

## 新增 capability 判断

不要让 provider 在运行时才报错。

伪代码：

```python
class ProviderCapability:

    def supports(self, case):

        if case.batch_size not in self.batch_sizes:
            return False

        if case.n_neuron % self.n_alignment != 0:
            return False

        return True
```

例如：

```python
MHSpGEMM:

batch_sizes=[32,64,128]

DTC:

batch_multiple=16

VDHA:

batch_sizes=[1]
```

输出：

```
provider          status
--------------------------------
persistent        run
vdha              run
sputnik           run
dtc_spmm          not_applicable(B=1)
mh_spgemm         not_applicable(B=1)
```

---

# Phase 2：统一 benchmark runner 生命周期

## 当前问题

现在：

```
correctness
    |
    v
warmup
    |
    v
timing
```

对于 stateful provider：

```
state0
 |
correctness
 |
state1
 |
warmup
 |
state2
 |
timing
```

不同 provider 实际运行状态不同。

---

## 修改目标

所有 provider 都遵循：

```
prepare()
 |
reset()
 |
correctness()
 |
reset()
 |
warmup()
 |
reset()
 |
timing()
```

---

## Runner 接口设计

新增：

```python
class BenchmarkRunner:

    def prepare(self):
        pass

    def reset(self):
        pass

    def run(self):
        pass
```

---

## benchmark 流程

修改为：

```python
provider.prepare()


if check:

    provider.reset()

    result = provider.run()

    check(result)


provider.reset()


for i in range(warmup):
    provider.run()


provider.reset()


samples = measure(
    provider.run
)
```

---

# Phase 3：修复 persistent state reset

## 当前风险

persistent：

```python
state = make_empty_state()
```

只创建一次。

如果：

```python
persistent_snn_forward()
```

修改：

```python
state.v
state.psc
workspace.queue
workspace.counter
```

那么 repeat 不等价。

---

## 修改方案

保存初始 snapshot：

prepare:

```python
initial_v = state.v.clone()
initial_psc = state.psc.clone()
initial_workspace = workspace.clone()
```

reset:

```python
def reset():

    state.v.copy_(initial_v)

    state.psc.copy_(initial_psc)

    workspace.reset()
```

---

## workspace reset

重点检查：

persistent kernel 可能修改：

* task queue head
* active spike list
* scheduler counter
* temporary buffer

因此：

```python
class PersistentWorkspace:

    def reset(self):

        self.queue_ptr.zero_()

        self.counter.zero_()

        self.metadata.copy_(initial_metadata)
```

---

## 验证实验

运行：

```
same input
same state
run 10 times
```

检查：

```python
for i in range(10):

    reset()

    output[i]=run()


assert all_equal(output)
```

否则 benchmark 不可信。

---

# Phase 4：统一 timing scope

## 当前问题

存在：

| provider   | timer        |
| ---------- | ------------ |
| torch      | CUDA event   |
| persistent | CUDA event   |
| MH-SpGEMM  | wall clock   |
| GeNN       | native timer |

不能直接 speedup。

---

## 第一阶段只保留：

## GPU execution latency

定义：

```
input already on GPU

state already allocated

output remains GPU

measure current stream execution
```

统一：

```python
cudaEventRecord(start)

provider.run()

cudaEventRecord(end)

cudaEventSynchronize(end)
```

---

## 对无法满足的 provider

例如：

MH-SpGEMM：

```
GPU->CPU
numpy
CPU->GPU
```

标记：

```
timing_scope="host_controlled"
```

不要进入主 speedup。

---

## CSV 新字段

增加：

```text
timing_scope

gpu_execution

host_controlled

public_api_e2e
```

例如：

```
persistent:
gpu_execution


mh_spgemm:
host_controlled
```

---

# Phase 5：拆分 operator benchmark 和 RSNN benchmark

## 当前问题

完整 RSNN：

```
UPDATE
 |
Spike generation
 |
Sparse propagation
 |
PSC update
```

不同 provider 可能 UPDATE kernel 数量不同。

---

## 第一阶段不完全重构，但增加标记。

增加：

```python
measure_mode
```

两个模式：

```
full_rsnn
operator_only
```

---

# 5.1 operator-only 模式

输入：

提前生成：

```
spike[t]
```

执行：

```python
for t in range(T):

    current = provider.sparse_op(spike[t])
```

计时：

```
only sparse propagation
```

伪代码：

```python
def benchmark_operator(provider, spikes):

    provider.prepare()

    for t in range(T):

        start.record()

        provider.run(spikes[t])

        end.record()
```

输出：

```
operator_latency_ms
```

---

# 5.2 full RSNN 模式

保留：

```
LIF
+
propagation
+
PSC
```

但记录：

```text
provider_type
fusion_level
```

例如：

| provider   | fusion      |
| ---------- | ----------- |
| torch csr  | none        |
| cuSPARSE   | sparse only |
| persistent | full fusion |

---

# Phase 6：修正 provider 输入布局问题

## 当前问题

例如 Sputnik：

```python
rhs = spikes.transpose().contiguous()
```

每 timestep 产生 layout conversion。

---

## 第一阶段最低修改

不改变 kernel，只记录。

增加：

```python
layout_conversion_time
```

或者：

```python
provider.prepare()

return:

{
 input_layout:"BN",

required_layout:"NB",

need_transform:true
}
```

---

## 第二阶段再优化

预分配：

```python
rhs_buffer=torch.empty(N,B)
output=torch.empty(N,B)
```

避免：

```python
torch.empty()
contiguous()
```

---

# Phase 7：重新定义 speedup 计算

## 当前：

所有 provider：

```python
latency_ms
```

直接比较。

---

## 修改：

speedup 只允许：

```python
same timing_scope
same benchmark_mode
same batch_size
```

伪代码：

```python
def compute_speedup(a,b):

    if a.scope != b.scope:
        return NaN

    if a.batch != b.batch:
        return NaN

    return b.latency/a.latency
```

---

# Phase 8：第一阶段验收实验

完成修改后，不马上跑大规模。

先：

```
N=4096
T=32
B=1
uniform graph
fanout=32
activity=1%
```

测试：

providers:

```
torch_csr_cudagraph

cusparse_direct_cudagraph

vdha_cudagraph

sputnik_cudagraph

persistent_plain

persistent_binning

persistent_spike_block
```

检查：

## 1. repeat稳定性

```
std/mean < 5%
```

## 2. persistent确定性

```
10次reset后输出一致
```

## 3. timing一致

所有：

```
timing_scope=gpu_execution
```

## 4. speedup合理

例如：

不会出现：

```
MH-SpGEMM
0.001ms
1000x faster
```

这种明显错误。

---

# 第一阶段完成标准

最终 benchmark v2 应满足：

## 实验定义

```
Task:
single RSNN inference

Batch:
B=1

State:
identical initial state

Input:
same spike trajectory/input

Timing:
GPU execution only

Comparison:
same scope only
```

## CSV 至少包含

```
provider

dataset

N

T

B

nnz

activity

timing_scope

fusion_level

input_layout

padding_ratio

workspace_bytes

latency_ms

std

correctness_status
```

---

# 推荐执行顺序（实际开发）

按优先级：

```
Day 1:
    删除auto batch
    B统一1
    capability系统

Day 2:
    runner reset机制
    persistent state检查

Day 3:
    timing_scope重构
    speedup过滤

Day 4:
    operator/full RSNN模式拆分

Day 5:
    小规模验证
    重新生成baseline结果
```
