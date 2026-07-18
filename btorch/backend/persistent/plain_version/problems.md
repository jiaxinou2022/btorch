
>真正严重的架构问题，是输出与临时缓冲区在稀疏发放场景下被 dense 化，以及 persistent kernel 每个时间步存在过多全网格阶段和同步。**

下面按严重程度展开。

---

# 二、最严重的问题：事件输出缓冲区实际上是 `T × B × N` dense 存储

虽然连接矩阵没有 dense 化，但代码无条件分配了：

```cpp
event_indices_full =
    torch::empty({t_steps * batch_size * n_neuron}, options_i);
```

位置：

* 普通版本：`persistent_snn.cpp:291-294`
* binned 版本：`persistent_snn.cpp:488-491`

每次神经元发放时，又无条件执行：

```cpp
const int rank = atomicAdd(event_counts + bucket, 1);
event_indices_full[bucket * n_neuron + rank] = n;
```

位置：

* 普通版本：`persistent_snn_kernel.cu:101-107`
* binned 版本：`persistent_snn_kernel.cu:257-263`

这本质上是：

```text
每个 (time, batch) bucket 预留 N 个 spike 槽位
```

即使发放率只有 1%，仍然分配整个：

[
TBN
]

规模的缓冲区。

更关键的是，**即使 `return_mode="dense"`、根本不需要事件输出，这部分分配、计数原子和写入仍然执行**。C++ 只是在 kernel 结束后跳过了 compact：

```cpp
if (return_events) {
    ...
} else {
    ...
}
```

但 kernel 内并不知道 `return_events`。

## 影响

这不只是显存浪费，还包含：

1. 每个 spike 多一次 `atomicAdd(event_counts)`；
2. 每个 spike 多一次全局写；
3. 每次 forward 多一个 `TBN` 临时分配；
4. 增加 allocator 压力；
5. 扩大工作集，污染 L2；
6. 和真正用于 recurrent fanout 的 spike queue 重复保存同一批 spike。

例如 `T=128, N=4166`：

* `B=1`：约 2.13 MB；
* `B=32`：约 68.3 MB；
* `B=64`：约 136.5 MB。

而这只是 `event_indices_full` 一个数组。

## 最小改法

这是最值得首先修改的部分。

### `return_mode="dense"`

完全不分配：

```text
event_counts
event_indices_full
```

kernel 中也完全删除相应的 atomic 和写入。

只保留当前时间步用于 recurrent fanout 的 spike queue。

### `return_mode="events"`

不建议保留 `TBN` 临时区。可以考虑两种架构。

第一种是窗口级全局 append queue：

```text
event_indices_capacity
event_bucket
global_event_count
```

每个 block/warp 先局部 compact，再一次性申请一段全局空间。空间按预计 spike 数或可扩展 capacity 分配，而不是按 `TBN` 分配。

第二种是复用 dense spike 输出进行后处理，但这只适合本来就必须返回 dense spikes 的 `"both"` 模式。

## 优先级

**最高。**

这是明确存在、容易隔离、又不涉及 prespan 主算法语义的修改。

---

# 三、另一个 dense 化：无条件生成完整 `dense_spikes[T,B,N]`

C++ 无条件分配：

```cpp
auto dense_spikes =
    torch::empty({t_steps, batch_size, n_neuron}, options_f);
```

位置：

* 普通版本：`persistent_snn.cpp:285`
* binned 版本：`persistent_snn.cpp:478`

kernel 又对每个细胞每个时间步写一次：

```cpp
dense_spikes[(t * batch_size + b) * n_neuron + n] = spike;
```

位置：

* 普通版本：`persistent_snn_kernel.cu:82`
* binned 版本：`persistent_snn_kernel.cu:235`

即使调用：

```python
return_mode="events"
```

Python 最终把 dense 输出丢掉，kernel 仍然完整写了 `TBN` 个 float。

这正是典型的“大量 0 浪费”：发放率可能只有几个百分点，但每个神经元每个时间步都写一个 `0.0f`。

## 修改建议

把返回模式真正下沉到 CUDA kernel：

```text
write_dense_spikes
write_event_spikes
```

形成三条专门路径，最好通过模板或独立 kernel 实例编译，避免运行时分支污染主循环：

```text
dense-only:
    写 dense_spikes
    不写 event history

events-only:
    不写 dense_spikes
    只写 compact event stream

both:
    两者都写
```

这个修改不会破坏 recurrent fanout，因为 fanout 使用的是当前时间步的：

```text
spike_queue_batch
spike_queue_pre
```

它与最终输出格式完全可以解耦。

---

# 四、persistent 的核心架构缺陷：每个时间步有太多 grid-wide barrier


位置：

* 普通版本：`persistent_snn_kernel.cu:57,71,110,155`
* binned 版本：`persistent_snn_kernel.cu:210,224,266,290,319`

这可能是当前最高层次的性能问题。

## 如何改而不大幅重构

优先考虑减少阶段，而不是立刻做复杂的异步流水。

### 可以直接合并的阶段

下一步可以处理 external input：

* 可以保证外部事件保证同一 `(b,n)` 不重复，直接在 LIF update 时查询或 merge；
* 避免每步清空整个 `input_current[B,N]`。

这样可能去掉：

```text
清零 input_current
第一处 grid.sync
external event 阶段
第二处 grid.sync
```

### 更进一步

对于仅少量 external events 的情况，不要先散射到 dense `input_current`，再让所有 cell 读取它。

现在路径是：

```text
sparse input event
    ↓ atomic scatter
dense input_current[B,N]
    ↓ full LIF scan
```

这既写 dense buffer，又需要清零。

可改成：

* 稀疏输入与 recurrent PSC 分开；
* 输入事件直接作用到目标状态；
* 或按 cell block 建局部输入事件表；
* 至少使用 generation tag，避免每步清零整个数组。

---

# 五、binned 版本存在结构性串行化：high 和 low 完全不能重叠

binned kernel 的执行顺序是：

```text
所有 high-fanout task 执行完
grid.sync()
所有 low-fanout task 执行完
grid.sync()
```

位置：`persistent_snn_kernel.cu:268-319`

这会导致：

* high queue 很小时，整个 grid 参与抢少量任务；
* high queue 最后一条超长行拖住全部 block；
* low queue 即使有很多可并行工作，也必须等待；
* high/low 分桶增加了 barrier，却没有形成真正异步调度。

换句话说，现在的 binning 更像：

```text
先批量处理重任务，再批量处理轻任务
```

而不是：

```text
用不同资源并行消化重任务与轻任务
```

## 建议

不一定需要真正 warp specialization，先做较低风险的统一调度：

```text
task descriptor:
    queue type
    pre
    batch
    edge begin
    edge end
```

warps 从统一任务池领取任务，根据 fanout 决定使用：

* 32-lane warp；
* 8-lane subwarp；
* 极长行切片任务。

这样 high/low 的区别只影响“任务执行宽度”，而不需要两轮全网格阶段。

更好的分桶方式是：

```text
low row       → 一个 8-lane task
normal row    → 一个 warp task
very long row → 多个 edge-range task
```

而不是为 high 和 low 分别建完整队列并全局串行执行。

---

# 六、极长行虽然不增加存储，却仍会制造尾部效应

当前 high-fanout 行仍然是：

```cpp
一个 warp 领取一个 pre
整个 warp 走完 [indptr[pre], indptr[pre+1])
```

即使 `fanout=4165`，仍然只有一个 warp 处理。该 warp 大约要执行：

[
\lceil4165/32\rceil \approx 131
]

轮 edge loop。

与此同时，一个 fanout=256 的行只需要约 8 轮。

所以 high-fanout 分桶并没有真正解决极长行，只是让它们先执行。最后一个超长行仍可能造成全局 barrier 前的长尾。

## 对应你最近修正的“第一个分桶”

你的新定义是合理的：

> 第一分桶不是普通 bucket，而是筛出极长行，并把极长行切成多个并行任务。

应改成类似：

```text
normal spike task:
    (batch, pre)

long-row segment task:
    (batch, edge_begin, edge_end)
```

例如每段 256、512 或 1024 条边。

这样：

```text
fanout=4165
```

可以拆成多个 edge segment，由多个 warp 并行处理，而不是一个 warp 独占。

连接仍保持原始 CSR，不需要复制或补零，只增加少量 segment metadata。

这是解决 max fanout 真正需要修改的地方：**不是存储，而是调度粒度。**

---

# 七、每次调用都有 GPU→CPU 同步，削弱了 persistent 的端到端收益

有至少两处明确同步。

## 输入 validation

```python
offsets_first, offsets_last = offsets[[0, -1]].tolist()
```

位置：`persistent_snn.py:156-161`

对 CUDA tensor 调用 `.tolist()` 会要求把结果拿回 CPU。

## overflow 检查

```cpp
if (overflow.item<int>() != 0)
```

位置：

* 普通版本：`persistent_snn.cpp:365`
* binned 版本：`persistent_snn.cpp:564`

这会在每个 forward/window 结束时强制等待 kernel 完成。

如果 benchmark 比较的是包含 Python/C++ wrapper 的端到端时间，这两处会破坏：

* stream 异步性；
* 多窗口流水；
* 与其他算子的 overlap；
* CUDA Graph capture 的可能性。

## 建议

验证类检查应移出热路径：

* graph/event 结构初始化时验证一次；
* benchmark 模式关闭边界同步检查；
* `overflow` 使用 debug 模式检查；
* 正常模式保证 capacity 足够，不做每次 `.item()`。

由于 queue capacity 已经是 `B×N`，一个时间步最多也只有 `B×N` 个神经元发放，因此在当前语义下 recurrent queue 理论上不会溢出。这里甚至可以用静态推理直接取消运行时 overflow 检查。

---

# 八、每次 forward 都重新分配、clone 和构造图元数据

每次调用都会：

```cpp
v.clone()
psc.clone()
dense_spikes allocation
input_current allocation
queue allocation
counter allocation
event history allocation
```

位置：`persistent_snn.cpp:281-294` 和 `474-491`

binned Python 路径还每次重新计算：

```python
graph_high_fanout = (
    (graph.indptr[1:] - graph.indptr[:-1]) >= threshold
).to(torch.int32)
```

位置：`persistent_snn.py:319-325, 421-430` 附近。

这意味着所谓 persistent 只发生在单个 window 内，window 之间：

* 状态重新 clone；
* workspace 重新分配；
* fanout metadata 重新构造；
* 再进行 host validation 和同步。

## 建议

建立长期存在的执行计划对象，例如概念上的：

```text
PersistentSNNPlan
    graph CSR
    fanout class
    long-row segments
    reusable queues
    counters
    input workspace
    output workspace
    cooperative grid configuration
```

state 则采用原地更新或 caller-provided output：

```text
v_inout
psc_inout
```

这样不会改变 kernel 主算法，却能减少大量框架开销。

对于性能 benchmark，至少应区分：

```text
setup time
steady-state window execution time
end-to-end API time
```

否则当前测到的可能不只是 kernel。

---

# 九、binned 版本将 queue 空间翻倍，但大部分时间只使用很小一部分

普通版本有：

```text
spike_queue_batch[B×N]
spike_queue_pre[B×N]
```

binned 版本变成：

```text
high_queue_batch[B×N]
high_queue_pre[B×N]
low_queue_batch[B×N]
low_queue_pre[B×N]
```

即四个 `B×N` int32 数组。

位置：`persistent_snn.cpp:480-483`

实际 high 和 low 总 spike 数仍然不会超过 `B×N`，因此没有必要让两个 queue 各自拥有完整 capacity。

可以使用：

```text
queue_pre[B×N]
queue_batch[B×N]
queue_class[B×N]
```

或者采用两端写入：

```text
low 从前向后写
high 从后向前写
```

不过两端写入会增加实现复杂度。更稳妥的是统一任务 queue 加 class/row length metadata。

这不是最大运行时瓶颈，但说明当前 binning 是“复制整套数据结构”，工程扩展性较差。以后增加更多 fanout 桶时不能继续每桶建立一整套 `B×N` queue。

---

# 十、external input 使用 atomicAdd，可能是不必要的

当前外部事件执行：

```cpp
atomicAdd(input_current + b * n_neuron + pre, value);
```

位置：

* 普通版本：`persistent_snn_kernel.cu:59-69`
* binned 版本：`persistent_snn_kernel.cu:212-223`

从 benchmark 的 event 生成方式看，每个 `(t,b,n)` 至多出现一次事件，因此没有重复目标，理论上不需要 atomic。

如果正式输入格式同样保证 bucket 内 neuron index 唯一，可以直接写：

```text
input_current[...] = value
```

或者直接让对应 cell 消费事件。

若真实输入允许同一 cell 同一步多个事件，才需要：

* 预聚合；
* block/warp reduce；
* atomicAdd。

这个问题通常不如 recurrent atomic 严重，但它发生在一个额外的 sparse-to-dense scatter 阶段，和全量清零结合后成本会放大。

---

# 十一、cell update 中的整数除法不是主要架构缺陷，但可以顺手避免

每个 cell 都执行：

```cpp
const int b = cell / n_neuron;
const int n = cell - b * n_neuron;
```

位置：

* 普通版本：`persistent_snn_kernel.cu:73-75`
* binned 版本：`persistent_snn_kernel.cu:226-228`

`n_neuron` 是运行时变量，编译器不能总是用简单位移替代。

相比全局访存和原子，这通常不是第一优先级，但可以通过二维映射或按 batch 外层划分避免：

```text
block/warp 固定 batch
cell index 只递增 neuron
```

不过不要为了省一个除法破坏负载均衡或合并访存。

---

# 十二、当前最可能的性能主因排序

只根据静态代码，我会这样排序：

## 第一梯队：很可能是重大问题

1. **每个时间步 4–5 次 `grid.sync()`**
2. **无条件写 `dense_spikes[T,B,N]`**
3. **无条件维护 `event_indices_full[T,B,N]`**
4. **high/low 两阶段全局串行**
5. **极长行没有切片，仍由单 warp 独占**
6. **每次 forward 的 `.tolist()` / `.item()` 强制同步**

## 第二梯队：明显存在，但要靠测量判断占比

7. 每次 forward clone 和大量 workspace 分配
8. high-fanout mask 每次重算
9. external sparse input 先 scatter 到 dense buffer，并每步全量清零
10. 统一全局 work counter 的原子争用
11. recurrent `atomicAdd(psc)` 冲突

## 第三梯队：局部工程问题

12. binned queue 空间重复
13. cell update 中整数除法
14. 边界判断：

```cpp
if (post >= 0 && post < n_neuron)
```

如果 CSR 已验证合法，可以在 release kernel 去掉。

---

# 十三、建议的架构修改顺序

## 第一阶段：先修纯浪费，不改变算法

修改输出路径：

```text
dense-only:
    不分配 event_counts/event_indices_full
    不执行事件输出 atomic/write

events-only:
    不分配、不写 dense_spikes

both:
    才同时维护两种输出
```

同时：

* cache `graph_high_fanout`；
* workspace 复用；
* benchmark 模式取消每次 D2H validation；
* overflow 检查改为 debug-only。

这些修改风险较低，也不会影响 prespan 的计算部分。

## 第二阶段：修极长行

保留 CSR，增加：

```text
long_row_flag
long_row_segment_offsets
segment_edge_begin/end
```

发放时：

* normal row 入普通 spike queue；
* extreme row 生成多个 edge segment task。

这正好对应你现在定义的“第一个分桶”。

## 第三阶段：取消 high/low 两轮全局串行

第二种 fanout bucket 不再对应多套 queue，而对应 task 标签：

```text
task.execution_width = 8 / 16 / 32 lanes
```

统一调度，避免 high 阶段结束后再 `grid.sync()` 才进入 low 阶段。

## 第四阶段：减少每时间步 barrier

重点改造：

```text
input_current 清零
external input scatter
LIF update
```

争取把前两个同步阶段压成一个，最终让每步接近：

```text
cell update + spike generation
grid.sync
recurrent fanout
grid.sync
```

两次 barrier 仍然不完美，但比当前 4–5 次更合理。

---

# 最终判断