下面给出一份可以**直接交给没有参与此前讨论的人制图**的执行说明。两张都是 **Method Figure**，不需要新实验；但制图前必须根据最终 CUDA 实现核对少量字段，尤其 Figure B 中 UPDATE 到底有哪些数据真的跨 timestep 驻留 on-chip，不能把“稳定 ownership”画成“实际 shared-memory residency”。

两张图可以借鉴 SpInfer 的两种表达范式：Figure 6 用层次结构把“数据/工作单元如何逐级组织”讲清楚；Figure 7 则把优化落到 Global Memory → Shared/Register → Compute 的具体数据路径上。

---

# Figure A：Hierarchical Block-Oriented Propagation

## 1. 图片目的

这张图回答一个核心问题：

> **RSNN 的 spike-driven propagation 如何从大量细粒度 neuron-level work，逐级重组为适合 GPU 执行和聚合的 coarse-grained work？**

重点不是单纯展示“block 化”，而是展示一个**三级 hierarchy**：

[
\text{Neuron-level work}
\rightarrow
\text{Block-level task}
\rightarrow
\text{Block-local accumulation}
]

三层分别解决：

1. **Neuron level**：原始实现中一个 firing neuron 产生一个小 propagation task，task 数多且访问细碎；
2. **Block level**：多个 neuron 组成一个 block/tile，将 active rows 的 edge spans 聚合成较粗粒度工作，再形成 bounded logical tasks；
3. **Aggregation level**：task 内多个 edge contribution 先在 block 内通过 shared hash 合并，把大量细粒度 global updates 转为较少 global atomic updates。

因此图的总体视觉思想应该是：

> **从上到下不断“收束”：先压缩 task issuing，再压缩 global update issuing。**

注意：第三级不是严格意义上的“task 划分”，所以图题和正文最好使用 **hierarchical work organization / task and accumulation hierarchy**，不要称为“three-level task tiling”。

---

# 2. 推荐整体版式

建议横向宽图，分为三层，由上到下或左到右均可；**更推荐纵向三级结构**，因为“逐级收束”更容易表现。

基本结构：

```text
(a) Fine-grained Neuron-level Work
        ↓
(b) Block-level Task Formation
        ↓
(c) Block-local Aggregation
```

右侧或底部可以有一个很小的最终结果：

```text
Dynamic PROP Task Pool / Global PSC
```

---

# 3. 第一层：Neuron-level work

## 要表达的概念

原始细粒度方式中：

> 每个 firing presynaptic neuron 独立对应一段 fanout edge span，并形成一个独立 propagation work unit/task。

绘制一个小型 toy network，例如 8 个 presynaptic neurons：

```text
N0 ● ───────── [ e e e e ]
N1 ○
N2 ● ───────── [ e e ]
N3 ● ───────── [ e e e ]
N4 ○
N5 ● ───────── [ e e e e e ]
N6 ○
N7 ● ───────── [ e e ]
```

其中：

* `●` = firing neuron；
* `○` = inactive neuron；
* 每个 active neuron 后面的连续小格代表 CSR/Prespan 中该 neuron 对应的 fanout edge span。

每个 active neuron 的 span 外面分别加一个小框：

```text
Task
```

视觉效果应该明显表现出：

```text
Task 0
Task 1
Task 2
Task 3
...
```

很多彼此独立的小 work units。

### 此层建议标签

主标签：

> **Neuron-level tasking**

小注释：

> One active neuron → one fine-grained propagation task

右侧可以用简短 consequence：

> Many small tasks and fine-grained memory accesses

不要在这张图里放性能数字。

---

# 4. 第二层：Block-level task formation

这是整张 Figure A 的核心。

## 4.1 先画 Neuron/Spike Block

把多个连续 neuron 用一个大框框起来。真实实现若为 32-neuron block，则直接写：

> **32-neuron block**

视觉上不必真的画 32 个，可以画 8 个代表，右上角标：

> 32 neurons in implementation

例如：

```text
┌──────────── Neuron / Spike Block ────────────┐
│ N0 ●  N1 ○  N2 ●  N3 ●  N4 ○ ... N31 ●    │
└───────────────────────────────────────────────┘
```

这一层表达的是：

> blockization 首先在 neuron/work formation 这一侧建立更大的处理单位。

---

## 4.2 把 block 中 active rows 的 edge span 画出来

下面对应几个 firing neuron：

```text
N0 : █████
N2 : ███
N3 : ██████
N7 : ██
```

然后通过箭头进入一个**logical active-edge stream**：

```text
|---N0---|--N2--|----N3----|-N7-|
```

这里要强调：

> 并不是改变原始 connectivity，也不是把 CSR 行物理合并成一行；这是 task formation 时对 block 内 active-edge work 的逻辑组织。

最好在图里出现：

> **active-edge prefix**

或者：

> Active-row prefix / logical edge offsets

用竖线表现 prefix boundaries：

```text
0       5    8       14   16
|  N0   | N2 |  N3    | N7 |
```

这样执行者不会错误地画成“直接把权重矩阵重新存成一个 dense block”。

---

# 5. 普通 block 的 logical task 划分

逻辑 edge stream 随后按照固定 edge budget 切成真正可调度的 bounded tasks：

```text
Logical active-edge stream
████████████████████████████████

          ↓ edge budget

┌──────────┐ ┌──────────┐ ┌───────┐
│ Task 0   │ │ Task 1   │ │Task 2 │
│ ≤ Budget │ │ ≤ Budget │ │       │
└──────────┘ └──────────┘ └───────┘
```

这里必须明确：

* **Neuron Block** 是上一级 tile/work aggregation unit；
* **Logical Task** 是 scheduler 最终消费的 bounded work unit。

不要把两者都写成 “BlockTask” 而造成歧义。

如果当前代码有确定名称，则最终全部替换成代码/论文统一术语。

---

# 6. Extremely long row 的独立分支

在普通 block 路径旁边画一个 parallel branch。

例如：

```text
Exceptional high-fanout neuron

Nlong ● ───────────────────────────────────────────────
         ██████████████████████████████████████████
                         ↓
              fixed-size fragmentation
                         ↓
        ┌────────┐ ┌────────┐ ┌────────┐
        │ Frag 0 │ │ Frag 1 │ │ Frag 2 │ ...
        └────────┘ └────────┘ └────────┘
```

核心表达：

> 普通 neuron rows 通过 block-oriented task formation 处理；极长行不强行塞进普通 block workload，而是直接切成 bounded fragments，每个 fragment 独立成为 task。

最终两路汇入同一个：

```text
Dynamic PROP Task Pool
```

即：

```text
Ordinary block logical tasks ──┐
                               ├──► Dynamic PROP scheduling
Long-row fragments ────────────┘
```

---

# 7. 第三层：Block-local hash aggregation

这一层应从一个已经形成的 logical task 中展开。

绘制 task 里的若干 edge：

```text
edge 0 → post 7,  w0
edge 1 → post 3,  w1
edge 2 → post 7,  w2
edge 3 → post 9,  w3
edge 4 → post 7,  w4
edge 5 → post 3,  w5
```

然后所有 contribution 进入：

```text
┌─────────────────────────┐
│ Shared-memory Hash      │
│                         │
│ post 3 : w1+w5          │
│ post 7 : w0+w2+w4       │
│ post 9 : w3             │
└─────────────────────────┘
```

这里可以把多个输入箭头画得很多：

```text
│ │ │ │ │ │ │ │
▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼
   Shared Hash
```

输出只剩：

```text
post3 Σ
post7 Σ
post9 Σ
 │     │    │
 ▼     ▼    ▼
global atomic
```

这一层最核心的视觉关系：

[
\text{many edge contributions}
\rightarrow
\text{many cheap/local shared updates}
\rightarrow
\text{few global atomic updates}
]

### 图中建议明确写

> **Block-local aggregation**

以及：

> Merge repeated postsynaptic destinations before global writeback

---

# 8. Figure A 的视觉主线

最好让整张图天然呈现一个漏斗：

```text
many neuron tasks
│ │ │ │ │ │ │
▼ ▼ ▼ ▼ ▼ ▼ ▼

   block task formation
        ╲ │ ╱
         ▼

   fewer coarse tasks
        │ │ │
        ▼ ▼ ▼

 many edge contributions
│ │ │ │ │ │ │ │
╲ ╲ ╲ │ ╱ ╱ ╱
   shared hash
       │ │ │
       ▼ ▼ ▼

 few global updates
```

也就是说：

### 第一次收束

**task/work issuing contraction**

由 blockization 完成。

### 第二次收束

**global update contraction**

由 hash aggregation 完成。

这就是 Figure A 最重要的总体 intuition。

---

# 9. Figure A 不应该出现什么

不要：

* 画 GPU memory hierarchy；
* 画 shared/register/global memory 路径；
* 画 UPDATE；
* 画 persistent timeline；
* 放 benchmark 数字；
* 放 NCU counter；
* 把 blockization 说成减少“数据量”；
* 暗示 connectivity 本身被重新压缩。

这些属于 Figure B 或后续实验。

Figure A 只回答：

> **work 和 update 在算法/执行层面怎样逐级 coarsen。**

---

# Figure B：Heterogeneous Data Movement in Persistent RSNN Execution

## 10. 图片目的

这张图回答：

> **为什么 persistent RSNN 中 UPDATE 和 PROP 不使用相同的 ownership/data-movement policy？这种差异如何改变 GPU 的实际 memory path？**

核心设计思想：

[
\boxed{
\text{UPDATE prefers locality}
\qquad
\text{PROP prefers dynamic balance}
}
]

更加具体地：

### UPDATE

工作对象稳定，neuron range 可长期绑定给固定 CTA/block，因此可以：

* 保持稳定 ownership；
* 利用 temporal locality；
* 对代码中确实长期驻留的 state/metadata，跳过重复 Global Memory round trips。

### PROP

active neurons 和 fanout workload 每 timestep 动态变化且 heavy-tailed，因此：

* 不应强行保持 static ownership；
* 从 dynamic task pool/bin 领取 Figure A 产生的 coarse tasks；
* connectivity 仍按需从 Global Memory 获取；
* 用 blockization 减少细粒度 memory issuing；
* 用 shared hash 将大量 global atomics 转为 local/shared updates + fewer global atomics。

因此 Figure B 的主题不是“pipeline”，而是：

> **不同数据生命周期与 workload 特征对应不同 data paths。**

这正是学习 SpInfer Figure 7 时应抓住的核心：画具体数据经过哪些 memory levels，而不只是画算法框。

---

# 11. Figure B 的整体版式

推荐横向宽图。

纵向分三层：

```text
GLOBAL MEMORY
──────────────────────────────────────────────────────

ON-CHIP / PERSISTENT STATE
Shared Memory + Registers
──────────────────────────────────────────────────────

EXECUTION
──────────────────────────────────────────────────────
```

横向分两大区域：

```text
          UPDATE                         PROP
      Locality-oriented              Balance-oriented
```

因此形成一个 **2 × 3** 的基本构图。

---

# 12. 左半：UPDATE data path

标题：

> **UPDATE: Static Ownership for Temporal Locality**

首先画 Global Memory 中实际 neuron-related arrays，例如最终根据代码确认：

```text
V
PSC / current
neuron parameters
other per-neuron state
```

不要统称成模糊的 “Data”。

---

## 12.1 Static ownership 要非常直观

画：

```text
CTA 0 → neuron range [0, N0)
CTA 1 → neuron range [N0, N1)
CTA 2 → neuron range [N1, N2)
```

用不交叉的固定实线。

旁边 annotation：

> Fixed contiguous neuron ownership across timesteps

这样读者一眼知道 UPDATE 为什么具有 data affinity。

---

# 13. UPDATE 的核心视觉：跨 timestep reuse

如果代码确认某些数据真的跨 timestep 保存在 shared/register 中，则画：

```text
Global Memory
     │
     │ initial load
     ▼
┌────────────────────────┐
│ Persistent local state │
│ Shared / Registers     │
└────────────────────────┘
     │
     ├──► UPDATE t
     │
     ├──► UPDATE t+1
     │
     ├──► UPDATE t+2
     │
     └──► ...
```

这里最重要的是：

**只有第一根 Global → Local 箭头。**

不要每 timestep 都再画 Global Memory。

可以用一个循环箭头：

```text
local state
    ↺
 UPDATE
    ↺
next timestep
```

标：

> **Cross-timestep reuse**

---

# 14. 如果代码没有真正 residency，必须降级画法

这是 Figure B 最重要的真实性要求。

如果当前实现只是：

> 同一个 CTA 长期负责相同 neuron range，但 `V/PSC` 每 timestep 仍从 global load/store，

则绝对不能画：

```text
load once → shared forever
```

这时应该改画：

```text
same Global neuron range
      ↓
same persistent CTA
      ↓
UPDATE
```

并写：

> Stable ownership preserves data affinity / cache locality

而不是：

> On-chip residency

因此制图前必须做一次 code audit。

---

# 15. UPDATE → PROP 的接口

UPDATE 计算后会产生 firing/spike information。

中间明确画一根从 UPDATE 指向 PROP 的箭头：

```text
UPDATE
   │
   │ spike/activity information
   ▼
Task formation
```

实际名称按代码核对，可为：

* spike flags；
* active-neuron IDs；
* block descriptors；
* ready flags；
* BlockTask descriptor。

这里最好直接引用 Figure A：

> Block/fragment task formation (Fig. A)

从而两张图形成连续故事。

---

# 16. 右半：PROP 的 overall path

标题：

> **PROP: Dynamic Scheduling for Load Balance**

完整数据路径建议画：

```text
Dynamic task pool / bins
          │
          ▼
   task descriptor
          │
          ▼
Global connectivity
(indptr / indices / weights)
          │
          ▼
block-oriented edge processing
          │
          ▼
Shared-memory hash
          │
          ▼
few global atomic updates
          │
          ▼
Global PSC / recurrent current
```

与 UPDATE 相比，它故意保留大量 Global Memory interaction。

图边上明确写 trade-off：

> Dynamic ownership sacrifices persistent data affinity to accommodate time-varying, heavy-tailed propagation work.

---

# 17. PROP 输入端一定画“多束 → 粗束”

这是你当前 Figure B 最关键的新元素。

## Baseline/fine-grained concept

在 Global Connectivity 到 execution 之间先用多个细箭头表示：

```text
Neuron Task 0 ─────────► row / edges
Neuron Task 1 ─────────► row / edges
Neuron Task 2 ─────────► row / edges
Neuron Task 3 ─────────► row / edges
Neuron Task 4 ─────────► row / edges
```

旁边小标签：

> Fine-grained memory issuing

---

## Block-oriented path

对应最终方法：

```text
            BlockTask
               ║
               ║
               ▼
      grouped edge processing
```

用**更少、更粗的箭头束**。

视觉含义：

[
\text{many fine-grained memory commands}
\rightarrow
\text{fewer coarse-grained commands}
]

这里必须谨慎使用文字。

推荐：

> **Fewer fine-grained global-memory instructions/requests**

暂时不要写：

> Less DRAM traffic

除非后续 NCU 已经证明 bytes/transactions 也下降。

---

# 18. PROP 输出端再画一次“多束 → 一束/少束”

这次是 hash。

## 无 hash 的概念路径

可以在背景/浅灰 inset 中：

```text
edge 0 ─────────► global atomic
edge 1 ─────────► global atomic
edge 2 ─────────► global atomic
edge 3 ─────────► global atomic
edge 4 ─────────► global atomic
...
```

即许多粗长箭头直接跨到 Global Memory。

---

## Hash 路径

最终方案：

```text
edge contributions
 │ │ │ │ │ │ │ │
 ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼
┌──────────────────┐
│ Shared Hash      │
└──────────────────┘
   │   │   │
   ▼   ▼   ▼
unique aggregated posts
   │   │   │
   ▼   ▼   ▼
Global atomic writes
```

视觉上非常明确地表现：

> **many local/shared atomics → few global atomics**

建议在图旁直接写：

> Shared-memory aggregation replaces redundant global atomics with local updates and a reduced global flush.

注意不要画成：

> many global atomics → one global atomic

除非 toy example 所有 edge 恰好写同一个 post。

一般情况应该是：

[
|E|
\rightarrow
|\mathrm{UniquePost}(E)|
]

所以视觉上应该是“很多 → 少数”，而不是固定“很多 → 一个”。

---

# 19. Figure B 的 PROP 最好形成一个沙漏

推荐最终视觉：

```text
                    GLOBAL CONNECTIVITY

                  │ │ │ │ │ │ │ │
                  ╲ ╲ │ │ ╱ ╱ ╱
                     ╲│╱
                  [ BlockTask ]
                       ║
                       ║
                       ▼
                 Edge Processing

                  │ │ │ │ │ │ │
                  ▼ ▼ ▼ ▼ ▼ ▼ ▼
                 [Shared Hash]
                    ╲ │ ╱
                     ╲│╱
                      ▼
                 unique posts
                   │ │ │
                   ▼ ▼ ▼

                     GLOBAL PSC
```

也就是：

### 上半

blockization 收束 input-side requests。

### 下半

hash 收束 output-side global updates。

这应该成为 PROP 区域最突出的视觉特征。

---

# 20. UPDATE 和 PROP 在视觉上应该形成“对称但不同”的设计

整张 Figure B 最终应该让人一眼看到：

## 左边 UPDATE：纵向复用

```text
Global
   ↓
Local
   ↺
   ↺ timestep
   ↺
```

它减少的是：

> **同一份数据沿时间维度重复搬运。**

---

## 右边 PROP：横向收束

```text
many requests
   ╲ │ ╱
    block
      ↓
   compute
      ↓
    hash
   ╱ │ ╲
few global updates
```

它减少的是：

> **同一 timestep 内大量细粒度 work/memory requests。**

因此整张图背后的高层思想可以总结为：

[
\boxed{
\text{Temporal locality for UPDATE}
+
\text{request aggregation for PROP}
}
]

这个信息不用作为自造术语写进论文标题，但应成为制图时的视觉原则。

---

# 21. Figure B 是否需要画 baseline

不建议占一半空间做完整 baseline。

SpInfer Fig. 7 右侧用了小型 path comparison，而不是复制两张完整 workflow。

你也可以采用：

### 主图

只画最终 persistent design。

### 两个小 inset

UPDATE 下方：

```text
Conventional:
Global → UPDATE → Global
Global → UPDATE → Global
Global → UPDATE → Global

Persistent:
Global → [UPDATE ↺ UPDATE ↺ UPDATE]
```

PROP 下方：

```text
Fine-grained:
│ │ │ │ │ global requests
│ │ │ │ │ global atomics

Blocked + hash:
══ coarse requests
│ │ │ shared
▼ ▼ ▼
few global atomics
```

这样主图不会过于拥挤。

---

# 22. 两张图需要统一的视觉语言

这是实际制图时很重要的一点。

建议两张图统一：

### Neuron

圆点。

* filled：active/spiking
* hollow：inactive

### Edge/span

连续小矩形或短横条。

### Task

圆角矩形。

### Block/Tile

较大的虚线或粗边框矩形。

### Shared Hash

带槽位的小表格结构。

### Global Memory

长条形大框。

### Shared/Register

中层长条框。

### Normal global memory access

细实线箭头。

### Coarse/block memory access

粗箭头。

### Shared/local update

短箭头。

### Global atomic

使用统一特殊箭头/标记，例如箭头旁 `atomic`。

不要依赖颜色单独传递语义，因为论文可能灰度打印。

---

# 23. 两张图术语必须统一

制图前确定最终词表，然后全篇一致。

建议候选：

| 概念                         | 推荐名称                       |
| -------------------------- | -------------------------- |
| 一个 neuron 对应的原始工作          | Neuron-level task          |
| 32-neuron grouping         | Neuron Block / Spike Block |
| block active edges 的逻辑范围   | Active-edge stream         |
| budget 后真正进入 scheduler 的任务 | Logical propagation task   |
| 超长行切片                      | Long-row fragment          |
| 动态调度结构                     | Dynamic task pool / bins   |
| hash                       | Block-local shared hash    |
| 最终写回                       | Global atomic flush        |

尤其不要在同一篇里交替使用：

* fragment
* tile
* block
* chunk
* task

却没有明确区分层级。

---

# 24. 制图前唯一必要的代码核查

Figure A 基本可以直接按当前算法结构画；Figure B 必须先查代码。

执行者应在 `.cu` 中确认以下内容，然后填写到图里。

### A. UPDATE

确认：

* UPDATE blocks 的 neuron range 是否在整个 persistent execution 中固定；
* `V` 是否每步从 global load；
* `PSC/current` 是否每步从 global load/store；
* 哪些 neuron parameters 在 timestep loop 外加载；
* 哪些真正保存于 registers/shared 跨 timestep；
* 哪些仅依赖 stable CTA ownership/cache affinity。

### B. UPDATE→PROP

确认实际传递：

* spike flag；
* active neuron；
* block descriptor；
* queue descriptor；
* ready flag；

究竟是哪一种。

### C. PROP

确认 task descriptor 实际包含什么；

确认 connectivity path：

```text
indptr
indices
weights
```

是否全部 global；

确认 block path 如何读取；

确认 hash：

* table 在 shared memory；
* edge contribution 如何插入/atomic accumulate；
* 最后按 used slots 还是扫描整表 flush；
* 最终写 `PSC` / `recurrent_delta` 的哪个数组。

**图上的每个数据名和每一根跳过的 memory path都必须能在代码里找到对应依据。**

---

# 25. 两张图分别应让 reviewer 在 5 秒内得到什么

## Figure A

第一眼：

> “原来 baseline 是 one-neuron-one-task；他们先把 neuron work tile 成 block tasks，对极长行单独 fragment，然后又在 block 内把重复 postsynaptic updates 聚合。”

第二眼：

> “所以这是两次 coarse-graining：task side 和 update side。”

---

## Figure B

第一眼：

> “UPDATE 和 PROP 故意采取不同 policy：UPDATE 固定 ownership 保 locality，PROP 动态领取任务保 balance。”

第二眼：

> “block 减少 input-side 细碎 memory issuing，而 hash 用 shared operations 换掉大量 global atomics。”

第三眼：

> “这些是实际 Global/Shared/Register data paths，而不是抽象算法框图。”

---

# 26. 最终可直接交给制图者的简化草图

### Figure A

```text
                 Hierarchical Propagation Work Organization

 ┌──────────────────────────────────────────────────────────────┐
 │ Level 1 — Neuron-level Work                                 │
 │                                                              │
 │ N0 ● ─ [edges] → Task     N1 ○                              │
 │ N2 ● ─ [edges] → Task     N3 ● ─ [edges] → Task             │
 │                                                              │
 │         many fine-grained neuron tasks                       │
 └──────────────────────────────┬───────────────────────────────┘
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ Level 2 — Block-level Task Formation                         │
 │                                                              │
 │ Ordinary neurons                     Exceptional long row    │
 │ ┌──── 32-neuron block ────┐          ███████████████████     │
 │ │ ● ○ ● ● ○ ...          │                ↓ fragment        │
 │ └─────────────────────────┘          [Frag][Frag][Frag]      │
 │          ↓                                                   │
 │ active-edge stream                                          │
 │ |row0|row2|row3|...|                                        │
 │          ↓ edge budget                                      │
 │ [Task 0][Task 1][Task 2]                                    │
 │            \                  /                              │
 │             └→ Dynamic PROP Pool ←┘                          │
 └──────────────────────────────┬───────────────────────────────┘
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ Level 3 — Block-local Aggregation                            │
 │                                                              │
 │ edge contributions                                           │
 │ │ │ │ │ │ │                                                 │
 │ ▼ ▼ ▼ ▼ ▼ ▼                                                 │
 │       Shared Hash                                            │
 │       post3: Σ                                               │
 │       post7: Σ                                               │
 │       post9: Σ                                               │
 │          │ │ │                                               │
 │          ▼ ▼ ▼                                               │
 │      few global atomics                                      │
 └──────────────────────────────────────────────────────────────┘
```

---

### Figure B

```text
             Heterogeneous Data Movement in Persistent RSNN

                     GLOBAL MEMORY
──────────────────────────────────────────────────────────────────
 Neuron state / params       Task pool     Connectivity       PSC
          │                     │            │ │ │ │ │         ▲
          │                     ▼            ╲ │ │ │ ╱         │
          ▼                dynamic acquire     ╲│ │╱            │

               ON-CHIP / PERSISTENT STATE
──────────────────────────────────────────────────────────────────
 ┌──────── UPDATE ─────────┐      ┌──────── PROP ───────────────┐
 │ fixed neuron ownership  │      │       BlockTask             │
 │                         │      │           ║                  │
 │ local state / metadata  │      │           ▼                  │
 │       ↺ timestep        │      │     edge processing          │
 │       ↺ reuse           │      │       │ │ │ │ │              │
 │                         │      │       ▼ ▼ ▼ ▼ ▼              │
 │                         │      │      Shared Hash             │
 │                         │      │          ╲│╱                 │
 └──────────┬──────────────┘      │           ▼                  │
            │ spike/activity     │     aggregated posts          │
            └───────────────────►│          │ │ │                │
                                └──────────┼─┼─┼────────────────┘
                                           ▼ ▼ ▼
──────────────────────────────────────────────────────────────────
          UPDATE                         PROP
   locality-oriented             balance-oriented

 Static ownership:               Dynamic ownership:
 temporal reuse                  fewer coarse reads +
 / stable affinity               shared aggregation +
                                 fewer global atomics
```

最终成图当然需要比 ASCII 更克制：**Figure A 强调三级 hierarchy 和两次“收束”；Figure B 强调 memory hierarchy、UPDATE 的纵向 temporal reuse，以及 PROP 的沙漏式 input/output request contraction。**

这样即使制图者没有经历此前讨论，也应该能够理解“每个区域画什么、为什么画、哪些东西绝对不能误画”，并能直接根据最终 CUDA 代码把数据名和具体 residency 补准确。
