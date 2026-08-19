可以直接收敛成 **3 个产出 + 4 组实验**。目标不是继续扩展方法，而是把现有 block / hash / persistent 的逻辑和证据补齐。

## 1. Figure A：Block-oriented task abstraction

**制图目标：** 解释 propagation task 如何生成，以及为什么需要 fragment。

图分两部分：

* **普通 row**：block 化后，一个 row 对应一个 task；
* **极长 row**：按固定 fragment size 切分，一个 fragment 对应一个 task。

图中只需要表现：

```text
active presyn neuron
    ↓
synaptic row
    ↓
regular row → one task

long row
    ↓
fragment 0 | fragment 1 | fragment 2
    ↓          ↓            ↓
 task 0      task 1        task 2
```

### 需要的数据

只需要证明“长尾确实存在，并且 fragmentation 把 task size 限制住”：

* 原始 active rows 的 fanout / edge count 分布；
* fragmentation 后 task edge count 分布；
* before/after 的 P50、P95、P99、Max、CV。

### 实验 1：Task size characterization

固定 1–2 个 representative dataset 和 firing rate，收集：

```text
row_edges
fragment_edges
```

输出一个简单表：

| Scheme     | P50 | P95 | P99 | Max | CV |
| ---------- | --: | --: | --: | --: | -: |
| row-level  |     |     |     |     |    |
| fragmented |     |     |     |     |    |

这组数据主要为 Figure A 和“heavy-tailed workload”论述服务，不需要做复杂性能测试。

---

## 2. Figure B：Persistent UPDATE / PROP data movement

**制图目标：** 说明 UPDATE 与 PROP 为什么采用不同 ownership。

核心表达：

```text
UPDATE
static ownership
→ locality / cross-timestep reuse
→ selected data stay local
→ fewer repeated global-memory accesses

PROP
dynamic task scheduling
→ weaker locality
→ better load balance for dynamic spike workload
```

图可以仿 SpInfer Fig. 7，左右画两个阶段。

### UPDATE

画：

```text
Global state / metadata
       ↓ once
persistent local storage
       ↓
UPDATE t
       ↓
UPDATE t+1
       ↓
UPDATE t+2
```

突出：

**Static ownership → cross-timestep reuse**

### PROP

画：

```text
Global task bins
      ↓
dynamic claim
      ↓
edge data
      ↓
propagation
```

突出：

**Dynamic ownership → load balance**

这里不要为了图好看而画尚未实现的“所有 neuron state 永久驻留 shared”；最终只画真实做了 residency 的数据。

### 需要的数据

这张图需要两个实验分别支撑 locality 和 balance。

---

## 3. 实验 2：UPDATE static ownership / residency

至少比较三个版本：

```text
U0: 原始 UPDATE assignment
U1: static contiguous ownership
U2: static ownership + 当前实际实现的 resident/cached data
```

测：

* UPDATE time；
* total step time；
* global read bytes / transactions；
* global write bytes / transactions；
* Long Scoreboard；
* shared memory / register usage；
* occupancy。

最关键数据只有两个：

[
\text{UPDATE latency}
]

和

[
\text{global-memory traffic}
]

如果 U2 确实降低 global traffic，就可以在 Figure B 里明确画：

> skip repeated global-memory access.

如果只是 static ownership 有收益，但 residency 没收益，则 Figure B 就只画：

> stable ownership / locality

不要画“load once”。

---

## 4. 实验 3：PROP static vs dynamic balance

比较：

```text
P0: static task assignment
P1: current dynamic binning
```

软件 instrumentation 记录每个 CTA：

```text
processed_tasks[cta]
processed_edges[cta]
```

然后算：

[
CV=\frac{\sigma(work)}{\mu(work)}
]

以及：

[
Imbalance=
\frac{\max(work)}
{\operatorname{mean}(work)}
]

再测：

* PROP time；
* CTA work CV；
* max/mean workload；
* 如果方便，再补 SM active / issue activity。

最终只需要形成这样的证据：

| Scheme          | PROP time | Work CV ↓ | Max/Mean ↓ |
| --------------- | --------: | --------: | ---------: |
| Static          |           |           |            |
| Dynamic binning |           |           |            |

这就足以支撑：

> PROP knowingly sacrifices some data affinity for substantially better dynamic balance.

---