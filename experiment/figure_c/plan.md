## 测试环境

测试数据集采用flybrain，在5090上测试，可进入服务器调试，是micromamba的ml-py312环境 zhanghan@162.105.95.95，私钥如果缺失，可到本地win环境中寻找。运行在micromamba的ml-py312，可用GPU，生成ncu报告可使用benchmark/benchmark_rsnn_roofline.py


# 5. Figure/Table C：Block + Hash micro-ablation

这是最重要的一组实验。

**制图目标：**

证明两个 claim：

1. **Blockization** 减少 fragmented memory issuing；
2. **Similarity + Hash** 聚合 repeated post updates，减少 global atomic。

实验版本固定成：

| Variant            | Block | Similarity | Hash |
| ------------------ | ----: | ---------: | ---: |
| V0 Binning         |     ✗ |          ✗ |    ✗ |
| V1 Block           |     ✓ |          ✗ |    ✗ |
| V2 Block+Sort      |     ✓ |          ✓ |    ✗ |
| V3 Block+Sort+Hash |     ✓ |          ✓ |    ✓ |

如果 V2 很难单独做，可以最终省掉，但实验阶段最好保留，用来分离 similarity 与 hash 的收益。

---

## 6. 实验 4：Block / Hash micro-ablation

### 第一层：runtime

所有版本统一 workload，测：

* PROP kernel time；
* end-to-end step time。

### 第二层：software counters

直接在代码里统计：

```text
edges_processed
tasks_generated
fragments_generated
global_updates_emitted
hash_entries_flushed
```

重点计算：

[
atomic_per_edge
===============

\frac{global_updates}
{edges}
]

以及：

[
merge_ratio
===========

1-
\frac{global_updates}
{edges}
]

这能直接证明 hash 在算法层到底消掉了多少 write。

### 第三层：NCU

只采与你 claim 直接相关的几项：

* global load/store instructions；
* global memory transactions 或 L2 sectors；
* atomic instructions / transactions；
* Long Scoreboard；
* Issue Slot Busy / issue activity。

不需要一开始采几十项指标。

---

# 7. 最终 Table C 建议长这样

全部 normalize 到 V0 Binning：

| Variant | PROP Time ↓ | Mem Inst ↓ | Atomic Ops ↓ | L2/DRAM Tx ↓ | Long Scoreboard ↓ | Issue Busy ↑ |
| ------- | ----------: | ---------: | -----------: | -----------: | ----------------: | -----------: |
| Binning |        1.00 |       1.00 |         1.00 |         1.00 |              1.00 |         1.00 |
| + Block |             |            |              |              |                   |              |
| + Sort  |             |            |              |              |                   |              |
| + Hash  |             |            |              |              |                   |              |

如果最终指标太多，就保留最有解释力的 4 个：

* PROP Time；
* Global Mem Instructions；
* Atomic Ops；
* Long Scoreboard。

这已经足够接近 SpInfer Table 1 的作用：不是单纯做 speedup ablation，而是说明每一步到底改变了什么 GPU 行为。

---

# 8. 最终执行顺序

按这个顺序做即可：

1. **Task characterization**
   收集 row/task size 分布，完成 Figure A 所需数据。

2. **UPDATE locality experiment**
   原始 → static ownership → residency，确定 Figure B 的 UPDATE 部分到底能 claim 到哪一步。

3. **PROP balance experiment**
   static vs dynamic binning，记录 CTA workload distribution。

4. **Block/hash ablation**
   V0–V3 跑 runtime + software counters + NCU。

5. **最后制图**

   * Figure A：任务组织方式；
   * Figure B：UPDATE locality vs PROP balance；
   * Table/Figure C：block/hash 的硬件效果。

最终三者的逻辑就是：

[
\boxed{\text{Figure A：任务怎么形成}}
]

[
\downarrow
]

[
\boxed{\text{Figure B：不同任务为什么用不同 ownership}}
]

[
\downarrow
]

[
\boxed{\text{Table C：这些设计具体减少了什么硬件开销}}
]

现阶段先不要再加入新的 pipeline 或 hash 变体。把这四组实验跑完，基本就能判断现有设计能否形成一条完整、可量化的论文主线。
