# Figure C：Block + Hash micro-ablation

## 实验配置

- GPU：NVIDIA GeForce RTX 5090（compute capability 12.0）
- 数据集：FlyBrain / FlyWire 783，138,639 neurons，15,091,983 edges
- workload：batch 1，128 timesteps，event rate 0.01
- 计时：10 warmup，30 repeats，CUDA Event median
- Block 配置：512-edge budget，512-edge long segment
- Similarity：`global_similarity`
- Hash：aggregation 512，capacity 512，4 probes，minimum 256 edges，
  used-slot flush
- NCU：每个版本采一个代表性 persistent launch；Profiler replay duration
  不作为性能数据

四个版本的 PyTorch correctness 均通过，spike mismatch 为 0。

## 归一化结果

所有数据归一化到 V0 Binning；除 Issue Busy 外越低越好。

| Variant | PROP Time | E2E Step | Global Mem Inst. | Atomic Tx. | L2 Sectors | Long Scoreboard | Issue Busy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Binning | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| + Block | 0.95 | 0.95 | 0.65 | 1.00 | 1.00 | 0.66 | 1.88 |
| + Sort | 0.94 | 0.97 | 0.65 | 0.97 | 0.98 | 0.57 | 1.91 |
| + Hash | 0.90 | 0.93 | 0.62 | 0.93 | 0.95 | 0.16 | 3.31 |

完整绝对值和额外 DRAM sectors 位于
`results/figure_c_metrics.csv` 与 `results/figure_c_normalized.csv`。

## 软件计数器

软件计数使用 8 个代表性时间步。`global_updates_emitted` 包含普通 Block
路径和 long-row 路径；因此 `atomic_per_edge` 与 `merge_ratio` 是整个 PROP
workload 的口径，而不只是进入 hash 的 edges。

| Variant | Edges | Tasks | Fragments | Global updates | Hash flushes | Hash fallbacks | Atomic/edge | Merge ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Binning | 32,792,092 | — | — | 32,792,092 | 0 | 0 | 1.000 | 0.00% |
| + Block | 32,792,092 | 68,681 | 90,579 | 32,792,092 | 0 | 0 | 1.000 | 0.00% |
| + Sort | 32,792,092 | 66,384 | 89,420 | 32,792,092 | 0 | 0 | 1.000 | 0.00% |
| + Hash | 32,792,092 | 66,384 | 89,420 | 31,539,478 | 13,768,425 | 2,085,714 | 0.962 | 3.82% |

V3 的 hash 覆盖 17,106,753 条普通 Block edges；在该路径内，
`v4_global_atomics / v4_input_edges = 0.934`。把不参与 hash 的 long-row
edges 纳入后，整体 merge ratio 为 3.82%。

## 结论边界

1. Blockization 的直接证据成立：相对 V0，global memory instructions
   减少 35.0%，Long Scoreboard 减少 34.2%，PROP 时间减少 4.5%，Issue
   Busy 提高到 1.88 倍。这支持“减少 fragmented memory issuing”。但 Block
   单独没有减少 L2 sectors，不能扩展表述为“减少 off-chip transactions”。
2. Similarity 单独使 Atomic Tx. 再减少 2.8%、L2 sectors 减少 2.2%，但
   permutation I/O 令 E2E 相对 V1 增加 1.5%。它主要为 Hash 创造 post
   locality，而不是独立的端到端优化。
3. Hash 相对 V2 使 Atomic Tx. 减少 4.1%、L2 sectors 减少 3.1%、Long
   Scoreboard 减少 71.9%，PROP 和 E2E 均减少约 4.3%。这与软件计数器的
   3.82% 整体 merge ratio 相互印证。
4. 完整 V3 相对 V0：PROP 减少 9.7%，E2E 减少 7.2%，global memory
   instructions 减少 38.1%，Atomic Tx. 减少 6.7%，L2 sectors 减少 5.0%。

## 指标定义

- Global Mem Inst.：global load、store 与 global reduction SASS instructions
  之和。
- Atomic Tx.：`l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum`。CUDA
  `atomicAdd` 在该 kernel 中编译为 global reduction；该指标比受 predication
  影响的静态/warp-level SASS 指令数更接近实际全局原子事务。
- L2 Sectors：`lts__t_sectors_srcunit_tex.sum`。
- Long Scoreboard：
  `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`。
- Issue Busy：`smsp__issue_active.avg.pct_of_peak_sustained_active`。

## 复现

```bash
micromamba run -n ml-py312 python experiment/figure_c/analyze_results.py
micromamba run -n ml-py312 python experiment/figure_c/plot_table.py
```

`raw/` 保存四个 timing CSV、三份 software-counter CSV 和四份原始
`.ncu-rep`，后续可重新选择指标而无需重新 profile。
