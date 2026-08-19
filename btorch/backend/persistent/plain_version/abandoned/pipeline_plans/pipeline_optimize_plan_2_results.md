# Pipeline 第三阶段测量结果

## 测试环境

- GPU：NVIDIA GeForce RTX 5090
- 数据集：FlyBrain，138,639 neurons，15,091,983 synapses
- 窗口：128 timesteps，batch size 1
- 生产计时：CUDA Event median，10 warmups，30 repeats
- 分段和 overlap 计时：instrumented build，5 warmups，20 repeats

instrumented 时间只用于结构归因，不与 production 时间直接混用。

## P4：Kernel 与 host 状态管理

| Component | Naive | Pipeline allocate | Pipeline preallocated |
| --- | ---: | ---: | ---: |
| Clone | 0.0153 ms | 0.0189 ms | 0.0158 ms |
| Delta init | 0.0004 ms | 0.0218 ms | 0.0167 ms |
| Queue state clear | 0.0017 ms | 0.0015 ms | 0.0015 ms |
| Core kernel | 11.8263 ms | 12.5055 ms | 12.5479 ms |
| PSC fold | 0.0012 ms | 0.0041 ms | 0.0041 ms |
| GPU interval | 11.8518 ms | 12.5587 ms | 12.5911 ms |

不同 GPU 上的 core kernel 小幅波动大于 delta 优化收益，因此 allocator
收益应只比较被独立 Event 包围的 delta init：预分配从 0.0218 ms 降到
0.0167 ms，节省约 0.0051 ms。取消 PSC fold 另节省约 0.003 ms。两项都
不足以解释 pipeline 与 naive 的差距。

关闭 instrumentation 后：

| Provider | Full forward |
| --- | ---: |
| Naive | 11.7603 ms |
| Pipeline | 12.4165 ms |

Pipeline core kernel 比 naive core kernel 慢约 5.7%，完整 forward 慢约
5.6%。因此属于计划中的情况 C：主要差距在 kernel 内，而不是 host 状态管理。

## P5：UPDATE--Propagation overlap

以下均为每 timestep 的 timestamp median：

| UPDATE:PROP | UPDATE | Startup | Overlap window | Tail | Pipeline |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7:1 | 27.392 us | 10.240 us | 12.544 us | 65.024 us | 92.416 us |
| 3:1 | 27.648 us | 10.752 us | 12.544 us | 65.024 us | 92.416 us |
| 2:1 | 27.392 us | 10.496 us | 12.288 us | 64.768 us | 92.160 us |
| 1:1 | 29.440 us | 9.728 us | 14.848 us | 62.976 us | 92.160 us |

Naive barrier reference：

- Serial UPDATE：20.992 us
- Serial propagation：70.400 us
- Serial total：91.648 us（timestamp interval）

以 1:1 pipeline 计算：

```text
overlap_gain = 20.992 + 70.400 - 92.160 = -0.768 us
overlap_efficiency = -0.768 / 20.992 = -3.66%
```

Pipeline 约隐藏了 7.4 us propagation，但 UPDATE 因 block 划分和并发资源
竞争增加了约 8.4 us，收益被完全抵消。首任务约在 timestep 开始后 4.6 us
发布，但 consumer 启动仍需约 9.7 us。

## 结论

下一阶段不应优先持久化 delta 或删除 PSC fold；它们的总潜在收益小于
0.01 ms/window。应优先处理：

1. consumer 启动延迟；
2. UPDATE 与 propagation 的资源竞争；
3. task 发布顺序和 queue 协议成本。

生产默认仍保持上一阶段扫描得到的 1:1、dedicated/helper 1/1、chunk 1。
