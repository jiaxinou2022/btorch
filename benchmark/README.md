benchmark/
├── benchmark_rsnn_cudagraph_compare.py   ← 主入口（本文档主角）
├── benchmark_data.py                     ← 工作负载生成、保存/恢复
├── benchmark_persistent_snn.py           ← BenchCase/RSNNResult 数据结构 + 输入生成
├── benchmark_workload_stats.py           ← 图结构/脉冲统计摘要
├── provider_common.py                    ← PreparedMetadata/BenchmarkRunner 合约
├── sota_rsnn_cudagraph.py               ← SOTA 与 in-tree SpMSpV CUDA Graph Provider
├── external_rsnn_simulators.py          ← GeNN/Brian2CUDA 隔离子进程 Provider
├── cusparse_rsnn/                        ← cuSPARSE 直接调用 CUDA 扩展
│   ├── loader.py                         ← JIT加载扩展
│   ├── cusparse_rsnn.cpp                 ← C++ 扩展入口
│   └── cusparse_rsnn_kernels.cu         ← CUDA 核函数
└── benchmark_provider_audit.py           ← Profiler 审计

In-tree `tilespmspv`, `sortspmspv`, `globalatomic`, `blockatomic`,
`blocksort`, `naivespmspv`, and `holaspmspv` each provide both an `_eager`
adapter and a `_cudagraph` full-window provider. The CUDA Graph variants consume
the recurrent dense spike tensor directly on the current CUDA stream. When both
variants are selected, the CSV reports `speedup_vs_same_eager`.
