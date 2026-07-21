# Persistent SNN Spike-Block 第一版优化方案

## 核心思路

当前 persistent kernel:

``` text
LIF update
→ spike list
→ 每个 spike 独立领取
→ CSR row 遍历
→ atomicAdd PSC
```

第一版改为:

``` text
warp 更新连续 32 neuron cell block
        ↓
warp ballot 得到 spike mask

fanout < 256:
    一个 block 生成一个 BlockTask

fanout >= 256:
    切成 SegmentTask

        ↓

warp 领取任务

SegmentTask:
    固定 edge 区间处理

BlockTask:
    多个 spike 共同处理 CSR
    warp 内合并相同 post
    global atomic


```

目标：

-   减少逐 spike 调度开销；
-   改善 warp 利用率；
-   缓解极长 fanout 长尾；
-   减少 atomic 冲突。

不做：

-   neuron 全局重排；
-   fanout 多级 bucket；
-   复杂 scheduler；
-   hash table；
-   TMA。

------------------------------------------------------------------------

## 数据结构

CSR 保持：

``` cpp
csr_indptr
csr_indices
csr_weights
```

新增：

``` cpp
BlockTask
队列1：某一block在队列2中开始的位置
队列2：spikes

领取时从队列1一次拿一个位置，再解析成spikes发放给lanes


SegmentTask 
    int pre;
    int edge_begin;
    int edge_end;



------------------------------------------------------------------------

## 预处理

第一版基本不需要预处理。

运行时：

``` cpp
degree =
    csr_indptr[n+1]
    -
    csr_indptr[n]
```

判断：

``` cpp
degree < 256
```

即可。

极长行 segment：

``` cpp
segment_num =
    ceil(degree / segment_size)
```

运行时生成。

------------------------------------------------------------------------

## Kernel 流程

``` cpp
for timestep:

    external input

    reset queues

    update cell blocks
        ↓
    generate tasks

    process tasks
```

------------------------------------------------------------------------

## Cell block update

一个 warp：

``` cpp
cell = block_id * 32 + lane
```

连续读取：

``` cpp
v[cell]
psc[cell]
input[cell]
```

执行：

``` cpp
LIF update
PSC decay
emit spike
```

生成：

``` cpp
spike_mask =
    ballot(spike)
```

------------------------------------------------------------------------

## Task 生成

普通 spike：

``` cpp
degree < 256
```

生成：

``` cpp
BlockTask {
    block_id,
    spike_mask
}
```

极长 spike：

``` cpp
degree >= 256
```

切：

``` text
segment_size = 256
```

例如：

``` text
fanout=1100

→ 5 SegmentTask
```

------------------------------------------------------------------------

## Queue

第一版不引入复杂 shared queue。

直接：

``` cpp
lane 0:

id = atomicAdd(counter,1)

queue[id]=task
```

原因：

原本：

``` text
每个 spike 一次 atomic
```

现在：

``` text
每个 block 一次 atomic
```

已经减少大量开销。

------------------------------------------------------------------------

## BlockTask 处理

一个 warp 领取一个 BlockTask。

得到 active neuron：

``` text
spike mask
↓
active pre neuron
```

简单版本：

``` cpp
for active neuron:

    for edge in CSR[row]:

        read post

        read weight

        warp reduce same post

        atomicAdd
```

------------------------------------------------------------------------

## Warp atomic reduction

第一版使用：

``` cpp
match_any_sync()
```

流程：

``` text
(post, weight)

↓

same post lanes

↓

warp sum

↓

leader atomicAdd
```
-

-------------------------------------------------------
------------------------------------------------------

## 总结

第一版核心只有：

1.  warp 固定处理连续 32 neuron；
2.  spike mask 转 block task；
3.  长 fanout 单独 segment；
4.  block task 内 warp 协同处理；
5.  warp 内合并后 atomic。

保持：

``` text
简单结构
少预处理
逐步 benchmark
```
