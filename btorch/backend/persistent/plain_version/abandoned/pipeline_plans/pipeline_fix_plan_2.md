修正 1：恢复 warp 独立执行，取消 block 级批同步

保留：

每个 warp 独立 claim
每个 warp 独立处理
每个 warp 独立结束

但不恢复原来的紧密 atomic 轮询。

每个 warp 使用：

acquire load；
ready-before-claim；
指数 backoff；
每次一个 CAS；
无 block-level __syncthreads()。
伪代码
const int lane = threadIdx.x & 31;

int backoff = 32;

while (true) {
    int task = -1;
    uint32_t state = 0;

    if (lane == 0) {
        const int head =
            load_acquire(queue_head);

        const int tail =
            load_acquire(queue_tail);

        if (head < tail) {
            state = load_acquire(
                spike_queue_state + head);

            if ((state >> 16) ==
                expected_epoch) {
                if (atomicCAS(
                        queue_head,
                        head,
                        head + 1) == head) {
                    task = head;
                }
            }
        }
    }

    task = __shfl_sync(
        0xffffffffu, task, 0);

    state = __shfl_sync(
        0xffffffffu, state, 0);

    if (task >= 0) {
        backoff = 32;

        process_fragment(task, state);

        __syncwarp();

        if (lane == 0) {
            atomicAdd(
                propagation_done_tasks,
                1);
        }

        continue;
    }

    bool exit = false;

    if (lane == 0) {
        const int update_done =
            load_acquire(update_done_blocks);

        const int tail =
            load_acquire(queue_tail);

        const int done =
            load_acquire(
                propagation_done_tasks);

        exit =
            update_done ==
                update_block_count &&
            done >= tail;

        if (!exit) {
            __nanosleep(backoff);
            backoff =
                min(backoff * 2, 1024);
        }
    }

    exit = __shfl_sync(
        0xffffffffu, exit, 0);

    if (exit)
        break;
}

这样热循环里只剩：

warp sync

而没有：

三次 block sync / 8 tasks
修正 2：逐 fragment 发布

把当前：

for fragment:
    queue_neuron = n;

for fragment:
    state_store_release();

改为：

for (int fragment = 0;
     fragment < fragment_count;
     ++fragment) {
    const int task =
        first_task + fragment;

    spike_queue_neuron[task] = n;

    const uint32_t state =
        (expected_epoch << 16) |
        static_cast<uint32_t>(
            fragment + 1);

    state_store_release(
        spike_queue_state + task,
        state);
}

这能让第一个 fragment 更快进入消费阶段。

修正 3：诊断并测量 head-of-line blocking

增加三个 debug counter：

unsigned long long* no_task_polls;
unsigned long long* head_not_ready_polls;
unsigned long long* failed_claims;

分别记录：

if (head >= tail)
    no_task++;

if (head < tail &&
    state epoch 不匹配)
    head_not_ready++;

if (state ready &&
    CAS 失败)
    failed_claims++;

如果：

head_not_ready_polls

很大，说明 reserve/publish 洞是主要问题。

如果：

failed_claims

很大，说明 consumer 数量仍过多。

如果：

no_task_polls

很大，则是预留 propagation blocks 与低发放率不匹配