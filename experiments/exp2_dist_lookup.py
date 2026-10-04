"""实验 2：row-shard lookup，AllReduce vs AllToAll。

torchrun --nproc_per_node=4 experiments/exp2_dist_lookup.py
"""
import argparse
import statistics
import time

import torch
import torch.distributed as dist

from mini_torchrec.comm import CommMeter
from mini_torchrec.data import ref_rows
from mini_torchrec.dist import (
    init_dist,
    lookup_allreduce,
    lookup_alltoall,
    make_shard,
    wire_bytes_allreduce,
    wire_bytes_alltoall,
)

p = argparse.ArgumentParser()
p.add_argument("--B", type=int, nargs="+", default=[1024, 8192, 65536])
p.add_argument("--rows", type=int, default=4_000_000)
p.add_argument("--D", type=int, default=64)
args = p.parse_args()

rank, world, dev = init_dist()
shard = make_shard(args.rows, args.D, rank, world, dev)
meter = CommMeter()


def time_dist(fn, warmup=5, iters=20):
    """各 rank 各自计时取中位数，再取所有 rank 的最大值（ms）。"""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    t = torch.tensor(statistics.median(times), device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def theory(fn, B):
    try:
        return f"{fn(world, B, args.D):>14,}"
    except NotImplementedError:
        return f"{'TODO':>14}"


if rank == 0:
    print(f"world={world} rows={args.rows} D={args.D}")
    print(f"{'B':>8} {'variant':>10} {'ms':>9} {'payload B/rank':>15} {'theory wire B':>14}")

for B in args.B:
    # 所有 rank 用同一个种子生成全局 id，rank r 的本地 batch 是其中第 r 段。
    g = torch.Generator(device="cpu").manual_seed(B)
    all_ids = torch.randint(0, args.rows, (world * B,), generator=g).to(dev)
    my_ids = all_ids[rank * B : (rank + 1) * B].contiguous()
    expect = ref_rows(my_ids, args.D)

    variants = [
        ("allreduce", lambda: lookup_allreduce(shard, all_ids, meter)[rank * B : (rank + 1) * B], wire_bytes_allreduce),
        ("alltoall", lambda: lookup_alltoall(shard, my_ids, meter), wire_bytes_alltoall),
    ]
    for name, fn, wire in variants:
        meter.reset()
        assert torch.allclose(fn(), expect, atol=1e-6), f"{name} wrong on rank {rank}"
        payload = meter.total_bytes()
        ms = time_dist(fn)
        if rank == 0:
            print(f"{B:>8} {name:>10} {ms:>9.3f} {payload:>15,} {theory(wire, B)}")

dist.destroy_process_group()
