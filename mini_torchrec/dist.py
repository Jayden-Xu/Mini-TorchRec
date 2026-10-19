import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from . import _C
from .comm import CommMeter
from .data import ref_rows


@dataclass
class RowShard:
    weight: torch.Tensor  # [shard_rows, D]，本卡持有的行
    row_offset: int       # 本卡第一行的全局行号
    shard_size: int       # 每卡的行数（最后一卡可能不满）
    rank: int
    world: int


def init_dist():
    """torchrun 下每个进程调用一次，返回 (rank, world, device)。"""
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return dist.get_rank(), dist.get_world_size(), torch.device("cuda")


def make_shard(num_rows, D, rank, world, device):
    shard_size = math.ceil(num_rows / world)
    lo = rank * shard_size
    hi = min(num_rows, lo + shard_size)
    weight = ref_rows(torch.arange(lo, hi, device=device), D).contiguous()
    return RowShard(weight, lo, shard_size, rank, world)


def lookup_allreduce(shard: RowShard, all_ids: torch.Tensor, meter: CommMeter):
    """方案 A：每卡持有全量 id，本地 lookup（未命中填 0）后 AllReduce(Sum)。

    all_ids: [world * B]，所有 rank 上内容相同。返回 [world * B, D]。
    """

    partial = _C.shard_lookup_forward(shard.weight, all_ids, shard.row_offset)
    meter.all_reduce(partial)
    return partial


def lookup_alltoall(shard: RowShard, my_ids: torch.Tensor, meter: CommMeter):
    """方案 B：每卡只有自己的 B 个 id，先把 id 路由给拥有者，再把行路由回来。

    my_ids: [B]，各 rank 不同。返回 [B, D]，行顺序与 my_ids 一致。
    """
    # TODO(你): 大致五步，全部通信走 meter.all_to_all_single
    #   1. 算每个 id 的 owner，按 owner 排序，记下排序置换
    #   2. 交换"我要发给你几个 id"，得到每个 rank 会收到多少
    #   3. all_to_all 发 id
    #   4. 对收到的 id 做本地 lookup（此时应该全部命中）
    #   5. all_to_all 把行发回去，用第 1 步的置换还原顺序

    world = shard.world
    D = shard.weight.shape[1]
    device = my_ids.device

    # calculate owner, permutation and send_counts
    owner = my_ids // shard.shard_size # e.g. [6, 1, 5] -> [1, 0, 1]
    perm = torch.argsort(owner)
    sorted_ids = my_ids[perm]
    send_counts = torch.bincount(owner, minlength=world)

    # exchange count
    recv_counts = torch.empty_like(send_counts)
    meter.all_to_all_single(recv_counts, send_counts, [1] * world, [1] * world)

    send_list = send_counts.tolist()
    recv_list = recv_counts.tolist()

    # exchange id
    recv_ids = torch.empty(sum(recv_list), dtype=my_ids.dtype, device=device)
    meter.all_to_all_single(recv_ids, sorted_ids, recv_list, send_list)

    # lookup
    rows = _C.shard_lookup_forward(shard.weight, recv_ids, shard.row_offset)

    # exchange row and restore order
    bank = torch.empty(my_ids.numel(), D, dtype=rows.dtype, device=device)
    meter.all_to_all_single(bank, rows, send_list, recv_list)

    out = torch.empty_like(bank)
    out[perm] = bank
    return out
    


def wire_bytes_allreduce(world, B, D):
    """方案 A 每个 rank 实际发送的字节数（按 ring AllReduce 推）。"""
    FLOAT = 4

    # AllReduce 的 buffer：全量 world*B 个 id 的输出 [world*B, D]
    buffer_bytes = world * B * D * FLOAT

    # ring 把 buffer 切成 world 块，每步每个 rank 给下一个 rank 发一块
    chunk_bytes = buffer_bytes / world

    # reduce-scatter：world-1 步，每个 rank 拿到一块的完整和
    reduce_scatter = (world - 1) * chunk_bytes

    # all-gather：world-1 步，把求好和的块转发给所有人
    all_gather = (world - 1) * chunk_bytes

    return round(reduce_scatter + all_gather)


def wire_bytes_alltoall(world, B, D):
    """方案 B 每个 rank 实际发送的字节数（id 均匀分布下的期望）。"""
    INT64, FLOAT = 8, 4

    # 第 0 轮：给每个其他 rank 发 1 个 int64 计数
    counts_bytes = (world - 1) * INT64

    # 第 1 轮：id 均匀分布，有 (world-1)/world 的 id 不归本卡，要发出去
    remote_ids = B * (world - 1) / world
    ids_bytes = remote_ids * INT64

    # 第 2 轮：发回的行数 = 从其他 rank 收到的 id 数。
    # 每个其他 rank 平均发来 B/world 个，共 world-1 个 rank，期望也是 B*(world-1)/world
    remote_rows = B / world * (world - 1)
    rows_bytes = remote_rows * D * FLOAT

    return round(counts_bytes + ids_bytes + rows_bytes)
