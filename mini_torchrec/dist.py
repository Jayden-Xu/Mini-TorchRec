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
    # TODO(你): _C.shard_lookup_forward + meter.all_reduce
    raise NotImplementedError


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
    raise NotImplementedError


def wire_bytes_allreduce(world, B, D):
    """方案 A 每个 rank 实际发送的字节数（按 ring AllReduce 推）。"""
    # TODO(你)
    raise NotImplementedError


def wire_bytes_alltoall(world, B, D):
    """方案 B 每个 rank 实际发送的字节数（id 均匀分布下的期望）。"""
    # TODO(你)
    raise NotImplementedError
