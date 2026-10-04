import torch.distributed as dist


class CommMeter:
    """torch.distributed 的薄封装，记录每次调用交给 NCCL 的 payload 字节数。

    注意这是"提交的 buffer 大小"，不是线上真实字节数：all_reduce 在 NCCL 内部
    走 ring/tree，真实发送量要自己推（见 dist.py 的 wire_bytes_*）。
    """

    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls = []

    def total_bytes(self):
        return sum(b for _, b in self.calls)

    def all_reduce(self, tensor):
        self.calls.append(("all_reduce", tensor.numel() * tensor.element_size()))
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    def all_to_all_single(self, output, input, output_split_sizes, input_split_sizes):
        """按 dim 0 切分：input 的第 r 段发给 rank r，output 的第 r 段来自 rank r。

        只统计发给其他 rank 的部分。
        """
        per_row = input[0].numel() * input.element_size() if input.size(0) > 0 else 0
        remote_rows = sum(input_split_sizes) - input_split_sizes[dist.get_rank()]
        self.calls.append(("all_to_all", remote_rows * per_row))
        dist.all_to_all_single(output, input, output_split_sizes, input_split_sizes)
