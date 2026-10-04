import torch
import torch.nn.functional as F

from . import _C


class EBagLoop(torch.nn.Module):
    """非 fused baseline：逐表 embedding_bag + cat，对应 EmbeddingBagCollection。"""

    def __init__(self, tables):
        super().__init__()
        self.tables = tables

    def forward(self, per_table):
        pooled = [
            F.embedding_bag(idx, w, off, mode="sum", include_last_offset=True)
            for w, (idx, off) in zip(self.tables, per_table)
        ]
        return torch.cat(pooled, dim=1)


class FusedTBE(torch.nn.Module):
    """所有表拼成一块连续内存，一次 kernel 完成 lookup + pooling。"""

    def __init__(self, tables):
        super().__init__()
        self.weights = torch.cat(tables, dim=0).contiguous()
        rows = torch.tensor([0] + [t.size(0) for t in tables[:-1]], device=self.weights.device)
        self.weights_offsets = torch.cumsum(rows, 0)

    def forward(self, indices, offsets, B):
        return _C.fused_tbe_forward(self.weights, self.weights_offsets, indices, offsets, B)


class FusedPerTable(torch.nn.Module):
    """同一个 fused kernel，但每张表单独 launch 一次再 cat。

    和 FusedTBE 的差别只有 launch 次数和 cat，用来把这部分开销单独量出来。
    """

    def __init__(self, tables):
        super().__init__()
        self.tables = tables
        self.zero = torch.zeros(1, dtype=torch.int64, device=tables[0].device)

    def forward(self, per_table):
        pooled = [
            _C.fused_tbe_forward(w, self.zero, idx, off, off.numel() - 1)
            for w, (idx, off) in zip(self.tables, per_table)
        ]
        return torch.cat(pooled, dim=1)
