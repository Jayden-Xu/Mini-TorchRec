import torch


def make_tables(T, rows, D, device, seed=0):
    """T 张 [rows, D] 的 float32 表。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randn(rows, D, generator=g).to(device) for _ in range(T)]


def make_jagged(T, B, rows, max_len, device, seed=0):
    """KJT 风格的输入：table-major 打平。

    返回 indices [total_L]（表内局部 id）和 offsets [T*B+1]。
    每个 bag 的长度在 [0, max_len] 均匀取，包含空 bag。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    lengths = torch.randint(0, max_len + 1, (T * B,), generator=g)
    indices = torch.randint(0, rows, (int(lengths.sum()),), generator=g)
    offsets = torch.zeros(T * B + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(lengths, 0)
    return indices.to(device), offsets.to(device)


def split_per_table(indices, offsets, T, B):
    """把打平的输入切回每张表各自的 (indices, offsets)，offsets 从 0 起。"""
    out = []
    for t in range(T):
        off = offsets[t * B : (t + 1) * B + 1]
        out.append((indices[off[0] : off[-1]].contiguous(), (off - off[0]).contiguous()))
    return out


def ref_rows(ids, D):
    """全局表的解析定义：第 r 行第 d 列的值只由 (r, d) 决定。

    任何 rank 都能不持有全表就算出期望结果，用来做分布式 lookup 的校验。
    """
    d = torch.arange(D, device=ids.device)
    h = (ids[:, None] * 2654435761 + d[None, :] * 40503) % 1000003
    return h.float() / 1000003 - 0.5
