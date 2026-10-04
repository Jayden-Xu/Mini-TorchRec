import pytest
import torch

from mini_torchrec import _C
from mini_torchrec.data import make_jagged, make_tables, ref_rows, split_per_table
from mini_torchrec.fused import EBagLoop, FusedPerTable, FusedTBE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
dev = torch.device("cuda")


@pytest.mark.parametrize("T,B,rows,D,max_len", [(1, 8, 50, 4, 3), (7, 64, 1000, 32, 10), (40, 256, 500, 128, 20)])
def test_fused_matches_embedding_bag(T, B, rows, D, max_len):
    tables = make_tables(T, rows, D, dev)
    indices, offsets = make_jagged(T, B, rows, max_len, dev)
    per_table = split_per_table(indices, offsets, T, B)

    ref = EBagLoop(tables)(per_table)
    assert torch.allclose(FusedTBE(tables)(indices, offsets, B), ref, atol=1e-5)
    assert torch.allclose(FusedPerTable(tables)(per_table), ref, atol=1e-5)


@pytest.mark.parametrize("world", [1, 2, 4])
def test_shard_lookup_partials_sum_to_full(world):
    # 单卡上模拟 world 个 shard：各 shard 的 partial 相加应等于完整 lookup。
    num_rows, D, n = 1000, 48, 512
    ids = torch.randint(0, num_rows, (n,), device=dev)
    shard_size = -(-num_rows // world)
    total = torch.zeros(n, D, device=dev)
    for r in range(world):
        lo, hi = r * shard_size, min(num_rows, (r + 1) * shard_size)
        table = ref_rows(torch.arange(lo, hi, device=dev), D).contiguous()
        total += _C.shard_lookup_forward(table, ids, lo)
    assert torch.allclose(total, ref_rows(ids, D), atol=1e-6)
