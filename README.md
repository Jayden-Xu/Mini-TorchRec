# Mini-TorchRec

用最小代码搞懂推荐系统 embedding 的两件事：

1. **Fused TBE**：多表合并 + pooling 进一个 kernel，为什么比逐表 `EmbeddingBag` 快
2. **分布式 lookup**：row-shard 后走 AllReduce 还是 AllToAll，通信量和延迟差在哪

## 环境

需要 CUDA 机器（实验 2 需要 ≥2 张卡）。

```bash
pip install torch pytest
pip install -e .          # 编译 mini_torchrec._C；改了 csrc/ 后重跑
```

## 结构

```
csrc/
  kernels.h           kernel 接口与数据布局
  ops.cpp             torch 绑定（已写好）
  fused_tbe.cu        TODO  实验 1 的 kernel
  shard_lookup.cu     TODO  实验 2 的本地 lookup kernel
mini_torchrec/
  data.py             表 / jagged 输入生成，ref_rows 校验用
  bench.py            计时
  fused.py            EBagLoop / FusedPerTable / FusedTBE 三个对照
  comm.py             CommMeter：记录 payload 字节数
  dist.py             TODO  lookup_allreduce / lookup_alltoall / wire_bytes_*
experiments/          两个实验脚本
tests/                kernel 正确性
```

## 要填的部分（按顺序）

| # | 位置 | 内容 | 验收 |
|---|---|---|---|
| 1 | `csrc/fused_tbe.cu` | fused kernel + launch | `pytest tests -k fused` |
| 2 | `csrc/shard_lookup.cu` | 命中写值、未命中写 0 | `pytest tests -k shard` |
| 3 | `dist.py: lookup_allreduce` | 本地 lookup + AllReduce | 实验 2 的 assert 通过 |
| 4 | `dist.py: lookup_alltoall` | id 路由 → lookup → 行路由回 | 实验 2 的 assert 通过 |
| 5 | `dist.py: wire_bytes_*` | 两种方案的理论发送字节数 | 与 payload 列对得上 |

## 实验 1：fused

```bash
python experiments/exp1_fused.py --T 1 10 100 1000
```

三列对照：`ebag_loop`（T 次 PyTorch kernel + cat）、`fused_per_table`（同一个 fused kernel 调 T 次 + cat）、`fused`（1 次）。

跑完要能回答：

- T 从 1 到 1000，三者各自怎么涨？`fused` 是否接近只随总 id 数增长？
- `fused_per_table` 和 `fused` 的差距是纯 launch + cat 开销，占 `ebag_loop` 总耗时多少？
- 把 `--B` 调大到 kernel 本身成为主要耗时，差距还剩多少？说明 fusion 的收益在什么区间最大。

## 实验 2：分布式 lookup

```bash
torchrun --nproc_per_node=4 experiments/exp2_dist_lookup.py
```

场景：每个 rank 有自己的 B 个 id，要拿到这 B 行。

- **AllReduce**：所有 rank 持有全量 `world*B` 个 id，各自输出 `[world*B, D]`（未命中填 0）后求和。
- **AllToAll**：只把 id 发给拥有者，只把命中的行发回。

跑完要能回答：

- 两种方案每个 rank 发送的字节数各是多少，比值随 `world` 怎么变？
- 实测延迟的比值和通信量的比值一致吗？不一致的部分来自哪（排序、多轮通信、lookup 本身）？
- B 很小时 AllToAll 还占优吗？
- id 分布倾斜（热点集中在一个 shard）时，两种方案各受什么影响？
