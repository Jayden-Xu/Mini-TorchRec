"""实验 1：fused TBE vs 逐表 lookup。

python experiments/exp1_fused.py
python experiments/exp1_fused.py --T 1 10 100 1000 --B 1024
"""
import argparse
import csv
import os

import torch

from mini_torchrec.bench import time_cuda
from mini_torchrec.data import make_jagged, make_tables, split_per_table
from mini_torchrec.fused import EBagLoop, FusedPerTable, FusedTBE

p = argparse.ArgumentParser()
p.add_argument("--T", type=int, nargs="+", default=[1, 10, 100, 1000])
p.add_argument("--B", type=int, default=1024)
p.add_argument("--rows", type=int, default=10000)
p.add_argument("--D", type=int, default=64)
p.add_argument("--max-len", type=int, default=20)
p.add_argument("--out", default="results/exp1_fused.csv")
args = p.parse_args()

dev = torch.device("cuda")
rows_out = []
print(f"{'T':>6} {'ebag_loop':>12} {'fused_per_table':>16} {'fused':>10}   (ms)")
for T in args.T:
    tables = make_tables(T, args.rows, args.D, dev)
    indices, offsets = make_jagged(T, args.B, args.rows, args.max_len, dev)
    per_table = split_per_table(indices, offsets, T, args.B)

    ebag, per, fused = EBagLoop(tables), FusedPerTable(tables), FusedTBE(tables)
    t_ebag = time_cuda(lambda: ebag(per_table))
    t_per = time_cuda(lambda: per(per_table))
    t_fused = time_cuda(lambda: fused(indices, offsets, args.B))

    print(f"{T:>6} {t_ebag:>12.3f} {t_per:>16.3f} {t_fused:>10.3f}")
    rows_out.append([T, args.B, args.rows, args.D, args.max_len, t_ebag, t_per, t_fused])

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["T", "B", "rows", "D", "max_len", "ebag_loop_ms", "fused_per_table_ms", "fused_ms"])
    w.writerows(rows_out)
