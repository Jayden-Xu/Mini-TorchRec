import statistics
import time

import torch


def time_cuda(fn, warmup=10, iters=50):
    """返回 fn 单次耗时的中位数（ms），每次计时前后都 synchronize。"""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)
