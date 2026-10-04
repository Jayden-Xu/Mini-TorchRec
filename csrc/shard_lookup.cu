#include "kernels.h"

// TODO(你): row-shard lookup kernel，逻辑见笔记 Multi-GPU Embedding Lookup。
// 命中本卡写真值，未命中写 0（output 是未初始化内存）。
__global__ void shard_lookup_kernel(const float* table, const int64_t* indices,
                                    float* output, int64_t row_offset,
                                    int64_t shard_rows, int64_t n, int64_t D) {
}

void launch_shard_lookup(const float* table, const int64_t* indices,
                         float* output, int64_t row_offset, int64_t shard_rows,
                         int64_t n, int64_t D, cudaStream_t stream) {
  // TODO(你): 选 grid / block 维度，在 stream 上 launch shard_lookup_kernel。
}
