#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// 所有表共用同一个 D。weights 是 T 张表按行拼接后的连续内存 [total_rows, D]。
// indices 是表内局部 id，按 table-major 排列；bag (t, b) 的 id 区间是
// indices[offsets[t*B+b] : offsets[t*B+b+1]]。输出 output[b, t*D : (t+1)*D]，sum pooling。
void launch_fused_tbe(const float* weights, const int64_t* weights_offsets,
                      const int64_t* indices, const int64_t* offsets,
                      float* output, int64_t T, int64_t B, int64_t D,
                      cudaStream_t stream);

// 本卡持有全局行 [row_offset, row_offset + shard_rows)。
// 命中写 table 的对应行，未命中写 0。输出 [n, D]。
void launch_shard_lookup(const float* table, const int64_t* indices,
                         float* output, int64_t row_offset, int64_t shard_rows,
                         int64_t n, int64_t D, cudaStream_t stream);
