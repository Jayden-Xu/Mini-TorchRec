#include "kernels.h"

// TODO(你): fused TBE kernel。接口和数据布局见 kernels.h。
// 写之前先想清楚：
//   1. 一个 block / 一个 warp / 一个 thread 各负责什么？(bag？dim？)
//   2. 怎么从 bag 编号反推 t，再用 weights_offsets[t] 定位到这张表的行？
//   3. 一个 bag 内多个 id 的 pooling 在哪里累加，结果只写一次 output。
//   4. 空 bag（offsets 相邻两项相等）必须输出 0，output 是未初始化内存。
__global__ void fused_tbe_kernel(const float* weights,
                                 const int64_t* weights_offsets,
                                 const int64_t* indices, const int64_t* offsets,
                                 float* output, int64_t T, int64_t B, int64_t D) {
}

void launch_fused_tbe(const float* weights, const int64_t* weights_offsets,
                      const int64_t* indices, const int64_t* offsets,
                      float* output, int64_t T, int64_t B, int64_t D,
                      cudaStream_t stream) {
  // TODO(你): 选 grid / block 维度，在 stream 上 launch fused_tbe_kernel。
}
