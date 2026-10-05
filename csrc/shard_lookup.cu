#include "kernels.h"

// row-shard lookup kernel
// __global__ void shard_lookup_kernel(const float* table, const int64_t* indices,
//                                     float* output, int64_t row_offset,
//                                     int64_t shard_rows, int64_t n, int64_t D) {

//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= n) {
//         return;
//     }

//     int64_t row = indices[idx];
//     if (row >= row_offset && row < row_offset + shard_rows) {
//         row -= row_offset;
//         for (int d = 0; d < D; d++) {
//             output[idx * D + d] = table[row * D + d];
//         }
//     }
//     else {
//         for (int d = 0; d < D; d++) {
//             output[idx * D + d] = 0;
//         }
//     }
    
// } // poor perf due to non-coalesced read & write

__global__ void shard_lookup_kernel(const float* table, const int64_t* indices,
  float* output, int64_t row_offset,
  int64_t shard_rows, int64_t n, int64_t D) { // every thread handles one row(sample_idx)'s one dimension -> (sample_idx, dim_idx)

    int64_t sample_idx = blockIdx.x;
    int64_t dim_idx = (int64_t)blockIdx.y * blockDim.x + threadIdx.x;

    if (dim_idx >= D) {
      return;
    }

    int64_t local_idx = indices[sample_idx] - row_offset;
    if (local_idx >= 0 && local_idx < shard_rows) {
      output[sample_idx * D + dim_idx] = table[local_idx * D + dim_idx];
    }
    else {
      output[sample_idx * D + dim_idx] = 0;
    }
  }

void launch_shard_lookup(const float* table, const int64_t* indices,
                         float* output, int64_t row_offset, int64_t shard_rows,
                         int64_t n, int64_t D, cudaStream_t stream) {
  if (n == 0 || D == 0) {
    return;
  }

  const dim3 num_of_threads = dim3(256);
  const dim3 num_of_blocks = dim3(n, (D + num_of_threads.x - 1) / num_of_threads.x);
  shard_lookup_kernel<<<num_of_blocks, num_of_threads, 0, stream>>>(table, indices, output, row_offset, shard_rows, n, D);
}
