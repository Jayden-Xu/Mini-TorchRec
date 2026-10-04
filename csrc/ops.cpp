#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "kernels.h"

namespace {

void check(const torch::Tensor& t, torch::ScalarType dtype, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == dtype, name, " has wrong dtype");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor fused_tbe_forward(const torch::Tensor& weights,
                                const torch::Tensor& weights_offsets,
                                const torch::Tensor& indices,
                                const torch::Tensor& offsets, int64_t B) {
  check(weights, torch::kFloat32, "weights");
  check(weights_offsets, torch::kInt64, "weights_offsets");
  check(indices, torch::kInt64, "indices");
  check(offsets, torch::kInt64, "offsets");
  TORCH_CHECK(weights.dim() == 2, "weights must be [total_rows, D]");

  const int64_t T = weights_offsets.numel();
  const int64_t D = weights.size(1);
  TORCH_CHECK(offsets.numel() == T * B + 1, "offsets must have T*B+1 entries");

  const c10::cuda::CUDAGuard guard(weights.device());
  auto output = torch::empty({B, T * D}, weights.options());
  launch_fused_tbe(weights.data_ptr<float>(), weights_offsets.data_ptr<int64_t>(),
                   indices.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
                   output.data_ptr<float>(), T, B, D,
                   at::cuda::getCurrentCUDAStream().stream());
  return output;
}

torch::Tensor shard_lookup_forward(const torch::Tensor& table,
                                   const torch::Tensor& indices,
                                   int64_t row_offset) {
  check(table, torch::kFloat32, "table");
  check(indices, torch::kInt64, "indices");
  TORCH_CHECK(table.dim() == 2, "table must be [shard_rows, D]");

  const int64_t n = indices.numel();
  const int64_t D = table.size(1);

  const c10::cuda::CUDAGuard guard(table.device());
  auto output = torch::empty({n, D}, table.options());
  launch_shard_lookup(table.data_ptr<float>(), indices.data_ptr<int64_t>(),
                      output.data_ptr<float>(), row_offset, table.size(0), n, D,
                      at::cuda::getCurrentCUDAStream().stream());
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_tbe_forward", &fused_tbe_forward);
  m.def("shard_lookup_forward", &shard_lookup_forward);
}
