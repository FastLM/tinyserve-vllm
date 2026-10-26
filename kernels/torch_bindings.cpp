#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "tinyserve_kernels.h"

namespace {

void check_launch(int code) {
  TORCH_CHECK(code == 0, "TinyServe kernel failed with code ", code,
              " (1 = unsupported shape, 2 = CUDA error)");
}

}  // namespace

std::vector<torch::Tensor> fused_sparse_attention(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor k_min,
    torch::Tensor k_max,
    torch::Tensor page_lens,
    torch::Tensor num_pages,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    int64_t k_pages,
    int64_t n_recent,
    double scale) {
  TORCH_CHECK(q.is_cuda(), "q must be CUDA");
  TORCH_CHECK(q.dtype() == torch::kFloat32, "q must be float32");
  TORCH_CHECK(q.dim() == 3, "q must be [batch, heads, dim]");
  TORCH_CHECK(k.dim() == 5 && v.dim() == 5, "k and v must be [batch, kv_heads, pages, page_size, dim]");
  TORCH_CHECK(k.dtype() == v.dtype(), "k and v dtypes must match");
  TORCH_CHECK(k.dtype() == torch::kFloat16 || k.dtype() == torch::kInt8, "k must be fp16 or int8");
  const at::cuda::OptionalCUDAGuard guard(device_of(q));

  q = q.contiguous();
  k = k.contiguous();
  v = v.contiguous();
  k_min = k_min.contiguous();
  k_max = k_max.contiguous();
  page_lens = page_lens.to(torch::kInt32).contiguous();
  num_pages = num_pages.to(torch::kInt32).contiguous();

  const int batch = static_cast<int>(q.size(0));
  const int num_q_heads = static_cast<int>(q.size(1));
  const int head_dim = static_cast<int>(q.size(2));
  const int num_kv_heads = static_cast<int>(k.size(1));
  const int max_pages = static_cast<int>(k.size(2));
  const int page_size = static_cast<int>(k.size(3));
  TORCH_CHECK(k.size(0) == batch && k.size(4) == head_dim, "k must be [B, Hkv, P, S, D]");
  TORCH_CHECK(v.sizes() == k.sizes(), "v shape must match k");
  TORCH_CHECK(page_lens.dim() == 2 && num_pages.dim() == 1, "page tables must be [B, P] and [B]");
  const int kv_kind = k.dtype() == torch::kInt8 ? 1 : 0;

  auto out = torch::empty({batch, num_q_heads, head_dim}, q.options());
  auto selected = torch::empty(
      {batch, num_q_heads, k_pages}, q.options().dtype(torch::kInt32));

  const float* k_scale_ptr =
      (k_scale.defined() && k_scale.numel() > 0) ? k_scale.data_ptr<float>() : nullptr;
  const float* v_scale_ptr =
      (v_scale.defined() && v_scale.numel() > 0) ? v_scale.data_ptr<float>() : nullptr;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  check_launch(tinyserve_fused_sparse_attention(
      q.data_ptr<float>(), k.data_ptr(), v.data_ptr(), k_min.data_ptr<float>(),
      k_max.data_ptr<float>(), page_lens.data_ptr<int>(), num_pages.data_ptr<int>(),
      k_scale_ptr, v_scale_ptr, out.data_ptr<float>(), selected.data_ptr<int>(), batch,
      num_q_heads, num_kv_heads, head_dim, page_size, max_pages, static_cast<int>(k_pages),
      static_cast<int>(n_recent), static_cast<float>(scale), kv_kind, stream));
  return {out, selected};
}

void update_bounds(
    torch::Tensor key_token,
    torch::Tensor k_min,
    torch::Tensor k_max,
    torch::Tensor page_index,
    torch::Tensor reset) {
  TORCH_CHECK(key_token.is_cuda() && key_token.dtype() == torch::kFloat16, "key_token must be fp16 CUDA");
  const at::cuda::OptionalCUDAGuard guard(device_of(key_token));
  key_token = key_token.contiguous();
  k_min = k_min.contiguous();
  k_max = k_max.contiguous();
  page_index = page_index.to(torch::kInt32).contiguous();
  reset = reset.to(torch::kInt32).contiguous();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  check_launch(tinyserve_update_bounds(
      key_token.data_ptr(), k_min.data_ptr<float>(), k_max.data_ptr<float>(),
      page_index.data_ptr<int>(), reset.data_ptr<int>(), static_cast<int>(key_token.size(0)),
      static_cast<int>(key_token.size(1)), static_cast<int>(k_min.size(2)),
      static_cast<int>(key_token.size(2)), stream));
}

void reduce_page_bounds(
    torch::Tensor key_page,
    torch::Tensor k_min,
    torch::Tensor k_max,
    torch::Tensor lengths) {
  TORCH_CHECK(key_page.is_cuda() && key_page.dtype() == torch::kFloat16, "key_page must be fp16 CUDA");
  const at::cuda::OptionalCUDAGuard guard(device_of(key_page));
  key_page = key_page.contiguous();
  k_min = k_min.contiguous();
  k_max = k_max.contiguous();
  lengths = lengths.to(torch::kInt32).contiguous();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  check_launch(tinyserve_reduce_page_bounds(
      key_page.data_ptr(), k_min.data_ptr<float>(), k_max.data_ptr<float>(),
      lengths.data_ptr<int>(), static_cast<int>(key_page.size(0)),
      static_cast<int>(key_page.size(1)), static_cast<int>(key_page.size(2)),
      static_cast<int>(key_page.size(3)), stream));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_sparse_attention", &fused_sparse_attention, "Fused query-aware sparse attention");
  m.def("update_bounds", &update_bounds, "Incremental per-page key bounds");
  m.def("reduce_page_bounds", &reduce_page_bounds, "Reduce one KV page into key bounds");
}
