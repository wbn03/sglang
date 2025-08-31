#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include "pytorch_extension_utils.h"

template <typename T>
__global__ void meanpooling_kernel(
    const T* k_cache,
    T* compressed_cache,
    const int* begin_pos,
    const int* compressed_len,
    int batch_num,
    int k_cache_max_len,
    int compressed_cache_max_len,
    int dim,
    int kernel_size,
    int stride) {
  int batch_id = blockIdx.x;
  int compressed_pos = blockIdx.y;
  if (compressed_pos >= compressed_len[batch_id]) {
    return;
  }
  int orig_left = (begin_pos[batch_id] + compressed_pos) * stride;
  T* c = compressed_cache + (batch_id * compressed_cache_max_len + begin_pos[batch_id] + compressed_pos) * dim;
  const T* k = k_cache + (batch_id * k_cache_max_len + orig_left) * dim;
  int offset = threadIdx.x;
  if (offset >= dim) {
    return;
  }
  float accum = 0.0f;
  for (int i = 0; i < kernel_size; i++) {
    accum += static_cast<float>(k[i * dim + offset]);
  }
  c[offset] = static_cast<T>(accum / kernel_size);
}

// meanpooling extension
torch::Tensor meanpooling(
    const torch::Tensor& k_cache,
    torch::Tensor& compressed_cache,
    const torch::Tensor& begin_pos,
    const torch::Tensor& compressed_len,
    const std::optional<at::Tensor>& cu_begin_pos_,
    const std::optional<at::Tensor>& cu_compressed_len_,
    int64_t kernel_size,
    int64_t stride) {
  // get param
  auto batch_num = k_cache.size(0);
  auto k_cache_max_len = k_cache.size(1);
  auto dim = k_cache.size(2);
  auto compressed_cache_max_len = compressed_cache.size(1);
  // check error
  TORCH_CHECK(begin_pos.dtype() == torch::kInt32, "begin_pos must have dtype int32");
  TORCH_CHECK(compressed_len.dtype() == torch::kInt32, "compressed_len must have dtype int32");
  TORCH_CHECK(k_cache.is_cuda(), "k_cache must be a CUDA tensor");
  TORCH_CHECK(begin_pos.is_cpu(), "begin_pos must be a CPU tensor");
  TORCH_CHECK(compressed_len.is_cpu(), "compressed_len must be a CPU tensor");
  TORCH_CHECK(batch_num == compressed_cache.size(0), "k_cache must has the same batch_num");
  TORCH_CHECK(dim == compressed_cache.size(2), "k_cache must has the same dim");
  TORCH_CHECK(batch_num == begin_pos.size(0), "begin_pos must has the shape of batch_num");
  TORCH_CHECK(batch_num == compressed_len.size(0), "compressed_len must has the shape of batch_num");

  // copy pos and len info to device
  at::Tensor cu_begin_pos;
  at::Tensor cu_compressed_len;
  // get cuda stream
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (cu_begin_pos_.has_value()) {
    TORCH_CHECK(cu_begin_pos_.value().dtype() == torch::kInt32, "cu_begin_pos must have dtype int32");
    TORCH_CHECK(cu_begin_pos_.value().is_cuda(), "cu_begin_pos must be a CUDA tensor");
    cu_begin_pos = cu_begin_pos_.value();
  } else {
    cu_begin_pos.reshape({batch_num});
    cudaError_t err = cudaMemcpyAsync(
        cu_begin_pos.data_ptr<int>(),
        begin_pos.data_ptr<int>(),
        batch_num * sizeof(int),
        cudaMemcpyHostToDevice,
        stream);
    if (err != cudaSuccess) {
      TORCH_CHECK(false, "CUDA error in h2d copy async: ", cudaGetErrorString(err));
    }
  }
  if (cu_compressed_len_.has_value()) {
    TORCH_CHECK(cu_compressed_len_.value().dtype() == torch::kInt32, "cu_compressed_len must have dtype int32");
    TORCH_CHECK(cu_compressed_len_.value().is_cuda(), "cu_compressed_len must be a CUDA tensor");
    cu_compressed_len = cu_compressed_len_.value();
  } else {
    cu_compressed_len.reshape({batch_num});
    cudaError_t err = cudaMemcpyAsync(
        cu_compressed_len.data_ptr<int>(),
        compressed_len.data_ptr<int>(),
        batch_num * sizeof(int),
        cudaMemcpyHostToDevice,
        stream);
    if (err != cudaSuccess) {
      TORCH_CHECK(false, "CUDA error in h2d copy async: ", cudaGetErrorString(err));
    }
  }

  // config grid and block
  dim3 block(1024);
  dim3 grid(batch_num, compressed_cache_max_len);

  // dispatch by type
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(k_cache.scalar_type(), "meanpooling_kernel", [&] {
    meanpooling_kernel<scalar_t><<<grid, block, 0, stream>>>(
        k_cache.data_ptr<scalar_t>(),
        compressed_cache.data_ptr<scalar_t>(),
        cu_begin_pos.data_ptr<int>(),
        cu_compressed_len.data_ptr<int>(),
        batch_num,
        k_cache_max_len,
        compressed_cache_max_len,
        dim,
        kernel_size,
        stride);
  });

  // check error
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "CUDA error in meanpooling_forward: ", cudaGetErrorString(err));
  }

  return compressed_cache;
}
