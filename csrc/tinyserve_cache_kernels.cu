/*
 * TinyServe Cache Kernels Integration with vLLM
 * This file integrates TinyServe optimization kernels with vLLM's cache system
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "tinyserve_kernels.h"
#include "cuda_utils.h"
#include "cuda_compat.h"

#include "cache.h"

// Fragmentation Analysis for vLLM Block Tables
void tinyserve_fragmentation_analysis(
    torch::Tensor& block_table,             // [num_seqs, max_blocks_per_seq]
    torch::Tensor& fragmentation_scores,    // [num_chunks] output
    torch::Tensor& free_block_runs,         // [max_runs] output
    torch::Tensor& free_block_run_lengths,  // [max_runs] output
    torch::Tensor& num_free_runs,           // [1] output
    int total_blocks, int block_size) {
  TORCH_CHECK(block_table.device().is_cuda(), "block_table must be on CUDA");
  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be int32");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(block_table));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Create TinyServeBlockTable structure
  TinyServeBlockTable table;
  table.physical_block_ids = block_table.data_ptr<int>();
  table.max_blocks_per_seq = block_table.size(1);
  table.total_physical_blocks = total_blocks;
  table.block_size = block_size;

  // Allocate device memory for block_ref_counts (simplified - using block_table
  // as reference) In real implementation, this would come from vLLM's block
  // manager
  int* d_block_ref_counts;
  cudaMalloc(&d_block_ref_counts, total_blocks * sizeof(int));
  cudaMemset(d_block_ref_counts, 0, total_blocks * sizeof(int));
  table.block_ref_counts = d_block_ref_counts;

  // Allocate other required fields
  int* d_block_access_patterns;
  int* d_block_priority;
  float* d_block_weights;
  cudaMalloc(&d_block_access_patterns,
             block_table.size(0) * table.max_blocks_per_seq * sizeof(int));
  cudaMalloc(&d_block_priority, total_blocks * sizeof(int));
  cudaMalloc(&d_block_weights, total_blocks * sizeof(float));
  cudaMemset(d_block_access_patterns, 0,
             block_table.size(0) * table.max_blocks_per_seq * sizeof(int));
  cudaMemset(d_block_priority, 0, total_blocks * sizeof(int));
  cudaMemset(d_block_weights, 0.0f, total_blocks * sizeof(float));

  table.block_access_patterns = d_block_access_patterns;
  table.block_priority = d_block_priority;
  table.block_weights = d_block_weights;
  table.cache_size = total_blocks / 4;  // 25% cache

  // Launch fragmentation analysis
  tinyserve_launch_fragmentation_analysis(
      &table, fragmentation_scores.data_ptr<int>(),
      free_block_runs.data_ptr<int>(), free_block_run_lengths.data_ptr<int>(),
      num_free_runs.data_ptr<int>(), total_blocks, stream);

  cudaStreamSynchronize(stream);

  // Cleanup
  cudaFree(d_block_ref_counts);
  cudaFree(d_block_access_patterns);
  cudaFree(d_block_priority);
  cudaFree(d_block_weights);
}

// Fragmentation-Aware Block Allocation
void tinyserve_fragmentation_aware_allocation(
    torch::Tensor& block_table,  // [num_seqs, max_blocks_per_seq] input/output
    torch::Tensor& seq_ids,      // [num_allocations]
    torch::Tensor& logical_block_ids,  // [num_allocations]
    torch::Tensor& num_blocks_needed,  // [num_allocations]
    torch::Tensor& allocated_blocks,   // [num_allocations, max_blocks] output
    torch::Tensor& fragmentation_scores,    // [num_chunks]
    torch::Tensor& free_block_runs,         // [max_runs]
    torch::Tensor& free_block_run_lengths,  // [max_runs]
    torch::Tensor& num_free_runs,           // [1]
    int total_blocks, int block_size) {
  TORCH_CHECK(block_table.device().is_cuda(), "block_table must be on CUDA");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(block_table));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Create TinyServeBlockTable structure
  TinyServeBlockTable table;
  table.physical_block_ids = block_table.data_ptr<int>();
  table.max_blocks_per_seq = block_table.size(1);
  table.total_physical_blocks = total_blocks;
  table.block_size = block_size;

  // Allocate device memory for block management
  int* d_block_ref_counts;
  int* d_block_access_patterns;
  int* d_block_priority;
  float* d_block_weights;
  cudaMalloc(&d_block_ref_counts, total_blocks * sizeof(int));
  cudaMalloc(&d_block_access_patterns,
             block_table.size(0) * table.max_blocks_per_seq * sizeof(int));
  cudaMalloc(&d_block_priority, total_blocks * sizeof(int));
  cudaMalloc(&d_block_weights, total_blocks * sizeof(float));
  cudaMemset(d_block_ref_counts, 0, total_blocks * sizeof(int));
  cudaMemset(d_block_access_patterns, 0,
             block_table.size(0) * table.max_blocks_per_seq * sizeof(int));
  cudaMemset(d_block_priority, 0, total_blocks * sizeof(int));
  cudaMemset(d_block_weights, 0.0f, total_blocks * sizeof(float));

  table.block_ref_counts = d_block_ref_counts;
  table.block_access_patterns = d_block_access_patterns;
  table.block_priority = d_block_priority;
  table.block_weights = d_block_weights;
  table.cache_size = total_blocks / 4;

  // Get num_free_runs from device
  int h_num_free_runs;
  cudaMemcpy(&h_num_free_runs, num_free_runs.data_ptr<int>(), sizeof(int),
             cudaMemcpyDeviceToHost);

  // Launch fragmentation-aware allocation
  tinyserve_launch_fragmentation_aware_allocation(
      &table, seq_ids.data_ptr<int>(), logical_block_ids.data_ptr<int>(),
      num_blocks_needed.data_ptr<int>(), allocated_blocks.data_ptr<int>(),
      fragmentation_scores.data_ptr<int>(), free_block_runs.data_ptr<int>(),
      free_block_run_lengths.data_ptr<int>(), seq_ids.size(0), h_num_free_runs,
      stream);

  cudaStreamSynchronize(stream);

  // Cleanup
  cudaFree(d_block_ref_counts);
  cudaFree(d_block_access_patterns);
  cudaFree(d_block_priority);
  cudaFree(d_block_weights);
}

// Defragmentation for vLLM KV Cache
void tinyserve_defragmentation(torch::Tensor& key_cache,   // [num_blocks, block_size, head_dim]
                               torch::Tensor& value_cache,  // [num_blocks, block_size, head_dim]
                               torch::Tensor& block_table,  // [num_seqs, max_blocks_per_seq]
                               torch::Tensor& block_mapping,  // [num_blocks] old -> new mapping
                               torch::Tensor& blocks_to_move,  // [num_blocks_to_move]
                               int block_size, int head_dim) {
  TORCH_CHECK(key_cache.device().is_cuda(), "key_cache must be on CUDA");
  TORCH_CHECK(value_cache.device().is_cuda(), "value_cache must be on CUDA");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Create TinyServeBlockTable structure
  TinyServeBlockTable table;
  table.physical_block_ids = block_table.data_ptr<int>();
  table.max_blocks_per_seq = block_table.size(1);
  table.total_physical_blocks = key_cache.size(0);
  table.block_size = block_size;

  // Allocate device memory for block management
  int* d_block_ref_counts;
  int* d_block_access_patterns;
  int* d_block_priority;
  float* d_block_weights;
  cudaMalloc(&d_block_ref_counts, table.total_physical_blocks * sizeof(int));
  cudaMalloc(&d_block_access_patterns,
             block_table.size(0) * table.max_blocks_per_seq * sizeof(int));
  cudaMalloc(&d_block_priority, table.total_physical_blocks * sizeof(int));
  cudaMalloc(&d_block_weights, table.total_physical_blocks * sizeof(float));
  cudaMemset(d_block_ref_counts, 0,
             table.total_physical_blocks * sizeof(int));
  cudaMemset(d_block_access_patterns, 0,
             block_table.size(0) * table.max_blocks_per_seq * sizeof(int));
  cudaMemset(d_block_priority, 0,
             table.total_physical_blocks * sizeof(int));
  cudaMemset(d_block_weights, 0.0f,
             table.total_physical_blocks * sizeof(float));

  table.block_ref_counts = d_block_ref_counts;
  table.block_access_patterns = d_block_access_patterns;
  table.block_priority = d_block_priority;
  table.block_weights = d_block_weights;
  table.cache_size = table.total_physical_blocks / 4;

  // Defragment key cache
  tinyserve_launch_defragmentation(
      reinterpret_cast<half*>(key_cache.data_ptr()), &table,
      block_mapping.data_ptr<int>(), blocks_to_move.data_ptr<int>(),
      blocks_to_move.size(0), block_size, head_dim, stream);

  // Defragment value cache
  tinyserve_launch_defragmentation(
      reinterpret_cast<half*>(value_cache.data_ptr()), &table,
      block_mapping.data_ptr<int>(), blocks_to_move.data_ptr<int>(),
      blocks_to_move.size(0), block_size, head_dim, stream);

  cudaStreamSynchronize(stream);

  // Cleanup
  cudaFree(d_block_ref_counts);
  cudaFree(d_block_access_patterns);
  cudaFree(d_block_priority);
  cudaFree(d_block_weights);
}

// Continuous Block Allocation (Best-Fit)
void tinyserve_continuous_block_allocation(
    torch::Tensor& block_table,  // [num_seqs, max_blocks_per_seq] input/output
    torch::Tensor& seq_ids,     // [num_allocations]
    torch::Tensor& logical_block_ids,       // [num_allocations]
    torch::Tensor& num_consecutive_blocks,  // [num_allocations]
    torch::Tensor&
        allocated_block_ranges,  // [num_allocations, 2] output [start, end]
    int total_blocks, int block_size) {
  TORCH_CHECK(block_table.device().is_cuda(), "block_table must be on CUDA");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(block_table));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Create TinyServeBlockTable structure
  TinyServeBlockTable table;
  table.physical_block_ids = block_table.data_ptr<int>();
  table.max_blocks_per_seq = block_table.size(1);
  table.total_physical_blocks = total_blocks;
  table.block_size = block_size;

  // Allocate device memory for block management
  int* d_block_ref_counts;
  int* d_block_access_patterns;
  int* d_block_priority;
  float* d_block_weights;
  cudaMalloc(&d_block_ref_counts, total_blocks * sizeof(int));
  cudaMalloc(&d_block_access_patterns,
             block_table.size(0) * table.max_blocks_per_seq * sizeof(int));
  cudaMalloc(&d_block_priority, total_blocks * sizeof(int));
  cudaMalloc(&d_block_weights, total_blocks * sizeof(float));
  cudaMemset(d_block_ref_counts, 0, total_blocks * sizeof(int));
  cudaMemset(d_block_access_patterns, 0,
             block_table.size(0) * table.max_blocks_per_seq * sizeof(int));
  cudaMemset(d_block_priority, 0, total_blocks * sizeof(int));
  cudaMemset(d_block_weights, 0.0f, total_blocks * sizeof(float));

  table.block_ref_counts = d_block_ref_counts;
  table.block_access_patterns = d_block_access_patterns;
  table.block_priority = d_block_priority;
  table.block_weights = d_block_weights;
  table.cache_size = total_blocks / 4;

  // Launch continuous block allocation
  tinyserve_launch_continuous_block_allocation(
      &table, seq_ids.data_ptr<int>(), logical_block_ids.data_ptr<int>(),
      num_consecutive_blocks.data_ptr<int>(),
      allocated_block_ranges.data_ptr<int>(), seq_ids.size(0), stream);

  cudaStreamSynchronize(stream);

  // Cleanup
  cudaFree(d_block_ref_counts);
  cudaFree(d_block_access_patterns);
  cudaFree(d_block_priority);
  cudaFree(d_block_weights);
}
