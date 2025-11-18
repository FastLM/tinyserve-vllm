/*
 * TinyServe Optimized vLLM PagedAttention CUDA Kernels
 * Advanced Page Allocation and Memory Management for Large Language Models
 * 
 * This file contains the complete implementation of TinyServe's optimized
 * PagedAttention kernels with enhanced performance and memory efficiency.
 * 
 * Key Optimizations:
 * - FlashAttention integration
 * - Advanced memory coalescing
 * - Optimized block allocation strategies
 * - Enhanced kernel fusion
 * - Dynamic workload balancing
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <mma.h>
#include <stdio.h>
#include <algorithm>

// TinyServe Optimized Constants
#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define MAX_SEQ_LEN 131072
#define DEFAULT_BLOCK_SIZE 16
#define NUM_HEADS 32
#define HEAD_DIM 128
#define TINYSERVE_BLOCK_SIZE 64
#define SHARED_MEM_SIZE 49152  // 48KB shared memory
#define MAX_WARPS_PER_BLOCK 8
#define FLASH_ATTENTION_BLOCK_M 64
#define FLASH_ATTENTION_BLOCK_N 64

// TinyServe Enhanced Data Structures
struct TinyServeBlockTable {
    int* physical_block_ids;         // Physical block ID mapping
    int* logical_to_physical;        // Logical to physical mapping
    int* block_ref_counts;          // Block reference counts
    int* block_access_patterns;      // Access pattern optimization
    int* block_priority;            // Block priority for LRU
    int max_blocks_per_seq;          // Maximum blocks per sequence
    int total_physical_blocks;       // Total physical blocks
    int block_size;                  // Size of each block
    int cache_size;                  // Cache size for hot blocks
    float* block_weights;           // Block importance weights
};

struct TinyServeAttentionMetadata {
    int batch_size;
    int num_heads;
    int head_dim;
    int block_size;
    int max_seq_len;
    float scale;                     // 1.0f / sqrt(head_dim)
    bool use_flash_attention;        // Enable FlashAttention
    bool use_memory_optimization;    // Enable memory optimization
    int num_warps_per_block;         // Warps per block
    int shared_mem_size;            // Shared memory size
};

struct TinyServeKernelConfig {
    int block_dim_x;
    int block_dim_y;
    int block_dim_z;
    int grid_dim_x;
    int grid_dim_y;
    int grid_dim_z;
    int shared_mem_bytes;
    bool use_tensor_cores;
    bool enable_kernel_fusion;
};

// TinyServe Advanced Utility Functions
__device__ __forceinline__ float atomicMax(float* address, float val) {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ __forceinline__ float atomicAdd(float* address, float val) {
    return atomicAdd(address, val);
}

// TinyServe Memory Coalescing Optimization
__device__ __forceinline__ void coalesced_load(
    const half* src, half* dst, int size, int tid, int block_size
) {
    // Optimized memory coalescing for better bandwidth utilization
    for (int i = tid; i < size; i += block_size) {
        dst[i] = src[i];
    }
}

// TinyServe Warp-level Reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// TinyServe Block Access Pattern Optimization
__device__ __forceinline__ int get_optimal_block_id(
    TinyServeBlockTable* table, int seq_id, int logical_block_id
) {
    // Use access pattern to optimize block placement
    int access_pattern = table->block_access_patterns[seq_id * table->max_blocks_per_seq + logical_block_id];
    int priority = table->block_priority[seq_id * table->max_blocks_per_seq + logical_block_id];
    
    // Prefer blocks with higher access frequency and priority
    return (access_pattern > 0) ? logical_block_id : -1;
}

// TinyServe Dynamic Workload Balancing
__device__ __forceinline__ void balance_workload(
    int* work_distribution, int total_work, int num_workers, int worker_id
) {
    int base_work = total_work / num_workers;
    int extra_work = total_work % num_workers;
    
    int start_work = worker_id * base_work + min(worker_id, extra_work);
    int end_work = start_work + base_work + (worker_id < extra_work ? 1 : 0);
    
    work_distribution[worker_id * 2] = start_work;
    work_distribution[worker_id * 2 + 1] = end_work;
}

// Kernel 1: Fused Reshape and Block Write
__global__ void fused_reshape_and_block_write_kernel(
    const half* input_kv,            // Input KV cache [batch, seq_len, hidden_size]
    half* output_blocks,             // Output block storage [num_blocks, block_size, hidden_size]
    const int* block_table,          // Block table mapping
    const int* seq_lens,             // Sequence lengths
    const int batch_size,
    const int hidden_size,
    const int block_size,
    const int max_seq_len,
    const int num_blocks_per_seq
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * max_seq_len * hidden_size;
    
    if (tid >= total_elements) return;
    
    // Calculate position in sequence
    int seq_idx = tid / (max_seq_len * hidden_size);
    int pos_in_seq = (tid % (max_seq_len * hidden_size)) / hidden_size;
    int hidden_idx = tid % hidden_size;
    
    // Skip if position exceeds sequence length
    if (pos_in_seq >= seq_lens[seq_idx]) return;
    
    // Calculate block index
    int block_idx = pos_in_seq / block_size;
    int pos_in_block = pos_in_seq % block_size;
    
    // Get physical block ID
    int physical_block_id = block_table[seq_idx * num_blocks_per_seq + block_idx];
    
    // Calculate output address
    int output_idx = physical_block_id * block_size * hidden_size + 
                     pos_in_block * hidden_size + hidden_idx;
    
    // Perform write operation
    output_blocks[output_idx] = input_kv[tid];
}

// TinyServe Kernel 2: Optimized FlashAttention with PagedAttention
__global__ void tinyserve_flash_paged_attention_kernel(
    const half* query,               // Query matrix [batch, num_heads, head_dim]
    const half* key_blocks,          // Key blocks [num_blocks, block_size, head_dim]
    const half* value_blocks,        // Value blocks [num_blocks, block_size, head_dim]
    half* output,                    // Output attention [batch, num_heads, head_dim]
    const int* block_table,          // Block table
    const int* seq_lens,             // Sequence lengths
    const TinyServeAttentionMetadata* metadata
) {
    int batch_idx = blockIdx.y;
    int head_idx = blockIdx.z;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    if (batch_idx >= metadata->batch_size || head_idx >= metadata->num_heads) return;
    
    int seq_len = seq_lens[batch_idx];
    int num_blocks = (seq_len + metadata->block_size - 1) / metadata->block_size;
    
    // Shared memory for FlashAttention blocks
    extern __shared__ half shared_mem[];
    half* shared_key = shared_mem;
    half* shared_value = shared_mem + FLASH_ATTENTION_BLOCK_M * metadata->head_dim;
    half* shared_query = shared_mem + 2 * FLASH_ATTENTION_BLOCK_M * metadata->head_dim;
    
    // Initialize output
    for (int d = tid; d < metadata->head_dim; d += blockDim.x) {
        output[batch_idx * metadata->num_heads * metadata->head_dim + 
               head_idx * metadata->head_dim + d] = __float2half(0.0f);
    }
    __syncthreads();
    
    // FlashAttention-style tiling
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    
    // Load query block to shared memory
    for (int i = tid; i < FLASH_ATTENTION_BLOCK_M * metadata->head_dim; i += blockDim.x) {
        shared_query[i] = query[batch_idx * metadata->num_heads * metadata->head_dim + 
                               head_idx * metadata->head_dim + i];
    }
    __syncthreads();
    
    // Process blocks in FlashAttention style
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        int physical_block_id = block_table[batch_idx * metadata->max_seq_len / metadata->block_size + block_idx];
        
        // Coalesced load of key-value blocks
        coalesced_load(
            key_blocks + physical_block_id * metadata->block_size * metadata->head_dim,
            shared_key,
            metadata->block_size * metadata->head_dim,
            tid,
            blockDim.x
        );
        
        coalesced_load(
            value_blocks + physical_block_id * metadata->block_size * metadata->head_dim,
            shared_value,
            metadata->block_size * metadata->head_dim,
            tid,
            blockDim.x
        );
        __syncthreads();
        
        // Compute attention scores with warp-level optimization
        int block_start = block_idx * metadata->block_size;
        int block_end = min(block_start + metadata->block_size, seq_len);
        
        float local_max = -INFINITY;
        for (int k_pos = block_start; k_pos < block_end; k_pos++) {
            float score = 0.0f;
            for (int d = 0; d < metadata->head_dim; d++) {
                score += __half2float(shared_query[d]) *
                        __half2float(shared_key[(k_pos - block_start) * metadata->head_dim + d]);
            }
            score *= metadata->scale;
            local_max = fmaxf(local_max, score);
        }
        
        // Warp-level reduction for max
        local_max = warp_reduce_max(local_max);
        if (lane_id == 0) {
            max_score = fmaxf(max_score, local_max);
        }
        __syncthreads();
        
        // Compute softmax and accumulate
        float local_sum = 0.0f;
        for (int k_pos = block_start; k_pos < block_end; k_pos++) {
            float score = 0.0f;
            for (int d = 0; d < metadata->head_dim; d++) {
                score += __half2float(shared_query[d]) *
                        __half2float(shared_key[(k_pos - block_start) * metadata->head_dim + d]);
            }
            score = expf(score * metadata->scale - max_score);
            local_sum += score;
            
            // Accumulate to output
            for (int d = 0; d < metadata->head_dim; d++) {
                atomicAdd(&output[batch_idx * metadata->num_heads * metadata->head_dim + 
                                 head_idx * metadata->head_dim + d],
                         score * __half2float(shared_value[(k_pos - block_start) * metadata->head_dim + d]));
            }
        }
        
        // Warp-level reduction for sum
        local_sum = warp_reduce_sum(local_sum);
        if (lane_id == 0) {
            sum_exp += local_sum;
        }
        __syncthreads();
    }
    
    // Normalize output
    for (int d = tid; d < metadata->head_dim; d += blockDim.x) {
        float val = __half2float(output[batch_idx * metadata->num_heads * metadata->head_dim + 
                                       head_idx * metadata->head_dim + d]);
        output[batch_idx * metadata->num_heads * metadata->head_dim + 
               head_idx * metadata->head_dim + d] = __float2half(val / sum_exp);
    }
}

// Kernel 2: Enhanced PagedAttention Main Kernel
__global__ void paged_attention_kernel(
    const half* query,               // Query matrix [batch, num_heads, head_dim]
    const half* key_blocks,          // Key blocks [num_blocks, block_size, head_dim]
    const half* value_blocks,        // Value blocks [num_blocks, block_size, head_dim]
    half* output,                    // Output attention [batch, num_heads, head_dim]
    const int* block_table,          // Block table
    const int* seq_lens,             // Sequence lengths
    const AttentionMetadata* metadata
) {
    int batch_idx = blockIdx.y;
    int head_idx = blockIdx.z;
    int tid = threadIdx.x;
    
    if (batch_idx >= metadata->batch_size || head_idx >= metadata->num_heads) return;
    
    int seq_len = seq_lens[batch_idx];
    int num_blocks = (seq_len + metadata->block_size - 1) / metadata->block_size;
    
    // Shared memory for storing key-value blocks
    extern __shared__ half shared_mem[];
    half* shared_key = shared_mem;
    half* shared_value = shared_mem + metadata->block_size * metadata->head_dim;
    
    // Initialize output
    for (int d = tid; d < metadata->head_dim; d += blockDim.x) {
        output[batch_idx * metadata->num_heads * metadata->head_dim + 
               head_idx * metadata->head_dim + d] = __float2half(0.0f);
    }
    __syncthreads();
    
    // First pass: compute max attention scores
    float max_score = -INFINITY;
    
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        int physical_block_id = block_table[batch_idx * metadata->max_seq_len / metadata->block_size + block_idx];
        
        // Load key block to shared memory
        for (int i = tid; i < metadata->block_size * metadata->head_dim; i += blockDim.x) {
            shared_key[i] = key_blocks[physical_block_id * metadata->block_size * metadata->head_dim + i];
        }
        __syncthreads();
        
        // Compute attention scores for current block
        int block_start = block_idx * metadata->block_size;
        int block_end = min(block_start + metadata->block_size, seq_len);
        
        for (int k_pos = block_start; k_pos < block_end; k_pos++) {
            float score = 0.0f;
            for (int d = 0; d < metadata->head_dim; d++) {
                score += __half2float(query[batch_idx * metadata->num_heads * metadata->head_dim + 
                                           head_idx * metadata->head_dim + d]) *
                        __half2float(shared_key[(k_pos - block_start) * metadata->head_dim + d]);
            }
            score *= metadata->scale;
            max_score = fmaxf(max_score, score);
        }
        __syncthreads();
    }
    
    // Second pass: compute softmax and accumulate values
    float sum_exp = 0.0f;
    
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        int physical_block_id = block_table[batch_idx * metadata->max_seq_len / metadata->block_size + block_idx];
        
        // Load key-value blocks to shared memory
        for (int i = tid; i < metadata->block_size * metadata->head_dim; i += blockDim.x) {
            shared_key[i] = key_blocks[physical_block_id * metadata->block_size * metadata->head_dim + i];
            shared_value[i] = value_blocks[physical_block_id * metadata->block_size * metadata->head_dim + i];
        }
        __syncthreads();
        
        int block_start = block_idx * metadata->block_size;
        int block_end = min(block_start + metadata->block_size, seq_len);
        
        for (int k_pos = block_start; k_pos < block_end; k_pos++) {
            float score = 0.0f;
            for (int d = 0; d < metadata->head_dim; d++) {
                score += __half2float(query[batch_idx * metadata->num_heads * metadata->head_dim + 
                                           head_idx * metadata->head_dim + d]) *
                        __half2float(shared_key[(k_pos - block_start) * metadata->head_dim + d]);
            }
            score = expf(score * metadata->scale - max_score);
            sum_exp += score;
            
            // Accumulate to output
            for (int d = 0; d < metadata->head_dim; d++) {
                atomicAdd(&output[batch_idx * metadata->num_heads * metadata->head_dim + 
                                 head_idx * metadata->head_dim + d],
                         score * __half2float(shared_value[(k_pos - block_start) * metadata->head_dim + d]));
            }
        }
        __syncthreads();
    }
    
    // Normalize output
    for (int d = tid; d < metadata->head_dim; d += blockDim.x) {
        float val = __half2float(output[batch_idx * metadata->num_heads * metadata->head_dim + 
                                       head_idx * metadata->head_dim + d]);
        output[batch_idx * metadata->num_heads * metadata->head_dim + 
               head_idx * metadata->head_dim + d] = __float2half(val / sum_exp);
    }
}

// Kernel 3: Fused Block Copy
__global__ void fused_block_copy_kernel(
    const half* src_blocks,          // Source blocks
    half* dst_blocks,                // Destination blocks
    const int* copy_operations,      // Copy operations [src_block_id, dst_block_id, block_size]
    const int num_operations,
    const int block_size,
    const int hidden_size
) {
    int op_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (op_idx >= num_operations) return;
    
    int src_block_id = copy_operations[op_idx * 3];
    int dst_block_id = copy_operations[op_idx * 3 + 1];
    int copy_size = copy_operations[op_idx * 3 + 2];
    
    int total_elements = copy_size * hidden_size;
    
    // Parallel block data copy
    for (int i = tid; i < total_elements; i += blockDim.x) {
        dst_blocks[dst_block_id * block_size * hidden_size + i] = 
            src_blocks[src_block_id * block_size * hidden_size + i];
    }
}

// TinyServe Kernel 4: Advanced Block Allocation with LRU Cache
__global__ void tinyserve_advanced_block_allocation_kernel(
    TinyServeBlockTable* table,
    const int* seq_ids,
    const int* logical_block_ids,
    int* allocated_blocks,
    const int num_allocations,
    const int* access_frequencies
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_allocations) return;
    
    int seq_id = seq_ids[tid];
    int logical_block_id = logical_block_ids[tid];
    int access_freq = access_frequencies[tid];
    
    // Update access pattern
    atomicAdd(&table->block_access_patterns[seq_id * table->max_blocks_per_seq + logical_block_id], 1);
    
    // Find optimal physical block using LRU and access pattern
    int best_block = -1;
    int min_priority = INT_MAX;
    
    for (int i = 0; i < table->total_physical_blocks; i++) {
        if (table->block_ref_counts[i] == 0) {
            // Check if this block is in cache
            if (i < table->cache_size) {
                // Prefer cached blocks for frequently accessed data
                if (access_freq > 5) {
                    best_block = i;
                    break;
                }
            }
            
            // Use LRU policy
            if (table->block_priority[i] < min_priority) {
                min_priority = table->block_priority[i];
                best_block = i;
            }
        }
    }
    
    if (best_block != -1) {
        // Allocate block
        table->physical_block_ids[seq_id * table->max_blocks_per_seq + logical_block_id] = best_block;
        table->block_ref_counts[best_block] = 1;
        table->block_priority[best_block] = atomicAdd(&table->block_priority[best_block], 1);
        allocated_blocks[tid] = best_block;
    } else {
        // Allocation failed
        allocated_blocks[tid] = -1;
    }
}

// TinyServe Kernel 5: Intelligent Memory Compaction
__global__ void tinyserve_intelligent_memory_compaction_kernel(
    half* blocks,
    const int* old_to_new_mapping,
    const int* block_weights,
    const int num_blocks,
    const int block_size,
    const int hidden_size,
    const int compaction_threshold
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_blocks * block_size * hidden_size;
    
    if (tid >= total_elements) return;
    
    int block_idx = tid / (block_size * hidden_size);
    int element_in_block = tid % (block_size * hidden_size);
    
    // Only compact blocks with low weights (less frequently accessed)
    if (block_weights[block_idx] < compaction_threshold) {
        int new_block_idx = old_to_new_mapping[block_idx];
        if (new_block_idx != block_idx) {
            blocks[new_block_idx * block_size * hidden_size + element_in_block] = 
                blocks[block_idx * block_size * hidden_size + element_in_block];
        }
    }
}

// TinyServe Kernel 6: Dynamic Workload Balancing
__global__ void tinyserve_dynamic_workload_balancing_kernel(
    const int* seq_lens,
    int* work_distribution,
    const int batch_size,
    const int num_warps_per_block
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    
    if (warp_id >= num_warps_per_block) return;
    
    // Calculate total work
    int total_work = 0;
    for (int i = 0; i < batch_size; i++) {
        total_work += seq_lens[i];
    }
    
    // Balance workload across warps
    balance_workload(work_distribution, total_work, num_warps_per_block, warp_id);
}

// TinyServe Kernel 7: Fragmentation Detection and Analysis
__global__ void tinyserve_fragmentation_analysis_kernel(
    const TinyServeBlockTable* table,
    int* fragmentation_scores,      // Output fragmentation scores per region
    int* free_block_runs,          // Output: start index of free block runs
    int* free_block_run_lengths,   // Output: length of free block runs
    int* num_free_runs,            // Output: number of free runs found
    const int total_blocks
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= total_blocks) return;
    
    // Analyze fragmentation in chunks
    int chunk_size = 32;  // Analyze 32 blocks at a time
    int chunk_id = tid / chunk_size;
    int chunk_start = chunk_id * chunk_size;
    int chunk_end = min(chunk_start + chunk_size, total_blocks);
    
    if (tid % chunk_size != 0) return;  // Only first thread in chunk processes
    
    // Count free and used blocks in chunk
    int free_blocks = 0;
    int used_blocks = 0;
    int free_run_start = -1;
    int free_run_length = 0;
    int max_free_run = 0;
    
    for (int i = chunk_start; i < chunk_end; i++) {
        if (table->block_ref_counts[i] == 0) {
            free_blocks++;
            if (free_run_start == -1) {
                free_run_start = i;
                free_run_length = 1;
            } else {
                free_run_length++;
            }
        } else {
            used_blocks++;
            if (free_run_start != -1) {
                max_free_run = max(max_free_run, free_run_length);
                free_run_start = -1;
                free_run_length = 0;
            }
        }
    }
    
    // Calculate fragmentation score (higher = more fragmented)
    // Fragmentation = (number of free runs) / (total free blocks)
    float fragmentation = 0.0f;
    if (free_blocks > 0) {
        // More small free runs = higher fragmentation
        fragmentation = (float)used_blocks / (float)chunk_size;
    }
    
    // Store fragmentation score
    fragmentation_scores[chunk_id] = __float_as_int(fragmentation * 1000.0f);
    
    // Store largest free run in this chunk
    if (free_run_start != -1) {
        max_free_run = max(max_free_run, free_run_length);
        // Store the final free run
        int run_idx = atomicAdd(num_free_runs, 1);
        if (run_idx < total_blocks) {
            free_block_runs[run_idx] = free_run_start;
            free_block_run_lengths[run_idx] = free_run_length;
        }
    } else if (max_free_run > 0) {
        // Store the largest free run found during scanning
        int run_idx = atomicAdd(num_free_runs, 1);
        if (run_idx < total_blocks) {
            // Find the start position of the max free run
            int run_start = -1;
            int current_run = 0;
            for (int i = chunk_start; i < chunk_end; i++) {
                if (table->block_ref_counts[i] == 0) {
                    if (run_start == -1) run_start = i;
                    current_run++;
                    if (current_run == max_free_run) {
                        free_block_runs[run_idx] = run_start;
                        free_block_run_lengths[run_idx] = max_free_run;
                        break;
                    }
                } else {
                    run_start = -1;
                    current_run = 0;
                }
            }
        }
    }
}

// TinyServe Kernel 8: Fragmentation-Aware Block Allocation
__global__ void tinyserve_fragmentation_aware_allocation_kernel(
    TinyServeBlockTable* table,
    const int* seq_ids,
    const int* logical_block_ids,
    const int* num_blocks_needed,      // Number of consecutive blocks needed
    int* allocated_blocks,             // Output: allocated block IDs
    const int* fragmentation_scores,   // Fragmentation scores
    const int* free_block_runs,        // Free block run starts
    const int* free_block_run_lengths, // Free block run lengths
    const int num_allocations,
    const int num_free_runs
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_allocations) return;
    
    int seq_id = seq_ids[tid];
    int logical_block_id = logical_block_ids[tid];
    int blocks_needed = num_blocks_needed[tid];
    
    // Strategy: Find the best free run that can fit the required blocks
    int best_run_idx = -1;
    int best_run_length = 0;
    int best_run_start = -1;
    
    // First, try to find a free run that exactly fits or is slightly larger
    for (int i = 0; i < num_free_runs; i++) {
        int run_length = free_block_run_lengths[i];
        int run_start = free_block_runs[i];
        
        if (run_length >= blocks_needed) {
            // Check if all blocks in this run are actually free
            bool all_free = true;
            for (int j = 0; j < blocks_needed; j++) {
                if (table->block_ref_counts[run_start + j] != 0) {
                    all_free = false;
                    break;
                }
            }
            
            if (all_free) {
                // Prefer runs that are close to the required size (reduce fragmentation)
                int waste = run_length - blocks_needed;
                if (best_run_idx == -1 || waste < (best_run_length - blocks_needed)) {
                    best_run_idx = i;
                    best_run_length = run_length;
                    best_run_start = run_start;
                }
            }
        }
    }
    
    // If no suitable run found, try individual block allocation (fallback)
    if (best_run_idx == -1) {
        // Allocate blocks individually, trying to keep them contiguous
        int allocated_count = 0;
        int start_block = -1;
        
        for (int i = 0; i < table->total_physical_blocks && allocated_count < blocks_needed; i++) {
            if (table->block_ref_counts[i] == 0) {
                if (start_block == -1) {
                    start_block = i;
                }
                
                // Check if this block is contiguous with previous allocation
                if (allocated_count == 0 || i == start_block + allocated_count) {
                    allocated_blocks[tid * blocks_needed + allocated_count] = i;
                    table->block_ref_counts[i] = 1;
                    table->physical_block_ids[seq_id * table->max_blocks_per_seq + logical_block_id + allocated_count] = i;
                    allocated_count++;
                }
            }
        }
        
        if (allocated_count < blocks_needed) {
            // Allocation failed
            for (int i = 0; i < allocated_count; i++) {
                table->block_ref_counts[allocated_blocks[tid * blocks_needed + i]] = 0;
            }
            allocated_blocks[tid * blocks_needed] = -1;
        }
    } else {
        // Allocate from the best free run
        for (int i = 0; i < blocks_needed; i++) {
            int block_id = best_run_start + i;
            allocated_blocks[tid * blocks_needed + i] = block_id;
            table->block_ref_counts[block_id] = 1;
            table->physical_block_ids[seq_id * table->max_blocks_per_seq + logical_block_id + i] = block_id;
        }
    }
}

// TinyServe Kernel 9: Defragmentation Kernel
__global__ void tinyserve_defragmentation_kernel(
    half* blocks,                    // Memory blocks to defragment
    TinyServeBlockTable* table,      // Block table
    const int* block_mapping,        // Mapping: old_block_id -> new_block_id
    const int* blocks_to_move,      // List of blocks that need to be moved
    const int num_blocks_to_move,
    const int block_size,
    const int hidden_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_blocks_to_move * block_size * hidden_size;
    
    if (tid >= total_elements) return;
    
    int move_idx = tid / (block_size * hidden_size);
    int element_in_block = tid % (block_size * hidden_size);
    
    if (move_idx >= num_blocks_to_move) return;
    
    int old_block_id = blocks_to_move[move_idx];
    int new_block_id = block_mapping[old_block_id];
    
    if (old_block_id != new_block_id && new_block_id >= 0) {
        // Move block data
        int old_offset = old_block_id * block_size * hidden_size + element_in_block;
        int new_offset = new_block_id * block_size * hidden_size + element_in_block;
        
        blocks[new_offset] = blocks[old_offset];
        
        // Update block table if this is the first element
        if (element_in_block == 0) {
            // Find all sequences using this block and update their references
            for (int seq_id = 0; seq_id < table->max_blocks_per_seq; seq_id++) {
                for (int log_block = 0; log_block < table->max_blocks_per_seq; log_block++) {
                    int idx = seq_id * table->max_blocks_per_seq + log_block;
                    if (table->physical_block_ids[idx] == old_block_id) {
                        table->physical_block_ids[idx] = new_block_id;
                    }
                }
            }
        }
    }
}

// TinyServe Kernel 10: Continuous Block Allocation (Best-Fit)
__global__ void tinyserve_continuous_block_allocation_kernel(
    TinyServeBlockTable* table,
    const int* seq_ids,
    const int* logical_block_ids,
    const int* num_consecutive_blocks,  // Number of consecutive blocks needed
    int* allocated_block_ranges,        // Output: [start_block, end_block] pairs
    const int num_allocations
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_allocations) return;
    
    int seq_id = seq_ids[tid];
    int logical_block_id = logical_block_ids[tid];
    int num_blocks = num_consecutive_blocks[tid];
    
    // Best-fit strategy: find the smallest free run that can fit
    int best_start = -1;
    int best_length = INT_MAX;
    int current_run_start = -1;
    int current_run_length = 0;
    
    for (int i = 0; i < table->total_physical_blocks; i++) {
        if (table->block_ref_counts[i] == 0) {
            if (current_run_start == -1) {
                current_run_start = i;
                current_run_length = 1;
            } else {
                current_run_length++;
            }
        } else {
            // End of free run
            if (current_run_start != -1 && current_run_length >= num_blocks) {
                if (current_run_length < best_length) {
                    best_start = current_run_start;
                    best_length = current_run_length;
                }
            }
            current_run_start = -1;
            current_run_length = 0;
        }
    }
    
    // Check last run
    if (current_run_start != -1 && current_run_length >= num_blocks) {
        if (current_run_length < best_length) {
            best_start = current_run_start;
            best_length = current_run_length;
        }
    }
    
    // Allocate from best fit
    if (best_start != -1) {
        for (int i = 0; i < num_blocks; i++) {
            int block_id = best_start + i;
            table->block_ref_counts[block_id] = 1;
            table->physical_block_ids[seq_id * table->max_blocks_per_seq + logical_block_id + i] = block_id;
        }
        
        allocated_block_ranges[tid * 2] = best_start;
        allocated_block_ranges[tid * 2 + 1] = best_start + num_blocks - 1;
    } else {
        // Allocation failed
        allocated_block_ranges[tid * 2] = -1;
        allocated_block_ranges[tid * 2 + 1] = -1;
    }
}

// Kernel 4: Enhanced Block Allocation
__global__ void block_allocation_kernel(
    BlockTable* table,
    const int* seq_ids,
    const int* logical_block_ids,
    int* allocated_blocks,
    const int num_allocations
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_allocations) return;
    
    int seq_id = seq_ids[tid];
    int logical_block_id = logical_block_ids[tid];
    
    // Find free physical block
    for (int i = 0; i < table->total_physical_blocks; i++) {
        if (atomicCAS(&table->block_ref_counts[i], 0, 1) == 0) {
            // Allocate block
            table->physical_block_ids[seq_id * table->max_blocks_per_seq + logical_block_id] = i;
            allocated_blocks[tid] = i;
            return;
        }
    }
    
    // Allocation failed
    allocated_blocks[tid] = -1;
}

// Kernel 5: Block Deallocation
__global__ void block_deallocation_kernel(
    BlockTable* table,
    const int* seq_ids,
    const int* logical_block_ids,
    const int num_deallocations
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_deallocations) return;
    
    int seq_id = seq_ids[tid];
    int logical_block_id = logical_block_ids[tid];
    
    int physical_block_id = table->physical_block_ids[seq_id * table->max_blocks_per_seq + logical_block_id];
    atomicSub(&table->block_ref_counts[physical_block_id], 1);
}

// Kernel 6: Memory Compaction
__global__ void memory_compaction_kernel(
    half* blocks,
    const int* old_to_new_mapping,
    const int num_blocks,
    const int block_size,
    const int hidden_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_blocks * block_size * hidden_size;
    
    if (tid >= total_elements) return;
    
    int block_idx = tid / (block_size * hidden_size);
    int element_in_block = tid % (block_size * hidden_size);
    
    int new_block_idx = old_to_new_mapping[block_idx];
    if (new_block_idx != block_idx) {
        blocks[new_block_idx * block_size * hidden_size + element_in_block] = 
            blocks[block_idx * block_size * hidden_size + element_in_block];
    }
}

// TinyServe Host-side wrapper functions
extern "C" {

// TinyServe Optimized Kernel Launchers
void tinyserve_launch_flash_paged_attention(
    const half* query,
    const half* key_blocks,
    const half* value_blocks,
    half* output,
    const int* block_table,
    const int* seq_lens,
    const TinyServeAttentionMetadata* metadata,
    cudaStream_t stream
) {
    dim3 grid(1, metadata->batch_size, metadata->num_heads);
    int shared_mem_size = 3 * FLASH_ATTENTION_BLOCK_M * metadata->head_dim * sizeof(half);
    
    tinyserve_flash_paged_attention_kernel<<<grid, BLOCK_SIZE, shared_mem_size, stream>>>(
        query, key_blocks, value_blocks, output, block_table, seq_lens, metadata
    );
}

void tinyserve_launch_advanced_block_allocation(
    TinyServeBlockTable* table,
    const int* seq_ids,
    const int* logical_block_ids,
    int* allocated_blocks,
    const int* access_frequencies,
    int num_allocations,
    cudaStream_t stream
) {
    int num_blocks = (num_allocations + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    tinyserve_advanced_block_allocation_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        table, seq_ids, logical_block_ids, allocated_blocks, num_allocations, access_frequencies
    );
}

void tinyserve_launch_intelligent_memory_compaction(
    half* blocks,
    const int* old_to_new_mapping,
    const int* block_weights,
    int num_blocks,
    int block_size,
    int hidden_size,
    int compaction_threshold,
    cudaStream_t stream
) {
    int total_elements = num_blocks * block_size * hidden_size;
    int num_blocks_kernel = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    tinyserve_intelligent_memory_compaction_kernel<<<num_blocks_kernel, BLOCK_SIZE, 0, stream>>>(
        blocks, old_to_new_mapping, block_weights, num_blocks, block_size, hidden_size, compaction_threshold
    );
}

void tinyserve_launch_dynamic_workload_balancing(
    const int* seq_lens,
    int* work_distribution,
    int batch_size,
    int num_warps_per_block,
    cudaStream_t stream
) {
    int num_blocks = (num_warps_per_block + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    tinyserve_dynamic_workload_balancing_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        seq_lens, work_distribution, batch_size, num_warps_per_block
    );
}

// TinyServe Fragmentation Reduction Kernels
void tinyserve_launch_fragmentation_analysis(
    const TinyServeBlockTable* table,
    int* fragmentation_scores,
    int* free_block_runs,
    int* free_block_run_lengths,
    int* num_free_runs,
    int total_blocks,
    cudaStream_t stream
) {
    int num_blocks = (total_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    tinyserve_fragmentation_analysis_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        table, fragmentation_scores, free_block_runs, free_block_run_lengths,
        num_free_runs, total_blocks
    );
}

void tinyserve_launch_fragmentation_aware_allocation(
    TinyServeBlockTable* table,
    const int* seq_ids,
    const int* logical_block_ids,
    const int* num_blocks_needed,
    int* allocated_blocks,
    const int* fragmentation_scores,
    const int* free_block_runs,
    const int* free_block_run_lengths,
    int num_allocations,
    int num_free_runs,
    cudaStream_t stream
) {
    int num_blocks = (num_allocations + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    tinyserve_fragmentation_aware_allocation_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        table, seq_ids, logical_block_ids, num_blocks_needed, allocated_blocks,
        fragmentation_scores, free_block_runs, free_block_run_lengths,
        num_allocations, num_free_runs
    );
}

void tinyserve_launch_defragmentation(
    half* blocks,
    TinyServeBlockTable* table,
    const int* block_mapping,
    const int* blocks_to_move,
    int num_blocks_to_move,
    int block_size,
    int hidden_size,
    cudaStream_t stream
) {
    int total_elements = num_blocks_to_move * block_size * hidden_size;
    int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    tinyserve_defragmentation_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        blocks, table, block_mapping, blocks_to_move, num_blocks_to_move,
        block_size, hidden_size
    );
}

void tinyserve_launch_continuous_block_allocation(
    TinyServeBlockTable* table,
    const int* seq_ids,
    const int* logical_block_ids,
    const int* num_consecutive_blocks,
    int* allocated_block_ranges,
    int num_allocations,
    cudaStream_t stream
) {
    int num_blocks = (num_allocations + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    tinyserve_continuous_block_allocation_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        table, seq_ids, logical_block_ids, num_consecutive_blocks,
        allocated_block_ranges, num_allocations
    );
}

// Original wrapper functions
void launch_fused_reshape_and_block_write(
    const half* input_kv,
    half* output_blocks,
    const int* block_table,
    const int* seq_lens,
    int batch_size,
    int hidden_size,
    int block_size,
    int max_seq_len,
    int num_blocks_per_seq,
    cudaStream_t stream
) {
    int total_elements = batch_size * max_seq_len * hidden_size;
    int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    fused_reshape_and_block_write_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        input_kv, output_blocks, block_table, seq_lens,
        batch_size, hidden_size, block_size, max_seq_len, num_blocks_per_seq
    );
}

void launch_paged_attention(
    const half* query,
    const half* key_blocks,
    const half* value_blocks,
    half* output,
    const int* block_table,
    const int* seq_lens,
    const AttentionMetadata* metadata,
    cudaStream_t stream
) {
    dim3 grid(1, metadata->batch_size, metadata->num_heads);
    int shared_mem_size = 2 * metadata->block_size * metadata->head_dim * sizeof(half);
    
    paged_attention_kernel<<<grid, BLOCK_SIZE, shared_mem_size, stream>>>(
        query, key_blocks, value_blocks, output, block_table, seq_lens, metadata
    );
}

void launch_fused_block_copy(
    const half* src_blocks,
    half* dst_blocks,
    const int* copy_operations,
    int num_operations,
    int block_size,
    int hidden_size,
    cudaStream_t stream
) {
    fused_block_copy_kernel<<<num_operations, BLOCK_SIZE, 0, stream>>>(
        src_blocks, dst_blocks, copy_operations, num_operations, block_size, hidden_size
    );
}

void launch_block_allocation(
    BlockTable* table,
    const int* seq_ids,
    const int* logical_block_ids,
    int* allocated_blocks,
    int num_allocations,
    cudaStream_t stream
) {
    int num_blocks = (num_allocations + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    block_allocation_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        table, seq_ids, logical_block_ids, allocated_blocks, num_allocations
    );
}

void launch_block_deallocation(
    BlockTable* table,
    const int* seq_ids,
    const int* logical_block_ids,
    int num_deallocations,
    cudaStream_t stream
) {
    int num_blocks = (num_deallocations + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    block_deallocation_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        table, seq_ids, logical_block_ids, num_deallocations
    );
}

void launch_memory_compaction(
    half* blocks,
    const int* old_to_new_mapping,
    int num_blocks,
    int block_size,
    int hidden_size,
    cudaStream_t stream
) {
    int total_elements = num_blocks * block_size * hidden_size;
    int num_blocks_kernel = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    memory_compaction_kernel<<<num_blocks_kernel, BLOCK_SIZE, 0, stream>>>(
        blocks, old_to_new_mapping, num_blocks, block_size, hidden_size
    );
}

} // extern "C"
