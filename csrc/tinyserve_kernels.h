/*
 * TinyServe Optimized vLLM PagedAttention CUDA Kernels Header
 * Advanced Page Allocation and Memory Management for Large Language Models
 * 
 * This header file provides the interface for TinyServe's optimized
 * PagedAttention kernels with enhanced performance and memory efficiency.
 */

#ifndef TINYSERVE_KERNELS_H
#define TINYSERVE_KERNELS_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

// TinyServe Enhanced Data Structures
typedef struct {
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
} TinyServeBlockTable;

typedef struct {
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
} TinyServeAttentionMetadata;

typedef struct {
    int block_dim_x;
    int block_dim_y;
    int block_dim_z;
    int grid_dim_x;
    int grid_dim_y;
    int grid_dim_z;
    int shared_mem_bytes;
    bool use_tensor_cores;
    bool enable_kernel_fusion;
} TinyServeKernelConfig;

// TinyServe Optimized Kernel Functions

/**
 * Launch TinyServe FlashAttention with PagedAttention
 * Combines FlashAttention's memory efficiency with PagedAttention's block management
 * 
 * @param query Query matrix [batch, num_heads, head_dim]
 * @param key_blocks Key blocks [num_blocks, block_size, head_dim]
 * @param value_blocks Value blocks [num_blocks, block_size, head_dim]
 * @param output Output attention [batch, num_heads, head_dim]
 * @param block_table Block table mapping
 * @param seq_lens Sequence lengths
 * @param metadata Attention metadata
 * @param stream CUDA stream
 */
void tinyserve_launch_flash_paged_attention(
    const half* query,
    const half* key_blocks,
    const half* value_blocks,
    half* output,
    const int* block_table,
    const int* seq_lens,
    const TinyServeAttentionMetadata* metadata,
    cudaStream_t stream
);

/**
 * Launch advanced block allocation with LRU cache
 * Uses access patterns and LRU policy for optimal block placement
 * 
 * @param table Block table
 * @param seq_ids Sequence IDs
 * @param logical_block_ids Logical block IDs
 * @param allocated_blocks Output allocated blocks
 * @param access_frequencies Access frequencies
 * @param num_allocations Number of allocations
 * @param stream CUDA stream
 */
void tinyserve_launch_advanced_block_allocation(
    TinyServeBlockTable* table,
    const int* seq_ids,
    const int* logical_block_ids,
    int* allocated_blocks,
    const int* access_frequencies,
    int num_allocations,
    cudaStream_t stream
);

/**
 * Launch intelligent memory compaction
 * Compacts memory based on block weights and access patterns
 * 
 * @param blocks Memory blocks
 * @param old_to_new_mapping Mapping from old to new block indices
 * @param block_weights Block importance weights
 * @param num_blocks Number of blocks
 * @param block_size Block size
 * @param hidden_size Hidden dimension size
 * @param compaction_threshold Compaction threshold
 * @param stream CUDA stream
 */
void tinyserve_launch_intelligent_memory_compaction(
    half* blocks,
    const int* old_to_new_mapping,
    const int* block_weights,
    int num_blocks,
    int block_size,
    int hidden_size,
    int compaction_threshold,
    cudaStream_t stream
);

/**
 * Launch dynamic workload balancing
 * Balances workload across warps based on sequence lengths
 * 
 * @param seq_lens Sequence lengths
 * @param work_distribution Output work distribution
 * @param batch_size Batch size
 * @param num_warps_per_block Number of warps per block
 * @param stream CUDA stream
 */
void tinyserve_launch_dynamic_workload_balancing(
    const int* seq_lens,
    int* work_distribution,
    int batch_size,
    int num_warps_per_block,
    cudaStream_t stream
);

/**
 * Launch fragmentation analysis
 * Analyzes memory fragmentation and identifies free block runs
 * 
 * @param table Block table
 * @param fragmentation_scores Output fragmentation scores
 * @param free_block_runs Output free block run starts
 * @param free_block_run_lengths Output free block run lengths
 * @param num_free_runs Output number of free runs
 * @param total_blocks Total number of blocks
 * @param stream CUDA stream
 */
void tinyserve_launch_fragmentation_analysis(
    const TinyServeBlockTable* table,
    int* fragmentation_scores,
    int* free_block_runs,
    int* free_block_run_lengths,
    int* num_free_runs,
    int total_blocks,
    cudaStream_t stream
);

/**
 * Launch fragmentation-aware block allocation
 * Allocates blocks while minimizing fragmentation
 * 
 * @param table Block table
 * @param seq_ids Sequence IDs
 * @param logical_block_ids Logical block IDs
 * @param num_blocks_needed Number of consecutive blocks needed
 * @param allocated_blocks Output allocated block IDs
 * @param fragmentation_scores Fragmentation scores
 * @param free_block_runs Free block run starts
 * @param free_block_run_lengths Free block run lengths
 * @param num_allocations Number of allocations
 * @param num_free_runs Number of free runs
 * @param stream CUDA stream
 */
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
);

/**
 * Launch defragmentation
 * Moves blocks to reduce memory fragmentation
 * 
 * @param blocks Memory blocks
 * @param table Block table
 * @param block_mapping Block mapping (old -> new)
 * @param blocks_to_move Blocks to move
 * @param num_blocks_to_move Number of blocks to move
 * @param block_size Block size
 * @param hidden_size Hidden dimension size
 * @param stream CUDA stream
 */
void tinyserve_launch_defragmentation(
    half* blocks,
    TinyServeBlockTable* table,
    const int* block_mapping,
    const int* blocks_to_move,
    int num_blocks_to_move,
    int block_size,
    int hidden_size,
    cudaStream_t stream
);

/**
 * Launch continuous block allocation (best-fit)
 * Allocates consecutive blocks using best-fit strategy
 * 
 * @param table Block table
 * @param seq_ids Sequence IDs
 * @param logical_block_ids Logical block IDs
 * @param num_consecutive_blocks Number of consecutive blocks needed
 * @param allocated_block_ranges Output allocated block ranges [start, end]
 * @param num_allocations Number of allocations
 * @param stream CUDA stream
 */
void tinyserve_launch_continuous_block_allocation(
    TinyServeBlockTable* table,
    const int* seq_ids,
    const int* logical_block_ids,
    const int* num_consecutive_blocks,
    int* allocated_block_ranges,
    int num_allocations,
    cudaStream_t stream
);

// Original vLLM Kernel Functions (for compatibility)

/**
 * Launch fused reshape and block write
 */
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
);

/**
 * Launch PagedAttention main kernel
 */
void launch_paged_attention(
    const half* query,
    const half* key_blocks,
    const half* value_blocks,
    half* output,
    const int* block_table,
    const int* seq_lens,
    const TinyServeAttentionMetadata* metadata,
    cudaStream_t stream
);

/**
 * Launch fused block copy
 */
void launch_fused_block_copy(
    const half* src_blocks,
    half* dst_blocks,
    const int* copy_operations,
    int num_operations,
    int block_size,
    int hidden_size,
    cudaStream_t stream
);

/**
 * Launch block allocation
 */
void launch_block_allocation(
    TinyServeBlockTable* table,
    const int* seq_ids,
    const int* logical_block_ids,
    int* allocated_blocks,
    int num_allocations,
    cudaStream_t stream
);

/**
 * Launch block deallocation
 */
void launch_block_deallocation(
    TinyServeBlockTable* table,
    const int* seq_ids,
    const int* logical_block_ids,
    int num_deallocations,
    cudaStream_t stream
);

/**
 * Launch memory compaction
 */
void launch_memory_compaction(
    half* blocks,
    const int* old_to_new_mapping,
    int num_blocks,
    int block_size,
    int hidden_size,
    cudaStream_t stream
);

// TinyServe Utility Functions

/**
 * Initialize TinyServe block table
 * 
 * @param table Block table to initialize
 * @param max_blocks_per_seq Maximum blocks per sequence
 * @param total_physical_blocks Total physical blocks
 * @param block_size Block size
 * @param cache_size Cache size
 * @return CUDA error code
 */
cudaError_t tinyserve_init_block_table(
    TinyServeBlockTable* table,
    int max_blocks_per_seq,
    int total_physical_blocks,
    int block_size,
    int cache_size
);

/**
 * Destroy TinyServe block table
 * 
 * @param table Block table to destroy
 * @return CUDA error code
 */
cudaError_t tinyserve_destroy_block_table(TinyServeBlockTable* table);

/**
 * Get optimal kernel configuration
 * 
 * @param metadata Attention metadata
 * @param config Output kernel configuration
 * @return CUDA error code
 */
cudaError_t tinyserve_get_optimal_config(
    const TinyServeAttentionMetadata* metadata,
    TinyServeKernelConfig* config
);

/**
 * Benchmark TinyServe kernels
 * 
 * @param metadata Attention metadata
 * @param num_iterations Number of benchmark iterations
 * @param results Output benchmark results
 * @return CUDA error code
 */
cudaError_t tinyserve_benchmark_kernels(
    const TinyServeAttentionMetadata* metadata,
    int num_iterations,
    float* results
);

#ifdef __cplusplus
}
#endif

#endif // TINYSERVE_KERNELS_H
