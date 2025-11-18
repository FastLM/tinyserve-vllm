## TinyServe-vLLM: Optimized vLLM with Advanced Kernel Optimizations

**TinyServe-vLLM** is an enhanced version of vLLM with advanced CUDA kernel optimizations for improved memory management, fragmentation reduction, and query-aware cache selection.

### Key TinyServe Optimizations

- **Query-Aware Cache Selection**: Intelligent cache management based on query characteristics and access patterns
- **Fragmentation Reduction**: Advanced techniques to minimize memory fragmentation (40-60% reduction)
- **FlashAttention Integration**: Combined FlashAttention with PagedAttention for maximum efficiency
- **Best-Fit Allocation**: Optimized block allocation strategies to reduce wasted space by 20-30%
- **Memory Utilization**: >96% memory utilization (vs 60-80% traditional)

### TinyServe Kernel Features

- Fragmentation detection and analysis
- Fragmentation-aware block allocation
- Intelligent defragmentation
- Continuous block allocation (best-fit strategy)
- Dynamic workload balancing
- LRU cache management

### New Files Added by TinyServe-vLLM

We have added the following files to enhance vLLM with TinyServe optimizations:

#### Core Kernel Files
- **`csrc/vllm_kernels.cu`**: Complete CUDA kernel implementation with TinyServe optimizations
  - Fragmentation detection and analysis kernel
  - Fragmentation-aware block allocation kernel
  - Defragmentation kernel
  - Continuous block allocation (best-fit) kernel
  - FlashAttention with PagedAttention integration
  - Advanced block allocation with LRU cache
  - Intelligent memory compaction
  - Dynamic workload balancing

- **`csrc/tinyserve_kernels.h`**: Header file for TinyServe kernels
  - Complete API definitions for all TinyServe optimization kernels
  - Data structures for block table management
  - Function declarations for kernel launchers

- **`csrc/tinyserve_example.cu`**: Example implementation demonstrating TinyServe usage
  - Complete example showing how to use TinyServe kernels
  - Performance benchmarking utilities
  - Memory compaction examples

#### Key Features Implemented

1. **Query-Aware Cache Selection**
   - Query complexity analysis kernel
   - Cache selection strategy kernel
   - Adaptive cache management kernel
   - Integration with PagedAttention

2. **Fragmentation Reduction**
   - Fragmentation analysis: Detects memory fragmentation patterns
   - Fragmentation-aware allocation: Best-fit strategy to minimize fragmentation
   - Defragmentation: Moves blocks to consolidate free space
   - Continuous block allocation: Allocates consecutive blocks efficiently

3. **Memory Management Optimizations**
   - LRU cache management for block allocation
   - Access pattern learning and optimization
   - Intelligent memory compaction based on block weights
   - Dynamic workload balancing across GPU warps

For detailed information about TinyServe optimizations, see the [TinyServe Kernels Documentation](csrc/tinyserve_kernels.h).

---

## Citation

If you use TinyServe-vLLM for your research, please cite our paper:

```bibtex
@inproceedings{liu2025tinyserve,
  title={TinyServe: Query-Aware Cache Selection for Efficient LLM Serving},
  author={Liu, Dong and Yu, Yanxuan},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={12529--12537},
  year={2025}
}
```

---

## About

**TinyServe-vLLM** is an enhanced version of vLLM with advanced CUDA kernel optimizations for improved memory management, fragmentation reduction, and query-aware cache selection.

vLLM is a fast and easy-to-use library for LLM inference and serving, originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley.

---

TinyServe-vLLM is fast with:

- State-of-the-art serving throughput with **Query-Aware Cache Selection** for intelligent cache management
- Efficient management of attention key and value memory with [**PagedAttention**] enhanced by **Fragmentation Reduction** (40-60% reduction)
- **Best-Fit Block Allocation** strategies that reduce wasted space by 20-30%
- **Memory Utilization** exceeding 96% (vs 60-80% traditional methods)
- Optimized CUDA kernels with **FlashAttention integration** and **PagedAttention** fusion
- **Fragmentation-aware allocation** that minimizes memory fragmentation
- **Intelligent defragmentation** for continuous memory optimization
- **Dynamic workload balancing** across GPU warps for maximum utilization

TinyServe-vLLM is flexible and easy to use with:

- Seamless integration with vLLM's existing infrastructure and popular Hugging Face models
- High-throughput serving with **query-aware cache selection** that adapts to query characteristics
- **Fragmentation detection and analysis** for proactive memory management
- **Continuous block allocation** using best-fit strategy for optimal memory layout
- **LRU cache management** with access pattern learning for intelligent block placement
- Support for NVIDIA GPUs with optimized CUDA kernels
- **Adaptive cache management** that continuously improves cache selection over time
- **Memory compaction** based on block weights and access frequencies

TinyServe-vLLM seamlessly supports all models compatible with vLLM, including:

- Transformer-like LLMs (e.g., Llama)
- Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
- Embedding Models (e.g., E5-Mistral)
- Multi-modal LLMs (e.g., LLaVA)
---
