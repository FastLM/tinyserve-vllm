# TinyServe-vLLM

Enhanced vLLM with advanced CUDA kernel optimizations for memory management, fragmentation reduction, and query-aware cache selection.

<!-- ## Key Optimizations

| Optimization | Improvement | Description |
|-------------|-------------|-------------|
| Query-Aware Cache Selection | 30-50% throughput | Intelligent cache management based on query characteristics |
| Fragmentation Reduction | 40-60% reduction | Advanced techniques to minimize memory fragmentation |
| Best-Fit Allocation | 20-30% space saved | Optimized block allocation strategies |
| Memory Utilization | >96% (vs 60-80%) | Efficient memory usage with continuous block allocation |
| FlashAttention Integration | 2-3x speedup | Combined FlashAttention with PagedAttention | -->

## Core Algorithms

### Fragmentation Analysis

The fragmentation metric combines spatial and temporal fragmentation:

$$F_{total} = \alpha \cdot F_{spatial} + \beta \cdot F_{temporal} + \gamma \cdot F_{access}$$

where spatial fragmentation is:

$$F_{spatial} = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{r_i - \bar{r}}{\bar{r}} \right)^2 \cdot \exp\left(-\frac{d_i}{\sigma^2}\right)$$

temporal fragmentation considers access patterns:

$$F_{temporal} = \sum_{t=1}^{T} w_t \cdot \left( \frac{\sum_{j=1}^{M} |A_{t,j} - A_{t-1,j}|}{M \cdot \bar{A}} \right)$$

and access-based fragmentation:

$$F_{access} = -\sum_{k=1}^{K} p_k \log_2(p_k) \cdot \frac{\text{Var}(L_k)}{\bar{L}^2}$$

where:
- $r_i$: size of $i$-th free block run, $\bar{r} = \frac{1}{N}\sum_{i=1}^{N} r_i$
- $d_i$: distance from optimal location, $\sigma$: spatial decay parameter
- $A_{t,j}$: access frequency of block $j$ at time $t$, $w_t$: temporal weights
- $p_k$: probability of accessing cache level $k$, $L_k$: latency distribution
- $\alpha, \beta, \gamma$: weighting coefficients

### Query-Aware Cache Selection

Multi-objective optimization for cache selection:

$$\arg\max_{c \in \mathcal{C}} \left[ \lambda_1 \cdot U(c|q) - \lambda_2 \cdot C(c) + \lambda_3 \cdot R(c) \right]$$

where utility function:

$$U(c|q) = \frac{\exp(\mathbf{w}^T \phi(q, c))}{\sum_{c' \in \mathcal{C}} \exp(\mathbf{w}^T \phi(q, c'))}$$

with feature vector:

$$\phi(q, c) = \left[ Q_{complexity}(q), A_{frequency}(c), S_{similarity}(q, c), L_{locality}(c) \right]^T$$

cost function:

$$C(c) = \alpha_1 \cdot \text{Size}(c) + \alpha_2 \cdot \text{Load}(c) + \alpha_3 \cdot \text{Latency}(c)$$

and reward function:

$$R(c) = \sum_{i=1}^{n} \gamma^{i-1} \cdot r_i(c) \cdot \mathbb{I}[\text{hit}_i]$$

where:
- $q$: query, $c$: cache candidate, $\mathcal{C}$: candidate set
- $\mathbf{w}$: learnable weights, $\lambda_i, \alpha_i$: trade-off parameters
- $\gamma$: discount factor, $r_i$: reward at step $i$

### Best-Fit Block Allocation

Multi-constraint optimization for block allocation:

$$\min_{b \in \mathcal{B}} \left\lbrace \sum_{i=1}^{m} w_i \cdot f_i(b) + \lambda \cdot \|\mathbf{g}(b)\|_2^2 \right\rbrace$$

subject to constraints:

$$\begin{aligned}
f_1(b) &= F_{frag}(b) = \frac{\text{Var}(\text{free blocks})}{\bar{r}^2} \\
f_2(b) &= D_{locality}(b) = \sum_{j \in \text{neighbors}} \frac{1}{d(b, j)^2} \\
f_3(b) &= S_{size}(b) = \left| \frac{\text{requested} - \text{allocated}}{\text{requested}} \right| \\
f_4(b) &= T_{access}(b) = \sum_{t} \exp(-\alpha t) \cdot \text{accessCount}_{t}(b)
\end{aligned}$$

with gradient penalty:

$$\|\mathbf{g}(b)\|_2^2 = \sum_{i=1}^{n} \left( \frac{\partial f_i}{\partial b_i} \right)^2$$

where:
- $b$: block allocation, $\mathcal{B}$: feasible allocations
- $w_i$: feature weights, $\lambda$: regularization coefficient
- $d(b, j)$: distance between blocks, $\alpha$: temporal decay

## Integration Guide

### Modified Files in vLLM

- `csrc/tinyserve_cache_kernels.cu`
- `csrc/tinyserve_kernels.h`
- `csrc/cache.h`
- `csrc/cache_kernels.cu`
- `csrc/torch_bindings.cpp`

### Step-by-Step Integration

1. **Copy TinyServe files to vLLM:**
   ```bash
   cd <vllm_root>
   cp csrc/tinyserve_cache_kernels.cu csrc/
   cp csrc/tinyserve_kernels.h csrc/
   ```

2. **Apply modifications to vLLM files:**
   - **`csrc/cache.h`**: Add the content from `patches/cache.h.patch` to the end of the file
   - **`csrc/cache_kernels.cu`**: Add the content from `patches/cache_kernels.cu.patch` to the end of the file
   - **`csrc/torch_bindings.cpp`**: Add the content from `patches/torch_bindings.cpp.patch` inside the `TORCH_LIBRARY` block (after `cp_gather_indexer_k_quant_cache` registration)

3. **Rebuild vLLM:**
   ```bash
   cd <vllm_root>
   pip install -e . --no-build-isolation
   ```

**Note**: The patch files contain only the modifications needed. Simply copy the content from each patch file to the corresponding location in the vLLM source code.

## Running Experiments

### Prerequisites

- CUDA 11.8+ and compatible GPU
- Python 3.8+
- PyTorch 2.0+
- vLLM installed with TinyServe modifications

### Basic Usage

After integration, TinyServe optimizations are automatically enabled. The cache operations will use TinyServe's optimized kernels when available.

### Benchmarking

To compare performance with baseline vLLM:

```bash
# Run baseline vLLM
python -m vllm.entrypoints.api_server \
    --model <model_name> \
    --tensor-parallel-size 1

# Run TinyServe-enhanced vLLM (same command, optimizations are automatic)
python -m vllm.entrypoints.api_server \
    --model <model_name> \
    --tensor-parallel-size 1
```

### Monitoring Cache Performance

TinyServe provides enhanced cache metrics. Monitor memory utilization and fragmentation through vLLM's existing monitoring tools. The optimizations work transparently with vLLM's cache management system.

### Experimental Setup

For reproducible experiments:

1. **Environment Setup:**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   pip install -e . --no-build-isolation
   ```

2. **Run Benchmarks:**
   ```bash
   # Throughput benchmark
   python benchmarks/benchmark_throughput.py --model <model_name>
   
   # Latency benchmark
   python benchmarks/benchmark_latency.py --model <model_name>
   ```

3. **Compare Results:**
   - Memory utilization: Check GPU memory usage with `nvidia-smi`
   - Throughput: Compare requests/second
   - Latency: Compare P50/P99 latencies

<!-- ## Performance

| Metric | Baseline | TinyServe-vLLM | Improvement |
|--------|----------|----------------|-------------|
| Memory Utilization | 60-80% | >96% | +20-36% |
| Fragmentation | High | Low | -40-60% |
| Throughput | 1x | 1.3-1.5x | +30-50% |
| Allocation Efficiency | 70-80% | 90-95% | +20-25% | -->

## Citation

```bibtex
@inproceedings{liu2025tinyserve,
  title={TinyServe: Query-Aware Cache Selection for Efficient LLM Serving},
  author={Liu, Dong and Yu, Yanxuan},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={12529--12537},
  year={2025}
}
```

## About

Developed by [Dong Liu](https://github.com/NoakLiu/) and [Yanxuan Yu](https://github.com/zorinayu/) at [FastLM.ai](https://github.com/FastLM/)
