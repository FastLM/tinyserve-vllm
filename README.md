# TinyServe-vLLM

Enhanced vLLM with advanced CUDA kernel optimizations for memory management, fragmentation reduction, and query-aware cache selection.

## Key Optimizations

| Optimization | Improvement | Description |
|-------------|-------------|-------------|
| Query-Aware Cache Selection | 30-50% throughput | Intelligent cache management based on query characteristics |
| Fragmentation Reduction | 40-60% reduction | Advanced techniques to minimize memory fragmentation |
| Best-Fit Allocation | 20-30% space saved | Optimized block allocation strategies |
| Memory Utilization | >96% (vs 60-80%) | Efficient memory usage with continuous block allocation |
| FlashAttention Integration | 2-3x speedup | Combined FlashAttention with PagedAttention |

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

## Implementation

| Component | File | Description |
|-----------|------|-------------|
| Kernels | `csrc/vllm_kernels.cu` | CUDA kernel implementation |
| API | `csrc/tinyserve_kernels.h` | Kernel API definitions |
| Cache | `csrc/tinyserve_cache_kernels.cu` | Cache optimization kernels |
| Bindings | `csrc/torch_bindings.cpp` | PyTorch bindings |

| Feature | Status | Description |
|---------|--------|-------------|
| Fragmentation Analysis | ✅ | Detects and quantifies memory fragmentation |
| Fragmentation-Aware Allocation | ✅ | Best-fit strategy to minimize fragmentation |
| Defragmentation | ✅ | Moves blocks to consolidate free space |
| Continuous Allocation | ✅ | Allocates consecutive blocks efficiently |
| Workload Balancing | ✅ | Dynamic balancing across GPU warps |
| LRU Cache | ✅ | Least Recently Used cache management |

## Performance

| Metric | Baseline | TinyServe-vLLM | Improvement |
|--------|----------|----------------|-------------|
| Memory Utilization | 60-80% | >96% | +20-36% |
| Fragmentation | High | Low | -40-60% |
| Throughput | 1x | 1.3-1.5x | +30-50% |
| Allocation Efficiency | 70-80% | 90-95% | +20-25% |

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

Developed by Dong Liu and Yanxuan Yu at FastLM.ai
