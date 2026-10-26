# TinyServe: Query-Aware Cache Selection for Efficient LLM Serving

TinyServe is a serving stack for small language models. During decode it scores KV pages with bounding-box metadata, keeps the newest pages, and runs attention only on that subset. The same selection runs as a C++ kernel and, when CUDA is present, as one fused GPU kernel.

## Architecture Overview

TinyServe has four pieces:

- **Query-Aware KV Retriever.** Each page stores the per-channel min and max of its keys. The score of page $j$ is an upper bound on the largest key dot product inside that page.
- **Page Selector.** A streaming bitonic top-$K$ picks pages. The newest causal pages are pinned so the token just written cannot be dropped.
- **Sparse Attention Executor.** The C++ kernel softmaxes the selected tokens. The CUDA kernel fuses scoring, top-$K$, asynchronous page loads, and online softmax into one launch.
- **Host Runtime.** A C API plus a Python decoder: exact prefill, query-aware decode, entropy early-exit, int8 or fp16 pages, and session placement across devices.

$$
r(q, \phi_j) = \sum_i q_i \cdot
\begin{cases}
M_{j,i} & q_i \ge 0 \\
m_{j,i} & q_i < 0
\end{cases}
$$

## Key Features

- Query-aware page selection from bounding-box metadata, with no change to model weights
- Fused CUDA kernel: page scoring, top-\(K\) selection, and online softmax in one pass
- Up to 3.4× decode speedup and over 2× KV memory savings in the paper's evaluation
- Plugin decode loop: entropy early-exit, token pruning, approximate attention, fp16 and int8 pages

## Memory

With cache length $L$, page size $S$, $K$ selected pages, reuse probability $\rho$, and $M$ bytes per token:

$$
\mathrm{Load} = 2M \left(\frac{L}{S} + \rho K S\right), \qquad \mathrm{fraction} = \frac{1}{S} + \rho \frac{K S}{L}.
$$

The page size $\sqrt{L/K}$ is the operating point stated in the paper. The page size that minimizes the fraction is $\sqrt{L/(\rho K)}$.

## Project Core Files Structure

```
TinyServe/
├── host/                    # C API, demo, Python serving runtime
├── selector/                # Bounding-box score and streaming top-K
├── attention/               # Dense and sparse decode
├── cache/                   # Incremental page bounds
├── kernels/                 # CUDA score + select + online softmax
├── include/                 # C++ headers
├── tests/                   # C++ kernel tests and Python runtime tests
└── docs/                    # Architecture and build notes
```

## Requirements

- Python 3.9+
- A C++17 compiler
- CMake 3.16+
- PyTorch 2.1+ for the Python runtime
- CUDA, optional, for `kernels/tinyserve_kernels.cu`

## Installation

```bash
pip install -r requirements.txt
mkdir build && cd build
cmake ..
cmake --build . -j
```

This creates:

- `libtinyserve.a`: C++ kernel library
- `tinyserve_demo`: One decode step over two pages
- `test_selector` and `test_attention`

If `nvcc` is on `PATH`, CMake also builds `libtinyserve_kernels.a`.

## Usage

### Run Tests

```bash
cd tests
make
make test
```

`make test` runs the C++ checks and the Python runtime tests. From the build directory, `ctest` runs the C++ checks only.

### Python

From the repository root:

```python
import torch
from host.python.tinyserve import TinyServeLM, generate, preset

model = TinyServeLM(preset("debug"))
prompt = torch.randint(0, model.config.vocab_size, (1, 16))
result = generate(model, prompt, max_new_tokens=8)
```

`result.lengths` is the valid length after entropy early-exit. `result.page_hit_rate` is the fraction of selected pages that were also selected on the previous step. Presets such as `gpt2-345m` set layer widths only. They are not pretrained checkpoints.

### C API

```c
#include "tinyserve.h"

tinyserve_bounding_box_score(query, k_min, k_max, dim, &score);
tinyserve_sparse_decode(query, &kv, k_pages, n_recent, scale, position, out, selected);
```

`./tinyserve_demo` prints the page scores, the pinned page, and the attention output for a fixed two-page cache.

## Technical Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) describes the page layout, the selector, and the fused kernel.
- [docs/BUILD.md](docs/BUILD.md) covers the CMake build and the optional CUDA library.

## Citation

```bibtex
@inproceedings{liu_tinyserve_2025,
  author = {Liu, Dong and Yu, Yanxuan},
  title = {TinyServe: Query-Aware Cache Selection for Efficient LLM Serving},
  year = {2025},
  isbn = {9798400720352},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3746027.3758181},
  doi = {10.1145/3746027.3758181},
  abstract = {Serving large language models (LLMs) efficiently remains challenging due to the high memory and latency overhead of key-value (KV) cache access during autoregressive decoding. We present TinyServe, a lightweight and extensible serving system for deploying tiny LLMs (e.g., TinyLLaMA, GPT2-345M) with support for structured KV sparsity, plugin-based token selection, and hardware-efficient attention kernels. Unlike prior simulation frameworks, TinyServe executes real-time decoding with configurable sparsity strategies and fine-grained instrumentation. To reduce decoding cost, we introduce a query-aware page selection mechanism that leverages bounding-box metadata to estimate attention relevance between the query and KV cache blocks. This enables selective KV loading with minimal overhead and no model modifications. Our fused CUDA kernel integrates page scoring, sparse memory access, and masked attention in a single pass. Experiments show that TinyServe achieves up to 3.4\times speedup and over 2\times memory savings with negligible accuracy drop. Additional analysis of cache reuse, page hit rate, and multi-GPU scaling confirms its practicality as an efficient system-level design for LLM training and inference research on resource-constrained hardware.},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
  pages = {12529--12537},
  numpages = {9},
  keywords = {cuda kernels, efficient inference, efficient training, kv cache, llm serving, runtime systems, sparsity},
  location = {Dublin, Ireland},
  series = {MM '25}
}
```

## About

Developed by [Dong Liu](https://github.com/NoakLiu/) and [Yanxuan Yu](https://github.com/zorinayu/) at [FastLM.ai](https://github.com/FastLM/)

