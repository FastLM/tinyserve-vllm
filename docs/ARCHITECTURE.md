# Architecture

TinyServe splits a KV cache into fixed-size pages. Each page keeps a bounding box of its keys: one minimum and one maximum per channel. At every decode step the current query scores those boxes, a top-$K$ selects pages, and attention runs on the tokens inside the selected pages.

## Page layout

For one KV head the cache is contiguous:

| Buffer | Shape |
| --- | --- |
| keys, values | `[num_pages, page_size, dim]` |
| `k_min`, `k_max` | `[num_pages, dim]` |
| `page_lens` | `[num_pages]` |

`page_lens[j]` is the number of valid tokens in page $j$. The last page is usually short. Padding inside a page is not part of the min, the max, or the softmax.

Appending a token updates one row of the box. The first token of a page replaces the box. Later tokens fold in with a min and a max. `reduce_bounds` recomputes a page from scratch and is what the tests compare against the incremental update.

## Score

$$
r(q, \phi_j) = \sum_i q_i \cdot \begin{cases} M_{j,i} & q_i \ge 0 \\ m_{j,i} & q_i < 0 \end{cases}
$$

If $q_i$ is positive the largest key channel maximizes the product. If $q_i$ is negative the smallest key channel does. The sum is therefore at least as large as $q \cdot k$ for every key in the page. Pages that start after the query position, or that contain no tokens, receive score $-\infty$.

## Selection

The C++ selector and the CUDA kernel use the same streaming top-$K$:

1. Walk pages in tiles of 128. The C++ tests also accept smaller power-of-two tiles.
2. Bitonic-sort each tile by score, breaking ties toward the smaller page id.
3. Merge the tile into a sorted top-$K$.
4. Pin the newest causal pages. If the budget is smaller than the pin set, keep the newest pages that fit.

Pinning is what makes $K = 1$ still attend to the token that was just written.

## Attention

`sparse_decode` gathers every valid token in the selected pages whose position is at most the query position, applies the attention scale, and writes the softmax-weighted sum of values. With a budget that covers every visible page, the output matches `dense_decode` on the same tokens.

The CUDA kernel in `kernels/tinyserve_kernels.cu` does not materialize the gathered KV. One block owns one query head. It keeps the top-$K$ in shared memory, copies selected pages with `cp.async` (double-buffered when the tile fits), and updates an online softmax. Supported head dimensions are 32, 64, and 128. Supported page sizes are 8, 16, 32, and 64. At most 256 pages are selected.

KV may be fp16 or int8. Bounds and the query stay fp32. An int8 page stores one scale per page. The scale is the max absolute value of the valid tokens divided by 127.

## Host runtime

`host/include/tinyserve.h` is the C API over the C++ kernels. The Python package in `host/python/tinyserve` is the serving loop:

- Prefill runs dense causal attention and writes pages plus bounds.
- Decode appends one token per layer and calls the sparse kernel.
- Entropy early-exit, token pruning, and approximate attention are plugins on that loop.
- `ServingScheduler` places a session on the least-loaded device and moves a small session only when the move shrinks the load gap.

Training uses the PyTorch implementation of the same sparse attention so gradients reach only the gathered pages. `page_importance` reads the value gradient of each page. The CUDA kernel is an inference path and is used when the extension is loaded, the cache is fp16 or int8, and the query is the last cached token.

## Memory fraction

With cache length $L$, page size $S$, $K$ selected pages, reuse probability $\rho$, and $M$ bytes per token:

$$
\mathrm{Load} = 2M \left(\frac{L}{S} + \rho K S\right), \qquad \mathrm{fraction} = \frac{1}{S} + \rho \frac{K S}{L}.
$$

The page size $\sqrt{L/K}$ is the operating point stated in the paper. The page size that minimizes the fraction is $\sqrt{L/(\rho K)}$.
