# Build

## Dependencies

- CMake 3.16 or newer
- A C++17 compiler
- Python 3.9 or newer and PyTorch 2.1 or newer, for the runtime tests
- CUDA, optional. The C++ kernels build without it.

```bash
pip install -r requirements.txt
```

## CMake

From the repository root:

```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j
ctest --output-on-failure
./tinyserve_demo
```

`ctest` runs `test_selector` and `test_attention`. The demo prints scores for a two-page cache and exits with status 0 when the newest page is the one selected under a one-page budget.

## Tests without the CMake tree

```bash
cd tests
make
make test
```

`make test` also runs the Python suite with `PYTHONPATH=../host/python`.

## CUDA library

When `nvcc` is visible to CMake, the configure step enables CUDA and builds `libtinyserve_kernels.a` from `kernels/tinyserve_kernels.cu`. The public entry points are declared in `kernels/tinyserve_kernels.h`.

`kernels/torch_bindings.cpp` is the PyTorch wrapper around that launch. It is not part of the default CPU library because it needs libtorch. Build it with the PyTorch C++ extension against the same sources when you want `import tinyserve_cuda`. The Python decode path calls that module only if it imports, the cache is fp16 or int8, and the query is the last cached token. Otherwise it uses the PyTorch reference in `host/python/tinyserve/attention.py`.

## Python path

The package directory is `host/python`. From the repository root this import works without installing:

```python
from host.python.tinyserve import TinyServeLM, generate
```

Scripts that do `import tinyserve` need `host/python` on `PYTHONPATH`.
