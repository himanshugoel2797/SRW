# Plan: GPU/CPU Installation Options for SRW

## Current State

- **Package name**: `srwpy` on PyPI
- **Build system**: CMake + setuptools (no pyproject.toml yet)
- **CUDA support**: 10 `.cu` kernel files exist but are NOT wired into the CMake build — only the legacy Makefile supports `MODE=cuda`
- **CI/CD**: GitHub Actions builds CPU-only wheels for Linux/macOS/Windows
- **No conda-forge recipe** exists yet

## Strategy Overview

The core challenge: you want **GPU as default** but **CPU as a fallback**, while handling CUDA DLLs/SOs for the GPU build. There are several approaches, each with trade-offs.

---

## Approach A: Split Packages (`srwpy` + `srwpy-cpu`)

**This is the recommended approach.** It follows the pattern used by PyTorch (`torch` / `torch-cpu`).

### How it works

| Package | Contains | When to use |
|---------|----------|-------------|
| `srwpy` (default) | GPU-enabled binary + bundled CUDA runtime libs | `pip install srwpy` — works on GPU machines |
| `srwpy-cpu` | CPU-only binary, no CUDA deps | `pip install srwpy-cpu` — works everywhere |

Both packages provide the same Python API (`import srwpy`). They conflict with each other (only one installed at a time).

### Implementation steps

#### 1. Add CUDA support to CMake (`CMakeLists.txt` + `cpp/cmake/CMakeLists.txt`)

- Add `option(USE_CUDA "Activate CUDA GPU build" OFF)` to root `CMakeLists.txt`
- When `USE_CUDA=ON`:
  - Enable CUDA language: `enable_language(CUDA)`
  - Add the 10 `.cu` files as sources to appropriate targets
  - Add definitions: `-D_OFFLOAD_GPU -DUSE_CUDA -D_FFTW3`
  - Link `cudart_static`, `cudadevrt`, `cufft`
  - Add `auxgpu` object library from `cpp/src/ext/auxgpu/`
- Use FFTW3 for CUDA builds (same as default, not OpenMP's FFTW2)

#### 2. Handle CUDA DLLs/SOs

For wheel distribution, CUDA runtime libraries must be bundled:

- **Option A (simpler)**: Depend on `nvidia-cuda-runtime-cu12`, `nvidia-cufft-cu12` PyPI packages (NVIDIA publishes these). This avoids bundling DLLs entirely.
- **Option B (self-contained)**: Use a `post_build` step to copy CUDA shared libraries into the wheel. Add them to `package_data` and set `RPATH`/`PATH` accordingly.

**Recommendation**: Option A for Linux (NVIDIA's PyPI packages work well there), Option B for Windows (where NVIDIA PyPI packages are less reliable). On macOS, CUDA is not supported.

#### 3. Modify `env/python/setup.py`

```python
# Detect GPU vs CPU build mode
use_cuda = os.environ.get('SRW_BUILD_CUDA', '0') == '1'

# Package name changes based on variant
package_name = 'srwpy' if use_cuda else 'srwpy-cpu'

# Base requirements
install_requires = [...]

# Add CUDA runtime deps for GPU build (Linux)
if use_cuda and sys.platform == 'linux':
    install_requires += [
        'nvidia-cuda-runtime-cu12',
        'nvidia-cufft-cu12',
    ]

setup(
    name=package_name,
    # ... rest stays the same
)
```

In `CMakeBuild.build_extension()`, pass `-DUSE_CUDA=ON` when `SRW_BUILD_CUDA=1`.

#### 4. Update CI/CD (`.github/workflows/pypi_publish.yml`)

Add a matrix dimension for CUDA:

```yaml
matrix:
  host-os: ["ubuntu-latest", "windows-latest"]
  python-version: ["3.9", "3.10", "3.11", "3.12"]
  cuda: ["on", "off"]
  exclude:
    - host-os: "macos-latest"
      cuda: "on"  # No CUDA on macOS
```

For CUDA builds:
- Install CUDA toolkit in CI (use `Jimver/cuda-toolkit` action)
- Set `SRW_BUILD_CUDA=1`
- Build wheels with CUDA-specific platform tags

#### 5. conda-forge recipe

Create `conda-forge/meta.yaml`:

```yaml
package:
  name: srwpy
  version: "4.1.0"

source:
  url: https://github.com/ochubar/SRW/archive/refs/tags/v4.1.0.tar.gz

build:
  number: 0

requirements:
  build:
    - cmake >=3.12
    - {{ compiler('c') }}
    - {{ compiler('cxx') }}
  host:
    - python
    - fftw
  run:
    - python
    - numpy
    - scipy
    - matplotlib
    - h5py
    - pillow
    - scikit-learn

# GPU variant via conda build variant
# In conda_build_config.yaml:
# cuda_compiler_version:
#   - None    # CPU
#   - 12.0    # GPU
```

conda-forge has built-in CUDA variant support via `cuda_compiler_version`. The GPU variant automatically gets `cudatoolkit` as a dependency and proper CUDA compiler setup.

---

## Approach B: Single Package with Extras (simpler but limited)

```
pip install srwpy          # CPU version (default)
pip install srwpy[gpu]     # GPU version
```

### How this would work

- Ship **only CPU binaries** in the default wheel
- The `[gpu]` extra installs a separate `srwpy-gpu-backend` package containing the CUDA-compiled `srwlpy.so`
- At runtime, `srwlib.py` tries to import the GPU backend first, falls back to CPU

### Downsides

- **Cannot ship two different compiled binaries in one package** — extras can only add Python dependencies, not swap out compiled extensions
- Requires a second package anyway (`srwpy-gpu-backend`), making it effectively the same as Approach A but with more indirection
- **GPU is NOT the default** — this conflicts with your preference

---

## Approach C: Single Package, Runtime Detection

```
pip install srwpy    # Ships BOTH CPU and GPU binaries
```

### How this would work

- Build two versions of `srwlpy`: `srwlpy_cpu.so` and `srwlpy_gpu.so`
- At import time, detect GPU availability and load the right one
- User can force CPU via `SRW_USE_CPU=1` env var

### Downsides

- **Doubles the wheel size** (GPU binary + bundled CUDA libs + CPU binary)
- Complex to maintain two binaries in one package
- CUDA DLLs always shipped even for CPU-only users

---

## Recommendation

**Go with Approach A (split packages)** because:

1. GPU is the default (`pip install srwpy` gets GPU)
2. CPU users explicitly opt in (`pip install srwpy-cpu`)
3. No wasted disk space — you only get the binaries you need
4. CUDA DLLs only ship in the GPU wheel
5. Follows established patterns (PyTorch, TensorFlow)
6. conda-forge has native CUDA variant support that maps cleanly to this

### Runtime CPU fallback within the GPU package

Even in the GPU package, you likely want graceful degradation. In `srwlib.py`:

```python
import os

_use_gpu = True

try:
    from . import srwlpy as srwl
    # Check if GPU is actually available at runtime
    if hasattr(srwl, 'gpu_available') and not srwl.gpu_available():
        _use_gpu = False
except ImportError:
    import srwlpy as srwl
    _use_gpu = False

if os.environ.get('SRW_USE_CPU', '0') == '1':
    _use_gpu = False
```

This way the GPU package still works on machines without a GPU — it just runs on CPU.

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `CMakeLists.txt` | Modify | Add `USE_CUDA` option |
| `cpp/cmake/CMakeLists.txt` | Modify | Add CUDA sources, auxgpu lib, CUDA linking |
| `env/python/setup.py` | Modify | CUDA build flag, package name switching, CUDA deps |
| `env/python/srwpy/srwlib.py` | Modify | Runtime GPU detection + CPU fallback |
| `.github/workflows/pypi_publish.yml` | Modify | Add CUDA matrix, CUDA toolkit setup |
| `conda/meta.yaml` | Create | conda-forge recipe with CUDA variant |
| `conda/conda_build_config.yaml` | Create | CUDA variant config |
