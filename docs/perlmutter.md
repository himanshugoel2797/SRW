# Building SRW on NERSC Perlmutter

This guide builds the `cuda-workflow-support` branch of SRW on
[NERSC Perlmutter](https://docs.nersc.gov/systems/perlmutter/) with a Conda
environment stored in `$SCRATCH` and Cray-MPICH-linked `mpi4py`. All commands
run from the repository root (you never `cd` out of `$SCRATCH/SRW`). Both the
Make and CMake build paths are covered.

## 1. Clone SRW into `$SCRATCH`

```bash
cd $SCRATCH
git clone https://github.com/himanshugoel2797/SRW.git
cd SRW
git checkout cuda-workflow-support
```

## 2. Load modules

```bash
module load python          # provides conda + mamba
module load PrgEnv-gnu      # GNU host compilers + Cray `cc`/`CC` wrappers
module load cudatoolkit     # nvcc, cudart, cufft, CUDA compat libs (not loaded by default)
```

Any time you start a fresh shell — new SSH session, new `salloc`, etc. —
re-run these three `module load` commands. Perlmutter interactive/batch
sessions start with a clean module set.

If `nvcc` complains about an unsupported host `gcc`, load `cpe-cuda` or a
matching `gcc/<version>` module.

## 3. Create the Conda environment in `$SCRATCH` by cloning `nersc-mpi4py`

`nersc-mpi4py` ships `mpi4py` linked against Cray MPICH (Slingshot-11,
`srun`-aware). Don't `pip install mpi4py` or
`conda install -c conda-forge mpi4py` — those bundle their own MPICH and fall
back to TCP.

```bash
export CONDA_PKGS_DIRS=$SCRATCH/conda/pkgs        # keep package cache off $HOME
mkdir -p $CONDA_PKGS_DIRS

conda create --prefix $SCRATCH/srw_build_env --clone nersc-mpi4py -y
conda activate $SCRATCH/srw_build_env

# `mamba` ships inside the cloned env — much faster than `conda install`:
mamba install -y numpy scipy h5py matplotlib pillow scikit-learn setuptools wheel cmake
```

Don't override the Python version when cloning — that would rebuild mpi4py
against mismatched libs.

## 4. Point the build at CUDA

`cudatoolkit` only sets `CUDA_HOME`; SRW's Makefile reads `CUDA_PATH` and
`CUDA_MATHLIBS_PATH`. On Perlmutter, cuFFT / cuBLAS live in a separate
`math_libs` tree:

```bash
export CUDA_PATH=$CUDA_HOME
export CUDA_MATHLIBS_PATH=$CUDA_HOME/../../math_libs/
```

Point at `math_libs/` (the directory containing `lib64/`), not
`math_libs/lib64/` — the Makefile appends `/lib64` itself.

## 5. (Optional) Grab an interactive compute node for faster parallel compilation

Perlmutter login nodes do have A100s, so you can build and smoke-test there,
but login nodes are shared and SRW's CUDA compile is heavy. An interactive
compute node gives you dedicated cores:

```bash
salloc -N 1 -C gpu -q interactive -t 01:00:00 -A <your_account> --gpus=4

# Re-load modules and re-export CUDA paths — sessions start clean:
module load python PrgEnv-gnu cudatoolkit
conda activate $SCRATCH/srw_build_env
export CUDA_PATH=$CUDA_HOME
export CUDA_MATHLIBS_PATH=$CUDA_HOME/../../math_libs/
cd $SCRATCH/SRW
```

## 6. Build

Two paths are supported; pick whichever you prefer. Both produce
`env/python/srwpy/srwlpy.so`.

### 6a. Make (from the repo root)

The root `Makefile` orchestrates FFTW, the C++ core, and the Python binding.

```bash
# CPU build (default, MODE=0):
make all                  # first build: also builds bundled FFTW3
make                      # subsequent rebuilds — core + pylib only, skips FFTW

# Switching modes always requires a clean:
make clean
MODE=cuda make all        # CUDA build; full build including FFTW
MODE=cuda make            # subsequent rebuilds
```

### 6b. CMake (from the repo root)

The root `CMakeLists.txt` auto-downloads and builds FFTW3 if it isn't found. A
`POST_BUILD` step copies `srwlpy.so` into `env/python/srwpy/`, so you still
never leave the repo root.

```bash
# CPU build:
cmake -B build \
    -DUSE_CUDA=OFF \
    -DBUILD_CLIENT_PYTHON=ON \
    -DPython_EXECUTABLE=$(which python)
cmake --build build -j

# CUDA build (A100 = sm_80):
cmake -B build \
    -DUSE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_CUDA_COMPILER=$CUDA_PATH/bin/nvcc \
    -DBUILD_CLIENT_PYTHON=ON \
    -DPython_EXECUTABLE=$(which python)
cmake --build build -j
```

Use `rm -rf build` before switching between CPU and CUDA configurations.

### 6c. Editable install via pip (CMake under the hood, still from root)

`env/python/setup.py` is a `CMakeExtension` wrapper that reads `MODE` and maps
it to the correct CMake flags for you.

```bash
pip install -e env/python                # CPU
MODE=cuda pip install -e env/python      # CUDA
```

## 7. Verify the build

```bash
# Import / CLI smoke test:
python -c "import srwpy, srwpy.srwlpy; print('srwpy ok')"
srw-viewer --help

# Verify mpi4py is using Cray MPICH (must run under srun):
srun -n 2 python -c "from mpi4py import MPI; print(MPI.Get_library_version())"
# Expect output containing "CRAY MPICH".
```

## 8. Running SRW with MPI

Always launch with `srun` on Perlmutter — never `mpirun` / `mpiexec`:

```bash
# Single-rank example:
python env/python/srwpy/examples/SRWLIB_Example01.py

# Multi-electron / parallel example across 16 ranks, 2 cores per rank:
srun -n 16 -c 2 python env/python/srwpy/examples/SRWLIB_Example12.py
```

## Troubleshooting

- **`cannot find -lcufft` or `-lcublas`** — `CUDA_MATHLIBS_PATH` is wrong. It
  must be `$CUDA_HOME/../../math_libs/`, pointing at the directory that
  *contains* `lib64/`.
- **`nvcc: gcc version not supported`** — `module load cpe-cuda` or load a
  matching `gcc/<version>`.
- **Switching build modes** — always `make clean` (Make) or `rm -rf build`
  (CMake) between CPU and CUDA builds.
- **mpi4py falling back to TCP / poor multi-node scaling** — you're using a
  conda-forge or PyPI mpi4py. Recreate the env by cloning `nersc-mpi4py`
  (step 3).
- **New shell loses everything** — re-run the three `module load` lines,
  `conda activate`, and the two `CUDA_PATH` / `CUDA_MATHLIBS_PATH` exports.
  They don't persist across `ssh` or `salloc`.
- **Over-quota on `$HOME`** — confirm `CONDA_PKGS_DIRS=$SCRATCH/conda/pkgs`
  and your env prefix is under `$SCRATCH`.
