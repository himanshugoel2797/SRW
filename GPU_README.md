## General Conda-specific Build Instructions

Create `environment.yml`:

```yml
name: py39_gcc11
channels:
    - conda-forge
dependencies:
    - python==3.9.0
    - gcc_linux-64==11.3.0
    - gxx_linux-64==11.3.0
    - numpy
    - scipy
    - matplotlib
    - openmpi
    - openmp
    - mpi4py
    - scikit-learn
```

Then:

```bash
cp /nsls2/users/hgoel/srw_gpu/environment.yml .
conda env create -f "environment.yml"
conda activate py39_gcc11

git clone https://github.com/himanshugoel2797/SRW.git
cd SRW
git checkout reorg_gpu #Switch to GPU supporting branch

MODE=cuda make fftw
MODE=cuda make
```

The GPU enabled SRW will then work as normal (from `env/python/srwpy`).

### Making a script use GPU

Change:

```py
srwl.PropagElecField(wfr, opBL)
```

to:

```py
srwl.PropagElecField(wfr, opBL, None, 1)
```

to select the first GPU etc. Set to 0 to explicitly disable GPU.

## Perlmutter Build Instructions

```bash
module load python
conda env create -f "environment.yml"
conda activate py39_gcc11

git clone https://github.com/himanshugoel2797/SRW.git
cd SRW
git checkout reorg_gpu #Switch to GPU supporting branch

#Environment variables I added for configuring hpc sdk location
export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda 
export CUDA_MATHLIBS_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/math_libs 
MODE=cuda make fftw
MODE=cuda make

```

### Perlmutter GPU Slurm file

```bash
#!/bin/bash
#SBATCH -A m2173
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:05:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"

module load python
conda activate srw_gpu
srun python ./SRWLIB_Example18.py
```

### Rough benchmarks

- Propagation on 1 NVIDIA A100 (Perlmutter): ~210ms
- Propagation on 1 NVIDIA RTX 3090 (Windows): ~680ms
- Propagation on 1 NVIDIA RTX 3090 (Linux): ~450ms

Each propagation involves:

- 5 transmission elements
- 4 'full' free-space propagators between the elements
- 1 semi-analytical free-space propagator at the end

All running on GPU

### Logging into Perlmutter

Current simplest way, from Cori login node:

```bash
ssh perlmutter
```

'Proper' way:

```bash
ssh username@perlmutter-p1.nersc.gov
```
