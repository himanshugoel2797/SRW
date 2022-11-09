# Perlmutter Build Instructions

```bash
module load python
conda create --name srw_gpu python=3.8 #Have not tested with newer versions of Python, although should work
conda activate srw_gpu
conda install numpy scikit-learn matplotlib

git clone https://github.com/himanshugoel2797/SRW.git
cd SRW
git checkout manualmemory_test #Switch to GPU supporting branch

#Environment variables I added for configuring hpc sdk location
export NVCOMPILERS=/opt/nvidia/hpc_sdk
export NVVERSION=22.5
export NVARCH=Linux_x86_64

MODE=cuda make
#just 'make' is fine too as MODE=cuda is already set in the root level Makefile
```

## Perlmutter GPU Slurm file:
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

## Rough benchmarks
- Propagation on 1 NVIDIA A100 (Perlmutter): ~210ms
- Propagation on 1 NVIDIA RTX 3090 (Windows): ~680ms
- Propagation on 1 NVIDIA RTX 3090 (Linux): ~450ms

Each propagation involves:
- 5 transmission elements
- 4 'full' free-space propagators between the elements
- 1 semi-analytical free-space propagator at the end

All running on GPU

# Logging into Perlmutter

Current simplest way, from Cori login node:
```bash
ssh perlmutter
```

'Proper' way:
```bash
ssh username@perlmutter-p1.nersc.gov
```