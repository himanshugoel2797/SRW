/************************************************************************//**
 * File: sroptelm_gpu.cu
 * Description: Optical element (general CUDA functions)
 * Project: Synchrotron Radiation Workshop
 * First release: 2023
 *
 * Copyright (C) Brookhaven National Laboratory
 * All Rights Reserved
 *
 * @author H.Goel
 * @version 1.0
 ***************************************************************************/

#ifdef _OFFLOAD_GPU
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include "auxgpu.h"

template<typename T> __global__ void Memset_kernel(T* p, T val, long long n)
{
    long long offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset < n) p[offset] = val;
}

void CAuxGPU::Memset_GPU(float* p, float val, long long n, long long streamIdx)
{
    const int bs = 256;
    dim3 blocks(n / bs + ((n & (bs - 1)) != 0));
    dim3 threads(bs);
    Memset_kernel<float> <<<blocks, threads, 0, (cudaStream_t)streamIdx >>> (p, val, n);
}

void CAuxGPU::Memset_GPU(double* p, double val, long long n, long long streamIdx)
{
    const int bs = 256;
    dim3 blocks(n / bs + ((n & (bs - 1)) != 0));
    dim3 threads(bs);
    Memset_kernel<double> <<<blocks, threads, 0, (cudaStream_t)streamIdx >>> (p, val, n);
}

#endif