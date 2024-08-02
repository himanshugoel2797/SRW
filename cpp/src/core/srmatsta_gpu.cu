/************************************************************************//**
 * File: srmatsta_gpu.cu
 * Description: Basic statistical characteristics of intensity distributions (CUDA implementation)
 * Project: Synchrotron Radiation Workshop
 * First release: 2024
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
#include "cooperative_groups.h"
#include "cooperative_groups/reduce.h"
#include "cooperative_groups/memcpy_async.h"


#include <stdio.h>
#include <iostream>
#include <chrono>
#include "srmatsta.h"


namespace cg = cooperative_groups;
const int PerThreadSum = 16; //Number of values a single thread accumulates

template<class T>
__global__ void SumVector_FixedStride_Kernel(T* data, long long start, long long end, double multiplier, double* sum)
{
    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta); //Split the thread block into warps

    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx = idx * PerThreadSum + start;
    double sum_tmp = 0.;
    if (idx <= end)
    {
        for (int i = 0; i < PerThreadSum; i++)
        {
            long long pos = (i + idx);
            if (pos > end) break;

            sum_tmp += data[pos];
        }
    }

    sum_tmp = cg::reduce(tile, sum_tmp, cg::plus<double>());
    if (tile.thread_rank() == 0)
        atomicAdd(sum, sum_tmp * multiplier);
}

template<class T>
__global__ void IntegrateOverX_Kernel(T* data, int ixStart, int ixEnd, double xStep, int Nx, int Ny, double* AuxArrIntOverX)
{
    int iy = blockIdx.x * blockDim.x + threadIdx.x;
    int ix = blockIdx.y * blockDim.y + threadIdx.y;
    ix = ix * PerThreadSum + ixStart;
    if (ix > ixEnd) return;
    int ixFin = min(ix + PerThreadSum - 1, ixEnd);
    double sum = 0.;
    if (iy < Ny)
    {
        for (; ix <= ixFin; ix++)
        {
            sum += data[iy * Nx + ix];
        }
        atomicAdd(&AuxArrIntOverX[iy], sum * xStep);
    }
}

template<class T>
__global__ void IntegrateOverY_Kernel(T* data, int iyStart, int iyEnd, double yStep, int Nx, double* AuxArrIntOverY)
{
    cg::thread_block cta = cg::this_thread_block();

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    iy = iy * PerThreadSum + iyStart;
    if (iy > iyEnd) return;
    int iyFin = min(iy + PerThreadSum - 1, iyEnd);
    double sum = 0.;


    if (ix < Nx)
    {
        for (int i = iy; i <= iyFin; i++)
        {
            sum += data[i * Nx + ix];
        }
        atomicAdd(&AuxArrIntOverY[ix], sum * yStep);
    }
}

template<class T>
int IntegrateOverX_GPU_base(T* p0, long long ixStart, long long ixEnd, double xStep, long long Nx, long long Ny, double* AuxArrIntOverX, TGPUUsageArg* pGPU)
{
    const int bs = 128;
    dim3 threads(1, bs);
    dim3 nblocks(Ny, 1);

    long long LenArr = ixEnd - ixStart + 1;
    if (LenArr > bs * PerThreadSum)
    {
        nblocks.y = (LenArr / (bs * PerThreadSum));
        if (LenArr % (bs * PerThreadSum) > 0) nblocks.y++;
    }

    p0 = (T*)CAuxGPU::ToDevice(pGPU, p0, Nx * Ny * sizeof(T));
    AuxArrIntOverX = (double*)CAuxGPU::ToDevice(pGPU, AuxArrIntOverX, Ny * sizeof(double), true, false, 2);

    CAuxGPU::EnsureDeviceMemoryReady(pGPU, p0);
    CAuxGPU::EnsureDeviceMemoryReady(pGPU, AuxArrIntOverX);

    IntegrateOverX_Kernel<T><<<nblocks, threads>>>(p0, (int)ixStart, (int)ixEnd, xStep, (int)Nx, (int)Ny, AuxArrIntOverX);

    CAuxGPU::MarkUpdated(pGPU, AuxArrIntOverX, true, false);
    CAuxGPU::ToHostAndFree(pGPU, AuxArrIntOverX, Ny * sizeof(double));
    return 0;
}

int srTAuxMatStat::IntegrateOverX_GPU(float* p0, long long ixStart, long long ixEnd, double xStep, long long Nx, long long Ny, double* AuxArrIntOverX, TGPUUsageArg* pGPU)
{
    return IntegrateOverX_GPU_base<float>(p0, ixStart, ixEnd, xStep, Nx, Ny, AuxArrIntOverX, pGPU);
}

int srTAuxMatStat::IntegrateOverX_GPU(double* p0, long long ixStart, long long ixEnd, double xStep, long long Nx, long long Ny, double* AuxArrIntOverX, TGPUUsageArg* pGPU)
{
    return IntegrateOverX_GPU_base<double>(p0, ixStart, ixEnd, xStep, Nx, Ny, AuxArrIntOverX, pGPU);
}

template <class T>
int IntegrateOverY_GPU_base(T* p0, long long iyStart, long long iyEnd, double yStep, long long Nx, double* AuxArrIntOverY, TGPUUsageArg* pGPU)
{
    const int bs = 128;
    dim3 threads(bs, 1);
    dim3 nblocks(Nx, iyEnd - iyStart + 1);

    if (Nx < bs)
    {
        threads.x = Nx;
        nblocks.x = 1;

        if (threads.x % 32 != 0) //Ensure that number of threads is always a multiple of the warp size (32)
            threads.x += 32 - (threads.x % 32);
    }
    else
    {
        nblocks.x = (Nx / bs);
        if (Nx % bs > 0) nblocks.x++;
    }

    p0 = (T*)CAuxGPU::ToDevice(pGPU, p0, Nx * (iyEnd+1) * sizeof(T));
    AuxArrIntOverY = (double*)CAuxGPU::ToDevice(pGPU, AuxArrIntOverY, Nx * sizeof(double), true, false, 2);

    CAuxGPU::EnsureDeviceMemoryReady(pGPU, p0);
    CAuxGPU::EnsureDeviceMemoryReady(pGPU, AuxArrIntOverY);

    IntegrateOverY_Kernel<T><<<nblocks, threads>>>(p0, (int)iyStart, (int)iyEnd, yStep, (int)Nx, AuxArrIntOverY);

    CAuxGPU::MarkUpdated(pGPU, AuxArrIntOverY, true, false);
    CAuxGPU::ToHostAndFree(pGPU, AuxArrIntOverY, Nx * sizeof(double));
    return 0;
}

int srTAuxMatStat::IntegrateOverY_GPU(float* p0, long long iyStart, long long iyEnd, double yStep, long long Nx, double* AuxArrIntOverY, TGPUUsageArg* pGPU)
{
    return IntegrateOverY_GPU_base<float>(p0, iyStart, iyEnd, yStep, Nx, AuxArrIntOverY, pGPU);
}

int srTAuxMatStat::IntegrateOverY_GPU(double* p0, long long iyStart, long long iyEnd, double yStep, long long Nx, double* AuxArrIntOverY, TGPUUsageArg* pGPU)
{
    return IntegrateOverY_GPU_base<double>(p0, iyStart, iyEnd, yStep, Nx, AuxArrIntOverY, pGPU);
}

template <class T>
int IntegrateSimple_GPU_base(T* p0, long long LenArr, double Multiplier, double* OutVal, TGPUUsageArg* pGPU)
{
    int bs = 256;
    int nblocks = 1;
    if (LenArr > bs * PerThreadSum)
    {
        nblocks = (LenArr / (bs * PerThreadSum));
        if (LenArr % (bs * PerThreadSum) > 0) nblocks++;
    }
    else
    {
        bs = LenArr / PerThreadSum;
        if (LenArr % PerThreadSum > 0) bs++;
        if (bs % 32 != 0) //Ensure that number of threads is always a multiple of the warp size (32)
            bs += 32 - (bs % 32);

        nblocks = 1;
    }

    p0 = (T*)CAuxGPU::ToDevice(pGPU, p0, LenArr * sizeof(T));
    OutVal = (double*)CAuxGPU::ToDevice(pGPU, OutVal, sizeof(double));

    CAuxGPU::EnsureDeviceMemoryReady(pGPU, p0);
    CAuxGPU::EnsureDeviceMemoryReady(pGPU, OutVal);

    SumVector_FixedStride_Kernel<T><<<nblocks, bs>>>(p0, 0LL, LenArr - 1, Multiplier, OutVal);

    CAuxGPU::MarkUpdated(pGPU, OutVal, true, false);
    OutVal = (double*)CAuxGPU::ToHostAndFree(pGPU, OutVal, sizeof(double));
    return 0;
}

int srTAuxMatStat::IntegrateSimple_GPU(float* p0, long long LenArr, double Multiplier, double* OutVal, TGPUUsageArg* pGPU)
{
    return IntegrateSimple_GPU_base<float>(p0, LenArr, Multiplier, OutVal, pGPU);
}

int srTAuxMatStat::IntegrateSimple_GPU(double* p0, long long LenArr, double Multiplier, double* OutVal, TGPUUsageArg* pGPU)
{
    return IntegrateSimple_GPU_base<double>(p0, LenArr, Multiplier, OutVal, pGPU);
}

#endif