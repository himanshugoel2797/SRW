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
#include "cooperative_groups/scan.h"
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
__global__ void IntegrateOverX_Kernel(T* data, int* ixBounds, double xStep, int Nx, int Ny, double* AuxArrIntOverX)
{
    int iy = blockIdx.x * blockDim.x + threadIdx.x;
    int ix = blockIdx.y * blockDim.y + threadIdx.y;
    int ixStart = ixBounds[0];
    int ixEnd = ixBounds[1];
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
__global__ void IntegrateOverY_Kernel(T* data, int* iyBounds, double yStep, int Nx, int Ny, double* AuxArrIntOverY)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iyStart = iyBounds[0];
    int iyEnd = iyBounds[1];
    iy = iy * PerThreadSum + iyStart;
    if (iy > Ny) return;
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


template<class T, int KernelMode, bool multiWarp = true>
__global__ void PrefixSum_Kernel(T* data, int* bounds, T* sum_l, T* sum_r, T* residual_sum_l = NULL, T* residual_sum_r = NULL, double RelPowLevel = 0, double* IntegratedIntens = NULL, int* bounds_l = NULL, int* bounds_r = NULL, int* final_bounds = NULL)
{
    //Compute a parallel prefix sum of the array using warp shuffles and store the result in sum
    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta); //Split the thread block into war
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long r_idx = idx + (31 - 2 * tile.thread_rank());
    int start = bounds[0];
    int end = bounds[1];
    long len = end - start + 1;
    
    T leftLimit = IntegratedIntens[0]*(1. - RelPowLevel)*0.25;
    T rightLimit = leftLimit;

    if (KernelMode == 0)
    {
        T value = (idx < len) ? data[start + idx] : 0;
        // Perform the prefix sum within the warp
        T left_sum = cg::inclusive_scan(tile, value, cg::plus<T>());
        
        value = tile.shfl(value, 31 - tile.thread_rank());
        T right_sum = cg::inclusive_scan(tile, value, cg::plus<T>());
    
        if (multiWarp) 
        {
            // Initialize shared memory for prefix sums within the block
            __shared__ T shared_data[33][2];
            if (tile.meta_group_rank() == 0) 
            {
                shared_data[tile.thread_rank()][0] = 0;
                shared_data[tile.thread_rank()][1] = 0;
            }
            
            // Synchronize threads within the warp
            cta.sync();
            
            // Store the results in shared memory
            if (tile.thread_rank() == 31) 
            {
                shared_data[tile.meta_group_rank() + 1][0] = left_sum;
                shared_data[tile.meta_group_rank()][1] = right_sum;
            }
            
            // Synchronize threads within the warp
            cta.sync();
            
            // Prefix sum the shared data across the meta group
            if(tile.meta_group_rank() == 0) 
            {
                T l_temp = shared_data[tile.thread_rank()][0];
                T r_temp = shared_data[31 - tile.thread_rank()][1];
                
                l_temp = cg::inclusive_scan(tile, l_temp, cg::plus<T>());
                r_temp = cg::inclusive_scan(tile, r_temp, cg::plus<T>());
                
                // Store the results in shared memory
                shared_data[tile.thread_rank()][0] = l_temp;
                shared_data[31 - tile.thread_rank()][1] = r_temp;
            }
    
            // Synchronize threads within the warp
            cta.sync();
    
            // Pull the sum for the current warp from shared memory
            left_sum += shared_data[tile.meta_group_rank()][0];
            right_sum += shared_data[tile.meta_group_rank() + 1][1];
    
            // Synchronize threads within the warp
            cta.sync();
        }
    
        // Store the block sum
        if (sum_l != NULL && idx < len) sum_l[idx] = left_sum;
        if (sum_r != NULL && r_idx < len) sum_r[r_idx] = right_sum;
        if (residual_sum_l != NULL && threadIdx.x == blockDim.x - 1) residual_sum_l[blockIdx.x + 1] = left_sum;
        if (residual_sum_r != NULL && threadIdx.x == 31 && blockIdx.x > 0) residual_sum_r[blockIdx.x - 1] = right_sum;
    }
    else if (KernelMode == 1)
    {
        T value = (idx < len) ? data[start + idx] : 0;
        //Read the prefix sum of the other thread blocks
        T left_sum = residual_sum_l[blockIdx.x];
        T right_sum = residual_sum_r[blockIdx.x];

        if (idx < len)
        {
            float left_sum_tmp = left_sum + sum_l[idx];
            float right_sum_tmp = right_sum + sum_r[idx];

            //TODO track index closest to leftLimit and rightLimit respectively, via reduction, first within the block, then across blocks
            T left_sum_min = abs(left_sum_tmp - leftLimit);
            T right_sum_min = abs(right_sum_tmp - rightLimit);
            //printf("[%d] left_sum_min: %f, right_sum_min: %f\n", idx, left_sum_min, right_sum_min);

            int l_idx = idx;
            int r_idx = idx;
            for (int i = 1; i < 32; i *= 2)
            {
                T left_sum_min_r = tile.shfl_down(left_sum_min, i);
                T right_sum_min_r = tile.shfl_down(right_sum_min, i);
                T l_idx_tmp = tile.shfl_down(l_idx, i);
                T r_idx_tmp = tile.shfl_down(r_idx, i);

                if (left_sum_min_r < left_sum_min)
                {
                    left_sum_min = left_sum_min_r;
                    l_idx = l_idx_tmp;
                }
                if (right_sum_min_r < right_sum_min)
                {
                    right_sum_min = right_sum_min_r;
                    r_idx = r_idx_tmp;
                }
            }

            //TODO Reduce within the block
            __shared__ T shared_data[32][2];
            __shared__ int shared_idx[32][2];
            if (tile.meta_group_rank() == 0) 
            {
                shared_data[tile.thread_rank()][0] = INFINITY;
                shared_data[tile.thread_rank()][1] = INFINITY;
                shared_idx[tile.thread_rank()][0] = INT_MAX;
                shared_idx[tile.thread_rank()][1] = INT_MAX;
            }
            
            // Synchronize threads within the warp
            cta.sync();
            
            // Store the results in shared memory
            if (tile.thread_rank() == 0) 
            {
                shared_data[tile.meta_group_rank()][0] = left_sum_min;
                shared_data[tile.meta_group_rank()][1] = right_sum_min;
                shared_idx[tile.meta_group_rank()][0] = l_idx;
                shared_idx[tile.meta_group_rank()][1] = r_idx;

                //printf("[%d, %d] [%d]=%f [%d]=%f\n", tile.meta_group_rank(), tile.thread_rank(), l_idx, left_sum_min, r_idx, right_sum_min);
            }
            
            // Synchronize threads within the warp
            cta.sync();
            
            // Prefix sum the shared data across the meta group
            if(tile.meta_group_rank() == 0) 
            {
                T l_temp = shared_data[tile.thread_rank()][0];
                T r_temp = shared_data[tile.thread_rank()][1];
                int l_idx_temp = shared_idx[tile.thread_rank()][0];
                int r_idx_temp = shared_idx[tile.thread_rank()][1];

                for (int i = 1; i < 32; i *= 2)
                {
                    T _l_temp = tile.shfl_down(l_temp, i);
                    T _r_temp = tile.shfl_down(r_temp, i);
                    int _l_idx_temp = tile.shfl_down(l_idx_temp, i);
                    int _r_idx_temp = tile.shfl_down(r_idx_temp, i);
                    
                    if (_l_temp < l_temp)
                    {
                        l_temp = _l_temp;
                        l_idx_temp = _l_idx_temp;
                    }
                    if (_r_temp < r_temp)
                    {
                        r_temp = _r_temp;
                        r_idx_temp = _r_idx_temp;
                    }
                }

                //TODO Reduce across blocks by storing the minimum value and index for each block
                if (tile.thread_rank() == 0)
                {
                    //Get the index of the minimum value among all threads which matched the condition
                    //printf("Left pass: [%d] %f @ %d\n", tile.thread_rank(), left_sum_min, l_idx_temp);
                        residual_sum_l[blockIdx.x] = l_temp;
                        if (bounds_l != NULL) bounds_l[blockIdx.x] = l_idx_temp;

                    //printf("Right pass: [%d] %f @ %d\n", tile.thread_rank(), right_sum_min, r_idx_temp);
                    residual_sum_r[blockIdx.x] = r_temp;
                    if (bounds_r != NULL) bounds_r[blockIdx.x] = r_idx_temp;
                }
            }

            sum_l[idx] = left_sum_tmp;
            sum_r[idx] = right_sum_tmp;
        }
    }
    else if (KernelMode == 2)
    {
        T l_value = (idx < len) ? residual_sum_l[idx] : INFINITY;
        int l_idx = (idx < len) ? bounds_l[idx] : INT_MAX;
        T r_value = (idx < len) ? residual_sum_r[idx] : INFINITY;
        int r_idx = (idx < len) ? bounds_r[idx] : INT_MAX;

        //TODO track index closest to leftLimit and rightLimit respectively, via reduction, first within the block, then across blocks
        T left_sum_min = abs(l_value - leftLimit);
        T right_sum_min = abs(r_value - rightLimit);
        for (int i = 1; i < 32; i *= 2)
        {
            T left_sum_min_r = tile.shfl_down(left_sum_min, i);
            T right_sum_min_r = tile.shfl_down(right_sum_min, i);
            T l_idx_tmp = tile.shfl_down(l_idx, i);
            T r_idx_tmp = tile.shfl_down(r_idx, i);

            if (left_sum_min_r < left_sum_min)
            {
                left_sum_min = left_sum_min_r;
                l_idx = l_idx_tmp;
            }
            if (right_sum_min_r < right_sum_min)
            {
                right_sum_min = right_sum_min_r;
                r_idx = r_idx_tmp;
            }
        }

        //TODO Reduce within the block
        __shared__ T shared_data[32][2];
        __shared__ int shared_idx[32][2];
        if (tile.meta_group_rank() == 0) 
        {
            shared_data[tile.thread_rank()][0] = INFINITY;
            shared_data[tile.thread_rank()][1] = INFINITY;
            shared_idx[tile.thread_rank()][0] = INT_MAX;
            shared_idx[tile.thread_rank()][1] = INT_MAX;
        }
        
        // Synchronize threads within the warp
        cta.sync();
        
        // Store the results in shared memory
        if (tile.thread_rank() == 0) 
        {
            shared_data[tile.meta_group_rank()][0] = left_sum_min;
            shared_data[tile.meta_group_rank()][1] = right_sum_min;
            shared_idx[tile.meta_group_rank()][0] = l_idx;
            shared_idx[tile.meta_group_rank()][1] = r_idx;
        }
        
        // Synchronize threads within the warp
        cta.sync();
        
        // Prefix sum the shared data across the meta group
        if(tile.meta_group_rank() == 0) 
        {
            T l_temp = shared_data[tile.thread_rank()][0];
            T r_temp = shared_data[tile.thread_rank()][1];
            int l_idx_temp = shared_idx[tile.thread_rank()][0];
            int r_idx_temp = shared_idx[tile.thread_rank()][1];

            for (int i = 1; i < 32; i *= 2)
            {
                T _l_temp = tile.shfl_down(l_temp, i);
                T _r_temp = tile.shfl_down(r_temp, i);
                int _l_idx_temp = tile.shfl_down(l_idx_temp, i);
                int _r_idx_temp = tile.shfl_down(r_idx_temp, i);
                
                if (_l_temp < l_temp)
                {
                    l_temp = _l_temp;
                    l_idx_temp = _l_idx_temp;
                }
                if (_r_temp < r_temp)
                {
                    r_temp = _r_temp;
                    r_idx_temp = _r_idx_temp;
                }
            }

            //TODO Reduce across blocks by storing the minimum value and index for each block
            if (tile.thread_rank() == 0)
            {
                //Get the index of the minimum value among all threads which matched the condition
                final_bounds[0] = l_idx_temp;
                final_bounds[1] = r_idx_temp;
            }
        }
    }
}

template<class T>
int IntegrateOverX_GPU_base(T* p0, int* ixBounds, double xStep, long long Nx, long long Ny, double* AuxArrIntOverX, TGPUUsageArg* pGPU)
{
    int minGridSize;
    int bs = 128;
    dim3 threads(1, bs);
    dim3 nblocks(Ny, 1);
    long long LenArr = Nx;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bs, IntegrateOverX_Kernel<T>, 0, (LenArr + PerThreadSum - 1) / PerThreadSum);
    nblocks.y = ((LenArr + PerThreadSum - 1)/PerThreadSum + bs - 1) / bs;
    threads.y = bs;

    p0 = (T*)CAuxGPU::ToDevice(pGPU, p0, Nx * Ny);
    AuxArrIntOverX = CAuxGPU::ToDevice(pGPU, AuxArrIntOverX, Ny, CAuxGPU::DONT_COPY);
    CAuxGPU::Memset(pGPU, AuxArrIntOverX, 0, Ny);
    ixBounds = CAuxGPU::ToDevice(pGPU, ixBounds, 2);
    CAuxGPU::EnsureDeviceMemoryReady(pGPU, p0, AuxArrIntOverX, ixBounds);
    IntegrateOverX_Kernel<T><<<nblocks, threads>>>(p0, ixBounds, xStep, (int)Nx, (int)Ny, AuxArrIntOverX);
    CAuxGPU::MarkUpdated(pGPU, AuxArrIntOverX, CAuxGPU::DEVICE);
    return 0;
}

int srTAuxMatStat::IntegrateOverX_GPU(float* p0, int* ixBounds, double xStep, long long Nx, long long Ny, double* AuxArrIntOverX, TGPUUsageArg* pGPU)
{
    return IntegrateOverX_GPU_base<float>(p0, ixBounds, xStep, Nx, Ny, AuxArrIntOverX, pGPU);
}

int srTAuxMatStat::IntegrateOverX_GPU(double* p0, int* ixBounds, double xStep, long long Nx, long long Ny, double* AuxArrIntOverX, TGPUUsageArg* pGPU)
{
    return IntegrateOverX_GPU_base<double>(p0, ixBounds, xStep, Nx, Ny, AuxArrIntOverX, pGPU);
}

template <class T>
int IntegrateOverY_GPU_base(T* p0, int* iyBounds, double yStep, long long Nx, long long Ny, double* AuxArrIntOverY, TGPUUsageArg* pGPU)
{
    int minGridSize;
    int bs = 128;
    dim3 threads(bs, 1);
    dim3 nblocks(Nx, Ny + 1);
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bs, IntegrateOverY_Kernel<T>, 0, Nx);
    nblocks.x = (Nx + bs - 1) / bs;
    nblocks.y = (Ny + PerThreadSum - 1) / PerThreadSum;
    threads.x = bs;

    p0 = CAuxGPU::ToDevice(pGPU, p0, Nx*Ny);
    AuxArrIntOverY = CAuxGPU::ToDevice(pGPU, AuxArrIntOverY, Nx);
    CAuxGPU::Memset(pGPU, AuxArrIntOverY, 0, Nx);
    iyBounds = CAuxGPU::ToDevice(pGPU, iyBounds, 2);
    CAuxGPU::Memset(pGPU, iyBounds, 0, 2);
    CAuxGPU::EnsureDeviceMemoryReady(pGPU, p0, AuxArrIntOverY, iyBounds);
    IntegrateOverY_Kernel<T><<<nblocks, threads>>>(p0, iyBounds, yStep, (int)Nx, (int)Ny, AuxArrIntOverY);
    CAuxGPU::MarkUpdated(pGPU, AuxArrIntOverY, CAuxGPU::DEVICE);
    return 0;
}

int srTAuxMatStat::IntegrateOverY_GPU(float* p0, int* iyBounds, double yStep, long long Nx, long long Ny, double* AuxArrIntOverY, TGPUUsageArg* pGPU)
{
    return IntegrateOverY_GPU_base<float>(p0, iyBounds, yStep, Nx, Ny, AuxArrIntOverY, pGPU);
}

int srTAuxMatStat::IntegrateOverY_GPU(double* p0, int* iyBounds, double yStep, long long Nx, long long Ny, double* AuxArrIntOverY, TGPUUsageArg* pGPU)
{
    return IntegrateOverY_GPU_base<double>(p0, iyBounds, yStep, Nx, Ny, AuxArrIntOverY, pGPU);
}

template <class T>
int IntegrateSimple_GPU_base(T* p0, long long LenArr, double Multiplier, double* OutVal, TGPUUsageArg* pGPU)
{
    int minGridSize;
    int bs = 1024;
    int nblocks = 1;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bs, SumVector_FixedStride_Kernel<T>, 0, (LenArr + PerThreadSum - 1) / PerThreadSum);
    nblocks = ((LenArr + PerThreadSum - 1)/PerThreadSum + bs - 1) / bs;

    p0 = CAuxGPU::ToDevice(pGPU, p0, LenArr);
    OutVal = CAuxGPU::ToDevice(pGPU, OutVal, 1);
    CAuxGPU::Memset(pGPU, OutVal, 0, 1);
    CAuxGPU::EnsureDeviceMemoryReady(pGPU, p0, OutVal);
    SumVector_FixedStride_Kernel<T><<<nblocks, bs>>>(p0, 0LL, LenArr - 1, Multiplier, OutVal);
    CAuxGPU::MarkUpdated(pGPU, OutVal, CAuxGPU::DEVICE);
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

template <class T>
int PrefixSum_GPU(T* data, int len, int* sum_bounds, int* final_bounds, double RelPowLevel, double* IntegratedIntens, TGPUUsageArg* pGPU)
{
    int minGridSize;
    int bs = 1024;
    dim3 threads(bs, 1);
    dim3 nblocks((len + bs - 1) / bs, 1);
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bs, PrefixSum<T>, 0, len);
    if (bs > 32) bs = ((bs + 31) / 32) * 32; //Round up block size to the nearest multiple of 32
    nblocks.x = (len + bs - 1) / bs;
    threads.x = bs;

    data = CAuxGPU::ToDevice(pGPU, data, len);
    sum_bounds = CAuxGPU::ToDevice(pGPU, sum_bounds, 2);
    final_bounds = CAuxGPU::ToDevice(pGPU, final_bounds, 2);
    T* residual_sum_l = CAuxGPU::ToDevice(pGPU, NULL, (nblocks.x + 1) * 2);
    CAuxGPU::Memset(pGPU, residual_sum_l, 0, (nblocks.x + 1) * 2);
    T* residual_sum_r = residual_sum_l + nblocks.x + 1;
    T* sum_l = CAuxGPU::ToDevice(pGPU, NULL, len);
    T* sum_r = sum_l + len;
    int* bounds_l = CAuxGPU::ToDevice(pGPU, NULL, nblocks.x * 2);
    int* bounds_r = bounds_l + nblocks.x;
    int tertiary_sum_bounds[2] { 0, nblocks.x - 1 };
    int* tertiary_sum_bounds_d = CAuxGPU::ToDevice(pGPU, tertiary_sum_bounds, 2);

    CAuxGPU::EnsureDeviceMemoryReady(pGPU, data, sum_bounds, final_bounds, residual_sum_l, sum_l, bounds_l, tertiary_sum_bounds_d);

    if (bs > 32) PrefixSum_Kernel<T, 0, true><<<nblocks, threads>>>(data, sum_bounds, sum_l, sum_r, residual_sum_l, residual_sum_r);
    else PrefixSum_Kernel<T, 0, false><<<nblocks, threads>>>(data, sum_bounds, sum_l, sum_r, residual_sum_l, residual_sum_r);

    int residual_thds = 32;
    if (nblocks.x > residual_thds) residual_thds = nblocks.x;
    if (nblocks.x > 1)
    {
        if (residual_thds > 32) PrefixSum_Kernel<T, 0, true><<<1, residual_thds>>>(residual_sum_l, tertiary_sum_bounds_d, residual_sum_l, NULL);
        else PrefixSum_Kernel<T, 0, false><<<1, residual_thds>>>(residual_sum_l, tertiary_sum_bounds_d, residual_sum_l, NULL);
        
        if (residual_thds > 32) PrefixSum_Kernel<T, 0, true><<<1, residual_thds>>>(residual_sum_r, tertiary_sum_bounds_d, NULL, residual_sum_r);
        else PrefixSum_Kernel<T, 0, false><<<1, residual_thds>>>(residual_sum_r, tertiary_sum_bounds_d, NULL, residual_sum_r);
    }     
    PrefixSum_Kernel<T, 1><<<nblocks, threads>>>(data, sum_bounds, sum_l, sum_r, residual_sum_l, residual_sum_r, RelPowLevel, IntegratedIntens, bounds_l, bounds_r);
    PrefixSum_Kernel<T, 2><<<1, residual_thds>>>(NULL, tertiary_sum_bounds_d, NULL, NULL, residual_sum_l, residual_sum_r, RelPowLevel, IntegratedIntens, bounds_l, bounds_r, final_bounds);

    CAuxGPU::MarkUpdated(pGPU, bounds_l, CAuxGPU::DEVICE);
    CAuxGPU::MarkUpdated(pGPU, sum_l, CAuxGPU::DEVICE);
    CAuxGPU::MarkUpdated(pGPU, residual_sum_l, CAuxGPU::DEVICE);
    CAuxGPU::MarkUpdated(pGPU, final_bounds, CAuxGPU::DEVICE);
    CAuxGPU::ToHostAndFree(pGPU, bounds_l);
    CAuxGPU::ToHostAndFree(pGPU, sum_l);
    CAuxGPU::ToHostAndFree(pGPU, residual_sum_l);
    CAuxGPU::ToHostAndFree(pGPU, tertiary_sum_bounds_d);
    return 0;
}

int FindIntensityLimits2D_GPU(srTWaveAccessData& InWaveData, double RelPowLevel, double* IntegratedIntens, int* IndLims, TGPUUsageArg* pGPU)
{
    long Nx = (long)InWaveData.DimSizes[0];
    long Ny = (long)InWaveData.DimSizes[1];
    double xStep = InWaveData.DimSteps[0];
    double yStep = InWaveData.DimSteps[1];
    double xStart = InWaveData.DimStartValues[0];
    double yStart = InWaveData.DimStartValues[1];

    float* pf0 = NULL;
    double* pd0 = NULL;
    if (*(InWaveData.WaveType) == 'f') pf0 = (float*)InWaveData.pWaveData;
    else pd0 = (double*)InWaveData.pWaveData;
    
    int *IndLims_d = (int*)CAuxGPU::ToDevice(pGPU, IndLims, 4 * sizeof(int));
    int *ixBounds_d = IndLims_d;
    int *iyBounds_d = IndLims_d + 2;

    //Integrate over X
    double *AuxArrIntOverX = CAuxGPU::ToDevice(pGPU, NULL, Ny);
    CAuxGPU::Memset(pGPU, AuxArrIntOverX, 0, Ny);
    CAuxGPU::EnsureDeviceMemoryReady(pGPU, IndLims_d);
    if (pf0 != NULL) IntegrateOverX_GPU(pf0, ixBounds_d, xStep, Nx, Ny, AuxArrIntOverX, pGPU);
    else IntegrateOverX_GPU(pd0, ixBounds_d, xStep, Nx, Ny, AuxArrIntOverX, pGPU);
    //Find the limits of integration over X
    if (pf0 != NULL) PrefixSum_GPU<float>(AuxArrIntOverX, Ny, ixBounds_d, iyBounds_d, RelPowLevel, IntegratedIntens, pGPU);
    else PrefixSum_GPU<double>(AuxArrIntOverX, Ny, ixBounds_d, iyBounds_d, RelPowLevel, IntegratedIntens, pGPU);
    
    //Integrate Y over the limits of integration over X
    double* AuxArrIntOverY = CAuxGPU::ToDevice(pGPU, NULL, Nx);
    CAuxGPU::Memset(pGPU, AuxArrIntOverY, 0, Nx);
    if (pf0 != NULL) IntegrateOverY_GPU(pf0, iyBounds_d, yStep, Nx, Ny, AuxArrIntOverY, pGPU);
    else IntegrateOverY_GPU(pd0, iyBounds_d, yStep, Nx, Ny, AuxArrIntOverY, pGPU);
    //Find the limits of integration over Y
    if (pf0 != NULL) PrefixSum_GPU<float>(AuxArrIntOverY, Nx, iyBounds_d, ixBounds_d, RelPowLevel, IntegratedIntens, pGPU);
    else PrefixSum_GPU<double>(AuxArrIntOverY, Nx, iyBounds_d, ixBounds_d, RelPowLevel, IntegratedIntens, pGPU);

    //The integer limits of integration over X and Y are now in ixBounds_d and iyBounds_d respectively
    CAuxGPU::MarkUpdated(pGPU, IndLims_d, CAuxGPU::DEVICE);
    CAuxGPU::MarkUpdated(pGPU, AuxArrIntOverX, CAuxGPU::DEVICE);
    CAuxGPU::MarkUpdated(pGPU, AuxArrIntOverY, CAuxGPU::DEVICE);
    CAuxGPU::ToHostAndFree(pGPU, AuxArrIntOverX);
    CAuxGPU::ToHostAndFree(pGPU, AuxArrIntOverY);
    
    return 0;
}

int srTAuxMatStat::FindIntensityLimitsInds_GPU(CHGenObj& hRad, int ie, double RelPow, int* IndLims, TGPUUsageArg* pGPU)
{
    srTSRWRadStructAccessData& Rad = *((srTSRWRadStructAccessData*)(hRad.ptr()));

	IndLims[0] = 0;
	IndLims[1] = Rad.nx - 1;
	IndLims[2] = 0;
	IndLims[3] = Rad.nz - 1;

	try //OC21022024: to rewrite, avoiding allocation: new float[Rad.nx*Rad.nz]; !
	{
		srTRadExtract RadExtract;
		RadExtract.PolarizCompon = 6;
		RadExtract.Int_or_Phase = 0;
		RadExtract.PlotType = 3;
		RadExtract.TransvPres = Rad.Pres;

		RadExtract.ePh = Rad.eStart + ie*Rad.eStep;
		RadExtract.pExtractedData = new float[Rad.nx*Rad.nz];

		//srTRadGenManip RadGenManip(Rad);
		srTRadGenManip RadGenManip(hRad);
		srTWaveAccessData ExtractedWaveData;
		int res = 0;
        if(res = RadGenManip.ExtractRadiation(RadExtract, ExtractedWaveData, pGPU))
		{
            CAuxGPU::ToHostAndFree(pGPU, RadExtract.pExtractedData);
			delete[] RadExtract.pExtractedData; return res;
		}

		float AuxArrF[5];
		srTWaveAccessData OutInfoData;
		(OutInfoData.WaveType)[0] = 'f';
		OutInfoData.AmOfDims = 1;
		(OutInfoData.DimSizes)[0] = 5;
		(OutInfoData.DimSizes)[1] = 0;
		(OutInfoData.DimStartValues)[0] = 0;
		(OutInfoData.DimSteps)[0] = 1;
		OutInfoData.pWaveData = (char*)AuxArrF;
		for(int i=0; i<5; i++) AuxArrF[i] = 0.;
        
        // Directly call IntegrateSimple_GPU so the result stays on the device
        long Nx = (long)ExtractedWaveData.DimSizes[0];
        long Ny = (long)ExtractedWaveData.DimSizes[1];
        double xStep = ExtractedWaveData.DimSteps[0];
        double yStep = ExtractedWaveData.DimSteps[1];

        double IntegratedIntens = 0.;
        if (*(ExtractedWaveData.WaveType) == 'f') IntegrateSimple_GPU((float*)ExtractedWaveData.pWaveData, Nx*Ny, xStep*yStep, &IntegratedIntens, pGPU);
        else IntegrateSimple_GPU((double*)ExtractedWaveData.pWaveData, Nx*Ny, xStep*yStep, &IntegratedIntens, pGPU);
		
        res = FindIntensityLimits2D_GPU(ExtractedWaveData, RelPow, &IntegratedIntens, IndLims, pGPU);
        CAuxGPU::ToHostAndFree(pGPU, RadExtract.pExtractedData);
        CAuxGPU::ToHostAndFree(pGPU, &IntegratedIntens);
        delete[] RadExtract.pExtractedData;
        if(res) return res;
	}
	catch(...)
	{ }
	return 0;
}

#endif