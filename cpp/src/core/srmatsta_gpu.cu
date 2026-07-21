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

#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/reverse.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/iterator/transform_iterator.h> //HG20072026

#include <stdio.h>
#include <iostream>
#include <chrono>
#include "srmatsta.h"
#include "srradmnp.h"

const int PerThreadSum = 16;

//HG20072026 Widens the scanned values to double so the prefix sums match the CPU's
//`double Sum` accumulator (srmatsta.cpp:404).
template<class T> struct CastToDouble
{
    __host__ __device__ double operator()(const T& v) const { return (double)v; }
};

template<class T>
__global__ void IntegrateOverX_Kernel(T* data, int ixStart, int ixEnd, double xStep, int Nx, int Ny, double* AuxArrIntOverX)
{
    int iy = blockIdx.x * blockDim.x + threadIdx.x;
    int ix = blockIdx.y * blockDim.y + threadIdx.y;
    ix = ix * PerThreadSum + ixStart;
    if (ix > Nx) return;
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
__global__ void IntegrateOverY_Kernel(T* data, int iyStart, int iyEnd, double yStep, int Nx, int Ny, double* AuxArrIntOverY)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    iy = iy * PerThreadSum + iyStart;
    if (iy > Ny) return;
    if (iy > iyEnd) return;
    int iyFin = min(iy + PerThreadSum - 1, iyEnd);
    double sum = 0.;
    if (ix < Nx)
    {
        for (; iy <= iyFin; iy++)
        {
            sum += data[iy * Nx + ix];
        }
        atomicAdd(&AuxArrIntOverY[ix], sum * yStep);
    }
}

template<class T>
int IntegrateOverX_GPU_base(T* p0, int ixStart, int ixEnd, double xStep, long Nx, long Ny, double* AuxArrIntOverX, TGPUUsageArg* pGPU) //HG09122025
//int IntegrateOverX_GPU_base(T* p0, int ixStart, int ixEnd, double xStep, long long Nx, long long Ny, double* AuxArrIntOverX, TGPUUsageArg* pGPU)
{
    dim3 nblocks(Ny, Nx / PerThreadSum + !!(Nx % PerThreadSum));
    dim3 threads(1);
    CAuxGPU::CalcLaunchDims(IntegrateOverX_Kernel<T>, nblocks, nblocks, threads, 1);

    p0 = (T*)CAuxGPU::ToDevice(pGPU, p0, Nx * Ny);
    AuxArrIntOverX = CAuxGPU::ToDevice(pGPU, AuxArrIntOverX, Ny, CAuxGPU::DONT_COPY);
    CAuxGPU::Memset(pGPU, AuxArrIntOverX, 0.0, Ny);
    CAuxGPU::EnsureDeviceMemoryReady(pGPU, p0, AuxArrIntOverX);
    IntegrateOverX_Kernel<T><<<nblocks, threads>>>(p0, ixStart, ixEnd, xStep, (int)Nx, (int)Ny, AuxArrIntOverX);
    CAuxGPU::MarkUpdatedBatch(pGPU, CAuxGPU::DEVICE, p0, AuxArrIntOverX);
    return 0;
}

template <class T>
int IntegrateOverY_GPU_base(T* p0, int iyStart, int iyEnd, double yStep, long Nx, long Ny, double* AuxArrIntOverY, TGPUUsageArg* pGPU) //HG09122025
//int IntegrateOverY_GPU_base(T* p0, int iyStart, int iyEnd, double yStep, long long Nx, long long Ny, double* AuxArrIntOverY, TGPUUsageArg* pGPU)
{
    dim3 nblocks(Nx, (iyEnd + 1 - iyStart) / PerThreadSum + !!((iyEnd + 1 - iyStart) % PerThreadSum));
    dim3 threads(1);
    CAuxGPU::CalcLaunchDims(IntegrateOverY_Kernel<T>, nblocks, nblocks, threads, 1);

    p0 = CAuxGPU::ToDevice(pGPU, p0, Nx*Ny);
    AuxArrIntOverY = CAuxGPU::ToDevice(pGPU, AuxArrIntOverY, Nx);
    CAuxGPU::Memset(pGPU, AuxArrIntOverY, 0.0, Nx);
    CAuxGPU::EnsureDeviceMemoryReady(pGPU, p0, AuxArrIntOverY);
    IntegrateOverY_Kernel<T><<<nblocks, threads>>>(p0, iyStart, iyEnd, yStep, (int)Nx, (int)Ny, AuxArrIntOverY);
    CAuxGPU::MarkUpdatedBatch(pGPU, CAuxGPU::DEVICE, p0, AuxArrIntOverY);
    return 0;
}

template <class T>
int PrefixSum_GPU(T* data, int len, double step, double RelPowLevel, double IntegratedIntens, int *leftIndex, int *rightIndex, TGPUUsageArg* pGPU)
{
    //HG20072026 The running sums are accumulated in DOUBLE, matching the CPU
    //(srTAuxMatStat::FindLimit1DLeft, srmatsta.cpp:404, uses `double Sum = 0.`).
    //They used to be scanned in T -- float32 for an intensity array -- which can cross
    //the power threshold at a different index than the CPU's double accumulator.
    //NOTE: this widening alone did NOT fix the ~1.3% second-order-moment gap (measured
    //bit-for-bit unmoved by it). The actual one-bin window shift was the CPU's float32
    //coordinate round-trip, replicated in FindIntensityLimitsInds_GPU below. This
    //change is CPU-parity hardening of the scan itself.
    data = CAuxGPU::ToDevice(pGPU, data, len);
    double* sum = new double[len * 2];
    double* sum_l = CAuxGPU::ToDevice<double>(pGPU, sum, len * 2, CAuxGPU::DONT_COPY);
    double* sum_r = sum_l + len;

    CAuxGPU::EnsureDeviceMemoryReady(pGPU, data, sum_l);

    auto stream0 = CAuxGPU::GetComputeStream(pGPU, 0);
    auto stream1 = CAuxGPU::GetComputeStream(pGPU, 1);
    auto async_policy0 = thrust::cuda::par_nosync.on((cudaStream_t)stream0);
    auto async_policy1 = thrust::cuda::par_nosync.on((cudaStream_t)stream1);

    CAuxGPU::SyncComputeStream(pGPU, 0, stream0);
    CAuxGPU::SyncComputeStream(pGPU, 0, stream1);

    CastToDouble<T> toD; //HG20072026 widen on the way into the scan, not after it
    thrust::inclusive_scan(async_policy0,
        thrust::make_transform_iterator(data, toD),
        thrust::make_transform_iterator(data + len, toD), sum_l);
    thrust::reverse_copy(async_policy1,
        thrust::make_transform_iterator(data, toD),
        thrust::make_transform_iterator(data + len, toD), sum_r);
    thrust::exclusive_scan(async_policy1, sum_r, sum_r + len, sum_r);
    CAuxGPU::SyncComputeStream(pGPU, stream0, 0);
    CAuxGPU::SyncComputeStream(pGPU, stream1, 0);
    CAuxGPU::MarkUpdatedBatch(pGPU, CAuxGPU::DEVICE, data, sum_l);
    sum_l = CAuxGPU::ToHostAndFree(pGPU, sum_l);
    sum_r = sum_l + len;
    double* a00 = sum_l;
    double* a10 = sum_r;
    //HG20072026 `>` not `>=`, to match the CPU test (`if(Sum > IntToStop)`,
    //srmatsta.cpp:412). With the sums now agreeing to double precision the boundary
    //case is reachable, so the comparison has to agree too.
    const double thresh = IntegratedIntens / step * (1. - RelPowLevel) * 0.25;
    for (int i = 0; i < len; i++)
    {
        if (a00 == sum_l && sum_l[i] > thresh) a00 = sum_l + i;
        if (a10 == sum_r && sum_r[i] > thresh) a10 = sum_r + i;
        if (a00 != sum_l && a10 != sum_r) break;
    }
    *leftIndex = (int)(a00 - sum_l);
    *rightIndex = (int)(len - (a10 - sum_r));
    delete[] sum; //HG20072026 was leaked on every call
    return 0;
}

int srTAuxMatStat::FindIntensityLimitsInds_GPU(CHGenObj& hRad, int ie, double RelPow, int* IndLims, void* pvGPU)
{
    TGPUUsageArg parGPU(pvGPU);
    srTSRWRadStructAccessData* Rad = ((srTSRWRadStructAccessData*)(hRad.ptr()));

	IndLims[0] = 0;
	IndLims[1] = Rad->nx - 1;
	IndLims[2] = 0;
	IndLims[3] = Rad->nz - 1;

	try
	{
		srTRadExtract RadExtract;
		RadExtract.PolarizCompon = 6;
		RadExtract.Int_or_Phase = 0;
		RadExtract.PlotType = 3;
		RadExtract.TransvPres = Rad->Pres;
		RadExtract.ePh = Rad->eStart + ie*Rad->eStep;
		RadExtract.pExtractedData = new float[Rad->nx*Rad->nz];

		srTRadGenManip RadGenManip(hRad);
		srTWaveAccessData ExtractedWaveData;
		int res = 0;
        if(res = RadGenManip.ExtractRadiation(RadExtract, ExtractedWaveData, pvGPU))
		{
            CAuxGPU::ToHostAndFree(&parGPU, RadExtract.pExtractedData);
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

        double *AuxArrIntOverX = CAuxGPU::ToDevice<double>(&parGPU, NULL, Ny);
        double* AuxArrIntOverY = CAuxGPU::ToDevice<double>(&parGPU, NULL, Nx);
        CAuxGPU::EnsureDeviceMemoryReady(&parGPU, AuxArrIntOverX, AuxArrIntOverY);

        double IntegratedIntens = 0.;
        float* pf0 = NULL;
        double* pd0 = NULL;
        if (*(ExtractedWaveData.WaveType) == 'f')
        {
            pf0 = CAuxGPU::ToDevice(&parGPU, (float*)ExtractedWaveData.pWaveData, Nx*Ny);
            CAuxGPU::EnsureDeviceMemoryReady(&parGPU, pf0);
            //HG20072026 Accumulate in double (the CPU IntegrateSimple sums into `double Sum`),
            //then collapse through float32: the CPU stores the total in `float AuxArrF[0]` and
            //derives every power threshold from that value (srmatsta.cpp:622,466).
            CastToDouble<float> toD;
            IntegratedIntens = thrust::reduce(thrust::device,
                thrust::make_transform_iterator(pf0, toD),
                thrust::make_transform_iterator(pf0 + Nx*Ny, toD), 0., thrust::plus<double>());
            IntegratedIntens = (double)((float)(IntegratedIntens * xStep*yStep));
            IntegrateOverX_GPU_base<float>(pf0, IndLims[0], IndLims[1], xStep, Nx, Ny, AuxArrIntOverX, &parGPU);
        }
        else
        {
            pd0 = CAuxGPU::ToDevice(&parGPU, (double*)ExtractedWaveData.pWaveData, Nx*Ny);
            CAuxGPU::EnsureDeviceMemoryReady(&parGPU, pd0);
            IntegratedIntens = thrust::reduce(thrust::device, pd0, pd0 + Nx*Ny, 0., thrust::plus<double>());
            IntegratedIntens = (double)((float)(IntegratedIntens * xStep*yStep)); //HG20072026 see above
            IntegrateOverX_GPU_base<double>(pd0, IndLims[0], IndLims[1], xStep, Nx, Ny, AuxArrIntOverX, &parGPU);
        }

        //Find the limits of integration over X
        PrefixSum_GPU<double>(AuxArrIntOverX, Ny, yStep, RelPow, IntegratedIntens, &IndLims[2], &IndLims[3], &parGPU);

        //HG20072026 CPU parity: FindIntensityLimits2D (srmatsta.cpp:501) fails out with
        //INCORRECT_ARGUMENTS if the vertical window is degenerate, and the caller then keeps
        //the full-range limits it initialized. Reproduce that, or the two paths pick
        //different windows for pathological wavefronts.
        if(IndLims[3] <= IndLims[2])
        {
            IndLims[0] = 0; IndLims[1] = Rad->nx - 1;
            IndLims[2] = 0; IndLims[3] = Rad->nz - 1;
            CAuxGPU::ToHostAndFree(&parGPU, AuxArrIntOverX);
            CAuxGPU::ToHostAndFree(&parGPU, AuxArrIntOverY);
            CAuxGPU::ToHostAndFree(&parGPU, RadExtract.pExtractedData);
            delete[] RadExtract.pExtractedData;
            return INCORRECT_ARGUMENTS;
        }

        //Integrate Y over the limits of integration over X
        if (*(ExtractedWaveData.WaveType) == 'f') IntegrateOverY_GPU_base<float>(pf0, IndLims[2], IndLims[3], yStep, Nx, Ny, AuxArrIntOverY, &parGPU);
        else IntegrateOverY_GPU_base<double>(pd0, IndLims[2], IndLims[3], yStep, Nx, Ny, AuxArrIntOverY, &parGPU);

        //Find the limits of integration over Y
        PrefixSum_GPU<double>(AuxArrIntOverY, Nx, xStep, RelPow, IntegratedIntens, &IndLims[0], &IndLims[1], &parGPU);

        //HG20072026 THE ~1.3% SECOND-ORDER MOMENT GAP LIVED HERE (KNOWN_ISSUES.md #1).
        //The CPU does not use its scanned integer limits directly: FindIntensityLimits2D
        //stores them as float32 COORDINATES (AuxArrF[1..4] in FindIntensityLimitsInds), and
        //the indices are re-derived as (int)((coordF - start)*1.0000001/step). Half a float
        //ulp of the coordinate can outweigh the 1.0000001 fudge (e.g. ix=22, x~3.3e-4 m,
        //step~3.1e-6 m: ulp/2/step ~ 4.6e-6 > 22*1e-7), so the CPU lands one bin BELOW the
        //index its own scan found. This GPU path kept the exact integers, so its 90%-power
        //window could exclude an edge column the CPU includes; with the x^2 weighting that
        //made the second-order moments differ ~1.3%, which srTDriftSpace amplified to ~12%
        //of peak after an aperture+drift. Replicate the CPU's float round-trip and clamps
        //bit-for-bit so both paths select the same window.
        {
            double xStartWave = ExtractedWaveData.DimStartValues[0];
            double yStartWave = ExtractedWaveData.DimStartValues[1];
            float xLoF = (float)(xStartWave + xStep*IndLims[0]);
            float xHiF = (float)(xStartWave + xStep*IndLims[1]);
            float yLoF = (float)(yStartWave + yStep*IndLims[2]);
            float yHiF = (float)(yStartWave + yStep*IndLims[3]);
            IndLims[0] = (int)((xLoF - Rad->xStart)*1.0000001/Rad->xStep);
            if(IndLims[0] < 0) IndLims[0] = 0;
            IndLims[1] = (int)((xHiF - Rad->xStart)*1.0000001/Rad->xStep);
            if(IndLims[1] >= Rad->nx) IndLims[1] = Rad->nx - 1;
            IndLims[2] = (int)((yLoF - Rad->zStart)*1.0000001/Rad->zStep);
            if(IndLims[2] < 0) IndLims[2] = 0;
            IndLims[3] = (int)((yHiF - Rad->zStart)*1.0000001/Rad->zStep);
            if(IndLims[3] >= Rad->nz) IndLims[3] = Rad->nz - 1;
        }

        //The integer limits of integration over X and Y are now in ixBounds_d and iyBounds_d respectively
        CAuxGPU::ToHostAndFree(&parGPU, AuxArrIntOverX);
        CAuxGPU::ToHostAndFree(&parGPU, AuxArrIntOverY);

        CAuxGPU::ToHostAndFree(&parGPU, RadExtract.pExtractedData);
        delete[] RadExtract.pExtractedData;

        if(res) return res;
	}
	catch(...)
	{ }
	return 0;
}

#endif