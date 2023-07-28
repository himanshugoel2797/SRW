/************************************************************************//**
 * File: sroptelm_gpu.h
 * Description: Optical element (general CUDA header)
 * Project: Synchrotron Radiation Workshop
 * First release: 2023
 *
 * Copyright (C) Brookhaven National Laboratory
 * All Rights Reserved
 *
 * @author H.Goel
 * @version 1.0
 ***************************************************************************/

#ifndef __SROPTELMGPU_H
#define __SROPTELMGPU_H

#include "cuda_runtime.h"
#include <sroptelm.h>
#include <srradstr.h>
#include <srstraux.h>

void TreatStronglyOscillatingTerm_GPU(srTSRWRadStructAccessData& RadAccessData, bool TreatPolCompX, bool TreatPolCompZ, int ieStart, int ieBefEnd, double ConstRx, double ConstRz, gpuUsageArg* pGpuUsage);
void MakeWfrEdgeCorrection_GPU(srTSRWRadStructAccessData& RadAccessData, float* pDataEx, float* pDataEz, srTDataPtrsForWfrEdgeCorr& DataPtrs, gpuUsageArg* pGpuUsage);

#ifdef __CUDACC__
template<class T> __global__ void RadPointModifierParallel_Kernel(srTSRWRadStructAccessData RadAccessData, void* pBufVars, T* tgt_obj)
{
	int ix = (blockIdx.x * blockDim.x + threadIdx.x); //nx range
	int iz = (blockIdx.y * blockDim.y + threadIdx.y); //nz range

	if (ix < RadAccessData.nx && iz < RadAccessData.nz)
	{
		srTEFieldPtrs EPtrs;
		srTEXZ EXZ;
		EXZ.z = RadAccessData.zStart + iz * RadAccessData.zStep;
		EXZ.x = RadAccessData.xStart + ix * RadAccessData.xStep;

		for (int iwfr = 0; iwfr < RadAccessData.nwfr; iwfr++)
			for (int ie = 0; ie < RadAccessData.ne; ie++) {
				EXZ.e = RadAccessData.eStart + ie * RadAccessData.eStep;
				EXZ.aux_offset = RadAccessData.ne * RadAccessData.nx * RadAccessData.nz * 2 * iwfr + RadAccessData.ne * RadAccessData.nx * 2 * iz + RadAccessData.ne * 2 * ix + ie * 2;
				if (RadAccessData.pBaseRadX != 0)
				{
					EPtrs.pExRe = RadAccessData.pBaseRadX + EXZ.aux_offset;
					EPtrs.pExIm = EPtrs.pExRe + 1;
				}
				else
				{
					EPtrs.pExRe = 0;
					EPtrs.pExIm = 0;
				}
				if (RadAccessData.pBaseRadZ != 0)
				{
					EPtrs.pEzRe = RadAccessData.pBaseRadZ + EXZ.aux_offset;
					EPtrs.pEzIm = EPtrs.pEzRe + 1;
				}
				else
				{
					EPtrs.pEzRe = 0;
					EPtrs.pEzIm = 0;
				}

				tgt_obj->RadPointModifierPortable(EXZ, EPtrs, pBufVars);
			}
	}
}

template<class T> int RadPointModifierParallelImpl(srTSRWRadStructAccessData* pRadAccessData, void* pBufVars, long pBufVarsSz, T* tgt_obj, gpuUsageArg *pGpuUsage)
{
	const int bs = 256;
	dim3 blocks(pRadAccessData->nx / bs + ((pRadAccessData->nx & (bs - 1)) != 0), pRadAccessData->nz);
	dim3 threads(bs, 1);
	
	if (pRadAccessData->pBaseRadX != NULL)
	{
		pRadAccessData->pBaseRadX = (float*)AuxGpu::ToDevice(pGpuUsage, pRadAccessData->pBaseRadX, 2*pRadAccessData->ne*pRadAccessData->nx*pRadAccessData->nz*pRadAccessData->nwfr*sizeof(float));
		AuxGpu::EnsureDeviceMemoryReady(pGpuUsage, pRadAccessData->pBaseRadX);
	}
	if (pRadAccessData->pBaseRadZ != NULL)
	{
		pRadAccessData->pBaseRadZ = (float*)AuxGpu::ToDevice(pGpuUsage, pRadAccessData->pBaseRadZ, 2*pRadAccessData->ne*pRadAccessData->nx*pRadAccessData->nz*pRadAccessData->nwfr*sizeof(float));
		AuxGpu::EnsureDeviceMemoryReady(pGpuUsage, pRadAccessData->pBaseRadZ);
	}

    T* local_copy = (T*)AuxGpu::ToDevice(pGpuUsage, tgt_obj, sizeof(T));
	AuxGpu::EnsureDeviceMemoryReady(pGpuUsage, local_copy);
    //cudaMalloc(&local_copy, sizeof(T));
    //cudaMemcpy(local_copy, tgt_obj, sizeof(T), cudaMemcpyHostToDevice);
	
	void* pBufVars_dev = NULL;
	if (pBufVarsSz > 0){
		pBufVars_dev = AuxGpu::ToDevice(pGpuUsage, pBufVars, pBufVarsSz);
		AuxGpu::EnsureDeviceMemoryReady(pGpuUsage, pBufVars_dev);
	}
	RadPointModifierParallel_Kernel<T> << <blocks, threads >> > (*pRadAccessData, pBufVars_dev, local_copy);
    //cudaDeviceSynchronize();
    //cudaFreeAsync(local_copy, 0);
	if (pBufVarsSz > 0) AuxGpu::ToHostAndFree(pGpuUsage, pBufVars_dev, pBufVarsSz, true);
	AuxGpu::ToHostAndFree(pGpuUsage, local_copy, sizeof(T), true);

	AuxGpu::MarkUpdated(pGpuUsage, pRadAccessData->pBaseRadX, true, false);
	AuxGpu::MarkUpdated(pGpuUsage, pRadAccessData->pBaseRadZ, true, false);

#ifndef _DEBUG
	if (pRadAccessData->pBaseRadX != NULL)
		pRadAccessData->pBaseRadX = (float*)AuxGpu::GetHostPtr(pGpuUsage, pRadAccessData->pBaseRadX);
	if (pRadAccessData->pBaseRadZ != NULL)
		pRadAccessData->pBaseRadZ = (float*)AuxGpu::GetHostPtr(pGpuUsage, pRadAccessData->pBaseRadZ);
#endif

#ifdef _DEBUG
	if (pRadAccessData->pBaseRadX != NULL)
		pRadAccessData->pBaseRadX = (float*)AuxGpu::ToHostAndFree(pGpuUsage, pRadAccessData->pBaseRadX, 2 * pRadAccessData->ne * pRadAccessData->nx * pRadAccessData->nz * pRadAccessData->nwfr * sizeof(float));
	if (pRadAccessData->pBaseRadZ != NULL)
		pRadAccessData->pBaseRadZ = (float*)AuxGpu::ToHostAndFree(pGpuUsage, pRadAccessData->pBaseRadZ, 2 * pRadAccessData->ne * pRadAccessData->nx * pRadAccessData->nz * pRadAccessData->nwfr * sizeof(float));
	cudaStreamSynchronize(0);
	auto err = cudaGetLastError();
	printf("%s\r\n", cudaGetErrorString(err));
#endif

	return 0;
}
#endif

#endif //__SROPTELMGPU_H