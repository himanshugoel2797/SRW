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

#ifdef _OFFLOAD_GPU
#ifndef __SROPTELMGPU_H
#define __SROPTELMGPU_H

#include "cuda_runtime.h"
#include <sroptelm.h>
#include <srradstr.h>
#include <srstraux.h>

#ifdef __CUDACC__
template<class T, bool combinedE> 
__global__ void RadPointModifierParallel_Kernel(srTSRWRadStructAccessData* pRadAccessData, void* pBufVars, T* tgt_obj, int xStart, int xFin, int zStart, int zFin) //HG27072024 Redesigned entire function
{
	int ie = (blockIdx.x * blockDim.x + threadIdx.x); //ne range
	int ix = (blockIdx.y * blockDim.y + threadIdx.y) + xStart; //nx range
	int iz = (blockIdx.z * blockDim.z + threadIdx.z) + zStart; //nz range
	
	int ne = 1;
	if (combinedE)
	{
		ne = pRadAccessData->ne;
		ie = 0;
	} 

	if (ix < xFin && iz < zFin && ie < pRadAccessData->ne) //HG27072024 changed RadAccessData to pRadAccessData
	{
		srTEFieldPtrs EPtrs;
		srTEXZ EXZ;
		EXZ.z = pRadAccessData->zStart + iz * pRadAccessData->zStep;
		EXZ.x = pRadAccessData->xStart + ix * pRadAccessData->xStep;
		EXZ.e = pRadAccessData->eStart + ie * pRadAccessData->eStep;
		EXZ.aux_offset = pRadAccessData->ne * pRadAccessData->nx * 2 * iz + pRadAccessData->ne * 2 * ix + ie * 2;
		if (pRadAccessData->pBaseRadX != 0)
		{
			EPtrs.pExRe = pRadAccessData->pBaseRadX + EXZ.aux_offset;
			EPtrs.pExIm = EPtrs.pExRe + 1;
		}
		else
		{
			EPtrs.pExRe = 0;
			EPtrs.pExIm = 0;
		}
		if (pRadAccessData->pBaseRadZ != 0)
		{
			EPtrs.pEzRe = pRadAccessData->pBaseRadZ + EXZ.aux_offset;
			EPtrs.pEzIm = EPtrs.pEzRe + 1;
		}
		else
		{
			EPtrs.pEzRe = 0;
			EPtrs.pEzIm = 0;
		}

		tgt_obj->RadPointModifierPortable(EXZ, EPtrs, pBufVars);

		for (ie=1; ie < ne; ie++)
		{
			EXZ.e += pRadAccessData->eStep;
			EXZ.aux_offset += 2;
			if (pRadAccessData->pBaseRadX != 0)
			{
				EPtrs.pExRe += 2;
				EPtrs.pExIm += 2;
			}
			if (pRadAccessData->pBaseRadZ != 0)
			{
				EPtrs.pEzRe += 2;
				EPtrs.pEzIm += 2;
			}
			tgt_obj->RadPointModifierPortable(EXZ, EPtrs, pBufVars);
		}
	}
}

//template<class T> int RadPointModifierParallelImpl(srTSRWRadStructAccessData* pRadAccessData, void* pBufVars, long pBufVarsSz, T* tgt_obj, TGPUUsageArg* pGPU)
template<class T> 
int RadPointModifierParallelImpl(srTSRWRadStructAccessData* pRadAccessData, void* pBufVars, long pBufVarsSz, T* tgt_obj, TGPUUsageArg* pGPU, int *pRegion=0, bool combinedE=false) //HG29072024
{
	if (pRadAccessData->pBaseRadX != NULL)
	{
		pRadAccessData->pBaseRadX = CAuxGPU::ToDevice(pGPU, pRadAccessData->pBaseRadX, 2*pRadAccessData->ne*pRadAccessData->nx*pRadAccessData->nz);
		CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadAccessData->pBaseRadX);
	}
	if (pRadAccessData->pBaseRadZ != NULL)
	{
		pRadAccessData->pBaseRadZ = CAuxGPU::ToDevice(pGPU, pRadAccessData->pBaseRadZ, 2*pRadAccessData->ne*pRadAccessData->nx*pRadAccessData->nz);
		CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadAccessData->pBaseRadZ);
	}
	
	srTSRWRadStructAccessData* pRadAccessData_dev = CAuxGPU::ToDevice(pGPU, pRadAccessData, 1);
    T* local_copy = CAuxGPU::ToDevice(pGPU, tgt_obj, 1);
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadAccessData_dev, local_copy);
	
	void* pBufVars_dev = NULL;
	if (pBufVarsSz > 0)
	{
		pBufVars_dev = CAuxGPU::ToDevice(pGPU, (char*)pBufVars, pBufVarsSz);
		CAuxGPU::EnsureDeviceMemoryReady(pGPU, pBufVars_dev);
	}
	
	int xStart = 0;
	int xFin = pRadAccessData->nx;
	int zStart = 0;
	int zFin = pRadAccessData->nz;
	
	//HG30072024 Allow for specifying a region to skip or to only process within the region, reduces extra operations for propagators like apertures and obstacles
	bool HandleInSingleLaunch = (pRegion == 0);
	if (!HandleInSingleLaunch)
	{
		if (pRegion[4] == 0)
		{
			xStart = pRegion[0];
			xFin = pRegion[1];
			zStart = pRegion[2];
			zFin = pRegion[3];
			HandleInSingleLaunch = true;
		}
	}
	
	void (*kern)(srTSRWRadStructAccessData*, void*, T*, int, int, int, int) = NULL;
	if (combinedE) kern = RadPointModifierParallel_Kernel<T, true>;
	else kern = RadPointModifierParallel_Kernel<T, false>;
	if (HandleInSingleLaunch)
	{
		dim3 blocks(combinedE ? 1 : pRadAccessData->ne, xFin - xStart, zFin - zStart);
		dim3 threads(1);
		printf("%s [%d, %d, %d] %d %d\r\n", __func__, blocks.x, blocks.y, blocks.z, xFin, zFin);
		CAuxGPU::CalcLaunchDims(kern, blocks, blocks, threads);
		printf("%s [%d, %d, %d][%d, %d, %d]", __func__, blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
		kern<<<blocks, threads >>> (pRadAccessData_dev, pBufVars_dev, local_copy, xStart, xFin, zStart, zFin);
	}
	else
	{
		//Have to split into 4 kernel launches to skip the specified region, run them in parallel
		cudaStream_t stream1 = (cudaStream_t)CAuxGPU::GetComputeStream(pGPU, 0);
		cudaStream_t stream2 = (cudaStream_t)CAuxGPU::GetComputeStream(pGPU, 1);
		cudaStream_t stream3 = (cudaStream_t)CAuxGPU::GetComputeStream(pGPU, 2);

		CAuxGPU::SyncComputeStream(pGPU, 0, (long long)stream1);
		CAuxGPU::SyncComputeStream(pGPU, 0, (long long)stream2);
		CAuxGPU::SyncComputeStream(pGPU, 0, (long long)stream3);

		dim3 blocks0(combinedE ? 1 : pRadAccessData->ne, pRegion[0], pRadAccessData->nx);
		dim3 blocks1(combinedE ? 1 : pRadAccessData->ne, pRadAccessData->nx - pRegion[1], pRadAccessData->nx);
		dim3 blocks2(combinedE ? 1 : pRadAccessData->ne, pRegion[1] - pRegion[0], pRadAccessData->nz - pRegion[3]);
		dim3 blocks3(combinedE ? 1 : pRadAccessData->ne, pRegion[1] - pRegion[0], pRegion[2]);
		dim3 threads0(1);
		dim3 threads1(1);
		dim3 threads2(1);
		dim3 threads3(1);

		if (blocks0.y > 0 && blocks0.z > 0)
		{
			CAuxGPU::CalcLaunchDims(kern, blocks0, blocks0, threads0);
			kern<<<blocks0, threads0 >>> (pRadAccessData_dev, pBufVars_dev, local_copy, 0, pRegion[0], 0, pRadAccessData->nx);
		}
		if (blocks1.y > 0 && blocks1.z > 0)
		{
			CAuxGPU::CalcLaunchDims(kern, blocks1, blocks1, threads1);
			kern<<<blocks1, threads1 >>> (pRadAccessData_dev, pBufVars_dev, local_copy, pRegion[1], pRadAccessData->nx, 0, pRadAccessData->nx);
		}
		if (blocks2.y > 0 && blocks2.z > 0)
		{
			CAuxGPU::CalcLaunchDims(kern, blocks2, blocks2, threads2);
			kern<<<blocks2, threads2 >>> (pRadAccessData_dev, pBufVars_dev, local_copy, pRegion[0], pRegion[1], pRegion[3], pRadAccessData->nz);
		}
		if (blocks3.y > 0 && blocks3.z > 0)
		{
			CAuxGPU::CalcLaunchDims(kern, blocks3, blocks3, threads3);
			kern<<<blocks3, threads3 >>> (pRadAccessData_dev, pBufVars_dev, local_copy, pRegion[0], pRegion[1], 0, pRegion[2]);
		}
		
		CAuxGPU::SyncComputeStream(pGPU, (long long)stream1, 0);
		CAuxGPU::SyncComputeStream(pGPU, (long long)stream2, 0);
		CAuxGPU::SyncComputeStream(pGPU, (long long)stream3, 0);
	}

	if (pBufVarsSz > 0) CAuxGPU::ToHostAndFree(pGPU, (char*)pBufVars_dev, CAuxGPU::DONT_COPY);
	CAuxGPU::ToHostAndFree(pGPU, pRadAccessData_dev, CAuxGPU::DONT_COPY); //HG27072024
	CAuxGPU::ToHostAndFree(pGPU, local_copy, CAuxGPU::DONT_COPY);
	
	CAuxGPU::MarkUpdated(pGPU, pRadAccessData->pBaseRadX, CAuxGPU::DEVICE);
	CAuxGPU::MarkUpdated(pGPU, pRadAccessData->pBaseRadZ, CAuxGPU::DEVICE);
	
//#ifndef _DEBUG //HG26022024 (commented-out)
	if (pRadAccessData->pBaseRadX != NULL)
		pRadAccessData->pBaseRadX = CAuxGPU::GetHostPtr(pGPU, pRadAccessData->pBaseRadX);
	if (pRadAccessData->pBaseRadZ != NULL)
		pRadAccessData->pBaseRadZ = CAuxGPU::GetHostPtr(pGPU, pRadAccessData->pBaseRadZ);
//#endif

//HG26022024 (commented-out)
//#ifdef _DEBUG
//	if (pRadAccessData->pBaseRadX != NULL)
//		pRadAccessData->pBaseRadX = (float*)CAuxGPU::ToHostAndFree(pGPU, pRadAccessData->pBaseRadX, 2*pRadAccessData->ne*pRadAccessData->nx*pRadAccessData->nz*sizeof(float));
//	if (pRadAccessData->pBaseRadZ != NULL)
//		pRadAccessData->pBaseRadZ = (float*)CAuxGPU::ToHostAndFree(pGPU, pRadAccessData->pBaseRadZ, 2*pRadAccessData->ne*pRadAccessData->nx*pRadAccessData->nz*sizeof(float));
//	cudaStreamSynchronize(0);
//	auto err = cudaGetLastError();
//	printf("%s\r\n", cudaGetErrorString(err));
//#endif

	return 0;
}
#endif

#endif //__SROPTELMGPU_H
#endif