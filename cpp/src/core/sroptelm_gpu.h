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
//template<class T> __global__ void RadPointModifierParallel_Kernel(srTSRWRadStructAccessData RadAccessData, void* pBufVars, T* tgt_obj)
template<class T, bool combinedE> __global__ void RadPointModifierParallel_Kernel(srTSRWRadStructAccessData* pRadAccessData, void* pBufVars, T* tgt_obj, int xStart, int xFin, int zStart, int zFin) //HG27072024 Redesigned entire function
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

//	if (ix < RadAccessData.nx && iz < RadAccessData.nz)
//	{
//		srTEFieldPtrs EPtrs;
//		srTEXZ EXZ;
//		EXZ.z = RadAccessData.zStart + iz * RadAccessData.zStep;
//		EXZ.x = RadAccessData.xStart + ix * RadAccessData.xStep;
//
//		for (int ie = 0; ie < RadAccessData.ne; ie++) {
//			EXZ.e = RadAccessData.eStart + ie * RadAccessData.eStep;
//			EXZ.aux_offset = RadAccessData.ne * RadAccessData.nx * 2 * iz + RadAccessData.ne * 2 * ix + ie * 2;
//			if (RadAccessData.pBaseRadX != 0)
//			{
//				EPtrs.pExRe = RadAccessData.pBaseRadX + EXZ.aux_offset;
//				EPtrs.pExIm = EPtrs.pExRe + 1;
//			}
//			else
//			{
//				EPtrs.pExRe = 0;
//				EPtrs.pExIm = 0;
//			}
//			if (RadAccessData.pBaseRadZ != 0)
//			{
//				EPtrs.pEzRe = RadAccessData.pBaseRadZ + EXZ.aux_offset;
//				EPtrs.pEzIm = EPtrs.pEzRe + 1;
//			}
//			else
//			{
//				EPtrs.pEzRe = 0;
//				EPtrs.pEzIm = 0;
//			}
//
//			tgt_obj->RadPointModifierPortable(EXZ, EPtrs, pBufVars);
//		}
//	}
}

template<class T> void RadPointModifierParallelImpl_Launcher(srTSRWRadStructAccessData* pRadAccessData, srTSRWRadStructAccessData* pRadAccessData_dev, void* pBufVars_dev, T* local_copy, int xStart, int xFin, int zStart, int zFin, bool combined_e)
{
	const int bs = 256;
	dim3 blocks(pRadAccessData->ne, xFin - xStart, zFin - zStart);
	dim3 threads(1, 1, 1);
	if (combined_e) blocks.x = 1;
	if (blocks.x > 1) //HG30072024 Attempt to pick an ideal grid size
	{
		if (blocks.x >= bs)
			threads.x = bs;
		else
			threads.x = blocks.x;


		if (blocks.x % threads.x > 0) blocks.x = blocks.x / threads.x + 1;
		else blocks.x = blocks.x / threads.x;

		if (bs / threads.x > 1)
		{
			threads.y = bs / threads.x;
			if (blocks.y % threads.y > 0) blocks.y = blocks.y / threads.y + 1;
			else blocks.y = blocks.y / threads.y;
		}
	}
	else
	{
		if (bs / blocks.y > 1)
		{
			threads.y = blocks.y;
			blocks.y = 1;
			threads.z = bs / threads.y;
			if (blocks.z % threads.z > 0) blocks.z = blocks.z / threads.z + 1;
			else blocks.z = blocks.z / threads.z;
		}
		else
		{
			threads.y = bs;
			if (blocks.y % threads.y > 0) blocks.y = blocks.y / threads.y + 1;
			else blocks.y = blocks.y / threads.y;
		}
	}

	if (combined_e) //Treating all energy points in one thread may be more efficient for simpler propagators
		RadPointModifierParallel_Kernel<T, true> << <blocks, threads >> > (pRadAccessData_dev, pBufVars_dev, local_copy, xStart, xFin, zStart, zFin);
	else
		RadPointModifierParallel_Kernel<T, false> << <blocks, threads >> > (pRadAccessData_dev, pBufVars_dev, local_copy, xStart, xFin, zStart, zFin);
}

//template<class T> int RadPointModifierParallelImpl(srTSRWRadStructAccessData* pRadAccessData, void* pBufVars, long pBufVarsSz, T* tgt_obj, TGPUUsageArg* pGPU)
template<class T> int RadPointModifierParallelImpl(srTSRWRadStructAccessData* pRadAccessData, void* pBufVars, long pBufVarsSz, T* tgt_obj, TGPUUsageArg* pGPU, int *region_params=0, bool combined_e=false) //HG29072024
{
	//region_params[0] = xmin, region_params[1] = xmax, region_params[2] = zmin, region_params[3] = zmax, region_params[4] = '1' to skip specified region or '0' to process only specified region

	if (pRadAccessData->pBaseRadX != NULL)
	{
		pRadAccessData->pBaseRadX = (float*)CAuxGPU::ToDevice(pGPU, pRadAccessData->pBaseRadX, 2*pRadAccessData->ne*pRadAccessData->nx*pRadAccessData->nz*sizeof(float));
		CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadAccessData->pBaseRadX);
	}
	if (pRadAccessData->pBaseRadZ != NULL)
	{
		pRadAccessData->pBaseRadZ = (float*)CAuxGPU::ToDevice(pGPU, pRadAccessData->pBaseRadZ, 2*pRadAccessData->ne*pRadAccessData->nx*pRadAccessData->nz*sizeof(float));
		CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadAccessData->pBaseRadZ);
	}

    T* local_copy = (T*)CAuxGPU::ToDevice(pGPU, tgt_obj, sizeof(T));
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, local_copy);
    //cudaMalloc(&local_copy, sizeof(T));
    //cudaMemcpy(local_copy, tgt_obj, sizeof(T), cudaMemcpyHostToDevice);
	
	void* pBufVars_dev = NULL;
	if (pBufVarsSz > 0){
		pBufVars_dev = CAuxGPU::ToDevice(pGPU, pBufVars, pBufVarsSz);
		CAuxGPU::EnsureDeviceMemoryReady(pGPU, pBufVars_dev);
	}

	srTSRWRadStructAccessData* pRadAccessData_dev = NULL; //HG27072024 Make pRadAccessData also passed by reference
	pRadAccessData_dev = (srTSRWRadStructAccessData*)CAuxGPU::ToDevice(pGPU, pRadAccessData, sizeof(srTSRWRadStructAccessData));
	CAuxGPU::EnsureDeviceMemoryReady(pGPU, pRadAccessData_dev);
	
	int xStart = 0;
	int xFin = pRadAccessData->nx;
	int zStart = 0;
	int zFin = pRadAccessData->nz;

	//HG30072024 Allow for specifying a region to skip or to only process within the region, reduces extra operations for propagators like apertures and obstacles
	bool HandleInSingleLaunch = (region_params == 0);
	if (!HandleInSingleLaunch)
	{
		if (region_params[4] == 0)
		{
			xStart = region_params[0];
			xFin = region_params[1];
			zStart = region_params[2];
			zFin = region_params[3];
			HandleInSingleLaunch = true;
		}
	}

	if (HandleInSingleLaunch)
	{
		RadPointModifierParallelImpl_Launcher<T>(pRadAccessData, pRadAccessData_dev, pBufVars_dev, local_copy, xStart, xFin, zStart, zFin, combined_e);
	}
	else
	{
		//Have to split into 4 kernel launches to skip the specified region
		RadPointModifierParallelImpl_Launcher<T>(pRadAccessData, pRadAccessData_dev, pBufVars_dev, local_copy, 0, region_params[0], 0, pRadAccessData->nx, combined_e);
		RadPointModifierParallelImpl_Launcher<T>(pRadAccessData, pRadAccessData_dev, pBufVars_dev, local_copy, region_params[1], pRadAccessData->nx, 0, pRadAccessData->nx, combined_e);
		RadPointModifierParallelImpl_Launcher<T>(pRadAccessData, pRadAccessData_dev, pBufVars_dev, local_copy, region_params[0], region_params[1], region_params[3], pRadAccessData->nz, combined_e);
		RadPointModifierParallelImpl_Launcher<T>(pRadAccessData, pRadAccessData_dev, pBufVars_dev, local_copy, region_params[0], region_params[1], 0, region_params[2], combined_e);
	}
	
	//RadPointModifierParallel_Kernel<T> << <blocks, threads >> > (*pRadAccessData, pBufVars_dev, local_copy); //HG30072024 commented-out
    //cudaDeviceSynchronize();
    //cudaFreeAsync(local_copy, 0);
	CAuxGPU::ToHostAndFree(pGPU, pRadAccessData_dev, sizeof(srTSRWRadStructAccessData), true); //HG27072024
	if (pBufVarsSz > 0) CAuxGPU::ToHostAndFree(pGPU, pBufVars_dev, pBufVarsSz, true);
	CAuxGPU::ToHostAndFree(pGPU, local_copy, sizeof(T), true);

	CAuxGPU::MarkUpdated(pGPU, pRadAccessData->pBaseRadX, true, false);
	CAuxGPU::MarkUpdated(pGPU, pRadAccessData->pBaseRadZ, true, false);

//#ifndef _DEBUG //HG26022024 (commented-out)
	if (pRadAccessData->pBaseRadX != NULL)
		pRadAccessData->pBaseRadX = (float*)CAuxGPU::GetHostPtr(pGPU, pRadAccessData->pBaseRadX);
	if (pRadAccessData->pBaseRadZ != NULL)
		pRadAccessData->pBaseRadZ = (float*)CAuxGPU::GetHostPtr(pGPU, pRadAccessData->pBaseRadZ);
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